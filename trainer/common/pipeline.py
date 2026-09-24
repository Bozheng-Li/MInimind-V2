"""训练流水线：把重构前 8 个脚本 ``__main__`` 里的「九段式」流程收拢到一处。

    1. 初始化分布式 + 随机种子          -> common.runtime.init_runtime
    2. 建目录 / 组配置 / 查续训档        -> 此处
    3. 混合精度上下文                    -> common.runtime.init_runtime
    4. wandb                            -> common.logging.init_wandb
    5. 模型 / 数据 / 优化器              -> init_model + algos + 此处
    6. 从续训档恢复                      -> common.checkpoint.restore
    7. torch.compile / DDP              -> common.runtime.wrap_model
    8. epoch 训练循环                    -> common.runner.run_epochs
    9. 清理分布式                        -> common.checkpoint.finalize
"""
from __future__ import annotations

import os
from pathlib import Path

from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import init_model

from .checkpoint import (
    CHECKPOINT_DIR,
    finalize,
    load_resume,
    resolve_from_weight,
    resolve_weight_prefix,
    restore,
)
from .data import build_sampler, split_holdout
from .logging import init_wandb
from .metrics import MetricsCollector
from .registry import RunRecorder
from .runner import TrainContext, build_metric_columns, run_epochs
from .runtime import (
    build_optimizer,
    build_scaler,
    init_runtime,
    synchronize_module,
    wrap_model,
)

#: tokenizer 所在目录（仓库根的 model/）。用绝对路径，避免依赖当前工作目录 ——
#: 从仓库根还是从 trainer/ 启动都必须能找到。
TOKENIZER_DIR = str(Path(__file__).resolve().parents[2] / "model")


def _metrics_path(args) -> str:
    """逐 step 指标 CSV 的路径。

    显式给了 ``--metrics_path`` 就用它；否则放在 ``{save_dir}/{权重前缀}_metrics.csv``。
    每组实验一个文件，便于后续汇总对比。
    """
    explicit = getattr(args, "metrics_path", None)
    if explicit:
        return explicit
    return os.path.join(args.save_dir, f"{resolve_weight_prefix(args)}_metrics.csv")


def _tokens_per_step(args) -> int:
    """单步实际处理的 token 数（含 DDP 的所有 rank），用于算吞吐。"""
    import torch.distributed as dist
    world = dist.get_world_size() if dist.is_initialized() else 1
    return int(args.batch_size) * int(args.max_seq_len) * world


def run_training(args, algo_name: str, cfg=None, config_path: str = None):
    """跑一个完整的训练任务。

    ``algo_name`` 见 ``trainer.algos.available()``。
    ``cfg`` 是分阶段配置（``configs/``）合并后的 dict；给了它就按 YAML 决定模型结构，
    不给则退回「仅用命令行参数」的旧路径。

    全过程登记到 ``test/log/runs/<run_id>/meta.json``（见 :mod:`trainer.common.registry`），
    训练是否正常结束、用了什么架构与超参、吃了多少资源，界面上都能直接读出来。
    """
    from ..trainer_utils import Logger

    _validate_training_args(args, algo_name)

    # 实验登记：任何异常都要落到 meta 里，否则页面上会留下一个永远「运行中」的幽灵 run
    recorder = RunRecorder(args, algo_name, lm_config=None, cfg=cfg, config_path=config_path,
                           data_path=getattr(args, "data_path", None))
    recorder.attach_artifacts(metrics_csv=_metrics_path(args),
                              weight=os.path.join(args.save_dir, f"{resolve_weight_prefix(args)}"
                                                  f"_{getattr(args, 'hidden_size', 768)}"
                                                  f"{'_moe' if getattr(args, 'use_moe', 0) else ''}.pth"))
    try:
        return _run_training_inner(args, algo_name, cfg, recorder)
    except Exception as exc:  # noqa: BLE001 登记失败原因后原样抛出，绝不吞掉训练异常
        import traceback
        tb = traceback.format_exc()
        Logger(f"❌ 训练异常终止: {type(exc).__name__}: {exc}")
        recorder.note(tb[-4000:])
        recorder.finish(status="failed", error=f"{type(exc).__name__}: {exc}")
        finalize()
        raise


def _validate_training_args(args, algo_name: str) -> None:
    """在加载大模型前校验会改变算法语义的关键参数。"""
    if args.batch_size <= 0 or args.accumulation_steps <= 0:
        raise ValueError("batch_size 与 accumulation_steps 必须为正整数")
    if args.epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs 与 learning_rate 必须为正数")
    if args.num_workers < 0:
        raise ValueError("num_workers 不能为负数")
    if args.log_interval <= 0 or args.save_interval <= 0:
        raise ValueError("log_interval 与 save_interval 必须为正整数")
    if args.max_steps < 0:
        raise ValueError("max_steps 不能为负数")
    if args.grad_clip < 0:
        raise ValueError("grad_clip 不能为负数；0 表示不裁剪")
    if args.max_seq_len <= 1:
        raise ValueError("max_seq_len 必须大于 1")
    if args.dtype not in {"bfloat16", "float16"}:
        raise ValueError("dtype 仅支持 bfloat16 或 float16")
    preference = {"dpo", "ipo", "simpo", "cpo", "kto"}
    if algo_name in preference and args.beta <= 0:
        raise ValueError(f"{algo_name.upper()} 的 beta 必须大于 0")
    grouped = {"grpo", "dapo", "rloo", "agent"}
    if algo_name in grouped and args.num_generations < 2:
        raise ValueError(f"{algo_name.upper()} 依赖组内 baseline，num_generations 必须 >= 2")
    online = {"grpo", "dapo", "rloo", "ppo", "agent"}
    if algo_name in online and args.max_gen_len <= 0:
        raise ValueError("在线 RL 的 max_gen_len 必须为正整数")
    if algo_name in online and args.rollout_temperature <= 0:
        raise ValueError("在线 RL 的 rollout_temperature 必须大于 0")
    if algo_name in online and not 0 <= args.epsilon < 1:
        raise ValueError("epsilon 必须位于 [0, 1)")
    if algo_name in online and args.epsilon_high <= 0:
        raise ValueError("epsilon_high 必须大于 0")
    if algo_name == "dapo" and args.loss_type != "grpo":
        raise ValueError("DAPO 必须使用 --loss_type grpo，CISPO 不具备 Clip-Higher 语义")
    if algo_name == "dapo" and (args.dapo_epsilon_high <= 0
                                  or args.dapo_max_resample <= 0
                                  or args.dapo_reward_std_threshold < 0
                                  or args.dapo_overlong_buffer <= 0
                                  or args.dapo_overlong_penalty < 0
                                  or args.dapo_kl_coef < 0):
        raise ValueError("DAPO 的裁剪、重采样、阈值、超长惩罚与 KL 参数必须为合法非负值")
    if algo_name == "ppo" and (args.mini_batch_size <= 0
                                or args.ppo_update_iters <= 0
                                or not 0 <= args.clip_epsilon < 1
                                or args.cliprange_value < 0
                                or args.vf_coef < 0
                                or args.kl_coef < 0
                                or not 0 <= args.gamma <= 1
                                or not 0 <= args.lam <= 1
                                or args.early_stop_kl <= 0):
        raise ValueError("PPO 的 minibatch、更新轮数、clip、GAE、KL 参数不合法")
    if algo_name == "agent" and args.max_total_len <= 1:
        raise ValueError("Agent 的 max_total_len 必须大于 1")
    if not 0 <= args.preference_label_smoothing < 0.5:
        raise ValueError("preference_label_smoothing 必须位于 [0, 0.5)")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("lora_dropout 必须位于 [0, 1)")
    if algo_name in {"lora", "qlora"} and (args.lora_rank <= 0 or args.lora_alpha <= 0):
        raise ValueError("LoRA rank 与 alpha 必须为正数")
    if algo_name == "cpo" and args.cpo_alpha < 0:
        raise ValueError("CPO 的 cpo_alpha 不能为负数")
    if algo_name == "orpo" and args.orpo_lambda < 0:
        raise ValueError("ORPO 的 orpo_lambda 不能为负数")
    if algo_name == "kto" and (args.kto_desirable_weight <= 0
                                or args.kto_undesirable_weight <= 0):
        raise ValueError("KTO 的 desirable/undesirable 权重必须为正数")
    if algo_name == "distill" and not 0 <= args.alpha <= 1:
        raise ValueError("蒸馏 alpha 必须位于 [0, 1]")
    if algo_name == "distill" and args.temperature <= 0:
        raise ValueError("蒸馏 temperature 必须大于 0")
    if algo_name == "ppo" and args.critic_learning_rate <= 0:
        raise ValueError("PPO 的 critic_learning_rate 必须大于 0")


def _run_training_inner(args, algo_name, cfg, recorder: RunRecorder):
    """真正的训练流程。异常由 :func:`run_training` 统一登记。"""
    from ..algos import get_algorithm

    # --- 1 + 3. 运行时（分布式 / 种子 / 精度）---
    runtime = init_runtime(args)

    # --- 2. 目录 / 配置 / 续训档 ---
    os.makedirs(args.save_dir, exist_ok=True)
    if cfg is not None:
        from configs.loader import build_lm_config
        lm_config = build_lm_config(args, cfg)
    else:
        lm_config = MiniMindConfig(hidden_size=args.hidden_size,
                                   num_hidden_layers=args.num_hidden_layers,
                                   use_moe=bool(args.use_moe))
    if algo_name in {"grpo", "dapo", "rloo", "ppo", "agent"} and lm_config.dropout > 0:
        raise ValueError(
            "在线 RL 要求模型 dropout=0：rollout 使用 eval，而训练前向的 dropout 会让"
            "未更新策略的 importance ratio 也偏离 1"
        )
    recorder.attach_lm_config(lm_config)
    ckp_data = load_resume(args, lm_config, resolve_weight_prefix(args))

    # 架构信息：既用于指标列清单（逐层/逐专家），也用于日志头的可复现快照
    arch = lm_config.to_arch_config()
    arch_n_layers = int(arch.model["num_hidden_layers"])
    _ffn = arch.component("feedforward")
    arch_n_experts = (int(_ffn.get("num_experts", 0))
                      if str(arch.type_of("feedforward")).startswith("moe") else 0)
    if runtime.is_main:
        _log_run_header(args, arch, cfg, recorder=recorder)

    # --- 4. wandb ---
    run_name = (f"{args.wandb_project}-Epoch-{args.epochs}"
                f"-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}")
    wandb = init_wandb(args, ckp_data, run_name=run_name)

    # --- 5. 模型 / 数据 / 优化器 ---
    # 基座前缀：蒸馏脚本用 from_student_weight，其余用 from_weight
    model, tokenizer = init_model(lm_config, resolve_from_weight(args),
                                  tokenizer_path=TOKENIZER_DIR,
                                  save_dir=args.save_dir, device=runtime.device)
    # 从零初始化时各 rank 的随机种子不同。必须在复制 reference/critic 之前以
    # rank 0 为准同步；仅依赖后面的 DDP 构造会留下“actor 已同步、reference 未同步”
    # 的隐蔽 KL 偏差。加载同一权重文件时天然一致，无需额外广播整模型。
    if resolve_from_weight(args) == "none":
        synchronize_module(model)
    algo = get_algorithm(algo_name)(args, lm_config, tokenizer, runtime)
    algo.model = model
    recorder.set_model(model)

    # 额外模型（ref / teacher / critic / reward）紧跟在主模型之后构造，与原实现顺序一致
    for attr, extra_model in algo.extra_models().items():
        setattr(algo, attr, extra_model)

    # 模型改造（如 LoRA 挂载与冻结）必须早于「收集参数建优化器」
    model = algo.configure_model(model)
    algo.model = model

    dataset = algo.build_dataset()
    # 留出固定尾部的若干条作验证集，用于观察是否过拟合
    dataset, val_ds = split_holdout(dataset, int(getattr(args, "val_samples", 0) or 0))
    sampler = build_sampler(dataset)
    scaler = build_scaler(args.dtype)
    optimizer = algo.build_optimizer(model) or build_optimizer(model, args.learning_rate)

    ctx = TrainContext(args=args, lm_config=lm_config, runtime=runtime, model=model,
                       optimizer=optimizer, scaler=scaler, dataset=dataset,
                       sampler=sampler, wandb=wandb,
                       save_weight=resolve_weight_prefix(args),
                       metrics_path=_metrics_path(args),
                       val_dataset=val_ds,
                       metric_columns=build_metric_columns(
                           n_layers=int(getattr(args, "num_hidden_layers", 0) or 0)
                           or arch_n_layers,
                           n_experts=arch_n_experts,
                           algo=algo_name,
                       ),
                       tokens_per_step=_tokens_per_step(args))

    # --- 6. 恢复续训档 ---
    start_epoch, start_step = restore(ctx, ckp_data, algorithm=algo)

    # --- 7. compile/DDP ---
    ctx.model = wrap_model(model, runtime, args.use_compile)
    algo.model = ctx.model      # 让算法的 compute_loss 用到（可能被包装过的）训练模型

    # 富指标采集器（拿到包装后的模型，内部会自动剥壳访问子模块）
    if getattr(args, "metrics_detail", 1):
        ctx.metrics = MetricsCollector(ctx.model, runtime, arch.model["hidden_size"])

    # --- 8. 训练 ---
    run_epochs(ctx, algo, start_epoch, start_step)

    # --- 9. 清理 ---
    if ctx.metrics is not None:
        ctx.metrics.close()
    finalize()

    # 登记收尾：把最终产物与结果摘要写回 meta.json
    recorder.attach_artifacts(
        weight=os.path.join(args.save_dir, f"{resolve_weight_prefix(args)}_"
                           f"{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"),
        checkpoints=CHECKPOINT_DIR,
    )
    recorder.finish(status="ok")
    return ctx.model


def _log_run_header(args, arch, cfg, recorder: RunRecorder = None):
    """把「这次到底跑的什么」写进日志开头 —— 保证事后可复现。

    包含：架构摘要、数据与超参、验证集设置、git 版本、torch/CUDA 环境。
    复现一次实验时先看这几行就够了。
    """
    from ..trainer_utils import Logger
    Logger("=" * 72)
    Logger(f"[run] {getattr(args, 'run_name', '') or args.save_weight}"
           f" | algo={getattr(args, 'algo', '?')} | seed={getattr(args, 'seed', '?')}")
    Logger(f"[arch] {arch.summary()}")
    # RL 的真实有效 batch 还要乘 num_generations（一份 prompt 采多条），头里单独说明
    algo = str(getattr(args, "algo", "") or "")
    online_rl = {"grpo", "dapo", "rloo", "ppo", "agent"}
    grouped_online = {"grpo", "dapo", "rloo", "agent"}
    preference = {"dpo", "ipo", "simpo", "cpo", "orpo", "kto"}
    # PPO 每个 prompt 固定只采一条；num_generations 只用于组相对算法和 Agent。
    n_gen = int(getattr(args, "num_generations", 1) or 1) if algo in grouped_online else 1
    batch_note = (f" | batch={args.batch_size} × accum={args.accumulation_steps}"
                  + (f" × gen={n_gen}" if n_gen > 1 else ""))
    Logger(f"[data] {getattr(args, 'data_path', '?')} | max_seq_len={args.max_seq_len}{batch_note}")
    Logger(f"[opt] lr={args.learning_rate} grad_clip={args.grad_clip} "
           f"epochs={args.epochs} max_steps={getattr(args, 'max_steps', 0) or '全部'}")
    if algo in online_rl:
        # 五个在线算法自建 train_epoch，不调用 evaluate_loss。
        Logger("[val] RL 自建循环不计算验证 loss（曲线看 reward / KL_ref）")
    else:
        Logger(f"[val] 每 {getattr(args, 'val_interval', 0) or 0} 步，留出 "
               f"{getattr(args, 'val_samples', 0) or 0} 条")
    # RL 的可复现性关键项：奖励从哪来、rollout 怎么做的、KL/裁剪怎么定的
    if algo in online_rl:
        Logger(f"[rl]  rollout={getattr(args, 'rollout_engine', '?')} "
               f"base={getattr(args, 'from_weight', '?')} "
               f"reward_model={getattr(args, 'reward_model_path', '?')}")
        Logger(f"[rl]  num_generations={n_gen} max_gen_len={getattr(args, 'max_gen_len', '?')} "
               f"thinking_ratio={getattr(args, 'thinking_ratio', '?')}")
    if algo in {"grpo", "rloo", "agent"}:
        Logger(f"[rl]  beta(KL)={getattr(args, 'beta', '?')} "
               f"loss_type={getattr(args, 'loss_type', '?')} "
               f"epsilon={getattr(args, 'epsilon', '?')}/{getattr(args, 'epsilon_high', '?')}")
    if algo == "dapo":
        Logger(f"[rl]  clip={getattr(args, 'epsilon', '?')}/"
               f"{getattr(args, 'dapo_epsilon_high', '?')} "
               f"resample={getattr(args, 'dapo_max_resample', '?')} "
               f"kl_coef={getattr(args, 'dapo_kl_coef', '?')}")
    if algo == "agent":
        Logger(f"[rl]  max_total_len={getattr(args, 'max_total_len', '?')} max_turns=3(硬编码)")
    if algo == "ppo":
        Logger(f"[rl]  clip_epsilon={getattr(args, 'clip_epsilon', '?')} "
               f"vf_coef={getattr(args, 'vf_coef', '?')} kl_coef={getattr(args, 'kl_coef', '?')} "
               f"gamma={getattr(args, 'gamma', '?')} lam={getattr(args, 'lam', '?')} "
               f"cliprange_value={getattr(args, 'cliprange_value', '?')} "
               f"ppo_update_iters={getattr(args, 'ppo_update_iters', '?')} "
               f"early_stop_kl={getattr(args, 'early_stop_kl', '?')} "
               f"mini_batch_size={getattr(args, 'mini_batch_size', '?')} "
               f"critic_lr={getattr(args, 'critic_learning_rate', '?')}")
    if algo in preference:
        Logger(f"[rl]  beta={getattr(args, 'beta', '?')} base={getattr(args, 'from_weight', '?')}")
    try:
        import subprocess
        git = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                      cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL
                                      ).decode().strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'],
                                        cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL
                                        ).decode().strip()
        Logger(f"[code] git {git}{' (有未提交改动)' if dirty else ''}")
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch as _t
        Logger(f"[env] torch {_t.__version__} | cuda {_t.version.cuda} | "
               f"GPU {_t.cuda.get_device_name(0) if _t.cuda.is_available() else 'cpu'}")
    except Exception:  # noqa: BLE001
        pass
    Logger("=" * 72)
