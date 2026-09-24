"""通用训练循环。

把旧训练脚本里重复的 ``train_epoch`` 骨架抽到这里：

    取 batch → (autocast) 前向算 loss → 反向 → 每 accumulation_steps 步做
    梯度裁剪+优化器更新 → 定期日志 → 定期存盘 → 循环末尾 flush 残余梯度

**顺序与原实现严格一致**（学习率写入 → autocast → loss 除以累积步数 →
backward → 条件更新），否则「同一配置跑出同一 loss」的兼容性保证会失效。
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass

import torch

from ..trainer_utils import Logger, get_lr, is_main_process
from .checkpoint import CHECKPOINT_DIR as _CKPT_DIR
from .checkpoint import save_resume
from .data import build_loader
from .logging import log_metrics
from .metrics import evaluate_loss

#: 续训档目录 —— 直接复用 ``checkpoint.py`` 的定义（同一环境变量、同一默认值），
#: 避免两处漂移。默认 <repo>/test/checkpoints，可用 MINIMIND_ARTIFACT_ROOT 覆盖。
CHECKPOINT_DIR = _CKPT_DIR


@dataclass
class TrainContext:
    """跑一次训练所需的全部对象。"""

    args: object
    lm_config: object
    runtime: object
    model: object
    optimizer: object
    scaler: object
    dataset: object
    sampler: object = None
    wandb: object = None
    save_weight: str = "model"

    #: 逐 step 指标写到这里（CSV）。为 None 时不写。
    metrics_path: str = None
    #: 富指标采集器（见 common/metrics.py）。为 None 时只记基础字段。
    metrics: object = None
    #: 留出验证集（用于定期算 val loss）。为 None 时跳过验证。
    val_dataset: object = None
    #: 指标 CSV 的列清单（由 build_metric_columns 按模型结构生成）
    metric_columns: object = None
    #: 最近一次记录的梯度范数与实际 batch token 数
    last_grad_norm: float = 0.0
    tokens_per_step: int = 0

    # 从 args 透出的常用项，省去到处 args.xxx
    @property
    def device(self):
        return self.runtime.device

    @property
    def autocast_ctx(self):
        return self.runtime.autocast_ctx


#: 逐 step 指标的基础列（其余列由 build_metric_columns 按模型结构补全）
BASE_METRIC_COLUMNS = (
    "step", "loss", "logits_loss", "aux_loss", "lr",
    "grad_norm", "grad_norm_attn", "grad_norm_ffn", "grad_norm_embed", "grad_norm_norm",
    "grad_norm_sum_groups",
    "weight_norm", "weight_norm_attn", "weight_norm_ffn", "weight_norm_embed",
    "update_ratio_attn", "update_ratio_ffn", "update_ratio_embed", "update_ratio_norm",
    "val_loss",
    "tokens_per_sec", "elapsed_sec",
    "gpu_mem_mb", "gpu_mem_alloc_mb", "gpu_mem_reserved_mb", "gpu_mem_peak_torch_mb",
    "gpu_mem_peak_mb", "gpu_util_mean", "gpu_util_min",
    "ram_used_mb", "ram_pct",
    # MoE 健康度
    "moe_load_mean", "moe_load_cv", "moe_load_maxmin", "moe_dead_experts",
    "moe_entropy_norm", "moe_bias_std", "moe_bias_absmean", "moe_bias_nonzero",
    "shared_share",
    # 注意力 / 组件中间量
    "q_rms", "k_rms", "v_rms", "out_rms",
    "gate_mean", "gate_std", "gate_sat_lo", "gate_sat_hi",
    "routed_rms", "shared_rms", "hidden_rms",
)


def build_metric_columns(n_layers: int = 0, n_experts: int = 0, per_layer: bool = True,
                         algo: str = None):
    """按模型结构生成完整的指标列清单。

    必须**预先声明**（而不是发现新字段再扩列）—— ``DictWriter`` 遇到表头外的
    字段会直接报错，而追加模式下改表头要重写整个文件。

    ``algo`` 给定时额外补上该 RL 算法的专属列（reward / kl_ref / pass_rate ...）。
    全部 RL 算法共用一张列清单会过宽；按算法裁剪后，同一算法跨 run 仍然可比。
    """
    cols = list(BASE_METRIC_COLUMNS)
    if n_experts:
        cols += [f"moe_load_e{i}" for i in range(n_experts)]
    if per_layer:
        for name in ("q_rms", "k_rms", "v_rms", "out_rms", "gate_mean", "gate_std",
                     "hidden_rms", "routed_rms", "shared_rms"):
            cols += [f"{name}_L{i}" for i in range(n_layers)]
    cols += list(rl_metric_columns(algo))
    return cols


#: 五个 online RL（GRPO / DAPO / RLOO / PPO / Agent）共有的列。
_RL_ONLINE_COLUMNS = (
    "reward", "reward_std",
    # reward 分项（rl.common.calculate_rewards 的拆解，用来诊断是哪一项在驱动奖励）
    "rew_len", "rew_think_len", "rew_think_close", "rew_rep", "rew_rm",
    "policy_loss", "kl_ref", "perplexity",
    "advantages_mean", "advantages_std", "adv_zero_frac",
    "avg_response_len", "gen_tokens", "eos_rate", "trunc_rate",
)

#: 各 RL 算法的专属列。缺的列在 CSV 里留空，不影响别的算法。
_RL_ALGO_COLUMNS = {
    "grpo": ("group_reward_std", "group_reward_zero_std", "ratio_mean", "clipfrac"),
    "dapo": ("group_reward_std", "group_reward_zero_std", "ratio_mean", "clipfrac"),
    "rloo": ("group_reward_std", "group_reward_zero_std", "ratio_mean", "clipfrac"),
    "ppo": ("critic_loss", "value_loss", "approx_kl", "clipfrac",
            "adv_raw_mean", "adv_raw_std", "kl_early_stop", "actor_lr", "critic_lr"),
    "agent": ("group_reward_std", "pass_rate", "unfinished_rate",
              "tool_calls_mean", "valid_call_rate", "tool_gap_mean", "turns_mean"),
    # dpo_loss 是历史兼容别名；新代码统一使用 preference_loss。
    "dpo": ("preference_loss", "dpo_loss", "reward_margin", "preference_acc"),
    "ipo": ("preference_loss", "reward_margin", "preference_acc"),
    "simpo": ("preference_loss", "reward_margin", "preference_acc"),
    "cpo": ("preference_loss", "reward_margin", "preference_acc"),
    "orpo": ("preference_loss", "reward_margin", "preference_acc"),
    "kto": ("preference_loss", "reward_margin", "preference_acc"),
}


def rl_metric_columns(algo: str):
    """某个 RL 算法的专属指标列（非 RL 算法返回空）。

    离线偏好算法没有 rollout / reward 链路，只声明偏好损失、间隔与准确率。
    """
    if algo not in _RL_ALGO_COLUMNS:
        return ()
    offline = {"dpo", "ipo", "simpo", "cpo", "orpo", "kto"}
    shared = () if algo in offline else _RL_ONLINE_COLUMNS
    return shared + _RL_ALGO_COLUMNS[algo]


def _append_metrics(ctx, row: dict) -> None:
    """把一步的指标追加到 CSV（每次打开追加，进程被杀也不丢已写部分）。

    ⚠️ 只在主进程写：DDP 下 4 个 rank 都会走到这里，不加判断会写出 4 份重复行
    （每行内容不同、step 相同），把指标文件污染成 4 倍大。

    ⚠️ 表头不一致时必须**归档旧文件另起新文件**：``DictWriter`` 配
    ``extrasaction="ignore"`` 会**静默丢弃**表头之外的列。如果沿用上一轮残留的
    旧表头（例如从 9 列升级到 139 列后），新增的富指标会被全部丢掉，
    而日志看起来一切正常 —— 实测踩过这个坑。
    """
    if not ctx.metrics_path or not is_main_process():
        return
    parent = os.path.dirname(ctx.metrics_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    cols = list(ctx.metric_columns or row.keys())

    if os.path.exists(ctx.metrics_path):
        with open(ctx.metrics_path, encoding="utf-8") as fh:
            existing = fh.readline().strip().split(",")
        if existing != cols:
            backup = ctx.metrics_path + ".old"
            os.replace(ctx.metrics_path, backup)
            Logger(f'[metrics] 表头与当前列清单不一致（旧 {len(existing)} 列 / 新 {len(cols)} 列），'
                   f'旧文件已归档为 {os.path.basename(backup)}')

    is_new = not os.path.exists(ctx.metrics_path)
    with open(ctx.metrics_path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in cols})


def _gpu_mem_mb(device) -> float:
    try:
        if "cuda" in str(device):
            return torch.cuda.max_memory_allocated() / 1024 ** 2
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def _log_line(ctx, algorithm, epoch, step, iters, start_step, loss_scaled, aux_loss, spend_time,
              extra: dict = None):
    """日志由算法自己格式化 —— 各算法重构前的日志格式并不相同。

    同时把这一步的完整指标追加到 ``ctx.metrics_path``（若已设置）。
    ``extra`` 是富采集器输出的组件级指标（QKV / 门控 / MoE / 系统），一并落盘。
    """
    lr = ctx.optimizer.param_groups[-1]['lr']
    line, metrics = algorithm.format_log(
        epoch=epoch, step=step, iters=iters, start_step=start_step,
        loss_scaled=loss_scaled, aux_loss=aux_loss, lr=lr, spend_time=spend_time,
    )
    done = max(step - start_step, 1)
    tps = ctx.tokens_per_step * done / max(spend_time, 1e-9)

    # stdout 只补最关键的几项，避免刷屏；完整字段在 CSV 里
    line += (f' | grad_norm: {ctx.last_grad_norm:.3f}, tokens/s: {tps:,.0f}')
    if extra:
        if 'val_loss' in extra:
            line += f", val_loss: {extra['val_loss']:.4f}"
        if 'moe_load_maxmin' in extra:
            line += f", moe_load_max/min: {extra['moe_load_maxmin']:.1f}"
    line += f', gpu_mem: {_gpu_mem_mb(ctx.device):,.0f}MB'
    Logger(line)
    log_metrics(ctx.wandb, metrics)

    row = {
        "step": step,
        "loss": metrics.get("loss", ""),
        "logits_loss": metrics.get("logits_loss", ""),
        "aux_loss": metrics.get("aux_loss", ""),
        "lr": lr,
        "grad_norm": ctx.last_grad_norm,
        "tokens_per_sec": round(tps, 1),
        "elapsed_sec": round(spend_time, 2),
        "gpu_mem_mb": round(_gpu_mem_mb(ctx.device), 1),
    }
    # 算法自己在 format_log 里算出来的标量也要落盘（如 preference_loss / preference_acc）；
    # 没在列清单里声明的键会被 DictWriter 忽略，所以这里放心全带上
    row.update({k: v for k, v in metrics.items()
                if k not in row and isinstance(v, (int, float))})
    if extra:
        row.update(extra)
    _append_metrics(ctx, row)


def record_rl_step(ctx, *, epoch, step, iters, start_step, spend_time, metrics, extra=None):
    """RL 算法自建循环里的逐 step 记录（CSV + 富指标）。

    GRPO / PPO / Agent 覆盖了 ``train_epoch``，不走 ``run_one_epoch``，因此拿不到
    ``_log_line``。但它们必须和 SFT / 预训练**写同一张表**，否则跨算法没法并排看 ——
    这个函数就是把 ``_log_line`` 的落盘那一半单独拿出来复用。

    ``metrics`` 是算法自己算出来的指标字典（reward / kl_ref / pass_rate ...），
    列清单由 ``build_metric_columns(algo=...)`` 预先声明，多余的键会被忽略。

    与 SFT 的两个口径差异（写在报告里，不要直接和 SFT 的吞吐并排比）：
      - ``tokens_per_sec`` 是**生成**吞吐（rollout 产出的 token / 本步墙钟），
        不是前向吞吐 —— RL 的墙钟大头在生成；
      - ``gen_tokens`` 给出本步实际生成的 token 数，便于换算。
    """
    if ctx.metrics is not None:
        lr_now = ctx.optimizer.param_groups[-1]['lr']
        ctx.metrics.set_step_info(lr_now)
        snap = ctx.metrics.snapshot()
        extra = {**(extra or {}), **snap}
    lr = ctx.optimizer.param_groups[-1]['lr']
    row = {
        "step": step,
        "lr": lr,
        "grad_norm": ctx.last_grad_norm,
        "elapsed_sec": round(spend_time, 2),
        "gpu_mem_mb": round(_gpu_mem_mb(ctx.device), 1),
    }
    gen_tokens = (metrics or {}).get("gen_tokens")
    if gen_tokens:
        # RL 的吞吐按生成量算：一步里前向很多次、优化一次，按 tokens_per_step 算会离谱。
        # 用**累计生成量 ÷ 累计耗时**，与 runner._log_line 里 tps = 累计token/累计时间 同口径
        # （早期版本写成「本步生成量 ÷ 累计耗时」，少乘了步数，尾部会小到 0.2 tok/s）。
        ctx._rl_gen_tokens_total = getattr(ctx, "_rl_gen_tokens_total", 0) + int(gen_tokens)
        row["tokens_per_sec"] = round(ctx._rl_gen_tokens_total / max(spend_time, 1e-9), 1)
    row.update({k: v for k, v in (metrics or {}).items() if v is not None})
    if extra:
        row.update(extra)
    _append_metrics(ctx, row)


#: 验证集 DataLoader 缓存（一个进程只训一个任务）
_VAL_LOADER = None


def _get_val_loader(ctx):
    global _VAL_LOADER
    if _VAL_LOADER is None:
        from torch.utils.data import DataLoader
        nw = max(1, min(2, int(getattr(ctx.args, "num_workers", 2))))
        _VAL_LOADER = DataLoader(ctx.val_dataset, batch_size=ctx.args.batch_size,
                                 shuffle=False, num_workers=nw, pin_memory=True)
    return _VAL_LOADER


def _save(ctx, algorithm, epoch, step):
    """定期存盘：推理权重 + 续训档（命名与字段与重构前完全一致）。"""
    ctx.model.eval()
    algorithm.save_weights(ctx, epoch, step)          # 保存方式由算法决定（LoRA 只存分支）
    save_resume(ctx.lm_config, algorithm.weight_prefix(), ctx.model, ctx.optimizer,
                scaler=ctx.scaler, epoch=epoch, step=step, wandb=ctx.wandb,
                save_dir=CHECKPOINT_DIR, **algorithm.checkpoint_extra())
    ctx.model.train()
    torch.cuda.empty_cache()


def run_one_epoch(ctx: TrainContext, algorithm, epoch: int, loader, iters: int, start_step: int = 0) -> bool:
    """跑一个 epoch。结构对应重构前的 ``train_epoch``。

    返回 ``True`` 表示触达了 ``--max_steps`` 上限，应当停止后续 epoch。
    """
    start_time = time.time()
    max_steps = getattr(ctx.args, "max_steps", 0) or 0
    hit_limit = False
    pending_micro_steps = 0
    save_requested = False
    # 固定预算模式下，学习率按 max_steps 走余弦衰减（而非整个 epoch 长度），
    # 否则冲刺跑几乎不衰减，不能代表真实训练
    schedule_total = max_steps if max_steps else ctx.args.epochs * iters

    def optimizer_step(micro_steps: int) -> None:
        """提交一组已经累积的梯度。

        每个 micro-batch 的 loss 都除以完整的 ``accumulation_steps``。若 epoch
        尾部或 ``max_steps`` 只剩不足一组的样本，需要把梯度乘回
        ``accumulation_steps / micro_steps``，才能得到“对实际样本求平均”的梯度。
        修正必须发生在记录梯度范数和裁剪之前，否则日志与真正更新的梯度口径不同。
        """
        ctx.scaler.unscale_(ctx.optimizer)
        params = list(algorithm.clip_parameters(ctx.model))
        correction = ctx.args.accumulation_steps / max(micro_steps, 1)
        if correction != 1.0:
            for parameter in params:
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        if ctx.metrics is not None:
            ctx.metrics.record_grads()
        max_norm = ctx.args.grad_clip if ctx.args.grad_clip > 0 else float("inf")
        ctx.last_grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm))
        ctx.scaler.step(ctx.optimizer)
        ctx.scaler.update()
        ctx.optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=start_step + 1):
        lr = get_lr(epoch * iters + step, schedule_total, ctx.args.learning_rate)
        for param_group in ctx.optimizer.param_groups:
            param_group['lr'] = lr

        with ctx.autocast_ctx:
            loss, aux_loss = algorithm.compute_loss(batch)   # loss 已除以 accumulation_steps

        # 富指标：每步累积 MoE 负载等（本地累积，到 log 点再跨卡汇总）
        if ctx.metrics is not None:
            ctx.metrics.record_step()

        ctx.scaler.scale(loss).backward()
        pending_micro_steps += 1

        if pending_micro_steps == ctx.args.accumulation_steps:
            optimizer_step(pending_micro_steps)
            pending_micro_steps = 0

        # ``step`` 是 epoch 内步数；``max_steps`` 是跨 epoch 的全局预算。
        # 先判断终止，再提交尾部梯度和保存，保证 checkpoint 精确包含最后一次更新。
        global_step = epoch * iters + step
        hit_limit = bool(max_steps and global_step >= max_steps)
        terminal = step == iters or hit_limit
        if terminal and pending_micro_steps:
            optimizer_step(pending_micro_steps)
            pending_micro_steps = 0

        if step % ctx.args.log_interval == 0 or terminal:
            extra = {}
            if ctx.metrics is not None:
                ctx.metrics.set_step_info(ctx.optimizer.param_groups[-1]['lr'])
                extra = ctx.metrics.snapshot()
            # 定期在留出集上算验证 loss（区分「还在学」与「过拟合」）
            val_interval = int(getattr(ctx.args, "val_interval", 0) or 0)
            if (ctx.val_dataset is not None and val_interval
                    and (step % val_interval == 0 or terminal)):
                extra['val_loss'] = round(evaluate_loss(
                    ctx.model, _get_val_loader(ctx), ctx.device, ctx.autocast_ctx,
                    max_batches=int(getattr(ctx.args, "val_batches", 16)),
                    algorithm=algorithm,
                ), 4)
            _log_line(ctx, algorithm, epoch, step, iters, start_step,
                      loss.item(), aux_loss, time.time() - start_time, extra)

        # 周期保存若落在梯度累积组中间，延迟到下一次 optimizer.step；否则续训会
        # 丢掉尚未写入 optimizer state 的半组梯度。终止点已经在上面强制 flush。
        save_requested = save_requested or step % ctx.args.save_interval == 0
        if (terminal or (save_requested and pending_micro_steps == 0)) and is_main_process():
            _save(ctx, algorithm, epoch, step)
            save_requested = False

        del batch, loss

        # 固定预算模式：到步数上限就停（用于架构对比的等 token 预算实验）
        if hit_limit:
            Logger(f'已达到 max_steps={max_steps}，停止训练')
            break

    return hit_limit


def run_epochs(ctx: TrainContext, algorithm, start_epoch: int = 0, start_step: int = 0):
    """跨 epoch 的主循环。"""
    for epoch in range(start_epoch, ctx.args.epochs):
        loader, skip = build_loader(ctx.dataset, ctx.args, epoch, sampler=ctx.sampler,
                                   start_step=start_step, start_epoch=start_epoch)
        algorithm.on_epoch_start(loader, len(loader) + skip)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{ctx.args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            hit_limit = algorithm.train_epoch(ctx, epoch, loader, len(loader) + skip, start_step)
        else:
            hit_limit = algorithm.train_epoch(ctx, epoch, loader, len(loader), 0)
        if hit_limit:
            break


def run_batch_epoch(ctx: TrainContext, algorithm, epoch, loader, iters, start_step=0) -> bool:
    """默认的 ``train_epoch``：每个 batch 一次前向 + 一次反向。

    算法只需实现 ``compute_loss``；RL 类算法可以覆盖 ``train_epoch`` 自建循环。
    """
    return run_one_epoch(ctx, algorithm, epoch, loader, iters, start_step)
