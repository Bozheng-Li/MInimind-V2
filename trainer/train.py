"""MiniMind 统一训练入口 —— 可插拔架构 × 可插拔算法。

历史 ``train_*.py`` 现在是兼容入口，统一转发到本文件；所有算法实现只保留在
``trainer/algos/`` 一处，把「模型结构」和「训练算法」都变成可切换参数：

    # 换架构：改 --config 指向的 YAML（不同 YAML 可以只差一个槽位）
    python trainer/train.py --algo sft --config configs/base.yaml
    python trainer/train.py --algo sft --config configs/mla.yaml

    # 换算法：同一个配置，不同 --algo
    python trainer/train.py --algo sft   --config configs/base.yaml
    python trainer/train.py --algo dpo   --config configs/base.yaml
    python trainer/train.py --algo grpo  --config configs/base.yaml

配置只提供**默认值**，命令行参数永远优先，因此下面这些也成立：

    python trainer/train.py --algo pretrain --epochs 3 --batch_size 16

不带 ``--config`` 时会按算法自动选择阶段配置：
``pretrain`` -> ``configs/pretrain.yaml``；``sft/lora/distill`` -> ``configs/sft.yaml``；
``dpo/ipo/simpo/cpo/orpo/kto/grpo/dapo/rloo/ppo/agent`` -> ``configs/rl.yaml``。

多卡：``torchrun --nproc_per_node N trainer/train.py --algo grpo``
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import warnings

import torch

from configs import apply_to_parser, stage_of
from trainer.algos import available as available_algos
from trainer.common import run_training

warnings.filterwarnings('ignore')


def build_parser():
    p = argparse.ArgumentParser(
        description="MiniMind 统一训练入口（可插拔架构 + 可插拔算法）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------- 入口选择 ----------------
    p.add_argument("--algo", required=True, choices=available_algos(),
                   help="训练算法")
    p.add_argument("--config", default=None,
                   help="配置文件的路径（不给则按 algo 自动选择对应阶段配置）")

    # ---------------- 通用 ----------------
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_dir", type=str, default="../test/out")
    p.add_argument("--save_weight", type=str, default="experiment")
    p.add_argument("--from_weight", type=str, default="none")
    p.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="MiniMind-Experiment")
    p.add_argument("--use_compile", type=int, default=0, choices=[0, 1])

    # ---------------- 数据 ----------------
    p.add_argument("--data_path", type=str, default=None,
                   help="训练数据路径（默认由配置决定）；配比模式下它是语料目录")
    p.add_argument("--data_mix", type=dict, default=None,
                   help="预训练配比 {文件名: 权重}（由配置的 data.mix 注入，一般不手填）")
    p.add_argument("--max_seq_len", type=int, default=768)

    # ---------------- 模型结构（也可完全交给 --config）----------------
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--use_moe", type=int, default=0, choices=[0, 1],
                   help="1 = 强制切到 MoE；0 = 沿用配置里 feedforward.type 的选择")

    # ---------------- 训练超参 ----------------
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=0,
                   help="步数上限（0=不限）。用于等 token 预算的架构对比实验：跑满即停，不看 epoch")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--critic_learning_rate", type=float, default=5e-7)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--save_interval", type=int, default=1000)

    # ---------------- LoRA ----------------
    p.add_argument("--lora_name", type=str, default="lora_experiment")
    p.add_argument("--lora_rank", type=int, default=16,
                   help="LoRA 低秩维度 r")
    p.add_argument("--lora_alpha", type=float, default=32.0,
                   help="LoRA 缩放系数 alpha，实际倍率为 alpha/r")
    p.add_argument("--lora_dropout", type=float, default=0.05,
                   help="仅作用于 LoRA 分支的 dropout")
    p.add_argument("--lora_target_modules", type=str, default=None,
                   help="逗号分隔的目标层叶子名；空值使用注意力+FFN投影默认集合")

    # QLoRA 专属：NF4 + double quant 是论文与工程实践中的常用组合。
    p.add_argument("--qlora_quant_type", choices=["nf4", "fp4"], default="nf4")
    p.add_argument("--qlora_double_quant", type=int, choices=[0, 1], default=1)
    p.add_argument("--qlora_compute_dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--qlora_optimizer", choices=["adamw", "paged_adamw_8bit"],
                   default="paged_adamw_8bit")

    # ---------------- 蒸馏 ----------------
    p.add_argument("--student_hidden_size", type=int, default=768)
    p.add_argument("--student_num_layers", type=int, default=8)
    p.add_argument("--teacher_hidden_size", type=int, default=768)
    p.add_argument("--teacher_num_layers", type=int, default=8)
    p.add_argument("--student_use_moe", type=int, default=0, choices=[0, 1])
    p.add_argument("--teacher_use_moe", type=int, default=1, choices=[0, 1])
    p.add_argument("--from_student_weight", type=str, default="full_sft")
    p.add_argument("--from_teacher_weight", type=str, default="full_sft")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=1.5)

    # ---------------- 偏好 / 强化 ----------------
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--preference_label_smoothing", type=float, default=0.0,
                   help="DPO 标签平滑比例，0 为标准 DPO")
    p.add_argument("--simpo_gamma", type=float, default=0.5,
                   help="SimPO 的目标偏好 margin")
    p.add_argument("--cpo_alpha", type=float, default=1.0,
                   help="CPO 中 chosen SFT 损失的权重")
    p.add_argument("--orpo_lambda", type=float, default=0.1,
                   help="ORPO 中 odds-ratio 偏好项的权重")
    p.add_argument("--kto_desirable_weight", type=float, default=1.0)
    p.add_argument("--kto_undesirable_weight", type=float, default=1.0)
    p.add_argument("--num_generations", type=int, default=6)
    p.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"])
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--epsilon_high", type=float, default=5.0)
    p.add_argument("--max_gen_len", type=int, default=1024)
    p.add_argument("--max_total_len", type=int, default=2500)
    p.add_argument("--thinking_ratio", type=float, default=0.9)
    p.add_argument("--rollout_temperature", type=float, default=0.8)
    p.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward")

    # DAPO 专属。epsilon_high 是相对 1 的上侧裁剪宽度，而不是 ratio 绝对值。
    p.add_argument("--dapo_epsilon_high", type=float, default=0.28)
    p.add_argument("--dapo_max_resample", type=int, default=4)
    p.add_argument("--dapo_reward_std_threshold", type=float, default=1e-6)
    p.add_argument("--dapo_overlong_buffer", type=int, default=128)
    p.add_argument("--dapo_overlong_penalty", type=float, default=1.0)
    p.add_argument("--dapo_kl_coef", type=float, default=0.0)

    # PPO 专属
    p.add_argument("--clip_epsilon", type=float, default=0.2)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--kl_coef", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--cliprange_value", type=float, default=0.2)
    p.add_argument("--ppo_update_iters", type=int, default=2)
    p.add_argument("--early_stop_kl", type=float, default=0.25)
    p.add_argument("--mini_batch_size", type=int, default=2)

    # ---------------- 采样后端（训推分离）----------------
    p.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"])
    p.add_argument("--sglang_base_url", type=str, default="http://localhost:8998")
    p.add_argument("--sglang_model_path", type=str, default="../model")
    p.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt")

    # ---------------- 调试 ----------------
    p.add_argument("--debug_mode", action="store_true")
    p.add_argument("--debug_interval", type=int, default=20)
    p.add_argument("--debug_log_ratio", action="store_true")

    # ---------------- 实验记录 ----------------
    p.add_argument("--metrics_path", type=str, default="",
                   help="逐 step 指标 CSV 的落盘路径（默认 {save_dir}/{save_weight}_metrics.csv）")
    p.add_argument("--run_name", type=str, default="",
                   help="本次实验的名字，仅用于日志标识")
    p.add_argument("--metrics_detail", type=int, default=1, choices=[0, 1],
                   help="1=采集富指标（参数组范数 / MoE 负载 / QKV / 门控 / 系统资源）")
    p.add_argument("--val_interval", type=int, default=500,
                   help="每隔多少步在留出集上算一次验证 loss（0=不算）")
    p.add_argument("--val_samples", type=int, default=512,
                   help="从数据尾部留出多少条作验证集（0=不切分）")
    p.add_argument("--val_batches", type=int, default=16,
                   help="每次验证跑多少个 batch（控制验证开销）")

    return p


if __name__ == "__main__":
    parser = build_parser()
    # 先解析出 --algo/--config，据此选定阶段配置，再把配置值灌成 argparse 默认值
    pre, _ = parser.parse_known_args()
    cfg = apply_to_parser(parser, stage_of(pre.algo), algo=pre.algo)
    args = parser.parse_args()

    run_training(args, args.algo, cfg, config_path=args.config)
