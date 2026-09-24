"""存盘与断点续训。

复用 ``trainer_utils.lm_checkpoint``，保证文件命名与字段**完全不变**：
- 推理权重 ``{save_dir}/{prefix}_{hidden_size}{_moe}.pth``（纯 state_dict，half/cpu）
- 续训档   ``{save_dir}/{prefix}_{hidden_size}{_moe}_resume.pth``
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist

from ..trainer_utils import lm_checkpoint
from .runtime import unwrap_model

#: 续训档目录。仓库根只放纯净代码，训练产物（权重/日志/续训档）默认落在
#: ``test/`` 下（见 ``test/README.md``）。设环境变量 ``MINIMIND_ARTIFACT_ROOT``
#: 可以改这个根，例如设成仓库根就是迁移前的行为。
#: 与 ``configs/loader.py`` 的 ``ARTIFACT_ROOT`` 同一口径，两者必须一致。
def _artifact_root() -> Path:
    env = os.environ.get("MINIMIND_ARTIFACT_ROOT")
    root = Path(env) if env else Path(__file__).resolve().parents[2] / "test"
    return root if root.is_absolute() else (Path(__file__).resolve().parents[2] / root)


CHECKPOINT_DIR = str(_artifact_root() / "checkpoints")


def weight_path(save_dir: str, prefix: str, lm_config) -> str:
    """推理权重路径 —— 命名规则与重构前逐字一致。"""
    moe_suffix = '_moe' if lm_config.use_moe else ''
    return f'{save_dir}/{prefix}_{lm_config.hidden_size}{moe_suffix}.pth'


def resolve_weight_prefix(args) -> str:
    """权重前缀名。

    LoRA 脚本历史上没有 ``--save_weight``，用的是 ``--lora_name``，这里统一取值。
    """
    if getattr(args, "algo", None) in {"lora", "qlora"}:
        return getattr(args, "lora_name", None) or getattr(args, "save_weight", "model")
    return getattr(args, "save_weight", None) or getattr(args, "lora_name", "model")


def resolve_from_weight(args) -> str:
    """基座权重前缀名。

    蒸馏脚本历史上没有 ``--from_weight``，用的是 ``--from_student_weight``。
    """
    if getattr(args, "algo", None) == "distill":
        return getattr(args, "from_student_weight", None) or getattr(args, "from_weight", "none")
    return getattr(args, "from_weight", None) or getattr(args, "from_student_weight", "none")


def save_inference_weights(model, save_dir: str, prefix: str, lm_config, ckp_path: str = None) -> str:
    """保存 half/cpu 的纯 state_dict（与重构前一致）。"""
    ckp = ckp_path or weight_path(save_dir, prefix, lm_config)
    state_dict = unwrap_model(model).state_dict()
    torch.save({
        k: (v.half().cpu() if torch.is_floating_point(v) else v.cpu())
        for k, v in state_dict.items()
    }, ckp)
    del state_dict
    return ckp


def save_resume(lm_config, weight: str, model, optimizer, scaler=None,
                epoch: int = 0, step: int = 0, wandb=None, save_dir: str = None, **extra):
    """保存完整续训档（含优化器状态），供 ``--from_resume 1`` 恢复。"""
    lm_checkpoint(lm_config, weight=weight, model=model, optimizer=optimizer,
                  scaler=scaler, epoch=epoch, step=step, wandb=wandb,
                  save_dir=save_dir or CHECKPOINT_DIR, **extra)


def load_resume(args, lm_config, weight: str, save_dir: str = None):
    """``--from_resume 1`` 时读取续训档；否则返回 None。"""
    if args.from_resume != 1:
        return None
    return lm_checkpoint(lm_config, weight=weight, save_dir=save_dir or CHECKPOINT_DIR)


def restore(ctx, ckp_data, algorithm=None) -> tuple:
    """把续训档内容灌回模型/优化器/其它对象。

    返回 ``(start_epoch, start_step)``。
    """
    if not ckp_data:
        return 0, 0
    if algorithm is None:
        ctx.model.load_state_dict(ckp_data['model'])
    else:
        algorithm.restore_model_state(ctx.model, ckp_data['model'])
    ctx.optimizer.load_state_dict(ckp_data['optimizer'])
    if ctx.scaler is not None and 'scaler' in ckp_data:
        ctx.scaler.load_state_dict(ckp_data['scaler'])
    return ckp_data['epoch'], ckp_data.get('step', 0)


def finalize():
    """训练结束后的分布式清理（"段9"）。"""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
