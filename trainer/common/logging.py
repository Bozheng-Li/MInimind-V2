"""日志与实验跟踪。

``wandb`` 在本项目里实际指向 swanlab（国内访问友好，接口兼容），
与重构前各脚本 "段4" 的行为保持一致。
"""
from __future__ import annotations

from ..trainer_utils import Logger


def init_wandb(args, ckp_data=None, run_name: str = None):
    """按 ``--use_wandb`` 初始化；仅在主进程生效。返回 wandb 模块或 None。"""
    if not args.use_wandb:
        return None
    import torch.distributed as dist
    if dist.is_initialized() and dist.get_rank() != 0:
        return None

    import swanlab as wandb

    wandb_id = ckp_data.get('wandb_id') if ckp_data else None
    resume = 'must' if wandb_id else None
    wandb.init(project=args.wandb_project, name=run_name, id=wandb_id, resume=resume)
    return wandb


def log_metrics(wandb, metrics: dict) -> None:
    if wandb:
        wandb.log(metrics)


__all__ = ["Logger", "init_wandb", "log_metrics"]
