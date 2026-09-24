"""运行时环境：分布式、精度、优化器、梯度缩放。

这些在原来的 8 个脚本里是逐字重复的样板（"段1 / 段3"），现在只写一次。
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel

from ..trainer_utils import Logger, init_distributed_mode, setup_seed


@dataclass
class Runtime:
    """一次训练运行的运行时环境。"""

    device: str
    dtype: torch.dtype
    autocast_ctx: object
    local_rank: int
    distributed: bool
    is_main: bool


def init_runtime(args) -> Runtime:
    """初始化 DDP、随机种子、混合精度上下文。

    与重构前各脚本的「段1 + 段3」逐字等价。
    注：蒸馏脚本历史上没有 ``--seed``（硬编码 42），故这里用 getattr 兜底。
    """
    local_rank = init_distributed_mode()
    distributed = dist.is_initialized()
    if distributed:
        args.device = f"cuda:{local_rank}"
    seed = getattr(args, "seed", 42)
    setup_seed(seed + (dist.get_rank() if distributed else 0))

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = contextlib.nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    return Runtime(
        device=args.device,
        dtype=dtype,
        autocast_ctx=autocast_ctx,
        local_rank=local_rank,
        distributed=distributed,
        is_main=(not distributed) or dist.get_rank() == 0,
    )


def build_optimizer(model, learning_rate: float):
    """AdamW，与重构前一致（不传 weight_decay 等额外参数）。"""
    return optim.AdamW(model.parameters(), lr=learning_rate)


def build_scaler(dtype: str):
    """仅在 fp16 下启用 GradScaler；bf16 不需要。"""
    return torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))


def wrap_model(model, runtime: Runtime, use_compile: int = 0):
    """torch.compile（可选）+ DDP 包装，顺序与重构前一致。"""
    if use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if runtime.distributed:
        model = DistributedDataParallel(model, device_ids=[runtime.local_rank])
    return model


def unwrap_model(model):
    """剥掉 DDP / torch.compile 的包装，拿到原始模型。"""
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(raw, "_orig_mod", raw)


def synchronize_module(module, src: int = 0):
    """把随机初始化模块的参数与 buffer 从 ``src`` 广播到所有 rank。

    DDP 构造时会同步主模型，但 reference/teacher 往往在 DDP 包装**之前**由主模型
    深拷贝。如果各 rank 使用 ``seed + rank`` 初始化而不先同步，主 actor 最终会以
    rank 0 为准，冻结的 reference 却仍各不相同，第一步 KL 就是假的。
    """
    if not dist.is_initialized():
        return module
    with torch.no_grad():
        for tensor in module.state_dict().values():
            if torch.is_tensor(tensor):
                dist.broadcast(tensor, src=src)
    return module
