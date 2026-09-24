"""零中心 RMSNorm（Zero-centered RMSNorm）。

Qwen3-Next 采用的形式：把可学习缩放写成 ``(1 + w)`` 而非 ``w``，且 ``w`` 零初始化。
好处是初始状态等价于恒等缩放（与 RMSNorm 一致），但参数天然围绕 0 分布，
配合权重衰减时不会把缩放系数往 0 拉 —— 这在深层网络里更稳定。
"""
import torch
from torch import nn

from ..registry import register


@register("norm", "rmsnorm_zero_centered")
class RMSNormZeroCentered(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 零初始化：初始缩放 = 1 + 0 = 1，与普通 RMSNorm 等价
        self.weight = nn.Parameter(torch.zeros(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return ((1.0 + self.weight) * self.norm(x.float())).type_as(x)
