"""RMSNorm —— 与重构前 ``model/model_minimind.py`` 的实现逐字一致。

数值细节：先升到 fp32 计算再转回原 dtype，避免 bf16 下平方均值的精度损失。
"""
import torch
from torch import nn

from ..registry import register


@register("norm", "rmsnorm")
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)
