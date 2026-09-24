"""标准 LayerNorm，作为 RMSNorm 的对照基线。

与 RMSNorm 的差异：会先减去均值再做缩放 + 平移，因此有两个参数
（``weight`` 与 ``bias``），参数量是 RMSNorm 的两倍。
"""
import torch
from torch import nn

from ..registry import register


@register("norm", "layernorm")
class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # 与 RMSNorm 一致，在 fp32 下计算再转回，避免 bf16 精度损失
        mean = x.float().mean(-1, keepdim=True)
        var = x.float().var(-1, unbiased=False, keepdim=True)
        normed = (x.float() - mean) / torch.sqrt(var + self.eps)
        return (self.weight * normed + self.bias).type_as(x)
