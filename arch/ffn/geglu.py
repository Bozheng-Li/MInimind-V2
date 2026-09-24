"""稠密 GeGLU 前馈网络 —— 与 SwiGLU 同构，只把激活函数换成 GELU。

GeGLU 出自 "GLU Variants Improve Transformer"，被 Gemma / Gemma-2 / PaLM 等
模型采用：门控分支用 ``gelu_pytorch_tanh``（即 tanh 近似的 GELU）而不是 SiLU，
其余结构与 SwiGLU 完全一致。

模块属性名刻意保持 ``gate_proj`` / ``up_proj`` / ``down_proj`` 不变，
这样可以和 SwiGLU 权重逐键互换做对照实验。
"""
from torch import nn
from transformers.activations import ACT2FN

from ..registry import register


@register("feedforward", "geglu")
class GeGLUFeedForward(nn.Module):
    def __init__(self, cfg, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or cfg.get("intermediate_size")
        self.hidden_size = cfg.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(cfg.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, cfg.hidden_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, intermediate_size, bias=False)
        # Gemma 系列用的就是 tanh 近似版 GELU，而非精确 erf 版
        self.act_fn = ACT2FN["gelu_pytorch_tanh"]

    def forward(self, x, token_mask=None):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
