"""稠密 SwiGLU 前馈网络 —— 与重构前 ``FeedForward`` 逐字一致。"""
from torch import nn
from transformers.activations import ACT2FN

from ..registry import register


@register("feedforward", "swiglu")
class SwiGLUFeedForward(nn.Module):
    def __init__(self, cfg, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or cfg.get("intermediate_size")
        self.hidden_size = cfg.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(cfg.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, cfg.hidden_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[cfg.hidden_act]

    def forward(self, x, token_mask=None):
        # 稠密 FFN 不需要 mask；保留参数是为了与 MoE 组件共享统一调用协议。
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
