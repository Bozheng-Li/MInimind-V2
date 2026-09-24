"""可组合的 Transformer Block。

子模块由 ``build.py`` 构造后注入，因此本文件不关心具体用了哪种注意力/前馈实现。
属性名（``self_attn`` / ``input_layernorm`` / ``post_attention_layernorm`` / ``mlp``）
与重构前保持一致，保证旧权重可严格加载。
"""
from torch import nn


class ArchBlock(nn.Module):
    def __init__(self, self_attn, input_layernorm, post_attention_layernorm, mlp):
        super().__init__()
        self.self_attn = self_attn
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        # token_mask 让 MoE 的负载均衡统计排除 padding；稠密 FFN 接收但忽略它。
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states), token_mask=attention_mask
        )
        return hidden_states, present_key_value
