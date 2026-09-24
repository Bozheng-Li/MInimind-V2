"""滑动窗口注意力（Sliding Window Attention）—— Mistral / Gemma / Laguna 的做法。

每个 query 只关注它**因果之前**且距离不超过 ``window_size`` 的 key：

    j <= i  and  j > i - window_size

与 ``gqa.py`` 的唯一区别就是掩码：因果掩码再叠一层窗口掩码。
窗口为 1 时退化成「只看自己」，窗口 >= 序列长度时与全因果注意力逐位一致。

⚠️ 增量解码时 key 里已经有 ``past_len`` 个历史 token，
掩码必须用**全局位置**比较（``j_global > i_global - window_size``），
用局部索引会把窗口整体右移，逐 token 生成结果与一次性前向不一致。
本实现以 ``past_key_value[0].shape[1]`` 作为 ``past_len``。
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from ..norm.rmsnorm import RMSNorm
from ..registry import register
from .gqa import repeat_kv


@register("attention", "sliding_window")
class SlidingWindowAttention(nn.Module):
    def __init__(self, cfg, positional=None):
        super().__init__()
        self.num_key_value_heads = cfg.num_attention_heads if cfg.get("num_key_value_heads") is None else cfg.num_key_value_heads
        self.n_local_heads = cfg.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = cfg.head_dim
        self.is_causal = True
        # 窗口大小：默认 4096，与 Mistral 一致
        self.window_size = max(int(cfg.get("window_size", 4096)), 1)
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.use_qk_norm = bool(cfg.get("qk_norm", True))
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.positional = positional
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.dropout = cfg.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and cfg.get("flash_attn", True)

    # ------------------------------------------------------------------ #
    @staticmethod
    def state_seq_len(state) -> int:
        """KV cache 里已缓存的序列长度（模型侧据此计算 start_pos）。"""
        return int(state[0].shape[1]) if state is not None else 0

    def _build_mask(self, bsz, seq_len, past_len, total_len, attention_mask, device):
        """(bsz, 1, seq_len, total_len) 的布尔掩码，True 表示可见。

        因果 + 窗口：``j_global <= i_global`` 且 ``j_global > i_global - window_size``。
        query 的全局位置是 ``[past_len, past_len + seq_len)``，
        key 的全局位置是 ``[0, total_len)``。
        """
        q_pos = torch.arange(past_len, past_len + seq_len, device=device).view(seq_len, 1)
        k_pos = torch.arange(total_len, device=device).view(1, total_len)
        allow = (k_pos <= q_pos) & (k_pos > q_pos - self.window_size)  # (seq_len, total_len)
        allow = allow.view(1, 1, seq_len, total_len).expand(bsz, 1, seq_len, total_len)
        # 叠加 padding 掩码（长度对得上才用，对不上说明调用方没按惯例传）
        if attention_mask is not None and attention_mask.shape[-1] == total_len:
            allow = allow & attention_mask.bool().view(bsz, 1, 1, total_len)
        return allow

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        if self.use_qk_norm:
            xq, xk = self.q_norm(xq), self.k_norm(xk)
        # 委托给位置编码组件；nope 会返回 None，此时跳过
        if self.positional is not None and position_embeddings is not None:
            cos, sin = position_embeddings
            xq, xk = self.positional.apply(xq, xk, cos, sin)
        past_len = int(past_key_value[0].shape[1]) if past_key_value is not None else 0
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        total_len = xk.shape[1]
        # 窗口掩码是自定义掩码，SDPA 的 is_causal 快路径用不了，统一走 attn_mask
        mask = self._build_mask(bsz, seq_len, past_len, total_len, attention_mask, x.device)
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        if self.flash:
            output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 全 False 的行才会得到 -inf（窗口 >= 1 时不会出现，因为 j == i 恒可见）
            scores = scores.masked_fill(~mask, float("-inf"))
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
