"""分组查询注意力（GQA）+ QK-Norm —— 与重构前实现逐字一致。

把 ``num_key_value_heads`` 设为与 ``num_attention_heads`` 相等，
``n_rep`` 即为 1，自然退化为标准多头注意力（MHA）；
设为 1 则退化为 MQA。因此无需单独的 MHA / MQA 实现。

位置编码**不**硬编码为 RoPE：attention 持有一个位置编码组件，
在 forward 里通过 ``self.positional.apply(...)`` 委托给它，
这样 ``nope`` / ``partial_rope`` 等实现才能即插即用。
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from ..norm.rmsnorm import RMSNorm
from ..registry import register


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


@register("attention", "gqa")
class GQAttention(nn.Module):
    def __init__(self, cfg, positional=None):
        super().__init__()
        self.num_key_value_heads = cfg.num_attention_heads if cfg.get("num_key_value_heads") is None else cfg.num_key_value_heads
        self.n_local_heads = cfg.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = cfg.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * self.head_dim, cfg.hidden_size, bias=False)
        # QK-Norm：Qwen3 风格的稳定性设计；关掉后该层不再有 q_norm/k_norm 权重
        self.use_qk_norm = bool(cfg.get("qk_norm", True))
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.positional = positional
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.dropout = cfg.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and cfg.get("flash_attn", True)
        # 显存开关：仅当显式开启时，带 padding 的 batch 也走 SDPA（默认关闭 -> 既有行为逐字不变）
        self.flash_attn_masked = bool(cfg.get("flash_attn_masked", False))

    @staticmethod
    def state_seq_len(state) -> int:
        """KV cache 里已缓存的序列长度（模型侧据此计算 start_pos）。"""
        return int(state[0].shape[1]) if state is not None else 0

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        # 记录 Q/K/V 的 RMS（只做逐元素平方均值，开销可忽略；不参与计算图）。
        # 训练时用它观察各投影的数值尺度是否健康、有无爆炸/塌缩。
        if self.training:
            with torch.no_grad():
                self.last_q_rms = float(xq.detach().float().pow(2).mean().sqrt())
                self.last_k_rms = float(xk.detach().float().pow(2).mean().sqrt())
                self.last_v_rms = float(xv.detach().float().pow(2).mean().sqrt())
        if self.use_qk_norm:
            xq, xk = self.q_norm(xq), self.k_norm(xk)
        # 委托给位置编码组件；nope 会返回 None，此时跳过
        if self.positional is not None and position_embeddings is not None:
            cos, sin = position_embeddings
            xq, xk = self.positional.apply(xq, xk, cos, sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        elif (self.flash and self.flash_attn_masked and seq_len > 1
              and past_key_value is None and attention_mask is not None):
            # 变长 batch（RL 的 rollout 必然如此）带 padding，上面的快路径要求 mask 全 1
            # 所以会被跳过；而 eager 路径要物化 [B, H, L, L] 的分数矩阵并一直留到 backward
            # —— 12 条 1792 token 的序列每层就是 GB 级，24G 卡上必爆显存。
            # 这里把「因果 + padding」合成一个布尔掩码交给 SDPA：数学等价，显存回到线性。
            causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=xq.device).tril()
            keep = (attention_mask == 1).view(bsz, 1, 1, seq_len)
            output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=causal.view(1, 1, seq_len, seq_len) & keep,
                dropout_p=self.dropout if self.training else 0.0)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal:
                scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        if self.training:
            with torch.no_grad():
                self.last_out_rms = float(output.detach().float().pow(2).mean().sqrt())
        return output, past_kv
