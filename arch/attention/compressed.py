"""压缩注意力（Compressed Attention）—— DeepSeek-V4 的 HCA（Hybrid Compressed
Attention）思路的简化实现。

核心想法：注意力的大部分开销来自 KV cache，而远处的 token 不需要逐位保留。
于是沿**序列维**把每 ``compress_rate`` 个连续 token 压成 1 项 KV，
query 只看压缩后的远处信息 + 自己所在块内的近处原始 token。

压缩方式（``compress_type``）
----------------------------
- ``mean``：窗口内 k/v 直接取平均（无参数）；
- ``conv``：可学习的 ``Conv1d(kernel=stride=compress_rate, groups=num_kv_heads)``
  逐头做窗口投影。

可见性规则（因果安全的版本）
----------------------------
设块 ``r`` 覆盖全局位置 ``[r*R, (r+1)*R)``（``R = compress_rate``），
query 的全局位置为 ``p``：

- **压缩项**：只有 ``r < p // R`` 的块可见，即**本 query 自己所在的块不参与压缩注意力**
  （它此刻还没定型）。这同时保证了「块完整地落在 p 之前」这一因果条件；
- **原始 token**：``keep_local=True``（默认）时可见**同块内位置 <= p** 的原始 token；
  ``keep_local=False`` 时只保留 query 自身那一个 token 作为兜底
  （否则序列开头、压缩块还没形成时会出现空 softmax → NaN）。

增量解码的状态设计
------------------
状态是一个 6 元组（不透明），只由 :meth:`state_seq_len` 解释：

``(comp_k, comp_v, raw_k, raw_v, n_seen, raw_start)``

- ``comp_k/comp_v``：已压满的块，序列维是**块数**；
- ``raw_k/raw_v``：尚未压满的最后一块的原始 token，起点是全局位置 ``raw_start``
  （恒为 ``R`` 的倍数）；每轮把新 token 拼到它后面，凑满一块就立即压缩并清空；
- ``n_seen``：已处理 token 总数 —— ``state_seq_len`` 返回它，模型据此算 ``start_pos``。

一致性：块边界恒按全局位置 ``R`` 对齐，压缩只发生在「块凑满」这一确定时刻，
而且压缩用到的那批原始 k/v 在两处完全相同（RoPE 只依赖绝对位置），
因此 ``use_cache=True`` 的逐 token 解码与一次性前向结果在数值上一致。
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from ..norm.rmsnorm import RMSNorm
from ..registry import register
from .gqa import repeat_kv


@register("attention", "compressed")
class CompressedAttention(nn.Module):
    def __init__(self, cfg, positional=None):
        super().__init__()
        self.num_key_value_heads = cfg.num_attention_heads if cfg.get("num_key_value_heads") is None else cfg.num_key_value_heads
        self.n_local_heads = cfg.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = cfg.head_dim
        # 压缩率：每 compress_rate 个连续 token 压成 1 项 KV
        self.compress_rate = max(int(cfg.get("compress_rate", 128)), 1)
        self.compress_type = str(cfg.get("compress_type", "mean")).lower()
        if self.compress_type not in ("mean", "conv"):
            raise ValueError(f"compressed 注意力的 compress_type 只支持 mean / conv，收到 {self.compress_type!r}")
        # 是否额外关注自己所在压缩块内的原始 token
        self.keep_local = bool(cfg.get("keep_local", True))
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.use_qk_norm = bool(cfg.get("qk_norm", True))
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        # conv 压缩：按头分组，kernel = stride = compress_rate（不重叠窗口，天然因果）
        if self.compress_type == "conv":
            channels = self.n_local_kv_heads * self.head_dim
            self.compress_conv = nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=self.compress_rate, stride=self.compress_rate,
                groups=self.n_local_kv_heads, bias=False,
            )
        self.positional = positional
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.dropout = cfg.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and cfg.get("flash_attn", True)

    # ------------------------------------------------------------------ #
    @staticmethod
    def state_seq_len(state) -> int:
        """已处理的 token 总数（不是压缩块数）。模型侧据此计算 start_pos。"""
        return int(state[4]) if state is not None else 0

    def _conv_compress(self, win, n_blocks: int):
        """(bsz, n_blocks*R, kv, D) -> (bsz, n_blocks, kv, D)，逐头窗口投影。"""
        bsz = win.shape[0]
        # (bsz, kv, D, T) 再摊平成通道，保证通道序是 (kv, D) 而不是 (kv, T, D)
        x = win.permute(0, 2, 3, 1).reshape(bsz, self.n_local_kv_heads * self.head_dim, n_blocks * self.compress_rate)
        y = self.compress_conv(x)  # (bsz, kv*D, n_blocks)
        return y.view(bsz, self.n_local_kv_heads, self.head_dim, n_blocks).permute(0, 3, 1, 2)

    def _compress(self, k, v, n_blocks: int, offset: int):
        """把 ``pending[offset : offset + n_blocks*R]`` 压成 ``n_blocks`` 项。

        ``offset`` 保证窗口对齐到全局块边界（``pending`` 的起点恒是块边界，故 offset 为 0，
        这里保留一般形式以防调用方在非对齐位置起头）。
        """
        R = self.compress_rate
        L = n_blocks * R
        win_k = k[:, offset:offset + L]
        win_v = v[:, offset:offset + L]
        bsz = win_k.shape[0]
        if self.compress_type == "conv":
            return self._conv_compress(win_k, n_blocks), self._conv_compress(win_v, n_blocks)
        # mean：把序列维按 R 分组取平均
        ck = win_k.reshape(bsz, n_blocks, R, self.n_local_kv_heads, self.head_dim).mean(dim=2)
        cv = win_v.reshape(bsz, n_blocks, R, self.n_local_kv_heads, self.head_dim).mean(dim=2)
        return ck, cv

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, q_len, _ = x.shape
        R = self.compress_rate
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, q_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, q_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, q_len, self.n_local_kv_heads, self.head_dim)
        if self.use_qk_norm:
            xq, xk = self.q_norm(xq), self.k_norm(xk)
        # 委托给位置编码组件；nope 会返回 None，此时跳过
        if self.positional is not None and position_embeddings is not None:
            cos, sin = position_embeddings
            xq, xk = self.positional.apply(xq, xk, cos, sin)

        # ---- 取出上一轮状态 ----
        if past_key_value is not None:
            comp_k, comp_v, raw_k, raw_v, n_seen_past, raw_start = past_key_value
        else:
            comp_k = xk.new_zeros(bsz, 0, self.n_local_kv_heads, self.head_dim)
            comp_v = xv.new_zeros(bsz, 0, self.n_local_kv_heads, self.head_dim)
            raw_k = raw_v = None
            n_seen_past, raw_start = 0, 0
        # pending = 上一轮遗留的未压满 token + 本次的 q_len 个 token
        if raw_k is not None:
            pending_k = torch.cat([raw_k, xk], dim=1)
            pending_v = torch.cat([raw_v, xv], dim=1)
            pend_start = raw_start
        else:
            pending_k, pending_v = xk, xv
            pend_start = 0
        n_seen = pend_start + pending_k.shape[1]
        pend_len = pending_k.shape[1]

        # ---- 只压缩「完全落在 pending 内」的块 ----
        r_start = (pend_start + R - 1) // R   # 第一个完整块
        r_end = n_seen // R                   # 最后一个完整块的下一个
        n_new = max(r_end - r_start, 0)
        if n_new > 0:
            new_k, new_v = self._compress(pending_k, pending_v, n_new, r_start * R - pend_start)
            comp_k = torch.cat([comp_k, new_k], dim=1)
            comp_v = torch.cat([comp_v, new_v], dim=1)
        n_comp = comp_k.shape[1]

        # ---- 掩码：(q_len, n_comp + pend_len) ----
        q_pos = torch.arange(n_seen_past, n_seen, device=x.device)   # query 的全局位置
        k_pos = torch.arange(pend_start, n_seen, device=x.device)    # 原始 key 的全局位置
        # 压缩块：只有 r < p // R 可见（本 query 所在的块尚未定型）
        comp_ok = torch.arange(n_comp, device=x.device).view(1, n_comp) < (q_pos // R).view(q_len, 1)
        causal = k_pos.view(1, -1) <= q_pos.view(-1, 1)
        if self.keep_local:
            # 注意加括号：Python 里 & 的优先级高于 ==
            raw_ok = ((k_pos // R).view(1, -1) == (q_pos // R).view(-1, 1)) & causal
        else:
            # 只保留自身作为兜底，避免序列开头没有可见 key 时 softmax 为空
            raw_ok = k_pos.view(1, -1) == q_pos.view(-1, 1)
        mask = torch.cat([
            comp_ok.unsqueeze(0).expand(bsz, q_len, n_comp),
            raw_ok.unsqueeze(0).expand(bsz, q_len, pend_len),
        ], dim=-1).unsqueeze(1)  # (bsz, 1, q_len, n_keys)
        # padding 掩码只作用于原始 token（压缩项是跨 token 的平均，无法逐位对应）
        if attention_mask is not None and attention_mask.shape[-1] == n_seen:
            pad = attention_mask[:, pend_start:n_seen].bool()
            mask = torch.cat([mask[..., :n_comp], mask[..., n_comp:] & pad[:, None, None, :]], dim=-1)

        # ---- 注意力：key = [压缩块, 本块原始 token] ----
        keys = torch.cat([repeat_kv(comp_k, self.n_rep), repeat_kv(pending_k, self.n_rep)], dim=1)
        vals = torch.cat([repeat_kv(comp_v, self.n_rep), repeat_kv(pending_v, self.n_rep)], dim=1)
        xq_t = xq.transpose(1, 2)
        xk_t = keys.transpose(1, 2)
        xv_t = vals.transpose(1, 2)
        if self.flash:
            output = F.scaled_dot_product_attention(
                xq_t, xk_t, xv_t, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            scores = (xq_t @ xk_t.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~mask, float("-inf"))
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv_t
        output = output.transpose(1, 2).reshape(bsz, q_len, -1)
        output = self.resid_dropout(self.o_proj(output))

        # ---- 新状态：raw 只保留还没压满的那一段 ----
        raw_start_new = r_end * R
        raw_off = max(raw_start_new - pend_start, 0)
        present = None
        if use_cache:
            present = (
                comp_k, comp_v,
                pending_k[:, raw_off:], pending_v[:, raw_off:],
                n_seen, pend_start + raw_off,
            )
        return output, present
