"""MLA —— Multi-head Latent Attention（DeepSeek-V2/V3/V4）。

它相对 GQA 的**唯一**价值在于缓存形态：不缓存完整 K/V，而是把 KV 压到一个低秩
latent 再缓存。单 token 的缓存量是 ``kv_lora_rank + qk_rope_head_dim``，而不是
``2 * n_kv_heads * head_dim``。

结构（训练/前向用朴素但正确的形态）::

    c_KV = x @ W_DKV                     # 下投影 -> [B, T, kv_lora_rank]（要缓存）
    k_C  = c_KV @ W_UK                   # 上投影 -> [B, T, H, qk_nope_head_dim]
    v_C  = c_KV @ W_UV                   # 上投影 -> [B, T, H, v_head_dim]
    k_R  = RoPE(x @ W_KR)                # 解耦 RoPE，所有头共享 -> [B, T, 1, qk_rope_head_dim]
    q_C  = x @ W_Q（或 x @ W_DQ @ W_UQ）  # -> [B, T, H, qk_nope_head_dim + qk_rope_head_dim]
    q    = [q_nope ; RoPE(q_rope)]，k = [k_C ; k_R] 广播到各头
    o    = softmax(q k^T / sqrt(qk_nope + qk_rope)) v   -> [B, T, H, v_head_dim]
    out  = o @ o_proj

**为什么缓存压缩形式能成立**：注意力只依赖 ``k_C/v_C`` 与 ``k_R``，而
``k_C = c_KV @ W_UK``、``v_C = c_KV @ W_UV`` 都是 ``c_KV`` 的**无参上下文**线性
映射（不含位置信息），因此「每步用缓存的 c_KV 重新上投影」与「缓存上投影后的
完整 K/V」在数学上完全等价。位置相关的部分只存在于 ``k_R``，它是逐 token 的
RoPE 结果，按 token 缓存即可（所有头共享，只有 qk_rope_head_dim 维）。
于是增量解码每步只搬运 ``kv_lora_rank + qk_rope_head_dim`` 个数，而不是
``2 * H * head_dim``。

矩阵吸收（matrix absorption，把 W_UK/W_UV 乘进 W_Q/o_proj，让推理期不再显式
构造 k_C/v_C）是**推理期加速**技巧，本实现不采用：缓存压缩形式已经达到省显存的
目的，而吸收会改变权重布局、让权重与训练时不再一一对应，收益（省几次小矩阵乘）
与复杂度不成正比。需要极致推理吞吐时再实现。

与位置编码组件的耦合：RoPE 的 cos/sin 由 ``positional`` 组件按 ``cfg.head_dim``
预计算，因此 ``qk_rope_head_dim`` 默认等于 ``head_dim``，保证两者维度一致。
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from ..registry import register


@register("attention", "mla")
class MLAttention(nn.Module):
    def __init__(self, cfg, positional=None):
        super().__init__()
        hidden = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim

        # ---- MLA 专属维度 ----
        # d_c：KV 压缩后的 latent 维度，缓存的主体
        self.kv_lora_rank = int(cfg.get("kv_lora_rank", 512))
        # d_c'：query 压缩后的维度，0 表示不做 query 压缩（直接用 hidden -> heads*head_dim）
        self.q_lora_rank = int(cfg.get("q_lora_rank", 0))
        # 不参与 RoPE 的 q/k 每头维度
        self.qk_nope_head_dim = int(cfg.get("qk_nope_head_dim", self.head_dim))
        # 参与 RoPE 的「解耦」每头维度；默认跟随 head_dim 以对齐位置编码的 cos/sin
        self.qk_rope_head_dim = int(cfg.get("qk_rope_head_dim", self.head_dim))
        self.v_head_dim = int(cfg.get("v_head_dim", self.head_dim))
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # softmax 的缩放按 q/k 的**总**每头维度算（nope + rope）
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)

        # ---- 权重：全部无 bias ----
        # KV 下投影（W_DKV）与解耦 RoPE 的 K 投影（W_KR）
        self.kv_dkv_proj = nn.Linear(hidden, self.kv_lora_rank, bias=False)
        self.k_rope_proj = nn.Linear(hidden, self.qk_rope_head_dim, bias=False)
        # KV 上投影（W_UK / W_UV）
        self.kv_uk_proj = nn.Linear(self.kv_lora_rank, self.num_heads * self.qk_nope_head_dim, bias=False)
        self.kv_uv_proj = nn.Linear(self.kv_lora_rank, self.num_heads * self.v_head_dim, bias=False)
        # Query：低秩压缩（W_DQ + W_UQ）或直接投影
        if self.q_lora_rank > 0:
            self.q_dq_proj = nn.Linear(hidden, self.q_lora_rank, bias=False)
            self.q_uq_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(hidden, self.num_heads * self.qk_head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden, bias=False)

        self.positional = positional
        self.is_causal = True
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.dropout = cfg.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and cfg.get("flash_attn", True)

    # ------------------------------------------------------------------ #
    @staticmethod
    def positional_dim(cfg) -> int:
        """MLA 的 RoPE 只作用在解耦的那部分维度上（``qk_rope_head_dim``）。

        框架据此预计算 cos/sin —— 与 ``head_dim`` 不同，必须显式声明，
        否则位置编码表会对不上。
        """
        return int(cfg.get("qk_rope_head_dim", cfg.head_dim))

    @staticmethod
    def state_seq_len(state) -> int:
        """缓存的是压缩 latent，但它的**序列维仍然是 token 数**。

        与线性注意力（DeltaNet，状态定长、无序列维 -> 0）不同，
        MLA 是 softmax 注意力，模型侧要据此算出 ``start_pos``。
        """
        return int(state[0].shape[1]) if state is not None else 0

    # ------------------------------------------------------------------ #
    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape

        # 1. KV 下投影：得到要缓存的压缩 latent
        c_kv = self.kv_dkv_proj(x)                                    # [B, T, d_c]
        # 解耦 RoPE 的那路 K：先压到单头，再做 RoPE（DeepSeek 设计：所有头共享）
        k_rope = self.k_rope_proj(x).view(bsz, seq_len, 1, self.qk_rope_head_dim)

        # 2. Query（可选低秩压缩）
        if self.q_lora_rank > 0:
            q = self.q_uq_proj(self.q_dq_proj(x))
        else:
            q = self.q_proj(x)
        q = q.view(bsz, seq_len, self.num_heads, self.qk_head_dim)
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # 3. 只对「解耦」部分施加位置编码；nope 部分走 softmax 不旋转
        if self.positional is not None and position_embeddings is not None:
            cos, sin = position_embeddings
            if cos.shape[-1] != self.qk_rope_head_dim:
                raise ValueError(
                    f"MLA 的 qk_rope_head_dim={self.qk_rope_head_dim} 与位置编码维度 "
                    f"{cos.shape[-1]} 不一致；请让 head_dim（位置编码按它预计算 cos/sin）"
                    f"与 qk_rope_head_dim 相等"
                )
            q_rope, k_rope = self.positional.apply(q_rope, k_rope, cos, sin)

        # 4. 缓存**压缩形式**：(c_KV, k_R)
        if past_key_value is not None:
            c_kv = torch.cat([past_key_value[0], c_kv], dim=1)
            k_rope = torch.cat([past_key_value[1], k_rope], dim=1)
        present_state = (c_kv, k_rope) if use_cache else None

        # 5. 上投影：增量解码时对整段 c_KV 重算（等价于缓存完整 K/V，但搬运量小得多）
        k_nope = self.kv_uk_proj(c_kv).view(bsz, -1, self.num_heads, self.qk_nope_head_dim)
        v = self.kv_uv_proj(c_kv).view(bsz, -1, self.num_heads, self.v_head_dim)
        # k_R 沿头维广播后与 k_nope 拼接
        k = torch.cat([k_nope, k_rope.expand(bsz, -1, self.num_heads, -1)], dim=-1)
        q = torch.cat([q_nope, q_rope], dim=-1)

        # 6. 标准因果 softmax 注意力
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (
            attention_mask is None or torch.all(attention_mask == 1)
        ):
            output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal
            )
        else:
            scores = (q @ k.transpose(-2, -1)) * self.scale
            if self.is_causal:
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len), float("-inf"), device=scores.device
                ).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(q)) @ v

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, present_state
