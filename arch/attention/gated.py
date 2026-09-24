"""门控注意力（Gated Attention）—— Qwen3-Next（NeurIPS'25 最佳论文）风格的 GQA。

在标准 GQA 之上叠加三个「稳定器」：

1. **输出门**：额外学一个 ``gate_proj``，``sigmoid`` 后逐元素乘在注意力输出上，
   再送 ``o_proj``：``output = o_proj(attn_out * sigmoid(gate(x)))``。
   相当于给注意力层一条「自适应调节写入强度」的通路。
2. **zero-centered QK-Norm**：把 q/k 的 RMSNorm 缩放写成 ``1 + w`` 且 ``w`` 零初始化
   （Qwen3-Next 的做法），权重衰减不会把缩放系数往 0 拉。
   配置 ``qk_norm_zero_centered: false`` 可退回普通 RMSNorm。
3. **partial RoPE 兼容**：只对 ``head_dim`` 的前 ``rotary_dim`` 维施加旋转。

位置编码的处理优先级（从高到低）
--------------------------------
1. ``positional is None`` 或 ``position_embeddings is None`` —— 完全不旋转
   （与 ``gqa.py`` 一致，``nope`` 走的就是这条）；
2. ``positional`` 自带 ``rotary_dim`` 属性（即 ``partial_rope`` 组件）——
   完全交给它，本组件的 ``rotary_dim`` 配置被忽略（避免二次旋转）；
3. 配置里给了 ``rotary_dim`` 且 ``0 < rotary_dim < head_dim`` —— 本组件自己建一张
   ``dim=rotary_dim`` 的频率表，只旋转前 ``rotary_dim`` 维、其余原样透传。
   ⚠️ 标准 partial RoPE 的频率用 ``rotary_dim``（而非 ``head_dim``）作分母，
   因此**不能**把传入的 cos/sin 直接切片复用；这里用 ``past_len`` 推得的全局位置
   从自建频率表取切片。``rope_theta`` / ``rope_scaling`` / ``max_position_embeddings``
   沿用 positional 组件的设置，保证与它一致；
4. 其余情况 —— 交给 ``positional.apply`` 做全维度 RoPE。
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from ..norm.rmsnorm import RMSNorm
from ..norm.rmsnorm_zero_centered import RMSNormZeroCentered
from ..positional.rope import apply_rotary_pos_emb, precompute_freqs_cis
from ..registry import register
from .gqa import repeat_kv


@register("attention", "gated")
class GatedAttention(nn.Module):
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
        # 输出门：hidden -> heads*head_dim，与注意力输出逐元素相乘（sigmoid 后）
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * self.head_dim, bias=False)
        # QK-Norm：zero-centered 形式是 Qwen3-Next 的默认；关掉则不再有 q_norm/k_norm 权重
        self.use_qk_norm = bool(cfg.get("qk_norm", True))
        self.qk_norm_zero_centered = bool(cfg.get("qk_norm_zero_centered", True))
        if self.use_qk_norm:
            norm_cls = RMSNormZeroCentered if self.qk_norm_zero_centered else RMSNorm
            self.q_norm = norm_cls(self.head_dim, eps=cfg.rms_norm_eps)
            self.k_norm = norm_cls(self.head_dim, eps=cfg.rms_norm_eps)
        # V-Norm（可选，默认关）。动机：Q/K 有 QK-Norm 把尺度归一化掉了，V 没有，
        # 于是模型可以放任 Q/K 漂移、却必须自己把 V 收回来 —— 实测预训练里
        # v/q 从 1.31 一路降到 0.61。给 V 也加 norm 是否更好，是本开关要验证的问题。
        self.use_v_norm = bool(cfg.get("v_norm", False))
        if self.use_v_norm:
            self.v_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
            if self.qk_norm_zero_centered:
                # ⚠️ transformers 的 post_init -> _init_weights 会把「类名里含 RMSNorm」
                # 的模块权重一律 fill_(1.0)，那会把零中心的 (1 + w) 变成 2.0，
                # 丢掉「初始等价于恒等缩放」这一性质。打上 HF 的「已初始化」标记
                # 让默认初始化跳过这两个模块（该标记只是个普通属性，不进 state_dict）。
                self.q_norm._is_hf_initialized = True
                self.k_norm._is_hf_initialized = True

        self.positional = positional
        # ---- partial RoPE 兼容层（见模块 docstring 的优先级说明）----
        # positional 自己声明了 rotary_dim（partial_rope 组件）时，本组件不插手
        self.positional_owns_rotary_dim = positional is not None and hasattr(positional, "rotary_dim")
        rotary_dim = int(cfg.get("rotary_dim") or 0)
        rotary_dim -= rotary_dim % 2  # RoPE 成对旋转，维度必须是偶数
        self.rotary_dim = 0
        if (not self.positional_owns_rotary_dim) and 0 < rotary_dim < self.head_dim:
            self.rotary_dim = rotary_dim
            # 自建频率表：dim 必须用 rotary_dim，这是与「切片复用」的关键区别。
            # 非持久化 buffer，不进 state_dict（与 rope.py 的约定一致）。
            rope_theta = float(getattr(positional, "rope_theta", cfg.get("rope_theta", 1e6)))
            rope_scaling = getattr(positional, "rope_scaling", cfg.get("rope_scaling"))
            max_pos = int(getattr(positional, "max_position_embeddings", cfg.get("max_position_embeddings", 32768)))
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.rotary_dim, end=max_pos, rope_base=rope_theta, rope_scaling=rope_scaling
            )
            self.register_buffer("partial_freqs_cos", freqs_cos, persistent=False)
            self.register_buffer("partial_freqs_sin", freqs_sin, persistent=False)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.dropout = cfg.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and cfg.get("flash_attn", True)
        # 显存开关：仅当显式开启时，带 padding 的 batch 也走 SDPA（默认关闭 -> 既有行为逐字不变）
        self.flash_attn_masked = bool(cfg.get("flash_attn_masked", False))

    # ------------------------------------------------------------------ #
    @staticmethod
    def state_seq_len(state) -> int:
        """KV cache 里已缓存的序列长度（模型侧据此计算 start_pos）。"""
        return int(state[0].shape[1]) if state is not None else 0

    def _apply_partial_rope(self, xq, xk, past_len: int):
        """只旋转前 ``rotary_dim`` 维：按全局位置查自建频率表。"""
        rd = self.rotary_dim
        seq_len = xq.shape[1]
        pos = torch.arange(past_len, past_len + seq_len, device=xq.device)
        # 形状 (seq, rd)：apply_rotary_pos_emb 内部会 unsqueeze(1) 成 (seq, 1, rd)，
        # 正好广播到 (bsz, seq, heads, rd)（与 gqa 传入的 cos/sin 形状一致）
        cos = self.partial_freqs_cos[pos].to(xq.dtype)
        sin = self.partial_freqs_sin[pos].to(xq.dtype)
        q_rot, q_pass = xq[..., :rd], xq[..., rd:]
        k_rot, k_pass = xk[..., :rd], xk[..., rd:]
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 输出门：与 q/k/v 同源，直接从输入 hidden 投影
        gate = torch.sigmoid(self.gate_proj(x))
        # 记录门控值的分布：门控饱和（全接近 0 或 1）是这类结构的典型失效模式，
        # 而它不会体现在 loss 上。不参与计算图。
        if self.training:
            with torch.no_grad():
                g = gate.detach().float()
                self.last_gate_mean = float(g.mean())
                self.last_gate_std = float(g.std(unbiased=False))
                self.last_gate_sat_lo = float((g < 0.01).float().mean())
                self.last_gate_sat_hi = float((g > 0.99).float().mean())
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        if self.use_v_norm:
            xv = self.v_norm(xv)
        # 记录 Q/K/V 的 RMS（廉价、不参与计算图），用于观察各投影的数值尺度
        if self.training:
            with torch.no_grad():
                self.last_q_rms = float(xq.detach().float().pow(2).mean().sqrt())
                self.last_k_rms = float(xk.detach().float().pow(2).mean().sqrt())
                self.last_v_rms = float(xv.detach().float().pow(2).mean().sqrt())
        if self.use_qk_norm:
            xq, xk = self.q_norm(xq), self.k_norm(xk)
        # ---- 位置编码：优先级见模块 docstring ----
        past_len = int(past_key_value[0].shape[1]) if past_key_value is not None else 0
        if self.positional is None or position_embeddings is None:
            pass  # 分支 1：nope / 无位置张量 —— 不做任何旋转
        elif self.rotary_dim:
            # 分支 3：本组件自己只旋转前 rotary_dim 维
            xq, xk = self._apply_partial_rope(xq, xk, past_len)
        else:
            # 分支 2/4：交给位置编码组件
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
        # 输出门：门控后再做输出投影
        output = self.resid_dropout(self.o_proj(output * gate))
        if self.training:
            with torch.no_grad():
                self.last_out_rms = float(output.detach().float().pow(2).mean().sqrt())
        return output, past_kv
