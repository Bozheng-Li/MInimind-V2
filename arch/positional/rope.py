"""RoPE（旋转位置编码），含 YaRN 长文本外推。

与重构前 ``model/model_minimind.py`` 的 ``precompute_freqs_cis`` /
``apply_rotary_pos_emb`` 逐字一致，只是包成了可插拔组件。

注意：RoPE 的 cos/sin 是 **非持久化 buffer**，不会进 state_dict。
因此 ``rope_theta`` / ``max_position_embeddings`` 这类改动不会导致权重加载报错，
但会静默改变数值行为 —— 换位置编码配置后请确认权重来源。
"""
import math

import torch

from ..registry import register

# YaRN 外推的默认参数（与重构前 MiniMindConfig 中的硬编码一致）
DEFAULT_YARN = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 16,
    "original_max_position_embeddings": 2048,
    "attention_factor": 1.0,
    "type": "yarn",
}


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:  # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


@register("positional_encoding", "rope")
class RotaryEmbedding:
    """RoPE 位置编码。"""

    #: 该实现需要模型注册 cos/sin 两个 buffer
    buffer_names = ("freqs_cos", "freqs_sin")

    def __init__(self, cfg):
        self.head_dim = int(cfg.head_dim)
        self.max_position_embeddings = int(cfg.max_position_embeddings)
        # 防御性 float()：手写 YAML 里 1.0e6 这种裸指数会被 PyYAML 当字符串
        self.rope_theta = float(cfg.get("rope_theta", 1e6))

        # 显式给了 rope_scaling 就用它；否则看 inference_rope_scaling 开关
        scaling = cfg.get("rope_scaling")
        if scaling is None and cfg.get("inference_rope_scaling", False):
            scaling = dict(DEFAULT_YARN)
        self.rope_scaling = scaling

    def build_buffers(self):
        """返回 (freqs_cos, freqs_sin)，由模型注册为持久化=False 的 buffer。"""
        return precompute_freqs_cis(
            dim=self.head_dim,
            end=self.max_position_embeddings,
            rope_base=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )

    @staticmethod
    def apply(q, k, cos, sin):
        return apply_rotary_pos_emb(q, k, cos, sin)
