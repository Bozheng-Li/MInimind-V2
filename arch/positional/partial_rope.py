"""部分维度 RoPE（Partial RoPE）。

只对 ``head_dim`` 的前 ``rotary_dim`` 维施加旋转，其余维度原样透传。
Qwen3-Next 等模型用它降低位置编码对表示空间的占用，
让一部分维度可以自由地编码与位置无关的内容。

``rotary_dim`` 默认取 ``head_dim // 4``（Qwen3-Next 的比例），必须是偶数。
"""
import torch

from ..registry import register
from .rope import RotaryEmbedding, apply_rotary_pos_emb, precompute_freqs_cis


@register("positional_encoding", "partial_rope")
class PartialRotaryEmbedding(RotaryEmbedding):
    def __init__(self, cfg):
        super().__init__(cfg)
        rotary_dim = int(cfg.get("rotary_dim") or max(self.head_dim // 4, 2))
        # RoPE 需要成对旋转，维度必须是偶数
        self.rotary_dim = max(rotary_dim - (rotary_dim % 2), 2)

    def build_buffers(self):
        # 频率表只需覆盖被旋转的那部分维度
        return precompute_freqs_cis(
            dim=self.rotary_dim,
            end=self.max_position_embeddings,
            rope_base=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )

    def apply(self, q, k, cos, sin):
        rd = self.rotary_dim
        q_rot, q_pass = q[..., :rd], q[..., rd:]
        k_rot, k_pass = k[..., :rd], k[..., rd:]
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)
