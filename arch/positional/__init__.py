"""位置编码组件。

组件接口约定
------------
- ``cls(cfg)``：从合并后的配置读参数
- ``build_buffers()``：返回 ``(cos, sin)`` 或 ``None``（无位置编码时）
- ``buffer_names``：需要注册的 buffer 名，``None`` 表示不需要
- ``apply(q, k, cos, sin) -> (q, k)``：把位置信息作用到 q/k 上

模型通过 ``self.positional`` 把这个组件注入每个注意力层，
注意力层再调用 ``apply`` —— 位置编码因此是真正可插拔的。
"""
from .nope import NoPositionalEncoding
from .partial_rope import PartialRotaryEmbedding
from .rope import RotaryEmbedding, apply_rotary_pos_emb, precompute_freqs_cis

__all__ = [
    "RotaryEmbedding",
    "PartialRotaryEmbedding",
    "NoPositionalEncoding",
    "apply_rotary_pos_emb",
    "precompute_freqs_cis",
]
