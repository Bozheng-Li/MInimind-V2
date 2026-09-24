"""注意力组件。

接口约定
--------
``cls(cfg)`` 且实现::

    forward(x, position_embeddings, past_key_value, use_cache, attention_mask)
        -> (output, present_state)

其中 ``present_state`` 是**不透明的**增量状态：

- softmax 注意力（gqa / sliding_window / mla...）返回 ``(k, v)`` 张量元组
- 线性注意力（deltanet...）返回定长循环状态矩阵

模型侧不解析状态内容，只在需要「已处理了多少个 token」时调用 ``state_seq_len``。
线性注意力的状态没有序列维，因此返回 0 —— 这也是它区别于 KV cache 的关键点。

属性命名约定（``q_proj`` / ``k_proj`` / ...）与重构前保持一致，
这样已有的 ``*.pth`` 权重仍能严格加载。
"""
from torch import nn

from .compressed import CompressedAttention
from .deltanet import GatedDeltaNet
from .gated import GatedAttention
from .gqa import GQAttention, repeat_kv
from .mla import MLAttention
from .sliding_window import SlidingWindowAttention

__all__ = [
    "GQAttention", "GatedAttention", "SlidingWindowAttention", "CompressedAttention",
    "MLAttention", "GatedDeltaNet",
    "repeat_kv", "get_state_seq_len",
]


def get_state_seq_len(attn: nn.Module, state) -> int:
    """向注意力组件询问「该状态里已缓存了多少个 token」。

    组件可自行实现 ``state_seq_len(state) -> int``；未实现时回退到
    「``state[0]`` 的第 1 维即序列长度」这一 softmax 注意力惯例。

    模型侧必须走这个函数，不要直接读 ``state[0].shape[1]`` ——
    否则线性注意力（状态定长、无序列维）会被算错。
    """
    if state is None:
        return 0
    fn = getattr(attn, "state_seq_len", None)
    if callable(fn):
        return int(fn(state))
    try:
        return int(state[0].shape[1])
    except Exception:
        return 0
