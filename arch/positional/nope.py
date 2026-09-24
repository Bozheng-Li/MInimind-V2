"""无位置编码（NoPE）。

不施加任何位置信息，模型只靠因果掩码区分 token 顺序。
近年研究发现：在因果注意力的 Decoder-only 结构里，位置信息其实已经隐含在
掩码结构中，显式位置编码并非必需。

本组件不需要任何 buffer，``apply`` 是恒等映射。
"""
from ..registry import register


@register("positional_encoding", "nope")
class NoPositionalEncoding:
    #: 不需要模型注册任何 buffer
    buffer_names = None

    def __init__(self, cfg):
        self.head_dim = int(cfg.head_dim)

    def build_buffers(self):
        return None

    @staticmethod
    def apply(q, k, cos, sin):
        return q, k
