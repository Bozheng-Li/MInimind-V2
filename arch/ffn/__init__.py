"""前馈网络组件。

接口约定：``cls(cfg)``，``forward(x) -> Tensor``；
MoE 实现额外暴露 ``aux_loss``（负载均衡损失），由模型聚合后返回给训练脚本。

注意：子模块必须在此**立即导入**，否则 ``@register`` 装饰器不会执行，
``resolve()`` 就找不到它们。
"""
from .geglu import GeGLUFeedForward
from .moe import MoEFeedForward
from .moe_finegrained import MoEFineGrainedFeedForward
from .moe_shared import MoESharedFeedForward
from .swiglu import SwiGLUFeedForward

__all__ = [
    "SwiGLUFeedForward",
    "GeGLUFeedForward",
    "MoEFeedForward",
    "MoESharedFeedForward",
    "MoEFineGrainedFeedForward",
]
