"""归一化组件。

接口约定：``cls(dim, eps=...)``。

注意：``bias`` 参数只有 LayerNorm 有，因此切换 norm 类型会改变
``state_dict`` 的键（这是预期行为，不是 bug）。
"""
from .layernorm import LayerNorm
from .rmsnorm import RMSNorm
from .rmsnorm_zero_centered import RMSNormZeroCentered

__all__ = ["RMSNorm", "RMSNormZeroCentered", "LayerNorm", "build_norm"]


def build_norm(norm_cls: type, dim: int, eps: float):
    """按约定构造一个归一化层。

    约定：归一化组件的构造函数接受 ``(dim, eps=...)``。
    若自定义归一化签名不同，可在此处适配或直接传工厂函数。
    """
    return norm_cls(dim, eps=eps)
