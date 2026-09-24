"""按配置构造各组件，并组装成完整模型。

组件的选择完全由 ``ArchConfig`` 的槽位决定，本文件只负责：
1. 用 ``registry.resolve`` 把 ``type`` 解析成具体类
2. 把「全局参数 + 槽内参数」合并成组件配置传进去
3. 处理跨槽依赖（位置编码需要 attention 的 ``head_dim``）
"""
from __future__ import annotations

from typing import Any, Optional

from .block import ArchBlock
from .norm import build_norm
from .registry import registered, resolve
from .schema import ArchConfig, arch_config_from, load_arch_config


def _norm_eps(arch: ArchConfig) -> float:
    cfg = arch.component("norm")
    return cfg.get("eps", cfg.get("rms_norm_eps", 1e-6))


def build_norm_layer(arch: ArchConfig, dim: int):
    """构造一个归一化层（用于层间 norm 与最终 norm）。"""
    cls = resolve("norm", arch.type_of("norm"))
    return build_norm(cls, dim, _norm_eps(arch))


def build_positional(arch: ArchConfig):
    """构造位置编码组件，并把**旋转维数**注入进去。

    旋转维数不一定是 ``head_dim``：MLA 只对解耦的那部分维度施加 RoPE
    （``qk_rope_head_dim``，通常远小于 head_dim）。因此注意力组件可以声明
    ``positional_dim(cfg) -> int``；不声明时退回 ``head_dim``。
    """
    attn_cfg = arch.component("attention")
    attn_cls = resolve("attention", arch.type_of("attention"))
    dim_fn = getattr(attn_cls, "positional_dim", None)
    rotary_dim = int(dim_fn(attn_cfg)) if callable(dim_fn) else int(attn_cfg.head_dim)

    cls = resolve("positional_encoding", arch.type_of("positional_encoding"))
    return cls(arch.component("positional_encoding", extra={"head_dim": rotary_dim}))


def build_attention(arch: ArchConfig, positional=None):
    """构造注意力组件，并把位置编码组件注入进去。

    接口约定：``cls(cfg, positional=None)``。注意力层在 forward 里
    通过 ``self.positional.apply(q, k, cos, sin)`` 委托位置编码，
    因此 ``nope`` / ``partial_rope`` 等实现才能即插即用。
    """
    cls = resolve("attention", arch.type_of("attention"))
    return cls(arch.component("attention"), positional=positional)


def build_feedforward(arch: ArchConfig):
    cls = resolve("feedforward", arch.type_of("feedforward"))
    return cls(arch.component("feedforward"))


def build_block(arch: ArchConfig, positional=None) -> ArchBlock:
    hidden = arch.model["hidden_size"]
    return ArchBlock(
        self_attn=build_attention(arch, positional=positional),
        input_layernorm=build_norm_layer(arch, hidden),
        post_attention_layernorm=build_norm_layer(arch, hidden),
        mlp=build_feedforward(arch),
    )


def describe(arch: ArchConfig) -> str:
    """可读的架构清单，便于日志与调试。"""
    lines = [arch.summary(), "  可用组件:"]
    for kind, names in registered().items():
        lines.append(f"    {kind:<20} {names}")
    return "\n".join(lines)


def build_model(config: Any = None, yaml_path: Optional[str] = None):
    """构造完整的 ``ArchForCausalLM``。

    参数二选一：
    - ``config``：``ArchConfig`` / 带旧属性的对象（如 ``MiniMindConfig``）/ dict
    - ``yaml_path``：``config.yaml`` 路径
    """
    if yaml_path is not None:
        arch = load_arch_config(yaml_path)
    else:
        arch = arch_config_from(config)

    from .config_class import ArchPretrainedConfig  # 延迟导入，避免循环
    from .model import ArchForCausalLM

    return ArchForCausalLM(ArchPretrainedConfig(arch))
