"""MiniMind 可组合模型架构（YAML 驱动）。

用法::

    from arch import build_model, load_arch_config

    # 方式一：从 YAML 组装
    model = build_model(yaml_path="configs/base.yaml")

    # 方式二：从 ArchConfig 组装
    model = build_model(load_arch_config("configs/base.yaml"))

    # 方式三：沿用旧参数（等价于重构前的结构）
    from model.model_minimind import MiniMindConfig
    model = build_model(MiniMindConfig(hidden_size=768, num_hidden_layers=8))

目录结构::

    arch/
    ├── config.yaml            全局配置：自由组合模型结构
    ├── registry.py            组件注册表（短名 / import 路径）
    ├── schema.py              YAML 解析、校验、默认值推导、旧参数合成
    ├── build.py               组件构造与组装
    ├── block.py               可组合 Transformer Block
    ├── model.py               ArchModel / ArchForCausalLM
    ├── norm/                  归一化组件
    ├── positional/            位置编码组件
    ├── attention/             注意力组件
    └── ffn/                   前馈网络组件
"""
from . import attention, ffn, norm, positional  # noqa: F401  导入即触发组件注册
from .block import ArchBlock
from .build import (
    build_attention,
    build_block,
    build_feedforward,
    build_model,
    build_norm_layer,
    build_positional,
    describe,
)
from .config_class import ArchPretrainedConfig
from .model import ArchForCausalLM, ArchModel
from .registry import available, kinds, register, registered, resolve
from .schema import (
    SLOTS,
    ArchConfig,
    SlotConfig,
    arch_config_from,
    dump_arch_config,
    load_arch_config,
    synthesize_arch_config,
)

__all__ = [
    # 配置
    "ArchConfig", "SlotConfig", "SLOTS",
    "load_arch_config", "dump_arch_config",
    "arch_config_from", "synthesize_arch_config",
    # 注册表
    "register", "resolve", "available", "kinds", "registered",
    # 组装
    "build_model", "build_block", "build_attention", "build_feedforward",
    "build_norm_layer", "build_positional", "describe",
    # 模型
    "ArchModel", "ArchForCausalLM", "ArchBlock",
]
