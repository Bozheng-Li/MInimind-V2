"""分阶段配置（pretrain / sft / rl）。

- ``base.yaml``      默认模型结构，被各阶段通过 ``defaults`` 引用
- ``pretrain.yaml``  预训练 = base + 数据 + 训练超参
- ``sft.yaml``       监督微调
- ``rl.yaml``        强化学习（PPO / GRPO / CISPO / Agent 共用，按算法分段）
- ``loader.py``      加载与合并逻辑

所有配置都在本目录，模型代码（``arch/``）不再自带配置。
"""
from .loader import (
    ARTIFACT_ROOT,
    REPO_ROOT,
    SLOT_KEYS,
    STAGE_OF_ALGO,
    STAGES,
    apply_to_parser,
    build_lm_config,
    data_options,
    deep_merge,
    default_config,
    is_moe,
    load_config,
    load_stage,
    stage_of,
    to_arch_config,
    train_options,
)

__all__ = [
    "ARTIFACT_ROOT",
    "REPO_ROOT",
    "SLOT_KEYS",
    "STAGES",
    "STAGE_OF_ALGO",
    "default_config",
    "stage_of",
    "load_config",
    "load_stage",
    "to_arch_config",
    "train_options",
    "data_options",
    "deep_merge",
    "is_moe",
    "build_lm_config",
    "apply_to_parser",
]
