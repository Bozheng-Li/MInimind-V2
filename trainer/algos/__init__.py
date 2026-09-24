"""训练算法模块集合。

每个算法一个模块、可独立 import 与测试::

    from trainer.algos import get_algorithm
    Algo = get_algorithm("dpo")        # -> DPOAlgorithm
    algo = Algo(args, lm_config, tokenizer, runtime)
"""
from __future__ import annotations

import importlib
from typing import Type

from .base import Algorithm

#: 算法名 -> (模块名, 类名)。按需 import，避免拉起全部依赖。
_ALGO_MODULES = {
    # 预训练：与指令微调分开，避免仅因损失都是 CE 就混为一类。
    "pretrain": ("pretrain.causal_lm", "PretrainAlgorithm"),
    # 监督微调 / 参数高效微调。
    "sft": ("sft.full", "SFTAlgorithm"),
    "lora": ("sft.lora", "LoRAAlgorithm"),
    "qlora": ("sft.qlora", "QLoRAAlgorithm"),
    "distill": ("sft.distill", "DistillAlgorithm"),
    # 离线偏好优化。
    "dpo": ("rl.preference", "DPOAlgorithm"),
    "ipo": ("rl.preference", "IPOAlgorithm"),
    "simpo": ("rl.preference", "SimPOAlgorithm"),
    "cpo": ("rl.preference", "CPOAlgorithm"),
    "orpo": ("rl.preference", "ORPOAlgorithm"),
    "kto": ("rl.preference", "KTOAlgorithm"),
    # 在线强化学习。
    "grpo": ("rl.grpo", "GRPOAlgorithm"),
    "dapo": ("rl.dapo", "DAPOAlgorithm"),
    "rloo": ("rl.rloo", "RLOOAlgorithm"),
    "ppo": ("rl.ppo", "PPOAlgorithm"),
    "agent": ("rl.agent", "AgentAlgorithm"),
}


def available() -> list:
    return sorted(_ALGO_MODULES)


def get_algorithm(name: str) -> Type[Algorithm]:
    """按名字取算法类。"""
    if name not in _ALGO_MODULES:
        raise KeyError(f"未知算法 {name!r}，可用: {available()}")
    module_name, class_name = _ALGO_MODULES[name]
    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, class_name)


__all__ = ["Algorithm", "get_algorithm", "available"]
