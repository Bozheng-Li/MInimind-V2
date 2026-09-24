"""架构配置：读取 ``config.yaml``，并把「全局参数 + 槽内参数」合并后交给组件。

设计要点
--------
- **扁平槽位**：每个槽只有 ``type`` + 该组件需要的任意参数。新增组件时
  *不需要* 改动本文件的 schema —— 在 YAML 里写参数即可。
- **全局参数优先**：``[model]`` 段里的 ``hidden_size``、``dropout`` 等属于
  模型级不变量，槽内写了也不允许覆盖，避免把结构改乱。
- **派生默认值**：``head_dim``、``intermediate_size`` 若未显式给出则按
  与旧实现一致的公式推导，保证默认配置精确复现原架构。
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

import yaml

# 四个可替换槽位
SLOTS = ("norm", "positional_encoding", "attention", "feedforward")

# 槽位缺失时的默认实现（与重构前的硬编码架构一致）
DEFAULT_SLOT_TYPE = {
    "norm": "rmsnorm",
    "positional_encoding": "rope",
    "attention": "gqa",
    "feedforward": "swiglu",
}

# 这些键来自 [model] 段，属于模型级不变量，不允许被槽内参数覆盖
PROTECTED = (
    "hidden_size",
    "num_hidden_layers",
    "vocab_size",
    "max_position_embeddings",
    "rms_norm_eps",
    "dropout",
    "tie_word_embeddings",
    "hidden_act",
    "bos_token_id",
    "eos_token_id",
)


class SlotConfig:
    """属性访问式配置容器，组件在 ``__init__`` 里按需读取。"""

    def __init__(self, data: Dict[str, Any], slot: str = ""):
        # 用 object.__setattr__ 避免触发下面的 __getattr__
        object.__setattr__(self, "_data", dict(data))
        object.__setattr__(self, "_slot", slot)

    def __getattr__(self, key: str) -> Any:
        # 仅当常规属性查找失败时才会走到这里
        data = object.__getattribute__(self, "_data")
        if key in data:
            return data[key]
        raise AttributeError(
            f"[{object.__getattribute__(self, '_slot')}] 配置缺少字段 {key!r}；"
            f"已有字段: {sorted(data)}"
        )

    def get(self, key: str, default: Any = None) -> Any:
        return object.__getattribute__(self, "_data").get(key, default)

    def override(self, **kwargs: Any) -> "SlotConfig":
        """派生一份改了几个字段的新配置（组件构造子组件时用，如 MoE 造专家）。"""
        data = dict(object.__getattribute__(self, "_data"))
        data.update(kwargs)
        return SlotConfig(data, object.__getattribute__(self, "_slot"))

    def has(self, key: str) -> bool:
        return key in object.__getattribute__(self, "_data")

    def as_dict(self) -> Dict[str, Any]:
        return dict(object.__getattribute__(self, "_data"))

    def __contains__(self, key: str) -> bool:
        return key in object.__getattribute__(self, "_data")

    def __repr__(self) -> str:
        return f"SlotConfig(slot={object.__getattribute__(self, '_slot')!r}, keys={sorted(object.__getattribute__(self, '_data'))})"


class ArchConfig:
    """一份完整的架构配置（全局参数 + 四个槽）。"""

    def __init__(self, raw: Dict[str, Any], source: str = "<dict>"):
        if not isinstance(raw, dict):
            raise TypeError(f"架构配置必须是 mapping，收到 {type(raw).__name__}")
        if "model" not in raw:
            raise ValueError(f"架构配置缺少 [model] 段（来源: {source}）")

        self.source = source
        self.raw = raw
        self.model: Dict[str, Any] = dict(raw["model"])

        if "hidden_size" not in self.model:
            raise ValueError(f"[model] 缺少 hidden_size（来源: {source}）")

        # 规整槽位：允许写成字符串简写（type: gqa）、dict，或省略
        self.slots: Dict[str, Dict[str, Any]] = {}
        for slot in SLOTS:
            value = raw.get(slot)
            if value is None:
                value = {"type": DEFAULT_SLOT_TYPE[slot]}
            elif isinstance(value, str):
                value = {"type": value}
            elif not isinstance(value, dict):
                raise TypeError(
                    f"槽 [{slot}] 必须是字符串或 mapping，收到 {type(value).__name__}"
                )
            if "type" not in value:
                raise ValueError(f"槽 [{slot}] 缺少 'type' 字段（来源: {source}）")
            self.slots[slot] = dict(value)

        self._derive_defaults()

    # ------------------------------------------------------------------ #
    def _derive_defaults(self) -> None:
        """补齐派生参数，公式与旧实现保持一致。"""
        hidden = int(self.model["hidden_size"])
        num_heads = int(self.slots["attention"].get("num_attention_heads", 8))

        # head_dim：旧实现 = hidden_size // num_attention_heads
        self.slots["attention"].setdefault("head_dim", hidden // num_heads)

        # intermediate_size：旧实现 = ceil(hidden * pi / 64) * 64
        ffn = self.slots["feedforward"]
        ffn.setdefault("intermediate_size", math.ceil(hidden * math.pi / 64) * 64)
        ffn.setdefault("moe_intermediate_size", ffn["intermediate_size"])

        # MoE 缺省参数（与旧 MiniMindConfig 默认一致）
        ffn.setdefault("num_experts", 4)
        ffn.setdefault("num_experts_per_tok", 1)
        ffn.setdefault("norm_topk_prob", True)
        ffn.setdefault("router_aux_loss_coef", 5e-4)

        # 全局缺省
        self.model.setdefault("dropout", 0.0)
        self.model.setdefault("rms_norm_eps", 1e-6)
        self.model.setdefault("hidden_act", "silu")
        self.model.setdefault("tie_word_embeddings", True)

    # ------------------------------------------------------------------ #
    def type_of(self, slot: str) -> str:
        """该槽选用的组件短名（或 import 路径）。"""
        return self.slots[slot]["type"]

    def component(self, slot: str, extra: Optional[Dict[str, Any]] = None) -> SlotConfig:
        """合并全局参数与该槽参数，产出传给组件构造函数的配置对象。

        ``extra`` 用于注入跨槽的派生量（例如位置编码需要 attention 槽的 ``head_dim``）。
        """
        merged: Dict[str, Any] = dict(self.model)
        merged.update(self.slots[slot])
        if extra:
            merged.update(extra)
        # 全局不变量不被槽覆盖
        for key in PROTECTED:
            if key in self.model:
                merged[key] = self.model[key]
        # type 只用于 resolve()，不必传进组件
        merged.pop("type", None)
        return SlotConfig(merged, slot)

    def summary(self) -> str:
        """一行可读摘要，训练日志里打印用。"""
        parts = [f"norm={self.type_of('norm')}", f"pe={self.type_of('positional_encoding')}",
                 f"attn={self.type_of('attention')}", f"ffn={self.type_of('feedforward')}"]
        if self.type_of("feedforward") == "moe":
            ffn = self.slots["feedforward"]
            parts.append(f"experts={ffn.get('num_experts')}x top{ffn.get('num_experts_per_tok')}")
        return (f"ArchConfig(d_model={self.model['hidden_size']}, "
                f"layers={self.model['num_hidden_layers']}, " + ", ".join(parts) + ")")

    def to_dict(self) -> Dict[str, Any]:
        """导出为可回写的 dict（含推导后的默认值，便于存档复现）。"""
        return {
            "model": dict(self.model),
            **{slot: dict(cfg) for slot, cfg in self.slots.items()},
        }

    def __repr__(self) -> str:
        return f"ArchConfig(source={self.source!r}, {self.summary()})"


def load_arch_config(path: str) -> ArchConfig:
    """从 YAML 文件读取架构配置。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到架构配置文件: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ArchConfig(raw, source=path)


def dump_arch_config(cfg: ArchConfig, path: str) -> None:
    """把（含默认值的）配置写回 YAML，便于实验复现。"""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, allow_unicode=True, sort_keys=False)


def synthesize_arch_config(src: Any = None) -> ArchConfig:
    """把「旧的参数对象」合成为等价 ArchConfig。

    目的是让 ``MiniMindConfig(hidden_size=..., num_hidden_layers=..., use_moe=...)``
    这条老路径产出的结构与重构前**逐位一致**，从而保证已训权重仍能严格加载。
    ``src`` 可以是任意带同名属性的对象（``MiniMindConfig``、``argparse.Namespace``、
    ``SimpleNamespace``...），缺失的键走旧实现的默认值。
    """

    def g(key: str, default: Any) -> Any:
        return getattr(src, key, default) if src is not None else default

    hidden = int(g("hidden_size", 768))
    num_heads = int(g("num_attention_heads", 8))
    use_moe = bool(g("use_moe", False))
    raw = {
        "model": {
            "hidden_size": hidden,
            "num_hidden_layers": int(g("num_hidden_layers", 8)),
            "vocab_size": int(g("vocab_size", 6400)),
            "max_position_embeddings": int(g("max_position_embeddings", 32768)),
            "rms_norm_eps": g("rms_norm_eps", 1e-6),
            "dropout": g("dropout", 0.0),
            "hidden_act": g("hidden_act", "silu"),
            "tie_word_embeddings": bool(g("tie_word_embeddings", True)),
            "bos_token_id": g("bos_token_id", 1),
            "eos_token_id": g("eos_token_id", 2),
        },
        "norm": {"type": "rmsnorm"},
        "positional_encoding": {
            "type": "rope",
            "rope_theta": g("rope_theta", 1e6),
            "rope_scaling": g("rope_scaling", None),
            "inference_rope_scaling": bool(g("inference_rope_scaling", False)),
        },
        "attention": {
            "type": "gqa",
            "num_attention_heads": num_heads,
            "num_key_value_heads": g("num_key_value_heads", 4),
            "head_dim": g("head_dim", hidden // num_heads),
            "qk_norm": True,                      # 旧实现恒有 q_norm / k_norm
            "flash_attn": bool(g("flash_attn", True)),
        },
        "feedforward": {
            "type": "moe" if use_moe else "swiglu",
            "intermediate_size": g("intermediate_size", math.ceil(hidden * math.pi / 64) * 64),
            "num_experts": g("num_experts", 4),
            "num_experts_per_tok": g("num_experts_per_tok", 1),
            "norm_topk_prob": bool(g("norm_topk_prob", True)),
            "router_aux_loss_coef": g("router_aux_loss_coef", 5e-4),
            "moe_intermediate_size": g(
                "moe_intermediate_size", math.ceil(hidden * math.pi / 64) * 64
            ),
        },
    }
    return ArchConfig(raw, source="<synthesized>")


def arch_config_from(obj: Any = None) -> ArchConfig:
    """从多种输入统一推导出 ``ArchConfig``。

    接受：``ArchConfig`` 本身 / 形如 ``{"model": {...}, ...}`` 的字典 /
    任何带 ``to_arch_config()`` 方法的对象 / 任何带旧属性名的对象。
    """
    if isinstance(obj, ArchConfig):
        return obj
    if isinstance(obj, dict):
        if "model" in obj:
            return ArchConfig(obj)
        return synthesize_arch_config(_AttrDict(obj))
    to_arch = getattr(obj, "to_arch_config", None)
    if callable(to_arch):
        return to_arch()
    arch = getattr(obj, "arch", None)
    if isinstance(arch, ArchConfig):
        return arch
    if isinstance(arch, dict) and "model" in arch:
        return ArchConfig(arch)
    return synthesize_arch_config(obj)


class _AttrDict:
    """把 dict 包成可 ``getattr`` 的对象，供 synthesize_arch_config 使用。"""

    def __init__(self, data: Dict[str, Any]):
        self.__dict__.update(data)
