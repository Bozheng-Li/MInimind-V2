"""分阶段配置的加载器。

用法::

    from configs import load_config, to_arch_config, train_options, default_config

    cfg = load_config(default_config("pretrain"))   # 或 load_config("configs/pretrain.yaml")
    arch = to_arch_config(cfg)                      # -> ArchConfig，交给 arch 组装模型
    opts = train_options(cfg)                       # -> dict，用作 argparse 默认值

``defaults`` 字段支持链式引用（``pretrain.yaml`` → ``base.yaml``），
每份引用的相对路径按**当前文件所在目录**解析。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import yaml

#: 仓库根目录（configs/ 的上一级）——用于把配置里的相对路径转成绝对路径
REPO_ROOT = Path(__file__).resolve().parent.parent

#: 训练产物的默认根目录。
#:
#: 仓库根只放纯净代码，权重/日志/续训档都在 ``test/`` 下（见 ``test/README.md``），
#: 所以配置里写 ``save_dir: out`` 默认落到 ``<repo>/test/out`` 而不是仓库根的 ``out/``。
#: 想让产物回到仓库根，设环境变量 ``MINIMIND_ARTIFACT_ROOT`` 即可（例如
#: ``MINIMIND_ARTIFACT_ROOT=/home/me/minimind`` 就是迁移前的行为）。
#:
#: 用环境变量而不是写死 ``test/``：这样 ``test/`` 被改名、被移走、或者换一台机器
#: 部署成「代码与产物同目录」时，都不需要改代码。
ARTIFACT_ROOT = Path(os.environ.get("MINIMIND_ARTIFACT_ROOT") or (REPO_ROOT / "test"))
if not ARTIFACT_ROOT.is_absolute():
    ARTIFACT_ROOT = (REPO_ROOT / ARTIFACT_ROOT).resolve()

#: 模型结构相关的段名（其余段如 data / train / algo 不参与 ArchConfig）
SLOT_KEYS = ("model", "norm", "positional_encoding", "attention", "feedforward")

#: 阶段 -> 默认配置文件
STAGES = ("pretrain", "sft", "rl")

#: 算法 -> 所属阶段（决定读哪份阶段配置）
STAGE_OF_ALGO = {
    "pretrain": "pretrain",
    "sft": "sft",
    "lora": "sft",
    "qlora": "sft",
    "distill": "sft",
    "dpo": "rl",
    "ipo": "rl",
    "simpo": "rl",
    "cpo": "rl",
    "orpo": "rl",
    "kto": "rl",
    "grpo": "rl",
    "dapo": "rl",
    "rloo": "rl",
    "ppo": "rl",
    "agent": "rl",
}


def stage_of(algo: str) -> str:
    """算法名 -> 阶段名（``configs/<阶段>.yaml``）。"""
    if algo not in STAGE_OF_ALGO:
        raise KeyError(f"未知算法 {algo!r}，可选: {sorted(STAGE_OF_ALGO)}")
    return STAGE_OF_ALGO[algo]


def default_config(stage: str) -> Path:
    """返回某个阶段的默认配置文件路径（与当前工作目录无关）。"""
    if stage not in STAGES:
        raise KeyError(f"未知阶段 {stage!r}，可选: {list(STAGES)}")
    return REPO_ROOT / "configs" / f"{stage}.yaml"


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并两个 dict：override 优先；list 与标量整体替换。

    ``mix`` 例外：**整体替换**而不是按键递归合并。它是「文件名 → 权重」的配比表，
    变体配置写它就是要换一整套配比（例如 mini 版的文件名与完整版完全不同），
    按键合并会把基线的旧文件名残留进来，训练时去找一个不存在的文件。
    """
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict) and key != "mix":
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """读取 YAML 配置，递归处理 ``defaults``，最后叠加 ``overrides``。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"找不到配置文件: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    defaults = raw.pop("defaults", None)

    merged: Dict[str, Any] = {}
    if defaults:
        refs: Iterable[str] = [defaults] if isinstance(defaults, str) else list(defaults)
        for ref in refs:
            merged = deep_merge(merged, load_config(p.parent / ref))
    merged = deep_merge(merged, raw)
    if overrides:
        merged = deep_merge(merged, overrides)
    return merged


def load_stage(stage: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """按阶段名加载（pretrain / sft / rl）。"""
    return load_config(default_config(stage), overrides)


def to_arch_config(cfg: Dict[str, Any]):
    """从合并后的配置里取出**模型结构**部分，构造 ``ArchConfig``。"""
    from arch.schema import ArchConfig

    if "model" not in cfg:
        raise KeyError(f"配置缺少 model 段；已有段: {sorted(cfg)}")
    return ArchConfig({k: cfg[k] for k in SLOT_KEYS if k in cfg})


def train_options(cfg: Dict[str, Any], algo: Optional[str] = None) -> Dict[str, Any]:
    """取训练超参。给了 ``algo`` 就再叠加 ``algo.<name>`` 段里的专属项。

    ``algo`` 段是**可选**的：像 ``pretrain`` / ``sft`` 这类没有专属超参的算法，
    配置里可以只写 ``train`` 段。
    """
    out = dict(cfg.get("train") or {})
    if algo:
        block = (cfg.get("algo") or {}).get(algo)
        if block is not None:
            out = deep_merge(out, block)

    # save_dir 解析成绝对路径：配置里写 "out" 即 ARTIFACT_ROOT/out/
    #（默认 <repo>/test/out，仓库根不落训练产物）。设 MINIMIND_ARTIFACT_ROOT
    # 可以改这个根。绝对路径原样使用，因此 --save_dir /tmp/x 仍然想去哪去哪。
    if out.get("save_dir"):
        p = Path(out["save_dir"])
        if not p.is_absolute():
            out["save_dir"] = str((ARTIFACT_ROOT / p).resolve())
    # 外部模型路径同样不能依赖启动 cwd。配置里的相对路径以仓库根为基准；SGLang
    # 权重交换目录属于训练产物，放到 ARTIFACT_ROOT 下。
    for key in ("reward_model_path", "sglang_model_path"):
        if out.get(key):
            p = Path(out[key])
            if not p.is_absolute():
                out[key] = str((REPO_ROOT / p).resolve())
    if out.get("sglang_shared_path"):
        p = Path(out["sglang_shared_path"])
        if not p.is_absolute():
            out["sglang_shared_path"] = str((ARTIFACT_ROOT / p).resolve())
    return out


def data_options(cfg: Dict[str, Any], algo: Optional[str] = None) -> Dict[str, Any]:
    """取数据配置，并把 ``path`` 解析成**绝对路径**（相对仓库根目录）。

    这样无论从哪个工作目录启动训练脚本都能找到数据；
    配置里无需写 ``../dataset/...`` 这种依赖 cwd 的相对路径。
    """
    out = dict(cfg.get("data") or {})
    if algo:
        block = (cfg.get("algo") or {}).get(algo) or {}
        # algo 段可以覆盖数据路径与序列长度（如 DPO 用 dpo.jsonl + 1024）
        if "data_path" in block:
            out["path"] = block["data_path"]
        if "max_seq_len" in block:
            out["max_seq_len"] = block["max_seq_len"]

    path = out.get("path")
    if path:
        p = Path(path)
        out["path"] = str(p if p.is_absolute() else (REPO_ROOT / p))

    # 配比混合：data.mix 的键是相对 data.path 目录的文件名。
    # 只有预训练用它（PretrainAlgorithm 读 args.data_mix）；其它算法的 argparse
    # 没有 data_mix，由 apply_to_parser 按 known 过滤，不会塞进去。
    mix = out.get("mix")
    if isinstance(mix, dict) and mix:
        out["mix"] = {str(k): float(v) for k, v in mix.items()}
    return out


def is_moe(cfg: Dict[str, Any]) -> bool:
    """配置里的前馈实现是否为 MoE 家族。"""
    return str((cfg.get("feedforward") or {}).get("type", "")).startswith("moe")


def build_lm_config(args, cfg: Dict[str, Any]):
    """按「YAML 定结构 + 旧命令行旋钮可覆盖」的规则构造 ``MiniMindConfig``。

    覆盖规则（对应 argparse 语义）：
    - ``--hidden_size`` / ``--num_hidden_layers`` 总是可用命令行覆盖 YAML
    - ``--use_moe 1`` 强制切到 MoE；``--use_moe 0`` 表示「沿用 YAML 的选择」
      （因为 argparse 默认值已由 :func:`apply_to_parser` 从 YAML 同步过来）
    """
    from model.model_minimind import MiniMindConfig

    arch = to_arch_config(cfg)
    # 蒸馏脚本用 student_* 命名，这里做一次兼容取值
    hidden = getattr(args, "hidden_size", None) or getattr(args, "student_hidden_size")
    layers = getattr(args, "num_hidden_layers", None) or getattr(args, "student_num_layers")
    use_moe = getattr(args, "use_moe", 0) or getattr(args, "student_use_moe", 0)
    arch.model["hidden_size"] = int(hidden)
    arch.model["num_hidden_layers"] = int(layers)
    # ⚠️ 只在配置本身是**稠密**前馈时才把 type 强切成 "moe"。
    # 因为 apply_to_parser 会把 --use_moe 从配置同步过来（moe_* 都以 "moe" 开头 ⇒ 1），
    # 若无条件覆写，就会把 moe_shared / moe_finegrained 一律降级成普通 top-k MoE ——
    # 共享专家、aux-loss-free 全部丢失，且外面看不出来（参数量只差共享专家那一份）。
    if use_moe and not str(arch.type_of("feedforward")).startswith("moe"):
        arch.slots["feedforward"]["type"] = "moe"
    return MiniMindConfig.from_arch_config(arch)


def apply_to_parser(parser, stage: str, algo: Optional[str] = None, argv=None) -> Dict[str, Any]:
    """把配置的值写成 argparse 的**默认值**，并返回合并后的配置。

    命令行显式传入的参数仍然优先 —— 这是 argparse 的固有语义：
    ``set_defaults`` 只改默认值，不影响已解析出的显式值。

    配置的合成方式（便于做架构对比实验）：

        configs/<阶段>.yaml   ← 基线（数据 + 训练超参 + 默认模型结构）
              ↓ 被叠加
        --config 指定的文件    ← 变体，通常只写要改的槽位

    因此一份「只换了注意力」的实验配置可以只有几行，数据与超参自动沿用阶段配置。
    若变体里也写了 ``data`` / ``train``，则按字段覆盖基线。

    同时会把模型结构里的三个旧旋钮（``hidden_size`` / ``num_hidden_layers`` /
    ``use_moe``）同步成默认值，避免它们以硬编码默认值压过配置。
    """
    import sys as _sys

    argv = _sys.argv[1:] if argv is None else argv
    pre, _ = parser.parse_known_args(argv)

    cfg = load_config(default_config(stage))
    variant = getattr(pre, "config", None)
    if variant:
        cfg = deep_merge(cfg, load_config(variant))

    updates = {}
    known = {action.dest for action in parser._actions}

    for key, value in train_options(cfg, algo).items():
        if key in known:
            updates[key] = value
    data = data_options(cfg, algo)
    if "path" in data and "data_path" in known:
        updates["data_path"] = data["path"]
    if "max_seq_len" in data and "max_seq_len" in known:
        updates["max_seq_len"] = data["max_seq_len"]
    if "mix" in data and "data_mix" in known:
        updates["data_mix"] = data["mix"]

    arch = to_arch_config(cfg)
    if "hidden_size" in known:
        updates["hidden_size"] = arch.model["hidden_size"]
    if "num_hidden_layers" in known:
        updates["num_hidden_layers"] = arch.model["num_hidden_layers"]
    if "use_moe" in known:
        updates["use_moe"] = 1 if is_moe(cfg) else 0

    # 蒸馏脚本用的是 student_* / teacher_* 命名（历史原因），单独映射一次
    ffn = arch.slots["feedforward"]
    if "student_hidden_size" in known:
        updates["student_hidden_size"] = arch.model["hidden_size"]
    if "student_num_layers" in known:
        updates["student_num_layers"] = arch.model["num_hidden_layers"]
    if "student_use_moe" in known:
        updates["student_use_moe"] = 1 if is_moe(cfg) else 0
    if "teacher_hidden_size" in known:
        updates["teacher_hidden_size"] = arch.model["hidden_size"]
    if "teacher_num_layers" in known:
        updates["teacher_num_layers"] = arch.model["num_hidden_layers"]

    parser.set_defaults(**updates)
    return cfg
