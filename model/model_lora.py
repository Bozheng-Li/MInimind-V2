"""MiniMind 的 LoRA / QLoRA 底层实现。

设计目标是同时满足三件事：

1. LoRA 遵循论文公式 ``W'x = Wx + (alpha / r) * BAx``；
2. 不改变基座权重在 ``state_dict`` 中的键名，旧权重仍可正常加载；
3. QLoRA 可把冻结的 ``nn.Linear`` 换成 bitsandbytes NF4 层，再挂同一套 LoRA。

训练算法入口位于 ``trainer/algos/sft/lora.py`` 和 ``qlora.py``。本文件只负责
模型变换、适配器存取和合并，不负责优化器或数据集。
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from types import MethodType

import torch
from torch import nn

# 默认覆盖注意力和 FFN 的投影层，但刻意排除 ``lm_head`` 与 MoE 路由器 ``gate``。
# 这比旧实现的“只匹配方阵”更准确：GQA 的 K/V、SwiGLU 的 up/down 都不是方阵，
# 旧逻辑会静默漏掉这些业界通常会训练的目标层。
DEFAULT_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "kv_dkv_proj", "k_rope_proj", "kv_uk_proj", "kv_uv_proj",
    "q_dq_proj", "q_uq_proj", "alpha_proj", "beta_proj",
)


def _parse_targets(target_modules: str | Sequence[str] | None) -> tuple[str, ...]:
    """把逗号分隔字符串或序列统一成模块叶子名元组。"""
    if target_modules is None:
        return DEFAULT_TARGET_MODULES
    if isinstance(target_modules, str):
        values = [item.strip() for item in target_modules.split(",") if item.strip()]
    else:
        values = [str(item).strip() for item in target_modules if str(item).strip()]
    if not values:
        raise ValueError("target_modules 不能为空")
    return tuple(values)


def _matches(name: str, targets: Iterable[str]) -> bool:
    """目标既可写叶子名（``q_proj``），也可写完整模块路径后缀。"""
    return any(name == target or name.endswith(f".{target}") for target in targets)


def _parent_and_child(model: nn.Module, module_name: str):
    """根据 ``a.b.0.c`` 路径返回父模块与最后一级属性名。"""
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


class LoRA(nn.Module):
    """低秩增量分支。

    A 用 Kaiming 初始化、B 置零，使适配器刚挂载时严格满足 ``BAx = 0``，模型
    输出不会发生突变。dropout 只作用于 LoRA 分支，推理时自动关闭。
    """

    def __init__(self, in_features: int, out_features: int, rank: int,
                 alpha: float | None = None, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank 必须为正整数，收到 {rank}")
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(self.A(self.dropout(x))) * self.scaling


def _linear_forward_with_lora(self, x):
    """线性基座与 LoRA 分支相加；以绑定方法安装，保留原模块/权重键名。"""
    return self._lora_base_forward(x) + self.lora(x)


def apply_lora(model: nn.Module, rank: int = 16, alpha: float | None = None,
               dropout: float = 0.0, target_modules: str | Sequence[str] | None = None):
    """给目标线性层挂载 LoRA，返回实际命中的完整模块名。

    函数可重复调用：已有 ``lora`` 子模块的层会跳过，避免二次叠加。LoRA 参数
    默认用 fp32 保存；autocast 会在前向时选择合适计算精度，这比直接继承基座的
    4-bit/半精度权重 dtype 更稳定。
    """
    targets = _parse_targets(target_modules)
    matched = []
    # 先转 list，避免遍历 named_modules 时新增 ``lora`` 子模块导致迭代结构变化。
    for name, module in list(model.named_modules()):
        if not name or not _matches(name, targets) or hasattr(module, "lora"):
            continue
        if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
            continue
        if not callable(getattr(module, "forward", None)):
            continue
        device = next(module.parameters()).device
        lora = LoRA(module.in_features, module.out_features, rank, alpha, dropout).to(device)
        module.add_module("lora", lora)
        module._lora_base_forward = module.forward
        module.forward = MethodType(_linear_forward_with_lora, module)
        matched.append(name)
    if not matched:
        raise ValueError(f"没有命中任何 LoRA 目标层，target_modules={targets}")
    return matched


def quantize_model_4bit(model: nn.Module, compute_dtype: torch.dtype = torch.bfloat16,
                        quant_type: str = "nf4", use_double_quant: bool = True):
    """把基座线性层原地替换为 bitsandbytes 4-bit 层。

    ``lm_head`` 通常与 embedding 权重绑定，不能直接替换，否则会破坏权重共享；
    其余线性层（包括 MoE 专家和路由器）都量化，以获得 QLoRA 主要的显存收益。
    量化权重始终冻结，梯度只流入稍后挂载的 LoRA 分支。
    """
    try:
        import bitsandbytes as bnb
    except ImportError as exc:  # pragma: no cover - 取决于用户是否安装可选依赖
        raise RuntimeError("QLoRA 需要 bitsandbytes，请安装 requirements-qlora.txt 中的依赖") from exc

    replaced = []
    for name, module in list(model.named_modules()):
        if not name or name == "lm_head" or not isinstance(module, nn.Linear):
            continue
        parent, child = _parent_and_child(model, name)
        old_device = module.weight.device
        quantized = bnb.nn.Linear4bit(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            compute_dtype=compute_dtype,
            compress_statistics=bool(use_double_quant),
            quant_type=quant_type,
            device="cpu",
        )
        # Params4bit 在迁移到 CUDA 时执行真正的块量化；先放 CPU 可避免保留一份
        # 额外的 GPU 全精度副本。赋值后再整体迁移，顺序不能反。
        quantized.weight = bnb.nn.Params4bit(
            module.weight.detach().float().cpu(),
            requires_grad=False,
            compress_statistics=bool(use_double_quant),
            quant_type=quant_type,
            module=quantized,
        )
        if module.bias is not None:
            quantized.bias = nn.Parameter(module.bias.detach().float().cpu(), requires_grad=False)
        quantized = quantized.to(old_device)
        setattr(parent, child, quantized)
        replaced.append(name)
    if not replaced:
        raise ValueError("模型中没有可量化的 nn.Linear 层")
    return replaced


def iter_lora_parameters(model: nn.Module):
    """迭代全部 LoRA 参数；优化器与梯度裁剪共用同一口径。"""
    for name, parameter in model.named_parameters():
        if ".lora." in name:
            yield parameter


def _unwrap(model: nn.Module) -> nn.Module:
    """依次剥掉 DDP 与 torch.compile 包装。"""
    raw = getattr(model, "module", model)
    return getattr(raw, "_orig_mod", raw)


def save_lora(model: nn.Module, path: str):
    """只保存适配器权重，并附带 rank/alpha 元数据。"""
    raw_model = _unwrap(model)
    state_dict = {}
    meta = None
    target_modules = []
    for name, module in raw_model.named_modules():
        if not hasattr(module, "lora"):
            continue
        clean_name = name[7:] if name.startswith("module.") else name
        target_modules.append(clean_name)
        state_dict.update({
            f"{clean_name}.lora.{key}": value.detach().cpu().half()
            for key, value in module.lora.state_dict().items()
        })
        if meta is None:
            meta = {"rank": module.lora.rank, "alpha": module.lora.alpha}
    if not state_dict:
        raise ValueError("模型中没有 LoRA 适配器，拒绝保存空权重")
    meta = dict(meta or {})
    meta["target_modules"] = target_modules
    state_dict["__lora_config__"] = meta
    torch.save(state_dict, path)


def load_lora(model: nn.Module, path: str):
    """加载 LoRA 权重；兼容旧版不带元数据的文件。

    当前模型必须已经挂载适配器。若 checkpoint 带 alpha 元数据，会同步更新每层
    的 ``scaling=alpha/r``；旧实现只加载 A/B，却沿用调用方默认 alpha，使用自定义
    alpha 训练的适配器在推理时会被错误缩放。
    """
    model = _unwrap(model)
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=device)
    meta = state_dict.get("__lora_config__", {})
    expected_rank = meta.get("rank") if isinstance(meta, dict) else None
    expected_alpha = meta.get("alpha") if isinstance(meta, dict) else None
    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items() if key != "__lora_config__"
    }
    loaded = 0
    for name, module in model.named_modules():
        if not hasattr(module, "lora"):
            continue
        if expected_rank is not None and module.lora.rank != int(expected_rank):
            raise ValueError(
                f"LoRA rank 不匹配：checkpoint={expected_rank}，当前层 {name}="
                f"{module.lora.rank}；请使用 apply_lora_from_checkpoint 自动挂载"
            )
        prefix = f"{name}.lora."
        local = {key[len(prefix):]: value for key, value in state_dict.items()
                 if key.startswith(prefix)}
        if local:
            module.lora.load_state_dict(local)
            if expected_alpha is not None:
                module.lora.alpha = float(expected_alpha)
                module.lora.scaling = module.lora.alpha / module.lora.rank
            loaded += 1
    if loaded == 0:
        raise ValueError(f"LoRA 文件 {path} 与当前模型的目标层不匹配")
    return loaded


def apply_lora_from_checkpoint(model: nn.Module, path: str, dropout: float = 0.0):
    """读取 checkpoint 元数据，按训练时的 rank/alpha/目标层自动挂载并加载。

    新版 checkpoint 直接记录目标模块完整路径；旧版则从 ``*.lora.A.weight`` 键名
    反推。这样推理和权重合并不再偷偷假设 ``rank=16, alpha=16``。
    """
    model = _unwrap(model)
    state = torch.load(path, map_location="cpu")
    meta = state.get("__lora_config__", {})
    targets = list(meta.get("target_modules", [])) if isinstance(meta, dict) else []
    if not targets:
        suffix = ".lora.A.weight"
        targets = [key[:-len(suffix)] for key in state if key.endswith(suffix)]
    if not targets:
        raise ValueError(f"LoRA 文件 {path} 中没有可识别的适配器权重")

    rank = meta.get("rank") if isinstance(meta, dict) else None
    if rank is None:
        first = next(state[f"{name}.lora.A.weight"] for name in targets
                     if f"{name}.lora.A.weight" in state)
        rank = int(first.shape[0])
    alpha = meta.get("alpha", rank) if isinstance(meta, dict) else rank
    apply_lora(model, rank=int(rank), alpha=float(alpha), dropout=dropout,
               target_modules=targets)
    return load_lora(model, path)


def merge_lora(model: nn.Module, lora_path: str, save_path: str):
    """把普通 LoRA 合并进浮点基座并保存；4-bit QLoRA 不支持直接合并。"""
    if not any(hasattr(module, "lora") for module in model.modules()):
        apply_lora_from_checkpoint(model, lora_path)
    else:
        load_lora(model, lora_path)
    raw_model = _unwrap(model)
    for module in raw_model.modules():
        if not hasattr(module, "lora"):
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError("QLoRA 的 4-bit 权重不能原地精确合并；请先反量化到浮点模型")
        delta = module.lora.B.weight @ module.lora.A.weight
        module.weight.data.add_(delta.to(module.weight.dtype), alpha=module.lora.scaling)
    state_dict = {
        key: value.detach().cpu().half()
        for key, value in raw_model.state_dict().items() if ".lora." not in key
    }
    torch.save(state_dict, save_path)
