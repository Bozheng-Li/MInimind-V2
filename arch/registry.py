"""可插拔组件的注册表。

组件按 ``(类别, 名字)`` 注册::

    @register("attention", "gqa")
    class GQAttention(nn.Module): ...

YAML 里写 ``type: gqa`` 即命中注册表。

若组件放在仓库之外，可以在 ``type`` 里写**完整 import 路径**（含点号），
例如 ``type: my_ext.attention.FlashAttn``，注册表会自动回退到 import。
这样新增组件既可以是本仓库内的文件，也可以是外部包，无需改动框架代码。
"""
from __future__ import annotations

import importlib
from typing import Dict, List, Type

# kind -> {name -> class}
_REGISTRY: Dict[str, Dict[str, type]] = {}


def register(kind: str, name: str):
    """把组件类登记到某个类别下（用作装饰器）。"""

    def deco(cls: type) -> type:
        bucket = _REGISTRY.setdefault(kind, {})
        if name in bucket and bucket[name] is not cls:
            raise ValueError(
                f"组件重复注册：{kind}/{name} 已被 {bucket[name].__name__} 占用，"
                f"现在又来了 {cls.__name__}"
            )
        bucket[name] = cls
        cls._arch_kind = kind
        cls._arch_name = name
        return cls

    return deco


def available(kind: str) -> List[str]:
    """列出某类别下已注册的短名（用于报错提示）。"""
    return sorted(_REGISTRY.get(kind, {}))


def kinds() -> List[str]:
    """列出所有已出现过的类别。"""
    return sorted(_REGISTRY)


def registered() -> Dict[str, List[str]]:
    """返回 {类别: [短名...]} 全貌，便于打印清单。"""
    return {k: sorted(v) for k, v in sorted(_REGISTRY.items())}


def resolve(kind: str, name: str) -> type:
    """按名解析组件类：先查注册表，再回退到 import 路径。"""
    name = str(name)
    bucket = _REGISTRY.get(kind, {})
    if name in bucket:
        return bucket[name]

    if "." in name:
        module_path, _, cls_name = name.rpartition(".")
        try:
            mod = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"无法导入 {kind} 组件 {name!r}：模块 {module_path!r} 导入失败（{exc}）。\n"
                f"已注册的 {kind} 短名: {available(kind)}"
            ) from exc
        cls = getattr(mod, cls_name, None)
        if cls is None:
            raise AttributeError(
                f"模块 {module_path!r} 中不存在 {cls_name!r}"
                f"（来自 {kind} 组件 {name!r}）"
            )
        return cls

    raise ValueError(
        f"未知的 {kind} 类型 {name!r}。已注册的 {kind}: {available(kind) or '(无)'}。\n"
        f"若组件位于仓库外，请在 type 里填写完整 import 路径（如 my_pkg.MyClass）"
    )
