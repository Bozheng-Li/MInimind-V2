"""MiniMind WebUI 服务端（FastAPI + SSE）。

为什么不用原来的 ``scripts/web_demo.py``（Streamlit）：它用
``AutoModelForCausalLM.from_pretrained`` 读 transformers 格式目录，而本项目新训的模型
（``gated`` 注意力 + ``moe_finegrained`` 前馈）是原生 torch ``.pth``，必须按 ``--config``
组装。本服务照搬 ``trainer/eval.py`` 的架构感知加载路径，因此能加载任意组合的架构。

启动::

    python webui/server.py --config configs/sft_moe_full.yaml \
        --weight test/out/full_sft_moe_full_768_moe.pth --device cuda:0 --port 7860

设计要点：

- **生成在独立线程里跑**（``model.generate`` 是阻塞的），通过 ``asyncio.Queue`` 把 token
  推给异步的 SSE 响应；一次只允许一个会话生成（``Engine.lock``），避免抢占显存。
- **自带增量解码器** ``TokenStreamer``：``transformers.TextIteratorStreamer`` 在
  ``skip_special_tokens=True`` 时会把 ``<tool_call>`` / ``<think>`` 一起吃掉（它们是
  added token），工具调用就再也解析不出来了。这里只过滤聊天控制符（``<|im_start|>`` 等），
  保留结构标签，同时用「pending 字节」策略处理 UTF-8 跨 token 截断。
- **多轮工具调用**：模型输出 ``<tool_call>`` → 服务端执行 → 结果以 ``tool`` 角色回灌 →
  继续生成，直到某一轮没有工具调用（上限 :data:`MAX_TOOL_ROUNDS`）。
- **实验台**（``/lab``）：``webui/experiments_data.py`` 把 ``test/log/`` 与 ``test/storage/report/``
  聚合成 JSON，同一個服务在另一个路由下提供，不占显存、不依赖模型是否加载。
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
#: 实验权重与日志的根。仓库根只放纯净代码，训练产物在 ``test/`` 下
#: （见 ``test/README.md``）。
#: 与 ``configs/loader.py`` 的 ``ARTIFACT_ROOT`` 用**同一个**环境变量
#: ``MINIMIND_ARTIFACT_ROOT``，保证「训练写哪里」和「UI 读哪里」永远一致。
#: ``MINIMIND_DATA_ROOT`` 是旧名，仍接受但优先用 ARTIFACT 那个。
DATA_ROOT = Path(os.environ.get("MINIMIND_ARTIFACT_ROOT")
                 or os.environ.get("MINIMIND_DATA_ROOT")
                 or (REPO_ROOT / "test"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from configs import build_lm_config, load_config, to_arch_config  # noqa: E402
from experiments_data import collect_experiments  # noqa: E402

import catalog  # noqa: E402
import experiments_data  # noqa: E402
import jobs as jobs_mod  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
TOKENIZER_DIR = REPO_ROOT / "model"

#: 单次请求最多几轮「生成 → 工具 → 回灌」
MAX_TOOL_ROUNDS = 16
#: 界面最多勾选几个工具（与 Streamlit 版一致）
MAX_TOOLS = 4
#: 聊天控制符：展示时丢弃（结构标签 <tool_call>/<think> 不在其中，必须保留）
CONTROL_TOKEN_IDS = frozenset({0, 1, 2})  # <|endoftext|> <|im_start|> <|im_end|>

SYSTEM_PROMPT = "你是MiniMind，一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题。"


# ====================================================================== #
# 工具：定义与执行（与 scripts/web_demo.py 保持一致）
# ====================================================================== #
TOOLS: List[Dict[str, Any]] = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}, "required": ["min", "max"]}}},
    {"type": "function", "function": {"name": "text_length", "description": "计算文本长度", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位转换", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "获取汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}}, "required": ["text", "target_lang"]}}},
]

TOOL_SHORT_NAMES = {
    "calculate_math": "数学", "get_current_time": "时间", "random_number": "随机",
    "text_length": "字数", "unit_converter": "单位", "get_current_weather": "天气",
    "get_exchange_rate": "汇率", "translate_text": "翻译",
}

#: 只放行数学表达式里会出现的字符，``eval`` 之前先卡一道
_SAFE_EXPR = re.compile(r"^[0-9+\-*/%.() \t^eE]+$")

#: 单位换算表：单位 -> 相对基准的倍率（长度基准米 / 质量基准千克）
_UNITS: Dict[str, Dict[str, float]] = {
    "length": {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, "mi": 1609.344, "ft": 0.3048, "in": 0.0254, "yd": 0.9144},
    "mass": {"kg": 1.0, "g": 0.001, "mg": 1e-6, "t": 1000.0, "lb": 0.45359237, "oz": 0.028349523125},
    "time": {"s": 1.0, "min": 60.0, "h": 3600.0, "day": 86400.0},
}
_UNIT_ALIAS = {"米": "m", "千米": "km", "公里": "km", "厘米": "cm", "毫米": "mm", "英尺": "ft", "英寸": "in",
               "码": "yd", "英里": "mi", "千克": "kg", "公斤": "kg", "克": "g", "吨": "t", "磅": "lb",
               "秒": "s", "分": "min", "分钟": "min", "时": "h", "小时": "h", "天": "day", "日": "day"}


def execute_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """执行一个内置工具，返回可直接回灌给模型的 JSON 对象。"""
    try:
        if tool_name == "calculate_math":
            expr = str(args.get("expression", "0")).replace("^", "**")
            if not _SAFE_EXPR.match(expr):
                return {"error": f"表达式含非法字符: {expr!r}"}
            return {"result": eval(expr, {"__builtins__": {}}, {})}  # noqa: S307 已做字符白名单
        if tool_name == "get_current_time":
            return {"result": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        if tool_name == "random_number":
            lo, hi = int(args.get("min", 0)), int(args.get("max", 100))
            return {"result": random.randint(lo, hi)}
        if tool_name == "text_length":
            return {"result": len(str(args.get("text", "")))}
        if tool_name == "unit_converter":
            value = float(args.get("value", 0))
            src = _UNIT_ALIAS.get(str(args.get("from_unit", "")).strip().lower(), str(args.get("from_unit", "")).strip().lower())
            dst = _UNIT_ALIAS.get(str(args.get("to_unit", "")).strip().lower(), str(args.get("to_unit", "")).strip().lower())
            for table in _UNITS.values():
                if src in table and dst in table:
                    converted = value * table[src] / table[dst]
                    return {"result": f"{value} {src} = {round(converted, 6)} {dst}"}
            # 未知单位：沿用旧版行为，返回占位串
            return {"result": f"{value} {src} = ? {dst}"}
        if tool_name == "get_current_weather":
            return {"result": f"{args.get('city', 'Unknown')}: 晴, 7~10°C"}
        if tool_name == "get_exchange_rate":
            return {"result": f"1 {args.get('from_currency', 'USD')} = 7.2 {args.get('to_currency', 'CNY')}"}
        if tool_name == "translate_text":
            return {"result": "[翻译结果]: hello world"}
        return {"error": f"未知工具 {tool_name!r}"}
    except Exception as exc:  # noqa: BLE001 工具执行失败不能让整轮对话崩掉
        return {"error": f"{type(exc).__name__}: {exc}"}


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """从模型输出里抽取所有 ``<tool_call>{...}</tool_call>``。"""
    calls = []
    for raw in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            obj = json.loads(raw.strip())
        except Exception:  # noqa: BLE001 模型可能吐出半截 JSON，忽略即可
            continue
        if isinstance(obj, dict) and obj.get("name"):
            calls.append(obj)
    return calls


# ====================================================================== #
# 增量解码器
# ====================================================================== #
class GenerationStopped(Exception):
    """用户点了「停止」，从 ``put`` 里抛出以中断阻塞中的 generate。"""


class TokenStreamer:
    """把 ``generate`` 吐出的 token 增量解码成文本片段，回调给上层。

    ``generate`` 约定 streamer 需要 ``put(tokens_tensor)`` 与 ``end()`` 两个方法
    （兼容 ``transformers.TextIteratorStreamer`` 的接口）。

    与 ``TextIteratorStreamer`` 的两点区别：

    1. 只丢弃 ``CONTROL_TOKEN_IDS``（聊天控制符），保留 ``<tool_call>`` / ``<think>``
       等结构标签 —— 否则工具调用循环无法解析。
    2. 单个 token 可能只承载一个 UTF-8 字节的一部分，解码出 ``\\ufffd``。这里把这类
       未完成的 token 暂存到 ``pending``，等后续字节到齐再一起解码。
    """

    def __init__(self, tokenizer, on_text, control_ids: Iterable[int] = CONTROL_TOKEN_IDS,
                 skip_prompt: bool = True, stop_event: Optional[threading.Event] = None):
        self.tokenizer = tokenizer
        self.on_text = on_text
        self.control_ids = frozenset(control_ids)
        self.special_ids = frozenset(getattr(tokenizer, "added_tokens_decoder", {}) or {})
        self.skip_prompt = skip_prompt
        self.stop_event = stop_event
        self.pending: List[int] = []
        self.next_tokens_are_prompt = skip_prompt
        #: 实际生成的 token 数（prompt 不计，控制符不计）——用于前端性能显示
        self.n_tokens = 0

    # ---------------------------------------------------------------- #
    def _decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def _flush_pending(self) -> None:
        if self.pending:
            text = self._decode(self.pending)
            self.pending = []
            if text:
                self.on_text(text)

    def put(self, value) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            raise GenerationStopped()
        if self.next_tokens_are_prompt:
            # generate 先 put 一次完整的 prompt，跳过它（不重复渲染用户输入）
            self.next_tokens_are_prompt = False
            return
        if hasattr(value, "dim") and value.dim() > 1:
            value = value[0]
        ids = value.tolist() if hasattr(value, "tolist") else list(value)
        self.n_tokens += len(ids)

        for tid in ids:
            if tid in self.control_ids:
                self._flush_pending()
                continue
            if tid in self.special_ids:
                # 结构标签是 added token：整块解码，不参与字节拼接
                self._flush_pending()
                self.on_text(self._decode([tid]))
                continue
            self.pending.append(tid)
            text = self._decode(self.pending)
            # 末尾是替换符 ⇒ 字节还没到齐，继续等（上限防御性截断）
            if text.endswith("�") and len(self.pending) < 8:
                continue
            self.pending = []
            if text:
                self.on_text(text)

    def end(self) -> None:
        self._flush_pending()


# ====================================================================== #
# 流式内容分段：把「思考」与「正文」拆开，供前端分别渲染
# ====================================================================== #
_TAG_OPEN, _TAG_CLOSE = "<think>", "</think>"


def segment(buf: str, pos: int, state: str, final: bool = False) -> Tuple[List[Tuple[str, str]], int, str]:
    """在增长中的文本 ``buf`` 上做增量分段。

    返回 ``(events, new_pos, new_state)``，``events`` 是 ``(kind, text)`` 列表，
    ``kind ∈ {"think", "content"}``。为了不把 ``<think>`` / ``</think>`` 从中间切开，
    非 final 时会扣住末尾最多 ``len(tag)-1`` 个字符等下一批。
    """
    events: List[Tuple[str, str]] = []
    while True:
        if state == "think":
            idx = buf.find(_TAG_CLOSE, pos)
            if idx >= 0:
                if idx > pos:
                    events.append(("think", buf[pos:idx]))
                pos, state = idx + len(_TAG_CLOSE), "content"
                continue
            safe = len(buf) if final else max(pos, len(buf) - (len(_TAG_CLOSE) - 1))
            if safe > pos:
                events.append(("think", buf[pos:safe]))
                pos = safe
            return events, pos, state
        idx = buf.find(_TAG_OPEN, pos)
        if idx >= 0:
            if idx > pos:
                events.append(("content", buf[pos:idx]))
            pos, state = idx + len(_TAG_OPEN), "think"
            continue
        safe = len(buf) if final else max(pos, len(buf) - (len(_TAG_OPEN) - 1))
        if safe > pos:
            events.append(("content", buf[pos:safe]))
            pos = safe
        return events, pos, state


# ====================================================================== #
# 模型装载
# ====================================================================== #
def weight_filename(prefix: str, hidden_size: int, use_moe: bool) -> str:
    """与 ``trainer/eval.py::weight_path`` 同一套命名规则。"""
    return f"{prefix}_{hidden_size}{'_moe' if use_moe else ''}.pth"


def arch_from_config(config_path: str):
    """照搬 ``trainer/eval.py``：配置 → ArchConfig + MiniMindConfig + 权重文件名。"""
    cfg = load_config(config_path)
    arch = to_arch_config(cfg)

    class _Ns:
        hidden_size = arch.model["hidden_size"]
        num_hidden_layers = arch.model["num_hidden_layers"]
        use_moe = 1 if str(arch.type_of("feedforward")).startswith("moe") else 0
        student_hidden_size = hidden_size
        student_num_layers = num_hidden_layers
        student_use_moe = use_moe

    lm_config = build_lm_config(_Ns(), cfg)
    train = cfg.get("train") or {}
    # 配置里写的是相对仓库根的 "out"；但权重实际在 test/out 下。
    # 优先试 DATA_ROOT（test/），不存在再退回 REPO_ROOT，两种布局都能用。
    save_dir = train.get("save_dir") or "out"
    if not os.path.isabs(save_dir):
        candidates = [DATA_ROOT / save_dir, REPO_ROOT / save_dir]
        save_dir = str(next((p for p in candidates if p.is_dir()), candidates[0]))
    prefix = train.get("save_weight") or Path(config_path).stem
    return cfg, arch, weight_filename(prefix, lm_config.hidden_size, bool(lm_config.use_moe)), save_dir, arch


class ModelCandidate(BaseModel):
    """占位：/api/models 直接返回 dict，这里保留类型别名以便前端对齐字段。"""

    id: str
    label: str
    config: str
    weight: str
    arch: str
    weight_exists: bool


def discover_models() -> List[Dict[str, Any]]:
    """扫描 ``configs/*.yaml`` 的 ``train.save_weight``，配对 ``test/out/`` 下的权重。

    这样不用在代码里硬编码文件名映射 —— 训练配置本身就是唯一事实来源。
    """
    out: List[Dict[str, Any]] = []
    seen = set()
    for cfg_path in sorted((REPO_ROOT / "configs").glob("*.yaml")):
        try:
            cfg, arch, fname, save_dir, _ = arch_from_config(str(cfg_path))
        except Exception:  # noqa: BLE001 有些 yaml 不是模型配置（或缺少 model 段）
            continue
        weight = os.path.join(save_dir, fname)
        exists = os.path.isfile(weight)
        if not exists:
            continue
        rel_cfg = str(Path(cfg_path).relative_to(REPO_ROOT))
        rel_weight = str(Path(weight).relative_to(REPO_ROOT)) if str(weight).startswith(str(REPO_ROOT)) else weight
        if rel_weight in seen:
            continue
        seen.add(rel_weight)
        prefix = (cfg.get("train") or {}).get("save_weight", cfg_path.stem)
        out.append({
            "id": rel_weight,
            "label": f"{prefix} · {rel_cfg}",
            "config": rel_cfg,
            "weight": rel_weight,
            "arch": arch.summary(),
            "weight_exists": True,
        })
    # 权重存在但没匹配到配置的，也列出来（用户可手填 config）
    out_dir = REPO_ROOT / "out"
    if out_dir.is_dir():
        for p in sorted(out_dir.glob("*.pth")):
            rel = str(p.relative_to(REPO_ROOT))
            if rel in seen:
                continue
            seen.add(rel)
            out.append({
                "id": rel,
                "label": f"{p.stem} · (未匹配配置)",
                "config": "",
                "weight": rel,
                "arch": "未知",
                "weight_exists": True,
            })
    return out


class Engine:
    """全局唯一的模型持有者：装载、信息查询、串行化生成。"""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.meta: Dict[str, Any] = {"loaded": False}
        self.lock = threading.Lock()          # 同一时刻只允许一个请求占用模型
        self.load_lock = threading.Lock()
        self.stop_event = threading.Event()

    # ---------------------------------------------------------------- #
    @property
    def loaded(self) -> bool:
        return self.model is not None

    def info(self) -> Dict[str, Any]:
        return dict(self.meta)

    def unload(self) -> None:
        with self.lock:
            self.model = None
            self.tokenizer = None
            self.meta = {"loaded": False}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self, config_path: str, weight_path: Optional[str] = None, device: Optional[str] = None) -> Dict[str, Any]:
        """按配置组装模型并加载权重。整个流程串行，避免并发装载打爆显存。"""
        device = device or self.device
        cfg_abs = config_path if os.path.isabs(config_path) else str(REPO_ROOT / config_path)
        if not os.path.isfile(cfg_abs):
            raise FileNotFoundError(f"找不到配置文件: {config_path}")

        _cfg, arch, default_fname, save_dir, _ = arch_from_config(cfg_abs)
        weight_abs = weight_path or os.path.join(save_dir, default_fname)
        if not os.path.isabs(weight_abs):
            weight_abs = str(REPO_ROOT / weight_abs)
        if not os.path.isfile(weight_abs):
            raise FileNotFoundError(f"找不到权重: {weight_abs}")

        with self.load_lock:
            self.unload()
            from arch import build_model

            model = build_model(arch)
            state = torch.load(weight_abs, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            del state
            model = model.half().eval().to(device)

            tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
            self.model, self.tokenizer = model, tokenizer
            self.device = device
            self.meta = {
                "loaded": True,
                "arch": arch.summary(),
                "arch_dict": arch.to_dict(),
                "config": str(Path(cfg_abs).relative_to(REPO_ROOT)) if str(cfg_abs).startswith(str(REPO_ROOT)) else cfg_abs,
                "weight": str(Path(weight_abs).relative_to(REPO_ROOT)) if weight_abs.startswith(str(REPO_ROOT)) else weight_abs,
                "params": sum(p.numel() for p in model.parameters()),
                "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "device": device,
                "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
                "hidden_size": arch.model["hidden_size"],
                "num_hidden_layers": arch.model["num_hidden_layers"],
                "vocab_size": arch.model["vocab_size"],
                "attention": arch.type_of("attention"),
                "feedforward": arch.type_of("feedforward"),
                "norm": arch.type_of("norm"),
                "positional_encoding": arch.type_of("positional_encoding"),
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
            }
            return self.info()


# ====================================================================== #
# 请求模型
# ====================================================================== #
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = []
    prompt: Optional[str] = None
    max_new_tokens: int = 1024
    temperature: float = 0.85
    top_p: float = 0.85
    top_k: int = 50
    repetition_penalty: float = 1.05
    history_rounds: int = 4          # 保留最近几轮（user+assistant 记一轮）
    open_thinking: bool = False
    tools: List[str] = []
    seed: Optional[int] = None
    stream: bool = True


class LoadRequest(BaseModel):
    config: str
    weight: Optional[str] = None
    device: Optional[str] = None


# ====================================================================== #
# FastAPI
# ====================================================================== #
app = FastAPI(title="MiniMind WebUI", version="1.0")
engine = Engine()


def sse(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def build_template_kwargs(tools: Optional[List[Dict[str, Any]]], open_thinking: bool) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if open_thinking:
        kwargs["open_thinking"] = True
    if tools:
        kwargs["tools"] = tools
    return kwargs


@app.get("/api/health")
async def health():
    return {"ok": True, "loaded": engine.loaded, "device": engine.device}


@app.get("/api/models")
async def list_models():
    return {"models": discover_models(), "current": engine.info()}


@app.get("/api/tools")
async def list_tools():
    return {
        "tools": [
            {"name": t["function"]["name"], "description": t["function"]["description"],
             "parameters": t["function"]["parameters"], "short": TOOL_SHORT_NAMES.get(t["function"]["name"], t["function"]["name"])}
            for t in TOOLS
        ],
        "max_select": MAX_TOOLS,
    }


# ---------------------------------------------------------------------- #
# 实验台：聚合 test/log/ 与 test/storage/report/ 的原始记录
# ---------------------------------------------------------------------- #
#: 结果缓存。日志文件在实验跑完前会一直变，所以按「最短 mtime」做 TTL，
#: 既不会每次刷新都重读几 MB 的 CSV，也不会让页面长期停在过期数字上。
_EXP_TTL_SEC = 30.0
_exp_cache: Dict[str, Any] = {"at": 0.0, "data": None}


def _experiments() -> Dict[str, Any]:
    now = time.monotonic()
    if _exp_cache["data"] is not None and (now - _exp_cache["at"]) < _EXP_TTL_SEC:
        return _exp_cache["data"]
    # ⚠️ 必须传 DATA_ROOT（实验根，默认 test/），不能传 REPO_ROOT。
    # 传 REPO_ROOT 时 root/"log"、root/"storage"/"report" 全都不存在，
    # 于是每个实验分区都静默变成空 —— 页面「看起来正常」，只是所有数字都是「—」。
    data = collect_experiments(DATA_ROOT)
    _exp_cache["at"] = now
    _exp_cache["data"] = data
    return data


@app.get("/api/experiments")
async def experiments():
    """全部实验数据（预训练 / SFT / RL / 架构 sweep / benchmark）。

    读盘 + 解析在后台线程做，避免阻塞事件循环（CSV 合起来几 MB）。
    """
    return await asyncio.to_thread(_experiments)


# ---------------------------------------------------------------------- #
# 目录：可用配置 / 数据 / 算法 / 训练策略选项
# ---------------------------------------------------------------------- #
_catalog_cache: Dict[str, Any] = {"at": 0.0, "data": None}
_CATALOG_TTL = 20.0


@app.get("/api/catalog")
async def catalog_endpoint():
    """配置 + 数据 + 算法 + 策略选项。构建控制台据此渲染整个表单。"""
    now = time.monotonic()
    if _catalog_cache["data"] is None or (now - _catalog_cache["at"]) > _CATALOG_TTL:
        data = await asyncio.to_thread(lambda: {
            "configs": catalog.list_configs(DATA_ROOT),
            "datasets": catalog.list_datasets(REPO_ROOT),
            "algos": [{"key": k, **v} for k, v in catalog.ALGO_INFO.items()],
            "choices": catalog.STRATEGY_CHOICES,
            "fields": catalog.field_spec(),
            "field_groups": catalog.FIELD_GROUPS,
            "artifact_root": str(DATA_ROOT),
        })
        _catalog_cache.update({"at": now, "data": data})
    return _catalog_cache["data"]


# ---------------------------------------------------------------------- #
# 训练控制台：解析某个 (配置, 算法) 组合下**实际生效**的默认值
# ---------------------------------------------------------------------- #
_defaults_cache: Dict[str, Any] = {}


def _train_defaults(config: Optional[str], algo: str) -> Dict[str, Any]:
    """用训练入口自己的 argparse 解析一遍，拿到这一组合的真实默认值。

    **为什么费这个劲**：控制台要展示「不改的话，这一跑会用哪些值」。这些值来自三层
    （argparse 硬编码 → configs/<阶段>.yaml → --config 变体），只有让**同一个解析器**
    走一遍才不会和真实训练产生偏差。命令行在训练时仍可覆盖，所以这里返回的是默认值、
    不是最终值。
    """
    from configs import apply_to_parser, stage_of
    from trainer.train import build_parser

    key = f"{algo}|{config or ''}"
    hit = _defaults_cache.get(key)
    if hit is not None:
        return hit

    parser = build_parser()
    argv = ["--algo", algo] + (["--config", config] if config else [])
    cfg = apply_to_parser(parser, stage_of(algo), algo=algo, argv=argv)
    ns = parser.parse_args(argv)

    params = {}
    for k, v in vars(ns).items():
        if k in jobs_mod.ALLOWED_FLAGS and v not in (None, "", False):   # False = 旗标未开，不当作「有值」
            params[k] = v
    out = {
        "ok": True,
        "algo": algo,
        "config": config,
        "params": params,
        "data": {k: v for k, v in (cfg.get("data") or {}).items()},
        "train": {k: v for k, v in (cfg.get("train") or {}).items()},
        "_keys": sorted(cfg.keys()),
    }
    _defaults_cache[key] = out
    if len(_defaults_cache) > 64:
        _defaults_cache.clear()
        _defaults_cache[key] = out
    return out


@app.get("/api/train/defaults")
async def train_defaults(algo: str, config: Optional[str] = None):
    """某个 (算法, 配置文件) 组合下的实际生效默认值，供控制台预填表单。"""
    if algo not in jobs_mod.ALGOS:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"未知算法 {algo!r}"})
    if config:
        from pathlib import Path as _P
        p = (_P(REPO_ROOT) / config).resolve() if not _P(config).is_absolute() else _P(config)
        if not str(p).startswith(str(_P(REPO_ROOT).resolve())):
            return JSONResponse(status_code=400, content={"ok": False, "error": "配置路径越界"})
        if not p.is_file():
            return JSONResponse(status_code=400, content={"ok": False, "error": f"找不到配置 {config}"})
    try:
        return await asyncio.to_thread(_train_defaults, config, algo)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"ok": False, "error": f"{type(exc).__name__}: {exc}"})



@app.get("/api/resources")
async def resources():
    """实时资源：GPU 占用 / 内存 / CPU / 磁盘 / 正在跑的训练进程。

    前端隔几秒轮询一次，用来回答「现在有没有空卡可以起训练」。
    """
    return await asyncio.to_thread(catalog.system_resources)


# ---------------------------------------------------------------------- #
# 训练登记表（trainer/common/registry.py 落盘）
# ---------------------------------------------------------------------- #
@app.get("/api/runs")
async def runs():
    """读 ``test/log/runs/``：每次训练的架构 / 超参 / 策略 / 资源 / 结果摘要。"""
    return await asyncio.to_thread(lambda: _runs_payload())


def _runs_payload() -> Dict[str, Any]:
    import experiments_data as ed
    return ed._runs(DATA_ROOT)


# ---------------------------------------------------------------------- #
# 日志浏览器
# ---------------------------------------------------------------------- #
def _safe_log_path(rel: str) -> Path:
    """把前端传来的相对路径解析成 ``test/log/`` 下的真实文件，拒绝越界。"""
    rel = (rel or "").lstrip("/")
    base = (DATA_ROOT / "log").resolve()
    p = (base / rel).resolve()
    if base != p and base not in p.parents:
        raise ValueError("路径越界")
    if not p.is_file():
        raise FileNotFoundError(f"找不到日志 {rel}")
    if p.suffix.lower() not in experiments_data.LOG_SUFFIXES:
        raise ValueError(f"不支持的文件类型 {p.suffix!r}")
    if p.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("文件过大，请用 tail 参数分段读取")
    return p


@app.get("/api/logs")
async def log_list():
    """可浏览的原始日志清单（``test/log/`` 下的 .log/.csv/.status…）。"""
    return await asyncio.to_thread(lambda: {
        "root": str(DATA_ROOT / "log"),
        "files": experiments_data._log_files(DATA_ROOT),
    })


@app.get("/api/logs/content")
async def log_content(path: str, offset: int = 0, limit: int = 200_000):
    """增量读取一个日志文件。``offset`` 是字节偏移，返回体里带下一个偏移。"""
    def _read():
        p = _safe_log_path(path)
        size = p.stat().st_size
        off = max(0, min(int(offset), size))
        with p.open("rb") as fh:
            fh.seek(off)
            chunk = fh.read(max(1, min(int(limit), 1 << 20)))
        text = chunk.decode("utf-8", errors="replace")
        return {"path": path, "text": text, "offset": off + len(chunk), "size": size,
                "eof": off + len(chunk) >= size}
    try:
        return await asyncio.to_thread(_read)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"ok": False, "error": f"{type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------------- #
# 训练控制台：启动 / 跟踪 / 停止真实训练进程
# ---------------------------------------------------------------------- #
class TrainRequest(BaseModel):
    algo: str
    params: Dict[str, Any] = {}


@app.post("/api/train/preview")
async def train_preview(req: TrainRequest):
    """只拼命令行、不执行 —— 界面上「将执行的命令」用它预览。"""
    try:
        cmd = jobs_mod.build_command(req.algo, req.params)
        return {"ok": True, "cmd": cmd, "pretty": jobs_mod.pretty_command(cmd)}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"ok": False, "error": f"{type(exc).__name__}: {exc}"})


@app.post("/api/train/start")
async def train_start(req: TrainRequest):
    """启动训练。参数经白名单校验后拼成 argv，不经过 shell。"""
    try:
        meta = await asyncio.to_thread(jobs_mod.start, req.algo, req.params)
        return {"ok": meta.get("status") != "failed", "job": meta,
                "error": meta.get("error")}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"ok": False, "error": f"{type(exc).__name__}: {exc}"})


@app.get("/api/train/jobs")
async def train_jobs(limit: int = 50):
    return {"jobs": await asyncio.to_thread(jobs_mod.list_jobs, limit)}


@app.get("/api/train/jobs/{job_id}")
async def train_job_detail(job_id: str):
    meta = await asyncio.to_thread(jobs_mod.job_detail, job_id)
    if not meta:
        return JSONResponse(status_code=404, content={"ok": False, "error": "找不到任务"})
    return {"ok": True, "job": meta}


@app.get("/api/train/jobs/{job_id}/log")
async def train_job_log(job_id: str, offset: int = 0, limit: int = 200_000):
    if not await asyncio.to_thread(jobs_mod.job_detail, job_id):
        return JSONResponse(status_code=404, content={"ok": False, "error": "找不到任务"})
    return await asyncio.to_thread(jobs_mod.tail, job_id, offset, limit)


@app.post("/api/train/jobs/{job_id}/stop")
async def train_job_stop(job_id: str):
    try:
        return {"ok": True, "job": await asyncio.to_thread(jobs_mod.stop, job_id)}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"ok": False, "error": f"{type(exc).__name__}: {exc}"})


@app.get("/lab", include_in_schema=False)
async def lab():
    return FileResponse(str(STATIC_DIR / "lab.html"))


@app.get("/train", include_in_schema=False)
async def train_page():
    return FileResponse(str(STATIC_DIR / "train.html"))


@app.post("/api/model/load")
async def load_model(req: LoadRequest):
    try:
        info = await asyncio.to_thread(engine.load, req.config, req.weight, req.device)
        return {"ok": True, "info": info}
    except Exception as exc:  # noqa: BLE001 装载失败要把原因原样告诉界面
        return JSONResponse(status_code=400, content={"ok": False, "error": f"{type(exc).__name__}: {exc}"})


@app.post("/api/model/unload")
async def unload_model():
    await asyncio.to_thread(engine.unload)
    return {"ok": True}


@app.post("/api/stop")
async def stop_generation():
    engine.stop_event.set()
    return {"ok": True}


# ---------------------------------------------------------------------- #
def _prepare_messages(req: ChatRequest, tokenizer) -> Tuple[List[Dict[str, str]], Optional[List[Dict[str, Any]]]]:
    """拼出交给 chat template 的消息列表（系统提示 + 截断后的历史 + 本轮输入）。"""
    selected = [t for t in TOOLS if t["function"]["name"] in set(req.tools or [])][:MAX_TOOLS]
    tools = selected or None

    history: List[Dict[str, str]] = [{"role": m.role, "content": m.content} for m in req.messages]
    if req.prompt is not None:
        history.append({"role": "user", "content": req.prompt})
    # 历史轮数：一轮 = user + assistant，因此保留末尾 2*rounds 条
    keep = max(0, int(req.history_rounds)) * 2
    if keep:
        history = history[-keep:]
    else:
        history = history[-1:]

    messages: List[Dict[str, str]] = []
    if not tools:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.extend(history)
    return messages, tools


def _run_round(engine: Engine, chat_messages: List[Dict[str, str]],
               template_kwargs: Dict[str, Any], req: ChatRequest, loop, aq: "asyncio.Queue",
               stats: Dict[str, Any]) -> None:
    """在工作线程里跑一轮生成，token 通过 ``aq`` 回传给事件循环。

    注意必须是**同步函数**：它由 ``threading.Thread`` 调用，写成 ``async def`` 只会
    创建一个永远不会被 await 的协程对象。
    """
    try:
        tokenizer = engine.tokenizer
        text = tokenizer.apply_chat_template(chat_messages, **template_kwargs)
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(engine.device)
        stats["prompt_tokens"] += int(inputs.input_ids.shape[1])

        def on_text(chunk: str) -> None:
            loop.call_soon_threadsafe(aq.put_nowait, ("delta", chunk))

        streamer = TokenStreamer(tokenizer, on_text, stop_event=engine.stop_event)
        with engine.lock:
            engine.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=int(req.max_new_tokens),
                do_sample=True,
                temperature=float(req.temperature),
                top_p=float(req.top_p),
                top_k=int(req.top_k),
                repetition_penalty=float(req.repetition_penalty),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                streamer=streamer,
            )
        stats["completion_tokens"] += streamer.n_tokens
    except GenerationStopped:
        loop.call_soon_threadsafe(aq.put_nowait, ("stopped", None))
    except Exception as exc:  # noqa: BLE001 生成异常要回传前端而不是 500
        loop.call_soon_threadsafe(aq.put_nowait, ("error", f"{type(exc).__name__}: {exc}"))
    finally:
        loop.call_soon_threadsafe(aq.put_nowait, ("end", None))


async def stream_chat(req: ChatRequest):
    """真正的 SSE 生成器。"""
    if not engine.loaded:
        yield sse({"type": "error", "message": "模型尚未加载，请先在左侧选择并加载模型"})
        return

    tokenizer = engine.tokenizer
    engine.stop_event.clear()
    if req.seed is not None:
        random.seed(req.seed)
        torch.manual_seed(req.seed)

    chat_messages, tools = _prepare_messages(req, tokenizer)
    template_kwargs = build_template_kwargs(tools, req.open_thinking)
    tool_names = [t["function"]["name"] for t in (tools or [])]

    stats = {"rounds": 0, "prompt_tokens": 0, "completion_tokens": 0, "started": time.time()}
    yield sse({"type": "start", "open_thinking": req.open_thinking, "tools": tool_names,
               "model": engine.meta.get("weight", ""), "rounds_max": MAX_TOOL_ROUNDS})

    loop = asyncio.get_running_loop()
    answer_parts: List[str] = []
    stopped = False

    for rnd in range(MAX_TOOL_ROUNDS):
        if engine.stop_event.is_set():
            stopped = True
            break
        stats["rounds"] = rnd + 1
        yield sse({"type": "round_start", "round": rnd})

        aq: asyncio.Queue = asyncio.Queue()
        worker = threading.Thread(
            target=_run_round,
            args=(engine, chat_messages, template_kwargs, req, loop, aq, stats),
            daemon=True,
        )
        worker.start()

        buf: List[str] = []
        pos, state = 0, ("think" if req.open_thinking else "content")
        round_error = None
        while True:
            kind, payload = await aq.get()
            if kind == "end":
                break
            if kind == "stopped":
                stopped = True
                continue
            if kind == "error":
                round_error = payload
                continue
            buf.append(payload)
            raw = "".join(buf)
            events, pos, state = segment(raw, pos, state)
            for ev_kind, ev_text in events:
                yield sse({"type": ev_kind, "text": ev_text})

        raw = "".join(buf)
        events, pos, state = segment(raw, pos, state, final=True)
        for ev_kind, ev_text in events:
            yield sse({"type": ev_kind, "text": ev_text})

        if round_error:
            yield sse({"type": "error", "message": round_error})
            break

        answer_parts.append(raw)
        calls = parse_tool_calls(raw)
        if not calls:
            if stopped:
                break
            break

        # ---- 有工具调用：执行 → 回灌 → 继续下一轮 ----
        chat_messages.append({"role": "assistant", "content": raw})
        for call in calls:
            name = str(call.get("name", ""))
            args = call.get("arguments", {}) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {"raw": args}
            if name not in tool_names:
                result = {"error": f"工具 {name!r} 未启用"}
            else:
                result = execute_tool(name, args)
            yield sse({"type": "tool_result", "name": name, "arguments": args, "result": result,
                       "ok": "error" not in result})
            chat_messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
        yield sse({"type": "round_end", "round": rnd})

    elapsed = max(time.time() - stats["started"], 1e-6)
    yield sse({
        "type": "done",
        "stopped": stopped,
        "stats": {
            "rounds": stats["rounds"],
            "prompt_tokens": stats["prompt_tokens"],
            "completion_tokens": stats["completion_tokens"],
            "elapsed": round(elapsed, 3),
            "tokens_per_second": round(stats["completion_tokens"] / elapsed, 2),
        },
        "text": "".join(answer_parts),
    })


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.stream:
        # 非流式：把事件收完一次性返回（便于 curl / 脚本化验证）
        kinds, think, content, tools_used, done = [], [], [], [], {}
        async for chunk in stream_chat(req):
            payload = json.loads(chunk[len("data: "):].strip())
            kinds.append(payload["type"])
            if payload["type"] == "think":
                think.append(payload["text"])
            elif payload["type"] == "content":
                content.append(payload["text"])
            elif payload["type"] == "tool_result":
                tools_used.append(payload)
            elif payload["type"] == "done":
                done = payload
            elif payload["type"] == "error":
                return JSONResponse(status_code=400, content={"ok": False, "error": payload["message"]})
        return {"ok": True, "thinking": "".join(think), "content": "".join(content),
                "tool_results": tools_used, "done": done, "events": kinds}
    return StreamingResponse(
        stream_chat(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------- #
@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main() -> int:
    parser = argparse.ArgumentParser(description="MiniMind WebUI（FastAPI + SSE）")
    parser.add_argument("--config", default="configs/sft_moe_full.yaml",
                        help="模型结构配置（必须与训练时一致，否则权重形状对不上）")
    parser.add_argument("--weight", default=None, help="权重路径；默认按配置的 save_dir/save_weight 推导")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="推理设备，例如 cuda:0；也可用 CUDA_VISIBLE_DEVICES 限制可见 GPU")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-load", action="store_true", help="启动时不加载模型（先在界面里选）")
    args = parser.parse_args()

    if not args.no_load:
        print(f"⏳ 正在加载模型: {args.config} / {args.weight or '(按配置推导)'} @ {args.device}")
        try:
            info = engine.load(args.config, args.weight, args.device)
            print(f"✅ 已加载 {info['weight']}")
            print(f"   架构: {info['arch']}")
            print(f"   参数量: {info['params']:,}  设备: {info['device']}  精度: {info['dtype']}")
            if info["missing_keys"] or info["unexpected_keys"]:
                print(f"⚠️  权重未完全匹配 missing={info['missing_keys']} unexpected={info['unexpected_keys']}")
        except Exception as exc:  # noqa: BLE001 启动期加载失败也允许先进界面
            print(f"⚠️  模型加载失败：{type(exc).__name__}: {exc}")
            print("   服务仍会启动，可在界面左栏手动选择模型后加载。")

    import uvicorn

    # 后台预热目录缓存：catalog 要解析十几份 YAML、读数据集元信息，冷启动要几秒。
    # 提前在子线程里跑一遍，用户打开页面时就是热的（失败无所谓，接口自己会重算）。
    def _warm_catalog() -> None:
        try:
            catalog.list_configs(DATA_ROOT)
            catalog.list_datasets(REPO_ROOT)
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_warm_catalog, name="warm-catalog", daemon=True).start()

    print(f"🚀 http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
