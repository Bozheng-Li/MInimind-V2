"""实验结果数据层 —— 把散落在 ``test/log/`` 与 ``test/storage/report/`` 里的原始记录
聚合成一份结构化的 JSON，供 WebUI 的「实验台」页面渲染。

设计约束：

- **只读**。绝不写日志目录、绝不改 CSV；页面刷新多少次结果都一样。
- **不猜**。每个数字都从一个明确的文件/列里读出来，找不到就留空（``None``），
  由前端显示成「—」，不编造、不用别的字段顶替。
- **不怕崩**。任何单个文件读失败只让那一块变空，不影响其余部分返回；
  这个模块本身抛异常会让整个实验台打不开，比缺一块数据严重得多。

数据来源:

============================  ============================================
路径                          内容
============================  ============================================
``test/log/pretrain.log``        Dense 预训练 stdout（从日志行正则抽 loss）
``test/log/pretrain_moe_metrics.csv``  MoE 预训练逐 50 步全量指标（139 列）
``test/log/sft_*_metrics.csv``  三个 SFT run 的逐 50 步指标
``test/log/rl/*_metrics.csv``   四种 RL 算法 × 两种架构的逐 step 指标
``test/log/rl/*.status``        OK / FAIL + 墙钟耗时
``test/storage/report/summary.csv`` 架构 sweep 18 组的最终排名
``test/storage/report/eval/summary.csv``  权重 × benchmark（含 n / 标准误）
``test/storage/report/eval/tasks.json``   任务表（单一真源在 eval_suite.py::TASKS）
============================  ============================================
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------- #
# 常量：阶段清单与算法展示名
# ---------------------------------------------------------------------- #
#: 偏好优化与在线 RL 算法。<key> 同时是 log/rl/<key>_<arch>_metrics.csv 的前缀。
RL_ALGOS: List[Tuple[str, str, str]] = [
    ("dpo", "DPO", "离线偏好优化（无 rollout / 无 reward model）"),
    ("ipo", "IPO", "有限最优偏好间隔的平方目标"),
    ("simpo", "SimPO", "长度归一化、无需 reference"),
    ("cpo", "CPO", "偏好目标 + chosen SFT 约束"),
    ("orpo", "ORPO", "SFT + odds-ratio 偏好目标"),
    ("kto", "KTO", "合意/不合意样本的前景效用"),
    ("grpo", "GRPO·CISPO", "组内采样 + CISPO 损失，num_gen=6"),
    ("dapo", "DAPO", "Clip-Higher + 动态采样 + token-level loss"),
    ("rloo", "RLOO", "leave-one-out reward baseline，无 critic"),
    ("ppo", "PPO", "Actor-Critic + GAE + KL 早停"),
    ("agent", "Agentic", "多轮工具调用，整轮延迟结算奖励"),
]

#: 两种基座架构。<key> 同时是权重/日志后缀。
ARCHS: List[Tuple[str, str, str]] = [
    ("dense", "Dense 63.9M", "gqa + swiglu + rmsnorm + rope"),
    ("moe", "MoE 214M/激活75M", "gated + moe_finegrained(16专家top4+1共享)"),
]

#: eval 任务的**历史兼容副本**。真源在 ``test/storage/eval_suite.py::TASKS``
#: —— 那份会导出 ``report/eval/tasks.json``，``_eval()`` 优先读它，读不到才退回这里。
#: 保留这个常量的意义：旧产物（只有 4 个 benchmark 的 summary.csv）仍能渲染。
EVAL_TASKS: List[Tuple[str, str, str]] = [
    ("ceval", "C-Eval acc", "中文多选（156 题）"),
    ("mmlu", "MMLU acc", "英文多选（174 题）"),
    ("gsm8k", "GSM8K EM", "小学数学（50 题，0-shot）"),
    ("mbpp", "MBPP pass@1", "代码生成（50 题，0-shot）"),
]

EVAL_RANDOM: Dict[str, float] = {"ceval": 0.25, "mmlu": 0.25, "gsm8k": 0.0, "mbpp": 0.0}

#: 回退路径下每个任务在旧 ``summary.csv`` 里写的**指标名**。
#: 旧口径的 `metric` 列是 `acc` / `acc_norm` / `exact_match` / `pass@1`，不是 task key。
#: 写成 key 会让 `_eval()` 的 metric 过滤把所有行丢光、整页显示「没有评测结果」。
_LEGACY_METRIC: Dict[str, str] = {
    "ceval": "acc", "mmlu": "acc", "gsm8k": "exact_match", "mbpp": "pass@1",
}

#: 主曲线（loss / val_loss）的点数 —— 页面的主角，多留一点。
MAX_POINTS = 220

#: MoE 健康度这类「看趋势就够」的曲线点数。它们比 loss 平滑得多，220 点在屏上
#: 已经比像素还密；降到 128 观感不变，整页 payload 少十几 KB。
TREND_POINTS = 128

#: 次要曲线（同一张图里的对照组、明细曲线）的降采样点数。
#: 比主曲线小一个量级 —— 一页要塞下十几条这样的曲线，用 220 点会让 payload 翻几倍，
#: 而这类曲线只是用来「看趋势对不对」，64 点足够。
CURVE_CAP = 64

#: 「系列 × 时间桶」矩阵的时间桶数。用于逐层（8 层）/ 逐专家（16 个）这类
#: 系列多、逐条画折线会糊成一团的指标。
MATRIX_BUCKETS = 28

#: RL 各算法都会记录的曲线（同一个训练循环里抽出来的通用量）。
#: 「奖励分解」五项与总的 ``reward`` 同量纲、可直接相加（实测 reward ≈ 各项之和），
#: 所以它们必须凑齐一起看 —— 单独一条 reward 曲线看不出「涨的是哪一项」。
#: （这些列在 DPO 里没有 rollout 来源，自然缺席。）

#: RL 里各算法都有的「生成健康度」曲线。只列**要单独画图**的那几项 ——
#: ``policy_loss`` / ``kl_ref`` / ``avg_response_len`` 等已有 tail 标量与
#: 头部曲线（``reward`` / ``response_len``），再各存一条 64 点曲线只是让 payload 翻倍。
#:
#: ``clipfrac`` / ``group_reward_std`` 放这里而不是算法专属表，因为它们**跨算法出现**
#: （GRPO 与 PPO 都记 clipfrac，GRPO 与 Agent 都记 group_reward_std）——
#: 归到某一个算法名下会让另一个算法的这一列在界面上凭空消失。
RL_GEN_CURVES: List[str] = [
    "eos_rate", "trunc_rate", "perplexity", "ratio_mean", "adv_zero_frac", "gen_tokens",
    "clipfrac", "group_reward_std", "policy_loss", "kl_ref",
]

#: RL 里**只有该算法才有**的曲线。缺失的列不会出现在结果里（见 ``_curves``），
#: 所以这里可以放心把「这一族该有的列」全写上 —— 有就画、没有就整族不出现。
RL_ALGO_CURVES: Dict[str, List[str]] = {
    "dpo": ["preference_loss", "dpo_loss", "reward_margin", "preference_acc"],
    "ipo": ["preference_loss", "reward_margin", "preference_acc"],
    "simpo": ["preference_loss", "reward_margin", "preference_acc"],
    "cpo": ["preference_loss", "reward_margin", "preference_acc"],
    "orpo": ["preference_loss", "reward_margin", "preference_acc"],
    "kto": ["preference_loss", "reward_margin", "preference_acc"],
    "grpo": ["group_reward_zero_std"],
    "dapo": ["group_reward_zero_std"],
    "rloo": ["group_reward_zero_std"],
    "ppo": ["critic_loss", "value_loss", "approx_kl", "kl_early_stop", "actor_lr", "critic_lr"],
    "agent": ["pass_rate", "unfinished_rate", "turns_mean", "tool_calls_mean",
              "valid_call_rate", "tool_gap_mean"],
}

#: RL 头部曲线（reward / 回复长度）的点数。它们原本走 ``MAX_POINTS``(220)，
#: 但 RL 只有 1000 步，220 点在屏幕上已经比像素还密；降到 128 不影响观感，
#: 却能让多组 run 的 payload 显著缩小。
RL_HEAD_POINTS = 128

#: RL 的逐 step 里「按算法分组后仍然该看」的优化动力学曲线（与预训练/SFT 同一口径，
#: 便于横向比较「RL 阶段在哪一层学得多」）。
RL_OPT_CURVES: List[str] = [
    "update_ratio_attn", "update_ratio_ffn", "update_ratio_embed", "update_ratio_norm",
]

#: 「算法 × 指标家族」覆盖矩阵的家族定义。
#: 每个家族给一组**代表性列名** —— 一个家族的覆盖率 = 该家族里有几列真的出现在
#: 该算法的 CSV 表头里。这就是「不同算法记录不同 log」的量化答案。
METRIC_FAMILIES: List[Tuple[str, List[str]]] = [
    ("优化动力学", ["update_ratio_attn", "update_ratio_ffn", "update_ratio_embed", "update_ratio_norm"]),
    ("梯度分组", ["grad_norm_attn", "grad_norm_ffn", "grad_norm_embed", "grad_norm_norm"]),
    ("权重范数", ["weight_norm_attn", "weight_norm_ffn", "weight_norm_embed"]),
    ("逐层激活", ["q_rms_L0", "hidden_rms_L0", "routed_rms_L0", "shared_rms_L0"]),
    ("逐层 Q/K/V/输出", ["q_rms_L3", "k_rms_L3", "v_rms_L3", "out_rms_L3"]),
    ("逐层门控", ["gate_mean_L0", "gate_std_L0", "gate_mean_L7", "gate_std_L7"]),
    ("MoE 负载", ["moe_load_cv", "moe_load_maxmin", "moe_entropy_norm", "moe_load_e0"]),
    ("MoE 逐专家", ["moe_load_e0", "moe_load_e7", "moe_load_e15"]),
    ("路由偏置", ["moe_bias_std", "moe_bias_absmean", "moe_bias_nonzero"]),
    ("系统", ["gpu_util_mean", "gpu_util_min", "ram_pct", "gpu_mem_peak_mb"]),
    ("奖励分解", ["rew_len", "rew_think_len", "rew_think_close", "rew_rep", "rew_rm"]),
    ("生成健康度", ["eos_rate", "trunc_rate", "perplexity", "adv_zero_frac", "ratio_mean", "gen_tokens"]),
    ("策略优化", ["policy_loss", "kl_ref", "clipfrac", "advantages_mean", "group_reward_std"]),
    ("PPO·Critic", ["critic_loss", "value_loss", "approx_kl", "kl_early_stop", "actor_lr", "critic_lr",
                    "adv_raw_mean", "adv_raw_std"]),
    ("Agent·工具", ["turns_mean", "tool_calls_mean", "valid_call_rate", "tool_gap_mean", "unfinished_rate"]),
    ("离线偏好", ["preference_loss", "reward_margin", "preference_acc"]),
]


# ---------------------------------------------------------------------- #
# 基础工具
# ---------------------------------------------------------------------- #
class _Rows(list):
    """CSV 行清单，额外挂一个「列名 → (steps, values)」的抽取消缓存。

    一份 metrics CSV 会被十几个 builder 反复读同一列（loss / val_loss / 每个逐层列…），
    朴素实现每次都把全表再扫一遍。行清单是 ``_read_csv`` 新建的、之后只读，
    所以把缓存挂在对象上既不会串味、也不会因为 ``id()`` 复用而出错。
    """
    __slots__ = ("cache",)

    def __init__(self, it=()):
        super().__init__(it)
        self.cache: Dict[str, Tuple[List[int], List[float]]] = {}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return _Rows()
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            return _Rows(r for r in csv.DictReader(fh) if r)
    except Exception:  # noqa: BLE001 单个文件损坏不该拖垮整个页面
        return _Rows()


def _floats(rows: List[Dict[str, str]], col: str) -> Tuple[List[int], List[float]]:
    """抽出一列 (step, value)，跳过空值与解析失败的行。结果按行清单缓存。"""
    cache = getattr(rows, "cache", None)
    if cache is not None and col in cache:
        return cache[col]
    steps: List[int] = []
    vals: List[float] = []
    if not rows or col not in rows[0]:
        if cache is not None:
            cache[col] = (steps, vals)
        return steps, vals
    for r in rows:
        raw = r.get(col)
        if raw in (None, ""):
            continue
        try:
            vals.append(float(raw))
            steps.append(int(float(r.get("step", len(vals)))))
        except (TypeError, ValueError):
            continue
    if cache is not None:
        cache[col] = (steps, vals)
    return steps, vals


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _tail_mean(vals: List[float], n: int = 100) -> Optional[float]:
    return _mean(vals[-n:]) if len(vals) >= 5 else (vals[-1] if vals else None)


def _downsample(steps: List[int], vals: List[float], cap: int = MAX_POINTS) -> Dict[str, List[float]]:
    """按块取均值降采样，保持首末点。

    直接抽第 i 个点会在震荡段漏掉峰谷；分块均值既压点数又保留趋势。
    """
    n = len(vals)
    if n == 0:
        return {"step": [], "value": []}
    if n <= cap:
        return {"step": [float(s) for s in steps], "value": list(vals)}
    bucket = n / float(cap)
    out_s: List[float] = []
    out_v: List[float] = []
    k = 0.0
    while k < n:
        lo = int(k)
        hi = min(n, int(k + bucket) or lo + 1)
        out_s.append(float(steps[lo]))
        out_v.append(_mean(vals[lo:hi]))
        k += bucket
    # 保证末点是真实末值（块均值会把最后的尖峰抹平）
    out_s[-1] = float(steps[-1])
    out_v[-1] = vals[-1]
    return {"step": out_s, "value": out_v}


def _round(x: Optional[float], nd: int = 4) -> Optional[float]:
    return None if x is None else round(x, nd)


def _fmt(x: Optional[float], nd: int = 4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _basename(v: Any) -> Optional[str]:
    """取路径的最后一段。``None`` / 空串 / 非字符串一律返回 ``None``。"""
    if not isinstance(v, str) or not v.strip():
        return None
    return PurePosixPath(v.replace("\\", "/")).name or None


def _has_cols(rows: List[Dict[str, str]], cols: List[str]) -> List[str]:
    """从 ``cols`` 里筛出**真的在这份 CSV 表头里、且至少有一个非空值**的列。

    两个条件缺一不可：
    - 不在表头 → 这个算法根本没记这一项（如 DPO 没有 ``rew_len``）；
    - 在表头但整列全空 → 记了位置没填数（如 ``agent`` 的 ``rew_*`` 实测全空）。

    两种情况的处理方式一样：**整族不出现**，而不是画一条空曲线或恒 0 直线。
    """
    if not rows or not cols:
        return []
    head = rows[0]
    out: List[str] = []
    for c in cols:
        if c not in head:
            continue
        for r in rows[:400]:          # 只看前 400 行就有定论，避免整列扫两遍
            if r.get(c) not in (None, ""):
                out.append(c)
                break
    return out


def _is_flat(vals: List[float]) -> bool:
    """整条曲线是不是一条水平线（实测 ``moe_bias_nonzero`` / ``kl_early_stop`` 这类）。

    恒定的列不该单独画图 —— 一条直线不传递任何信息，却会占掉一张卡片的位置，
    还容易被误读成「没有变化」的趋势。它的值仍然在统计牌 / 表视图里可读。
    """
    if len(vals) < 5:
        return False
    lo, hi = min(vals), max(vals)
    # 用「量程相对自身量级」判定，兼顾 0 附近的量与 1e4 量级的值
    scale = max(abs(lo), abs(hi), 1e-12)
    return (hi - lo) <= 1e-9 * scale + 1e-15


def _curves(rows: List[Dict[str, str]], cols: List[str], cap: int = CURVE_CAP,
            nd: int = 5, drop_flat: bool = True) -> Dict[str, List[Dict[str, float]]]:
    """把若干列各降采样成一条曲线，返回 ``{列名: {"step": [...], "value": [...]}}``。

    只返回**真的有数据**的列（见 ``_has_cols``）—— 前端因此可以「拿到什么画什么」，
    不用自己判断某条曲线该不该出现。``drop_flat`` 时再滤掉水平线（见 ``_is_flat``）。
    """
    out: Dict[str, List[Dict[str, float]]] = {}
    for c in _has_cols(rows, cols):
        steps, vals = _floats(rows, c)
        if not vals or (drop_flat and _is_flat(vals)):
            continue
        d = _downsample(steps, vals, cap)
        out[c] = {
            "step": [_round(s, 3) for s in d["step"]],
            "value": [_round(v, nd) for v in d["value"]],
        }
    return out


def _matrix(rows: List[Dict[str, str]], cols: List[str], buckets: int = MATRIX_BUCKETS,
            nd: int = 5, drop_flat: bool = True) -> Optional[Dict[str, Any]]:
    """把「一组同族列 × 时间」压成一个矩阵：``{"step", "labels", "values", "cols", "bounds"}``。

    用于逐层激活量（8 层）与逐专家负载（16 个专家）：系列太多，逐条折线会糊成一团，
    而热力图正好一眼看出「哪一层在放大、哪个专家被饿死」。

    ``values`` 的配色值按**每个系列自身**归一化（行内 min/max → [0,1]），
    这样量纲差几个数量级的行（如 L0 的 0.3 与 L7 的 4.0）不会互相压平；
    ``cols`` 保留原值供浮层读数，``bounds`` 给出每一行的真实量程。
    """
    labels: List[str] = []
    steps: List[float] = []
    series: List[List[float]] = []
    bounds: List[List[float]] = []
    for c in cols:
        if c not in (rows[0] if rows else {}):
            continue
        s_steps, vals = _floats(rows, c)
        if not vals or (drop_flat and _is_flat(vals)):
            continue
        d = _downsample(s_steps, vals, buckets)
        if not steps:
            steps = [_round(s, 3) for s in d["step"]]
        labels.append(c)
        series.append(d["value"])
        bounds.append([_round(min(vals), nd), _round(max(vals), nd)])
    if not series:
        return None
    n = min([len(s) for s in series] + [len(steps)])
    grid: List[List[float]] = []
    raw: List[List[float]] = []
    for s in series:
        lo, hi = min(s), max(s)
        span = (hi - lo) or 1e-12
        grid.append([_round((v - lo) / span, 4) for v in s[:n]])
        raw.append([_round(v, nd) for v in s[:n]])
    return {
        "step": steps[:n],
        "labels": labels,
        "values": grid,
        "cols": raw,
        "bounds": bounds,
    }


def _steps_of(rows: List[Dict[str, str]], buckets: int = MATRIX_BUCKETS) -> List[float]:
    """矩阵的横轴：任取一列真实存在的指标，按同样口径降采样出 step 网格。"""
    for probe in ("loss", "reward", "logits_loss"):
        if rows and probe in rows[0]:
            steps, vals = _floats(rows, probe)
            if vals:
                return [_round(s, 3) for s in _downsample(steps, vals, buckets)["step"]]
    return []



# ---------------------------------------------------------------------- #
# 预训练
# ---------------------------------------------------------------------- #
#: Dense 预训练只有 .log 没有 metrics csv，从 stdout 正则抽
_LOG_LINE = re.compile(
    r"Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\), loss: ([0-9.]+).*?"
    r"(?:val_loss: ([0-9.]+))?.*?lr: ([0-9.eE+-]+)"
)


def _dense_pretrain_log(root: Path) -> Dict[str, Any]:
    """从 test/log/pretrain.log 抽 loss / val_loss 曲线（每 100 步一行，本来就稀疏）。"""
    path = root / "log" / "pretrain.log"
    if not path.is_file():
        return {}
    steps: List[int] = []
    loss: List[float] = []
    val_steps: List[int] = []
    val_loss: List[float] = []
    epochs: set = set()
    try:
        for line in path.read_text(errors="replace").splitlines():
            m = _LOG_LINE.search(line)
            if not m:
                continue
            ep, _, it, total, ls, vl, _lr = m.groups()
            epochs.add(int(ep))
            steps.append(int(it))
            loss.append(float(ls))
            if vl:
                val_steps.append(int(it))
                val_loss.append(float(vl))
    except Exception:  # noqa: BLE001
        return {}
    if not steps:
        return {}
    return {
        "source": "test/log/pretrain.log",
        "epochs": max(epochs) if epochs else None,
        "steps_total": len(steps),
        "loss": _downsample(steps, loss),
        "val_loss": _downsample(val_steps, val_loss),
        "loss_first": _round(loss[0]),
        "loss_last": _round(loss[-1]),
        "loss_tail100": _round(_tail_mean(loss, 100)),
        "val_loss_last": _round(val_loss[-1]) if val_loss else None,
    }


def _moe_pretrain_csv(root: Path) -> Dict[str, Any]:
    rows = _read_csv(root / "log" / "pretrain_moe_metrics.csv")
    if not rows:
        return {}
    steps, loss = _floats(rows, "loss")
    v_steps, v_loss = _floats(rows, "val_loss")
    layers = [f"_L{i}" for i in range(8)]
    experts = [f"moe_load_e{i}" for i in range(16)]
    dead = _floats(rows, "moe_dead_experts")[1]
    return {
        "source": "test/log/pretrain_moe_metrics.csv",
        "steps_total": int(float(rows[-1]["step"])) if rows[-1].get("step") else len(rows),
        "loss": _downsample(steps, loss),
        "val_loss": _downsample(v_steps, v_loss),
        "loss_first": _round(loss[0]) if loss else None,
        "loss_last": _round(loss[-1]) if loss else None,
        "loss_tail100": _round(_tail_mean(loss, 200)),
        "val_loss_last": _round(v_loss[-1]) if v_loss else None,
        "tokens_per_sec": _round(_tail_mean(_floats(rows, "tokens_per_sec")[1], 200), 1),
        "grad_norm": _round(_tail_mean(_floats(rows, "grad_norm")[1], 200)),
        "gpu_mem_mb": _round(_tail_mean(_floats(rows, "gpu_mem_mb")[1], 200), 0),
        # 峰值显存（nvidia-smi 采样窗口内的最大值）—— 起训练前检查表拿它当实测依据
        "gpu_mem_peak_mb": _round(max(_floats(rows, "gpu_mem_peak_mb")[1] or [0]), 0) or None,
        # MoE 健康度四条：均衡、崩塌、死专家、共享专家占比
        "moe_load_cv": _downsample(*_floats(rows, "moe_load_cv"), TREND_POINTS),
        "moe_load_maxmin": _downsample(*_floats(rows, "moe_load_maxmin"), TREND_POINTS),
        "moe_dead_experts": _downsample(*_floats(rows, "moe_dead_experts"), TREND_POINTS),
        "moe_entropy_norm": _downsample(*_floats(rows, "moe_entropy_norm"), TREND_POINTS),
        "shared_share": _downsample(*_floats(rows, "shared_share"), TREND_POINTS),
        "dead_experts_max": _round(max(dead or [0]), 0),
        "dead_experts_last": _round(_tail_mean(dead, 200), 0),
        "moe_load_cv_last": _round(_tail_mean(_floats(rows, "moe_load_cv")[1], 200)),
        "moe_maxmin_first": _round(_floats(rows, "moe_load_maxmin")[1][0]) if _floats(rows, "moe_load_maxmin")[1] else None,
        "moe_maxmin_last": _round(_tail_mean(_floats(rows, "moe_load_maxmin")[1], 200)),
        "shared_share_first": _round(_floats(rows, "shared_share")[1][0]) if _floats(rows, "shared_share")[1] else None,
        "shared_share_last": _round(_tail_mean(_floats(rows, "shared_share")[1], 200)),
        "dead_experts_max": _round(max(dead or [0]), 0),
        "dead_experts_last": _round(_tail_mean(dead, 200), 0),
        "entropy_min": _round(min(_floats(rows, "moe_entropy_norm")[1] or [0])) if _floats(rows, "moe_entropy_norm")[1] else None,
        # 尾部均值口径：跨 run 横比（#library）要的是「这跑完之后路由有多均匀」，
        # 用 min 会把任意一次瞬时抖动当成结论。
        "moe_entropy_tail": _round(_tail_mean(_floats(rows, "moe_entropy_norm")[1], 200)),
        # 优化动力学：按参数组分组的更新比例（lr / 该组权重的逐元素 rms），
        # 用来判断「哪一部分还在学、哪一部分已经停滞」。
        "opt_curves": _curves(rows, ["update_ratio_attn", "update_ratio_ffn",
                                     "update_ratio_embed", "update_ratio_norm"]),
        # 按参数组分组的梯度范数（裁剪前记录，口径见 trainer/common/metrics.py）
        "grad_curves": _curves(rows, ["grad_norm_attn", "grad_norm_ffn",
                                      "grad_norm_embed", "grad_norm_norm", "grad_norm_sum_groups"]),
        "weight_curves": _curves(rows, ["weight_norm_attn", "weight_norm_ffn", "weight_norm_embed"]),
        # 门控健康：均值偏离 0.5、离散度、两侧饱和度
        "gate_curves": _curves(rows, ["gate_mean", "gate_std", "gate_sat_lo", "gate_sat_hi"]),
        # 逐层激活量：层 × step 的热力图（系列自身归一化配色，浮层读原值）
        "layer_hidden": _matrix(rows, [f"hidden_rms{s}" for s in layers]),
        "layer_q": _matrix(rows, [f"q_rms{s}" for s in layers]),
        "layer_out": _matrix(rows, [f"out_rms{s}" for s in layers]),
        "layer_gate_std": _matrix(rows, [f"gate_std{s}" for s in layers]),
        # 16 专家逐个负载：比 load_cv 一条线更直接地暴露「哪个专家被饿死」
        "expert_load": _matrix(rows, experts),
        # 系统侧：采样线程记录的 GPU 利用率与内存占用
        "sys_curves": _curves(rows, ["gpu_util_mean", "gpu_util_min", "ram_pct",
                                     "gpu_mem_alloc_mb", "gpu_mem_reserved_mb"]),
        # MoE 偏置（负载均衡 bias）的健康度
        "bias_curves": _curves(rows, ["moe_bias_std", "moe_bias_absmean", "moe_bias_nonzero"]),
    }


# ---------------------------------------------------------------------- #
# SFT
# ---------------------------------------------------------------------- #
SFT_RUNS: List[Tuple[str, str, str]] = [
    ("sft_dense", "Dense · mini", "63.9M · gqa+swiglu"),
    ("sft_moe", "MoE · mini", "214M/75M · 90 万条"),
    ("sft_moe_full", "MoE · full", "214M/75M · 510 万条"),
]


def _sft(root: Path) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    layers = [f"hidden_rms_L{i}" for i in range(8)]
    for key, label, note in SFT_RUNS:
        rows = _read_csv(root / "log" / f"{key}_metrics.csv")
        if not rows:
            continue
        steps, loss = _floats(rows, "loss")
        v_steps, v_loss = _floats(rows, "val_loss")
        tps = _floats(rows, "tokens_per_sec")[1]
        runs.append({
            "key": key,
            "label": label,
            "note": note,
            "steps": int(float(rows[-1]["step"])) if rows[-1].get("step") else len(rows),
            "loss": _downsample(steps, loss),
            "val_loss": _downsample(v_steps, v_loss),
            "loss_first": _round(loss[0]) if loss else None,
            "loss_last": _round(loss[-1]) if loss else None,
            "val_loss_first": _round(v_loss[0]) if v_loss else None,
            "val_loss_last": _round(v_loss[-1]) if v_loss else None,
            "val_loss_min": _round(min(v_loss)) if v_loss else None,
            "val_loss_min_step": v_steps[v_loss.index(min(v_loss))] if v_loss else None,
            "tokens_per_sec": _round(_tail_mean(tps, 100), 0),
            "gpu_mem_mb": _round(_tail_mean(_floats(rows, "gpu_mem_mb")[1], 100), 0),
            # 峰值显存（nvidia-smi 采样窗口内的最大值）：起训练前检查表拿它当
            # 「这套配置在这台机器上要多少显存」的实测依据 —— 比按参数量外推可靠。
            "gpu_mem_peak_mb": _round(max(_floats(rows, "gpu_mem_peak_mb")[1] or [0]), 0) or None,
            "grad_norm_max": _round(max(_floats(rows, "grad_norm")[1] or [0])),
            # 统计牌里的迷你趋势线（首值→末值的方向感，不用点开图）
            "loss_spark": _downsample(steps, loss, cap=24)["value"] or None,
            "val_loss_spark": _downsample(v_steps, v_loss, cap=24)["value"] or None,
            "tps_spark": _downsample(*_floats(rows, "tokens_per_sec"), cap=24)["value"] or None,
            # 与预训练同一口径的富指标：这样 #sft 页能直接和 #pretrain 页横向对照
            "opt_curves": _curves(rows, ["update_ratio_attn", "update_ratio_ffn",
                                         "update_ratio_embed", "update_ratio_norm"]),
            "grad_curves": _curves(rows, ["grad_norm_attn", "grad_norm_ffn", "grad_norm_embed"]),
            "gate_curves": _curves(rows, ["gate_mean", "gate_std", "gate_sat_lo", "gate_sat_hi"]),
            "layer_hidden": _matrix(rows, layers),
            "expert_load": _matrix(rows, [f"moe_load_e{i}" for i in range(16)]),
            "sys_curves": _curves(rows, ["gpu_util_mean", "ram_pct"]),
            # 跨 run 的 MoE 健康度横比（#library 的「哪次训练最健康」卡）用这几个标量：
            # 取尾部均值而不是最后一个点，避免被单步噪声带偏。
            "moe_load_cv_tail": _round(_tail_mean(_floats(rows, "moe_load_cv")[1], 100)),
            "moe_entropy_tail": _round(_tail_mean(_floats(rows, "moe_entropy_norm")[1], 100)),
            "dead_experts_max": _round(max(_floats(rows, "moe_dead_experts")[1] or [0]), 0),
            # 各组参数的更新量：掉到极小说明该组停了。
            # 注意 update_ratio 的量级在 1e-5~1e-3，默认 4 位小数会把它们全舍成 0.0，
            # 所以这里显式给到 8 位有效小数。
            "update_ratio_tail": {k: _round(_tail_mean(_floats(rows, k)[1], 100), 8)
                                  for k in ("update_ratio_attn", "update_ratio_ffn",
                                            "update_ratio_embed", "update_ratio_norm")},
        })
    return {"runs": runs, "compare": _sft_compare(runs)}


def _sft_compare(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Dense vs MoE（mini）的同口径对照。

    两臂同数据 / 同 seq_len / 同等效 batch / 同 epochs / 同 seed，唯一差异是结构，
    所以 val_loss 可以直接相减（test/storage/report/sft_compare/report.md 的同一口径）。

    等质量比算力：算力 ≈ 激活参数 × token 数。Dense 63.91M / MoE 79.94M，
    因此「等步数」对 MoE 略不公平，这里折算成倍数。
    """
    by_key = {r["key"]: r for r in runs}
    dense, moe = by_key.get("sft_dense"), by_key.get("sft_moe")
    if not (dense and moe):
        return {}
    ACT_DENSE, ACT_MOE = 63_912_192, 79_944_704  # 来自 test/storage/report/sft_compare/report.md
    return {
        "act_params": {"dense": ACT_DENSE, "moe": ACT_MOE},
        "ratio_act": _round(ACT_MOE / ACT_DENSE, 3),
        "val_loss_delta": _round(moe["val_loss_last"] - dense["val_loss_last"])
                          if (moe["val_loss_last"] is not None and dense["val_loss_last"] is not None) else None,
        "train_loss_delta": _round(moe["loss_last"] - dense["loss_last"])
                            if (moe["loss_last"] is not None and dense["loss_last"] is not None) else None,
        "throughput_ratio": _round(dense["tokens_per_sec"] / moe["tokens_per_sec"], 2)
                            if (dense["tokens_per_sec"] and moe["tokens_per_sec"]) else None,
        # 逐点 Δ（MoE − Dense），从两条降采样曲线上取共同 step
        "points": _paired_delta(dense["val_loss"], moe["val_loss"]),
    }


def _paired_delta(a: Dict[str, List[float]], b: Dict[str, List[float]]) -> List[Dict[str, float]]:
    """把两条曲线的 step 取交集，输出 [{step, dense, moe, delta}]。"""
    out: List[Dict[str, float]] = []
    bmap = {int(s): v for s, v in zip(b.get("step", []), b.get("value", []))}
    for s, v in zip(a.get("step", []), a.get("value", [])):
        k = int(s)
        if k in bmap:
            out.append({"step": k, "dense": _round(v), "moe": _round(bmap[k]),
                        "delta": _round(bmap[k] - v)})
    return out


# ---------------------------------------------------------------------- #
# RL
# ---------------------------------------------------------------------- #
def _rl(root: Path) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    for arch, arch_label, arch_note in ARCHS:
        for algo, algo_label, algo_note in RL_ALGOS:
            name = f"{algo}_{arch}"
            rows = _read_csv(root / "log" / "rl" / f"{name}_metrics.csv")
            status_file = root / "log" / "rl" / f"{name}.status"
            status = ""
            wall = None
            if status_file.is_file():
                try:
                    txt = status_file.read_text().strip()
                    status = "OK" if txt.startswith("OK") else ("FAIL" if txt.startswith("FAIL") else txt)
                    m = re.search(r"耗时\s+(\d+)s", txt)
                    if m:
                        wall = round(int(m.group(1)) / 3600.0, 2)
                except Exception:  # noqa: BLE001
                    pass

            entry: Dict[str, Any] = {
                "key": name,
                "arch": arch,
                "arch_label": arch_label,
                "arch_note": arch_note,
                "algo": algo,
                "algo_label": algo_label,
                "algo_note": algo_note,
                "status": status,
                "wall_hours": wall,
                "steps": len(rows),
            }
            if not rows:
                runs.append(entry)
                continue

            r_steps, reward = _floats(rows, "reward")
            l_steps, length = _floats(rows, "avg_response_len")
            # 算法专属列先算出来：这一族曲线**只有该算法才有**，缺失的整族不出现
            algo_curves = _curves(rows, RL_ALGO_CURVES.get(algo, []))
            rew_curves = _curves(rows, ["rew_len", "rew_think_len", "rew_think_close",
                                        "rew_rep", "rew_rm"])
            entry.update({
                "reward": _downsample(r_steps, reward, RL_HEAD_POINTS),
                "response_len": _downsample(l_steps, length, RL_HEAD_POINTS),
                "reward_first": _round(reward[0]) if reward else None,
                "reward_last": _round(_tail_mean(reward, 100)),
                "reward_tail": _round(_tail_mean(reward, 100)),
                "response_len_tail": _round(_tail_mean(length, 100), 1),
                "kl_ref_tail": _round(_tail_mean(_floats(rows, "kl_ref")[1], 100)),
                "clipfrac_tail": _round(_tail_mean(_floats(rows, "clipfrac")[1], 100)),
                "eos_rate_tail": _round(_tail_mean(_floats(rows, "eos_rate")[1], 100)),
                "trunc_rate_tail": _round(_tail_mean(_floats(rows, "trunc_rate")[1], 100)),
                "group_zero_std_tail": _round(_tail_mean(_floats(rows, "group_reward_zero_std")[1], 100)),
                "grad_norm_tail": _round(_tail_mean(_floats(rows, "grad_norm")[1], 100)),
                # 峰值显存优先读 gpu_mem_peak_mb（nvidia-smi 采样窗口内的最大值）；
                # 老 CSV 没这列时退回 gpu_mem_mb 的最大值（那是窗口均值，会偏低）。
                "gpu_mem_peak_mb": _round(max(_floats(rows, "gpu_mem_peak_mb")[1]
                                              or _floats(rows, "gpu_mem_mb")[1] or [0]), 0),
                "pass_rate_tail": _round(_tail_mean(_floats(rows, "pass_rate")[1], 100)),
                "policy_loss_tail": _round(_tail_mean(_floats(rows, "policy_loss")[1], 100)),
                # DPO 专属两列，其它算法为空 → 前端显示 —
                "preference_acc_tail": _round(_tail_mean(_floats(rows, "preference_acc")[1], 10)),
                "reward_margin_tail": _round(_tail_mean(_floats(rows, "reward_margin")[1], 10)),
                "turns_mean_tail": _round(_tail_mean(_floats(rows, "turns_mean")[1], 100), 2),
                "valid_call_rate_tail": _round(_tail_mean(_floats(rows, "valid_call_rate")[1], 100)),
                # ---- 本轮新增：把「后端记了、前端没画」的曲线全部补上 ----
                # 奖励分解：五个分项与总量同量纲可加，是「奖励涨了是 RM 真变好
                # 还是长度投机」的唯一直接证据（前端画成堆叠面积）。
                "reward_parts": rew_curves,
                # 优化动力学：RL 阶段在哪一层学得多（与 #pretrain 同口径）
                "opt_curves": _curves(rows, RL_OPT_CURVES),
                # 通用训练/生成健康度（DPO 无 rollout，其中几项自然缺席）
                "gen_curves": _curves(rows, RL_GEN_CURVES),
                # 算法专属族：PPO 的 critic / Agent 的工具轮次 / DPO 的偏好准确率
                "algo_curves": algo_curves,
                "algo_curve_keys": sorted(algo_curves.keys()),
                # MoE 健康度标量：只有 moe 架构的 CSV 才有这些列，dense 自然为 None。
                # #library 的跨 run 横比与 #rl 的诊断都读这一组。
                "moe_load_cv_tail": _round(_tail_mean(_floats(rows, "moe_load_cv")[1], 100)),
                "moe_entropy_tail": _round(_tail_mean(_floats(rows, "moe_entropy_norm")[1], 100)),
                "dead_experts_max": _round(max(_floats(rows, "moe_dead_experts")[1] or [0]), 0),
            })
            runs.append(entry)
    return {"runs": runs}


def _metric_coverage(root: Path, rl: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """算「算法 × 指标家族」的覆盖矩阵 —— 直接回答「不同算法到底记了哪些 log」。

    判据是**各算法 CSV 的真实表头**（不是文档、不是猜测）：
    ``test/log/rl/<algo>_{dense,moe}_metrics.csv`` 两架构取并集，
    再对 ``METRIC_FAMILIES`` 里每个家族数「代表列里有几个真的存在」。

    表格里同时给出「专属列」—— 只在该算法出现、其它算法都没有的列名。
    这就是「针对不同算法记录不同 log 信息」在数据上的确切含义。
    """
    heads: Dict[str, set] = {}
    for algo, _label, _note in RL_ALGOS:
        cols: set = set()
        for arch, _al, _an in ARCHS:
            rows = _read_csv(root / "log" / "rl" / f"{algo}_{arch}_metrics.csv")
            if rows:
                cols |= set(rows[0].keys())
        heads[algo] = cols

    algos = [a for a, _l, _n in RL_ALGOS]
    families: List[Dict[str, Any]] = []
    for fam, probes in METRIC_FAMILIES:
        row: Dict[str, Any] = {"family": fam, "total": len(probes), "by_algo": {}, "values": []}
        for a in algos:
            hit = [c for c in probes if c in heads.get(a, set())]
            row["by_algo"][a] = {"hit": len(hit), "cols": hit, "of": len(probes)}
            row["values"].append(_round(len(hit) / float(len(probes)), 3))
        families.append(row)

    # 专属列：只在某一算法出现、其它算法都没有
    others: Dict[str, set] = {}
    for a in algos:
        o: set = set()
        for b in algos:
            if b != a:
                o |= heads.get(b, set())
        others[a] = heads.get(a, set()) - o
    # 去掉纯序号 / 环境列，免得把「step」也算成某算法的专属能力
    NOISE = {"step", "epoch", "note", "wall_hours", "status"}
    exclusive: Dict[str, List[str]] = {}
    for a in algos:
        cols = [c for c in sorted(others.get(a, set())) if c not in NOISE]
        exclusive[a] = cols

    return {
        "algos": algos,
        "algo_labels": [{"key": a, "label": l, "note": n} for a, l, n in RL_ALGOS],
        "families": families,
        "exclusive": exclusive,
        "total_cols": {a: len(heads.get(a, set())) for a in algos},
    }


# ---------------------------------------------------------------------- #
# 架构 sweep
# ---------------------------------------------------------------------- #
#: 人类可读的配置说明，来自 configs/sweep/<name>.yaml 的语义
SWEEP_DESC: Dict[str, str] = {
    "t1-00-baseline": "基线：gqa + swiglu + rope + rmsnorm",
    "t1-01-attn-gated": "Qwen3-Next 门控注意力（输出门）",
    "t1-02-ffn-geglu": "GeGLU（GELU 版 SwiGLU）",
    "t1-03-ffn-moe-shared": "moe_shared：粗粒度共享专家 MoE",
    "t1-04-ffn-deepseekmoe": "moe_finegrained：16 专家 top-4 + 1 共享",
    "t1-05-pos-theta1e4": "rope_theta=1e4（默认 1e6）",
    "t1-06-pos-nope": "去掉位置编码（NoPE）",
    "t1-07-pos-partial-rope": "部分层 RoPE",
    "t1-08-norm-zero-centered": "zero-centered RMSNorm（权重零初始化）",
    "t1-09-norm-layernorm": "LayerNorm",
    "t2-00-attn-sliding": "滑窗注意力",
    "t2-01-attn-mla": "MLA 多头潜在注意力",
    "t2-02-attn-compressed": "压缩注意力",
    "t2-03-attn-deltanet": "DeltaNet 线性注意力",
    "t2-04-attn-mha": "标准 MHA（无 KV 分组）",
    "t3-00-combo-deepseek": "MLA + moe_finegrained 组合",
    "t3-01-combo-modern": "gated + swiglu + zero-centered + mha 组合",
    "t3-02-combo-readme-moe": "README 原版 MoE",
}


def _sweep(root: Path) -> Dict[str, Any]:
    rows = _read_csv(root / "storage" / "report" / "summary.csv")
    out: List[Dict[str, Any]] = []
    for r in rows:
        name = r.get("name", "")
        def g(col: str) -> Optional[float]:
            v = r.get(col)
            if v in (None, ""):
                return None
            try:
                return float(v)
            except ValueError:
                return None
        out.append({
            "name": name,
            "desc": SWEEP_DESC.get(name, ""),
            "tier": r.get("tier", ""),
            "loss": _round(g("final_loss")),
            "min_loss": _round(g("min_loss")),
            "tps": _round(g("median_tps"), 0),
            "grad_norm": _round(g("median_grad_norm")),
            "peak_mem_mb": _round(g("peak_mem_mb"), 0),
            "params_total_m": _round((g("params_total") or 0) / 1e6, 1),
            "params_act_m": _round((g("params_activated") or 0) / 1e6, 1),
            "cache_kb_per_token": _round((g("cache_bytes_per_token") or 0) / 1024, 1),
            "steps_to_target": int(g("steps_to_target")) if g("steps_to_target") else None,
        })
    out.sort(key=lambda x: x["loss"] if x["loss"] is not None else 9e9)
    baseline = next((x for x in out if x["name"] == "t1-00-baseline"), None)
    base_loss = baseline["loss"] if baseline else None
    for x in out:
        x["delta_vs_base"] = _round(x["loss"] - base_loss) if (x["loss"] is not None and base_loss) else None
    return {"configs": out, "baseline_name": baseline["name"] if baseline else None}


# ---------------------------------------------------------------------- #
# benchmark 评测
# ---------------------------------------------------------------------- #
def _eval(root: Path) -> Dict[str, Any]:
    """评测结果表。

    任务清单的**真源**在 ``test/storage/eval_suite.py::TASKS``，它导出
    ``report/eval/tasks.json``；这里优先读它。**读不到就退回内置的
    ``EVAL_TASKS`` / ``EVAL_RANDOM``** —— 旧产物（只有 4 个 benchmark 的
    ``summary.csv``）不会因此打不开。这与本模块一贯的「文件头只读数据」约束一致：
    多读一个同目录的数据文件，不 import ``test/`` 下的任何代码。

    ``metric`` 过滤是为了修掉旧 CSV 的坑：旧口径对同一个 ``(model, task)``
    写了 ``acc`` 与 ``acc_norm`` **两行**，字典后写覆盖先写，读的人不知道拿到哪个。
    新口径的 ``summary.csv`` 只写 primary metric 一行，这里的过滤对它是恒等的。
    """
    base = root / "storage" / "report" / "eval"
    rows = _read_csv(base / "summary.csv")

    # ---- 任务表：优先 tasks.json，回退内置常量 ----
    tasks: List[Dict[str, Any]] = []
    random: Dict[str, float] = dict(EVAL_RANDOM)
    try:
        meta = json.loads((base / "tasks.json").read_text(encoding="utf-8"))
        for t in meta.get("tasks") or []:
            if not t.get("key"):
                continue
            tasks.append({"key": t["key"], "label": t.get("label") or t["key"],
                          "note": t.get("note") or "", "group": t.get("group") or "",
                          "metric": t.get("metric") or t.get("key"),
                          "chance": t.get("chance")})
            if t.get("chance") is not None:
                random[t["key"]] = t["chance"]
    except Exception:  # noqa: BLE001 tasks.json 缺失/损坏 → 走内置副本
        # 回退时 metric 要写**真实指标名**，不能写成 task key：旧 summary.csv 里
        # `metric` 列是 `acc` / `acc_norm` / `exact_match` / `pass@1`，写成 `ceval`
        # 会让下面的过滤把所有行都丢掉、整页显示「没有评测结果」。
        tasks = [{"key": k, "label": l, "note": n, "group": "",
                  "metric": _LEGACY_METRIC.get(k, k),
                  "chance": EVAL_RANDOM.get(k)} for k, l, n in EVAL_TASKS]

    metric_of = {t["key"]: (t.get("metric") or t["key"]) for t in tasks}

    models: Dict[str, Dict[str, float]] = {}
    meta: Dict[str, Dict[str, Any]] = {}          # model → {depth, n_by_task}
    for r in rows:
        model, task = r.get("model", ""), r.get("task", "")
        if not task:
            continue
        # 只认 primary metric 那一行；任务表里没声明的列（旧 mbpp 等）按原样收下 ——
        # 它们是旧产物的一部分，丢掉会让历史数据在页面上凭空消失。
        want = metric_of.get(task)
        if want is not None and (r.get("metric") or want) != want:
            continue
        try:
            models.setdefault(model, {})[task] = float(r.get("value", 0) or 0)
        except (TypeError, ValueError):
            continue
        m = meta.setdefault(model, {"depth": "", "n": {}, "ci95": {}})
        d = (r.get("depth") or "").strip()
        if d:
            m["depth"] = d
        # n 与 CI 半宽逐任务带回前端：报告里要标「这个数背后是多少题、±多少」，
        # 光有点估计的表读不出可信度（旧口径的坑正是只报分数不报 n）。
        try:
            m["n"][task] = int(float(r.get("n") or 0))
        except (TypeError, ValueError):
            pass
        try:
            lo, hi = float(r.get("ci95_lo") or 0), float(r.get("ci95_hi") or 0)
            if hi > lo:
                m["ci95"][task] = round((hi - lo) / 2, 6)
        except (TypeError, ValueError):
            pass

    #: 阶段归属，用于在表里分组着色
    STAGE = {
        "pretrain": "预训练", "full_sft": "SFT", "dpo": "DPO",
        "grpo": "GRPO", "ppo": "PPO", "agent": "Agentic",
    }

    def stage_of(model: str) -> str:
        for k in sorted(STAGE, key=len, reverse=True):
            if model.startswith(k):
                return STAGE[k]
        return "其它"

    # 扫描目录下各权重的 JSON sidecar，提取 SFT / RL / BPB 等深层细粒度指标
    details_by_model: Dict[str, Dict[str, Any]] = {}
    for p in base.glob("*.json"):
        if p.name in ("tasks.json", "skipped.json"):
            continue
        try:
            jd = json.loads(p.read_text(encoding="utf-8"))
            m_name = jd.get("name")
            if m_name and jd.get("results"):
                d_info: Dict[str, Any] = {}
                res = jd["results"]
                if "agentic" in res and isinstance(res["agentic"], dict):
                    ag = res["agentic"]
                    d_info["agentic"] = {
                        "pass_rate": ag.get("pass_rate"),
                        "tool_name_acc": ag.get("tool_name_acc"),
                        "arg_valid_rate": ag.get("arg_valid_rate"),
                        "turns_mean": ag.get("turns_mean"),
                        "tool_calls_mean": ag.get("tool_calls_mean"),
                        "unfinished_rate": ag.get("unfinished_rate"),
                    }
                if "ifeval" in res and isinstance(res["ifeval"], dict):
                    ife = res["ifeval"]
                    d_info["ifeval"] = {
                        "prompt_acc": ife.get("prompt_acc"),
                        "prompt_acc_loose": ife.get("prompt_acc_loose"),
                        "inst_acc": ife.get("inst_acc"),
                        "n": ife.get("n"),
                        "n_skipped": ife.get("n_skipped"),
                    }
                if "bpb" in res and isinstance(res["bpb"], dict):
                    bpb = res["bpb"]
                    d_info["bpb"] = {
                        "bits_per_token": bpb.get("bits_per_token"),
                        "bits_per_byte": bpb.get("bits_per_byte"),
                    }
                norms = {k: v.get("acc_norm") for k, v in res.items() if isinstance(v, dict) and "acc_norm" in v}
                if norms:
                    d_info["norms"] = norms
                details_by_model[m_name] = d_info
        except Exception:
            continue

    table: List[Dict[str, Any]] = []
    for model, scores in models.items():
        row: Dict[str, Any] = {
            "model": model,
            "stage": stage_of(model),
            "is_moe": model.endswith("_moe"),
        }
        for t in tasks:
            row[t["key"]] = scores.get(t["key"])
        mm = meta.get(model, {})
        row["depth"] = mm.get("depth") or ""
        row["n"] = mm.get("n") or {}
        row["ci95"] = mm.get("ci95") or {}
        row["details"] = details_by_model.get(model, {})
        table.append(row)
    order = {"预训练": 0, "SFT": 1, "DPO": 2, "GRPO": 3, "PPO": 4, "Agentic": 5}
    table.sort(key=lambda x: (order.get(x["stage"], 9), x["is_moe"], x["model"]))
    return {"models": table, "random": random, "tasks": tasks}


# ---------------------------------------------------------------------- #
# 训练登记（trainer/common/registry.py 写的 meta.json）
# ---------------------------------------------------------------------- #
def _runs(root: Path) -> Dict[str, Any]:
    """读 ``<root>/log/runs/*/meta.json``，按开始时间倒序返回。

    登记表是**训练侧**写的（见 ``trainer/common/registry.py``），这里只负责读。
    目录里可能混着旧格式或写坏的 JSON —— 单条失败只跳过那一条。
    """
    rd = root / "log" / "runs"
    if not rd.is_dir():
        return {"runs": [], "counts": {}}
    out: List[Dict[str, Any]] = []
    for meta_path in rd.glob("*/meta.json"):
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        arch = m.get("arch") or {}
        strategy = m.get("train_strategy") or {}
        artifacts = dict(m.get("artifacts") or {})
        # 登记侧把逐 step CSV 记成 `metrics`，实验台其余页面统一叫 `metrics_csv`。
        # 这里补一个别名，让前端不必知道两套命名（只看登记表里到底有没有这个字段）。
        if artifacts.get("metrics") and not artifacts.get("metrics_csv"):
            artifacts["metrics_csv"] = artifacts["metrics"]
        out.append({
            "run_id": m.get("run_id", meta_path.parent.name),
            "status": m.get("status", "unknown"),
            "algo": m.get("algo", ""),
            "run_name": m.get("run_name", ""),
            "started_at": m.get("started_at"),
            "ended_at": m.get("ended_at"),
            "wall_seconds": m.get("wall_seconds"),
            "error": m.get("error"),
            "arch": arch,
            "data": m.get("data") or {},
            "strategy": strategy,
            "artifacts": artifacts,
            "env": m.get("env") or {},
            "git": m.get("git") or {},
            "summary": m.get("summary") or {},
            "notes": (m.get("notes") or [])[:3],
            # 权重产物的**文件名**（不含目录）。对话页要拿它跟「当前加载的权重路径」
            # 比对出「这个模型是哪次训练的产物」—— 比路径更稳：登记侧与部署侧
            # 的产物目录可以不同，文件名却一定是同一个。
            "weight_name": _basename((artifacts.get("weight") or artifacts.get("weights")
                                     or artifacts.get("checkpoint"))),
        })
    out.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    counts: Dict[str, int] = {}
    for r in out:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"runs": out[:80], "counts": counts, "total": len(out)}


# ---------------------------------------------------------------------- #
# 原始日志清单（供「日志浏览器」按需读取）
# ---------------------------------------------------------------------- #
#: 允许浏览的日志后缀。前端按这个清单列文件，读取接口也照此校验。
LOG_SUFFIXES = (".log", ".txt", ".csv", ".md", ".json", ".status", ".out")


def _log_files(root: Path, limit: int = 400) -> List[Dict[str, Any]]:
    """列出可浏览的原始日志（大小 / 修改时间 / 归属），按修改时间倒序。"""
    base = root / "log"
    if not base.is_dir():
        return []
    files: List[Dict[str, Any]] = []
    for p in base.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in LOG_SUFFIXES:
            continue
        if p.name.endswith(".json.tmp"):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        files.append({
            "path": str(p.relative_to(root)),
            "name": p.name,
            "group": str(p.parent.relative_to(base)) or ".",
            "size": st.st_size,
            "mtime": datetime.datetime.fromtimestamp(st.st_mtime).strftime("%m/%d %H:%M"),
            "mtime_ts": st.st_mtime,
        })
    files.sort(key=lambda f: f["mtime_ts"], reverse=True)
    return files[:limit]


# ---------------------------------------------------------------------- #
# 汇总入口
# ---------------------------------------------------------------------- #
#: 实验数据根目录。仓库根下只保留纯净代码，实验日志/产物/报告都在 ``test/`` 里
#: （见 ``test/README.md``）。可用参数或环境变量覆盖，便于指向别处的一份副本。
#: ``MINIMIND_ARTIFACT_ROOT`` 与 ``configs/loader.py`` / ``webui/catalog.py`` 同一口径，
#: 保证「训练写哪里」和「页面读哪里」永远一致。
_HERE = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("MINIMIND_ARTIFACT_ROOT")
                 or os.environ.get("MINIMIND_DATA_ROOT")
                 or (_HERE / "test"))


def collect_experiments(root: Optional[Path] = None) -> Dict[str, Any]:
    """扫一遍实验数据根，返回实验台页面需要的全部数据。任何子块失败都不抛。"""
    root = Path(root) if root else DATA_ROOT

    def safe(fn, *a) -> Any:
        try:
            return fn(*a)
        except Exception:  # noqa: BLE001 一块读失败不能让整个页面 500
            return {}

    return {
        "meta": {
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "artifact_root": str(root),
            "exists": root.is_dir(),
        },
        "pretrain": {
            "dense": safe(_dense_pretrain_log, root),
            "moe": safe(_moe_pretrain_csv, root),
        },
        "sft": safe(_sft, root),
        "rl": safe(_rl, root),
        "coverage": safe(_metric_coverage, root),
        "sweep": safe(_sweep, root),
        "eval": safe(_eval, root),
        "runs": safe(_runs, root),
        "logs": safe(_log_files, root),
    }


if __name__ == "__main__":  # 手工核对用：python webui/experiments_data.py
    import sys

    payload = collect_experiments(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    print(json.dumps(payload, ensure_ascii=False)[:4000])
    print(f"\n… 共 {len(json.dumps(payload, ensure_ascii=False))} 字节")
