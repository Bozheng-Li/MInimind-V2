"""训练任务管理 —— 在 WebUI 里启动、跟踪、停止一个真实的训练进程。

**为什么需要**：实验台的其它页面都是「读历史」。训练要调参就必须反复敲命令行，
超参、设备、策略散在 YAML 与 argparse 两层，改了哪一项事后也不好追溯。这里把
「选配置 → 调超参 → 起进程 → 看日志 → 停止」收进同一个服务，且**命令行由界面
用同一个函数拼出来**，因此界面上预览的命令就是真正执行的命令。

安全设计（这是本模块最重要的部分）：

- **参数白名单**：只有 :data:`ALLOWED_FLAGS` 里列出的键可以透传，其余一律丢弃。
  绝不把前端传来的裸字符串拼进命令行。
- **不经过 shell**：``Popen`` 收的是 argv 列表，``shell=False``，因此值里带
  ``;`` ``|`` ``$()`` 都只是普通字符，不构成注入。
- **值类型受控**：每个键声明类型（int/float/str/choice），超范围或类型不符直接 400。
- **独立进程组**：``start_new_session=True``，停止时对整个进程组发信号，
  torchrun / DDP 的子进程不会残留。
- **单写者日志**：每个 job 一个 ``.log`` 文件，stdout/stderr 合并重定向；
  WebUI 用字节偏移增量读取，不重复传已看过的内容。
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

ARTIFACT_ROOT = Path(os.environ.get("MINIMIND_ARTIFACT_ROOT")
                     or os.environ.get("MINIMIND_DATA_ROOT")
                     or (REPO_ROOT / "test"))

JOBS_DIR = ARTIFACT_ROOT / "log" / "jobs"

#: 允许从界面透传的参数。(类型, 取值集合或 None)
#: 类型决定前端控件与后端校验；集合非空时还会检查取值合法性。
ALLOWED_FLAGS: Dict[str, Tuple[str, Optional[List[str]]]] = {
    # ---- 入口 ----
    "config": ("str", None),
    "run_name": ("str", None),
    # ---- 资源 ----
    "device": ("str", None),
    "dtype": ("choice", ["bfloat16", "float16"]),
    "num_workers": ("int", None),
    "seed": ("int", None),
    "use_compile": ("choice", ["0", "1"]),
    # ---- 数据 ----
    "data_path": ("str", None),
    "max_seq_len": ("int", None),
    # ---- 模型结构 ----
    "hidden_size": ("int", None),
    "num_hidden_layers": ("int", None),
    "use_moe": ("choice", ["0", "1"]),
    # ---- 优化 ----
    "epochs": ("int", None),
    "max_steps": ("int", None),
    "batch_size": ("int", None),
    "learning_rate": ("float", None),
    "accumulation_steps": ("int", None),
    "grad_clip": ("float", None),
    "weight_decay": ("float", None),
    "log_interval": ("int", None),
    "save_interval": ("int", None),
    "save_dir": ("str", None),
    "save_weight": ("str", None),
    "from_weight": ("str", None),
    "from_resume": ("choice", ["0", "1"]),
    # ---- 实验记录 ----
    "metrics_detail": ("choice", ["0", "1"]),
    "val_interval": ("int", None),
    "val_samples": ("int", None),
    "val_batches": ("int", None),
    "use_wandb": ("flag", None),
    "wandb_project": ("str", None),
    # ---- LoRA / 蒸馏 ----
    "lora_name": ("str", None),
    "lora_rank": ("int", None),
    "lora_alpha": ("float", None),
    "lora_dropout": ("float", None),
    "lora_target_modules": ("str", None),
    "qlora_quant_type": ("choice", ["nf4", "fp4"]),
    "qlora_double_quant": ("choice", ["0", "1"]),
    "qlora_compute_dtype": ("choice", ["bfloat16", "float16"]),
    "qlora_optimizer": ("choice", ["adamw", "paged_adamw_8bit"]),
    "alpha": ("float", None),
    "temperature": ("float", None),
    "student_use_moe": ("choice", ["0", "1"]),
    "teacher_use_moe": ("choice", ["0", "1"]),
    "from_student_weight": ("str", None),
    "from_teacher_weight": ("str", None),
    # ---- 偏好 / 强化 ----
    "beta": ("float", None),
    "preference_label_smoothing": ("float", None),
    "simpo_gamma": ("float", None),
    "cpo_alpha": ("float", None),
    "orpo_lambda": ("float", None),
    "kto_desirable_weight": ("float", None),
    "kto_undesirable_weight": ("float", None),
    "num_generations": ("int", None),
    "loss_type": ("choice", ["grpo", "cispo"]),
    "epsilon": ("float", None),
    "epsilon_high": ("float", None),
    "max_gen_len": ("int", None),
    "max_total_len": ("int", None),
    "thinking_ratio": ("float", None),
    "rollout_temperature": ("float", None),
    "reward_model_path": ("str", None),
    "rollout_engine": ("choice", ["torch", "sglang"]),
    "sglang_base_url": ("str", None),
    "sglang_shared_path": ("str", None),
    # ---- DAPO ----
    "dapo_epsilon_high": ("float", None),
    "dapo_max_resample": ("int", None),
    "dapo_reward_std_threshold": ("float", None),
    "dapo_overlong_buffer": ("int", None),
    "dapo_overlong_penalty": ("float", None),
    "dapo_kl_coef": ("float", None),
    # ---- PPO ----
    "clip_epsilon": ("float", None),
    "vf_coef": ("float", None),
    "kl_coef": ("float", None),
    "gamma": ("float", None),
    "lam": ("float", None),
    "cliprange_value": ("float", None),
    "ppo_update_iters": ("int", None),
    "early_stop_kl": ("float", None),
    "mini_batch_size": ("int", None),
    "critic_learning_rate": ("float", None),
}

#: 键名合法性：只允许字母数字下划线，杜绝 `--x=...` 之类的花样
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
#: 字符串值里禁止出现的字符（换行会把日志/命令弄乱，空字符直接拒）
_BAD_VALUE_RE = re.compile(r"[\x00\n\r]")

ALGOS = ("pretrain", "sft", "lora", "qlora", "distill",
         "dpo", "ipo", "simpo", "cpo", "orpo", "kto",
         "grpo", "dapo", "rloo", "ppo", "agent")


# ---------------------------------------------------------------------- #
# 命令拼装
# ---------------------------------------------------------------------- #
def _coerce(key: str, value: Any) -> Any:
    kind, choices = ALLOWED_FLAGS[key]
    if kind == "flag":
        truthy = value in (True, 1, "1", "true", "True", "on", "yes")
        return truthy
    if kind == "int":
        return int(float(value))          # 前端可能给 "16" / 16 / 16.0
    if kind == "float":
        return float(value)
    text = str(value)
    if _BAD_VALUE_RE.search(text):
        raise ValueError(f"参数 {key} 含非法字符")
    if kind == "choice":
        text = str(text)
        if choices and text not in choices:
            raise ValueError(f"参数 {key} 只能取 {choices}，收到 {text!r}")
    return text


def build_command(algo: str, params: Dict[str, Any]) -> List[str]:
    """拼出真正的训练命令。**预览与执行共用这一个函数**，两者不可能不一致。

    布尔旗标（``use_wandb``）只在为真时出现；其余参数按 ``ALLOWED_FLAGS`` 白名单透传，
    未在白名单里的键被静默忽略（不是报错 —— 前端可能带上无关字段）。
    """
    if algo not in ALGOS:
        raise ValueError(f"未知算法 {algo!r}，可选 {list(ALGOS)}")

    cmd: List[str] = [sys.executable, "trainer/train.py", "--algo", algo]
    for key, raw in (params or {}).items():
        if raw is None or raw == "":
            continue
        if not _KEY_RE.match(str(key)) or key not in ALLOWED_FLAGS or key == "config":
            continue
        val = _coerce(key, raw)
        if ALLOWED_FLAGS[key][0] == "flag":
            if val:
                cmd.append(f"--{key}")
            continue
        cmd += [f"--{key}", str(val)]
    # --config 固定放在最后，便于阅读
    cfg = (params or {}).get("config")
    if cfg:
        text = str(cfg)
        if _BAD_VALUE_RE.search(text):
            raise ValueError("config 路径含非法字符")
        cmd += ["--config", text]
    return cmd


def pretty_command(cmd: List[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


# ---------------------------------------------------------------------- #
# 任务持久化
# ---------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _job_paths(job_id: str) -> Tuple[Path, Path]:
    return JOBS_DIR / f"{job_id}.json", JOBS_DIR / f"{job_id}.log"


def _write_meta(meta: Dict[str, Any]) -> None:
    meta_path, _ = _job_paths(meta["id"])
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, meta_path)


def _read_meta(job_id: str) -> Optional[Dict[str, Any]]:
    meta_path, _ = _job_paths(job_id)
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _alive(pid: Optional[int]) -> bool:
    """进程号还在不在。

    **注意**：``os.kill(pid, 0)`` 对一个**僵尸**（已退出但没人 wait）同样成功 ——
    所以这个函数只能回答「这个 pid 还被占用吗」，不能回答「它还在跑吗」。
    「还在跑吗」由 :func:`_exit_code` 回答。
    """
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except Exception:  # noqa: BLE001
        return False


#: 本进程起过的训练进程句柄：``job_id -> Popen``。
#:
#: **为什么必须留着**：``Popen`` 对象一旦被 GC，它的子进程退出后就没人回收，
#: 变成僵尸；而 :func:`_alive` 对僵尸返回 True。于是「已经正常跑完的训练」会被
#: :func:`_reconcile` 判成 alive → 永远停在 running；等 pid 终于被系统回收又会被
#: 判成 ``exit_code is None`` → 误标成 interrupted。两种情况都是错的。
#: 只有留住句柄，``poll()`` 才能给出真正的退出码。
#: 重启 WebUI 后这个表是空的，那才是「从 ps 推不出来」的真实场景。
_PROCS: Dict[str, "subprocess.Popen"] = {}


def _exit_code(meta: Dict[str, Any]) -> Optional[int]:
    """先问句柄（唯一可信来源），再退回 meta 里已记下的退出码。推不出来返回 None。"""
    proc = _PROCS.get(meta.get("id") or "")
    if proc is not None:
        rc = proc.poll()  # 进程还在跑时返回 None；退出后一直返回同一个码
        if rc is not None:
            return rc
    rc = meta.get("exit_code")
    return rc if isinstance(rc, int) else None


def _record_exit(meta: Dict[str, Any], rc: int) -> Dict[str, Any]:
    """把退出码落进 meta，并据此定状态。训练侧的 ``RunRecorder`` 是另一套登记。"""
    meta["exit_code"] = rc
    meta["status"] = "ok" if rc == 0 else "failed"
    meta["ended_at"] = meta.get("ended_at") or _now()
    if rc != 0:
        meta["error"] = meta.get("error") or f"训练进程退出码 {rc}"
    _write_meta(meta)
    return meta


def _record_stop(meta: Dict[str, Any], rc: Optional[int]) -> Dict[str, Any]:
    """落「已停止」。

    例外：若拿到的退出码是 0，说明用户点停止时它其实已经正常跑完了 ——
    这种情况如实记成「已完成」。退出码比「点过按钮」更接近事实。
    """
    if rc == 0:
        return _record_exit(meta, 0)
    meta["status"] = "stopped"
    if rc is not None:
        meta["exit_code"] = rc
    meta["ended_at"] = meta.get("ended_at") or _now()
    _write_meta(meta)
    return meta


def _reconcile(meta: Dict[str, Any]) -> Dict[str, Any]:
    """把「meta 写着 running」的状态与真实进程对齐。

    三步，顺序不能换：① 有句柄就先拿退出码（最准）；② 没退出码再问 pid 在不在；
    ③ 两者都说不出所以然（pid 被系统回收了）才标 ``interrupted``。
    """
    if meta.get("status") != "running":
        return meta
    rc = _exit_code(meta)
    if rc is not None:
        return _record_exit(meta, rc)
    if _alive(meta.get("pid")):
        return meta
    # 进程没了、也没有退出码 —— 只可能是本进程没持有句柄（WebUI 重启过，或进程被外部杀掉）
    meta["status"] = "interrupted"
    meta["ended_at"] = meta.get("ended_at") or _now()
    _write_meta(meta)
    return meta


def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    if not JOBS_DIR.is_dir():
        return []
    metas = []
    for p in sorted(JOBS_DIR.glob("*.json"), reverse=True)[:limit]:
        try:
            metas.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
    metas = [_reconcile(m) for m in metas]
    metas.sort(key=lambda m: m.get("started_at") or "", reverse=True)
    return metas


def job_detail(job_id: str) -> Optional[Dict[str, Any]]:
    meta = _read_meta(job_id)
    return _reconcile(meta) if meta else None


def tail(job_id: str, offset: int = 0, limit: int = 200_000) -> Dict[str, Any]:
    """按字节偏移增量读日志。前端只需记住上次的 ``next_offset``。"""
    _, log_path = _job_paths(job_id)
    if not log_path.is_file():
        return {"text": "", "offset": offset, "size": 0, "eof": True}
    size = log_path.stat().st_size
    with log_path.open("rb") as fh:
        fh.seek(max(0, int(offset)))
        chunk = fh.read(limit)
    # 按 UTF-8 边界回退：切在多字节字符中间会解码出乱码
    text = chunk.decode("utf-8", errors="replace")
    consumed = len(chunk)
    if consumed and offset + consumed < size:
        for _ in range(3):
            if text and text.endswith("�"):
                text = text[:-1]
            else:
                break
    return {"text": text, "offset": offset + consumed, "size": size,
            "eof": offset + consumed >= size}


# ---------------------------------------------------------------------- #
# 启动 / 停止
# ---------------------------------------------------------------------- #
def start(algo: str, params: Dict[str, Any], extra_env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """起一个训练进程。返回 job meta（含 id / pid / 命令 / 日志路径）。"""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = build_command(algo, params)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = str(params.get("run_name") or params.get("save_weight") or algo)
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", slug)[:40]
    job_id = f"{stamp}__{algo}__{slug}"

    log_path = JOBS_DIR / f"{job_id}.log"
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"                 # 日志必须实时落盘，否则界面看不到进度
    device = str(params.get("device") or "")
    m = re.search(r"cuda:(\d+)", device)
    if m and "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = m.group(1)  # 让训练进程只看见被选中的卡
    for k, v in (extra_env or {}).items():
        env[str(k)] = str(v)

    meta: Dict[str, Any] = {
        "id": job_id,
        "status": "running",
        "algo": algo,
        "cmd": cmd,
        "cmd_pretty": pretty_command(cmd),
        "params": {k: v for k, v in (params or {}).items()},
        "started_at": _now(),
        "ended_at": None,
        "pid": None,
        "exit_code": None,
        "log": str(log_path),
        "device": device,
        "run_name": params.get("run_name") or "",
        "save_weight": params.get("save_weight") or "",
        "cwd": str(REPO_ROOT),
    }

    fh = log_path.open("ab", buffering=0)
    fh.write(f"$ {meta['cmd_pretty']}\n# cwd={REPO_ROOT}  started={meta['started_at']}\n".encode())
    try:
        proc = subprocess.Popen(  # noqa: S603 参数已过白名单校验，shell=False
            cmd, cwd=str(REPO_ROOT), stdout=fh, stderr=subprocess.STDOUT,
            env=env, start_new_session=True,
        )
    except Exception as exc:  # noqa: BLE001
        fh.write(f"\n[启动失败] {type(exc).__name__}: {exc}\n".encode())
        fh.close()
        meta.update({"status": "failed", "ended_at": _now(),
                     "error": f"{type(exc).__name__}: {exc}"})
        _write_meta(meta)
        return meta

    meta["pid"] = proc.pid
    # 句柄必须留着：Popen 被 GC 后子进程就没人回收，退出后变僵尸，
    # 而僵尸的 pid 骗得过 os.kill(pid, 0)。留着它，poll() 才能给出真实退出码。
    _PROCS[job_id] = proc
    _write_meta(meta)
    fh.close()
    return meta


def stop(job_id: str, grace: float = 10.0) -> Dict[str, Any]:
    """停掉整个进程组（含 DDP / torchrun 的子进程）。先 SIGTERM，超时再 SIGKILL。"""
    meta = _read_meta(job_id)
    if not meta:
        raise FileNotFoundError(f"找不到任务 {job_id}")
    # 已经退出的（含跑完自己结束、或上次停过）直接落「已停止」，不再往死进程发信号
    rc = _exit_code(meta)
    if rc is not None:
        return _record_stop(meta, rc)
    pid = meta.get("pid")
    if not pid or not _alive(pid):
        return _record_stop(meta, None)
    # 训练脚本内部对 SIGTERM 没有特殊处理；但梯度累积/存盘中途被杀可能损坏权重，
    # 所以留一段宽限期，让它把当前 step 走完。
    for sig in (signal.SIGTERM,):
        try:
            os.killpg(os.getpgid(pid), sig)
        except Exception:  # noqa: BLE001
            try:
                os.kill(pid, sig)
            except Exception:  # noqa: BLE001
                pass
    t0 = time.time()
    while time.time() - t0 < grace and _alive(pid):
        time.sleep(0.3)
    if _alive(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    return _record_stop(meta, _exit_code(meta))


def prune(keep: int = 60) -> int:
    """保留最近 ``keep`` 个任务（含日志），其余删除。返回删除数。"""
    metas = list_jobs(limit=10_000)
    if len(metas) <= keep:
        return 0
    removed = 0
    for m in metas[keep:]:
        if m.get("status") == "running" and _alive(m.get("pid")):
            continue
        for p in _job_paths(m["id"]):
            try:
                p.unlink(missing_ok=True)
                removed += 1
            except Exception:  # noqa: BLE001
                pass
    return removed


__all__ = ["ALLOWED_FLAGS", "ALGOS", "build_command", "pretty_command",
           "start", "stop", "list_jobs", "job_detail", "tail", "prune", "JOBS_DIR"]
