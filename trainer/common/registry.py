"""实验登记表 —— 每次训练落一份机器可读的 run 记录。

**为什么需要**：训练日志（``.log``）是给人看的，超参散在 argparse / YAML 两层里，
事后要回答「这一跑到底用什么配置、什么策略、吃了多少资源」只能靠翻日志正文。
本模块在训练开始/结束时各写一次 ``{ARTIFACT_ROOT}/log/runs/<run_id>/meta.json``，
把**实际生效**的配置固化成结构化数据，供 WebUI 的训练控制台与实验台直接渲染。

设计要点：

- **只记录实际生效的值**。命令行的优先级高于 YAML，所以登记的是 argparse 解析后的
  ``Namespace``（即真正喂给训练循环的那一份），YAML 原文另存一份作为「声明」。
- **绝不打断训练**。写盘失败只打一行日志；``finalize`` 用 ``atexit`` 兜底，
  进程被异常带走时也能留下 ``status=failed`` 的记录，不会出现「幽灵 run」。
- **零依赖**。只用标准库，避免训练环境里多装东西。

目录布局::

    <ARTIFACT_ROOT>/log/runs/
        index.json                      # run_id -> 摘要，供列表页一次读全
        <run_id>/meta.json              # 单次 run 的完整登记
        <run_id>/stdout.log             # 由启动方重定向（可选）
"""
from __future__ import annotations

import atexit
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

#: 登记表格式版本。字段增删时自增，读取方据此决定兼容分支。
SCHEMA_VERSION = 1

#: 不写进 meta 的 argparse 项（路径噪声大、或体积无意义）
_SKIP_ARGS = frozenset({"config", "debug_mode", "debug_interval", "debug_log_ratio"})


def artifact_root() -> Path:
    """与 ``configs.loader.ARTIFACT_ROOT`` / ``checkpoint._artifact_root()`` 同一口径。"""
    env = os.environ.get("MINIMIND_ARTIFACT_ROOT")
    if env:
        p = Path(env)
        return p if p.is_absolute() else (Path(__file__).resolve().parents[2] / p)
    return Path(__file__).resolve().parents[2] / "test"


def runs_dir() -> Path:
    return artifact_root() / "log" / "runs"


def _jsonable(v: Any) -> Any:
    """把 argparse / YAML 里的值压成 JSON 可序列化的形式。"""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    return str(v)


def _git_info(cwd: Optional[Path] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    cwd = cwd or Path(__file__).resolve().parents[2]
    try:
        for key, cmd in (("commit", ["rev-parse", "--short", "HEAD"]),
                         ("branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
                         ("message", ["log", "-1", "--format=%s"])):
            out[key] = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL,
                                               timeout=5).decode().strip()
        dirty = subprocess.check_output(["status", "--porcelain"], cwd=cwd,
                                        stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        out["dirty"] = bool(dirty)
    except Exception:  # noqa: BLE001 git 不存在或无仓库都不能影响训练
        pass
    return out


def _env_info() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = torch.version.cuda
        out["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["gpu_count"] = torch.cuda.device_count()
            out["gpus"] = [{
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_mem_mb": round(torch.cuda.get_device_properties(i).total_memory / 1024 ** 2),
                "capability": list(torch.cuda.get_device_capability(i)),
            } for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        pass
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        out["cuda_visible_devices"] = cvd
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            out["world_size"] = dist.get_world_size()
            out["rank"] = dist.get_rank()
    except Exception:  # noqa: BLE001
        pass
    return out


def _count_rows(path: Optional[str]) -> Optional[int]:
    """数 jsonl 行数。大文件用分块读，避免把 10GB 数据读进内存。"""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        n = 0
        with p.open("rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                n += chunk.count(b"\n")
        return n
    except Exception:  # noqa: BLE001
        return None


def _algo_hyperparams(args, algo: str) -> Dict[str, Any]:
    """按算法挑出「这一跑的训练策略」——从 args 里取实际生效的值。

    通用项（lr / batch / 调度）对所有算法都重要；RL 再叠上它独有的那一组。
    这里显式列出而不是全量 dump，是为了让界面能按算法分组渲染、且语义清晰。
    """
    def g(*names, default=None):
        for n in names:
            if hasattr(args, n):
                return getattr(args, n)
        return default

    common = {
        "epochs": g("epochs"),
        "batch_size": g("batch_size"),
        "accumulation_steps": g("accumulation_steps"),
        "effective_batch": (g("batch_size") or 1) * (g("accumulation_steps") or 1),
        "learning_rate": g("learning_rate"),
        "scheduler": "cosine",
        "grad_clip": g("grad_clip"),
        "dtype": g("dtype"),
        "seed": g("seed"),
        "max_seq_len": g("max_seq_len"),
        "max_steps": g("max_steps"),
        "log_interval": g("log_interval"),
        "save_interval": g("save_interval"),
        "from_weight": g("from_weight", "from_student_weight"),
        "from_resume": g("from_resume"),
        "use_compile": g("use_compile"),
    }
    if algo in ("grpo", "dapo", "rloo", "ppo", "agent"):
        common.update({
            "num_generations": g("num_generations"),
            "beta_kl": g("beta"),
            "loss_type": g("loss_type"),
            "epsilon": g("epsilon"),
            "epsilon_high": g("epsilon_high"),
            "max_gen_len": g("max_gen_len"),
            "thinking_ratio": g("thinking_ratio"),
            "rollout_engine": g("rollout_engine"),
            "reward_model_path": g("reward_model_path"),
            "kl_coef": g("kl_coef"),
        })
    if algo == "agent":
        common.update({"max_total_len": g("max_total_len")})
    if algo == "ppo":
        common.update({
            "clip_epsilon": g("clip_epsilon"), "vf_coef": g("vf_coef"),
            "gamma": g("gamma"), "lam": g("lam"),
            "cliprange_value": g("cliprange_value"),
            "ppo_update_iters": g("ppo_update_iters"),
            "early_stop_kl": g("early_stop_kl"),
            "mini_batch_size": g("mini_batch_size"),
            "critic_learning_rate": g("critic_learning_rate"),
        })
    if algo in ("dpo", "ipo", "simpo", "cpo", "orpo", "kto"):
        common.update({
            "beta": g("beta"), "label_smoothing": g("preference_label_smoothing"),
            "simpo_gamma": g("simpo_gamma"), "cpo_alpha": g("cpo_alpha"),
            "orpo_lambda": g("orpo_lambda"),
        })
    if algo == "distill":
        common.update({
            "alpha": g("alpha"), "temperature": g("temperature"),
            "student_use_moe": g("student_use_moe"), "teacher_use_moe": g("teacher_use_moe"),
        })
    if algo in ("lora", "qlora"):
        common.update({
            "lora_name": g("lora_name"), "lora_rank": g("lora_rank"),
            "lora_alpha": g("lora_alpha"), "lora_dropout": g("lora_dropout"),
            "lora_target_modules": g("lora_target_modules"),
        })
    if algo == "qlora":
        common.update({
            "quant_type": g("qlora_quant_type"),
            "double_quant": g("qlora_double_quant"),
            "compute_dtype": g("qlora_compute_dtype"),
            "qlora_optimizer": g("qlora_optimizer"),
        })
    if algo == "dapo":
        common.update({
            "epsilon_high": g("dapo_epsilon_high"),
            "max_resample": g("dapo_max_resample"),
            "overlong_buffer": g("dapo_overlong_buffer"),
            "overlong_penalty": g("dapo_overlong_penalty"),
            "dapo_kl_coef": g("dapo_kl_coef"),
        })
    return {k: _jsonable(v) for k, v in common.items() if v is not None}


class RunRecorder:
    """一次训练的登记句柄。``__init__`` 时写首版 meta，``finish`` 时补全。"""

    def __init__(self, args, algo: str, lm_config=None, cfg: Optional[Dict[str, Any]] = None,
                 config_path: Optional[str] = None, data_path: Optional[str] = None):
        from .checkpoint import resolve_weight_prefix

        self.args = args
        self.algo = algo
        self.t0 = time.time()
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.status = "running"
        self.error: Optional[str] = None
        self.notes: List[str] = []
        self.extra: Dict[str, Any] = {}

        prefix = resolve_weight_prefix(args)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{stamp}__{algo}__{prefix}"
        self.dir = runs_dir() / self.run_id
        self.meta_path = self.dir / "meta.json"

        weight_prefix = prefix
        arch_summary, arch_dict, n_layers, n_experts, params_total = "", {}, None, 0, None

        self.meta: Dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": "running",
            "started_at": self.started_at,
            "ended_at": None,
            "wall_seconds": None,
            "algo": algo,
            "run_name": getattr(args, "run_name", "") or prefix,
            "weight_prefix": weight_prefix,
            "args": {k: _jsonable(v) for k, v in vars(args).items() if k not in _SKIP_ARGS},
            "config": {
                "path": config_path,
                "raw": _jsonable(cfg) if cfg else None,
            },
            "train_strategy": _algo_hyperparams(args, algo),
            "arch": {
                "summary": arch_summary,
                "dict": arch_dict,
                "n_layers": n_layers,
                "n_experts": n_experts,
                "params_total": params_total,
                "is_moe": False,
            },
            "data": {
                "path": data_path,
                "rows": _count_rows(data_path),
                "max_seq_len": getattr(args, "max_seq_len", None),
            },
            "artifacts": {
                "save_dir": getattr(args, "save_dir", None),
                "metrics_csv": getattr(args, "metrics_path", None),
                "weight": None,
            },
            "env": _env_info(),
            "git": _git_info(),
            "summary": {},
            "notes": self.notes,
        }
        self._write()
        # 异常退出（含被 kill / OOM）也要留下痕迹，否则列表页会一直显示 running
        atexit.register(self._atexit_finish)

    # ------------------------------------------------------------------ #
    def _write(self) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.meta_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.meta_path)      # 原子替换，读取方永远看到完整 JSON
            self._reindex()
        except Exception as exc:  # noqa: BLE001
            print(f"[registry] 写 meta 失败（不影响训练）: {type(exc).__name__}: {exc}")

    def _reindex(self) -> None:
        """维护 index.json：列表页只读这一个文件，不必扫上千个子目录。"""
        idx_path = runs_dir() / "index.json"
        try:
            idx: Dict[str, Any] = {}
            if idx_path.is_file():
                idx = json.loads(idx_path.read_text(encoding="utf-8")) or {}
            entry = idx.setdefault("runs", {})
            entry[self.run_id] = {
                "status": self.meta["status"],
                "started_at": self.meta["started_at"],
                "ended_at": self.meta["ended_at"],
                "algo": self.algo,
                "run_name": self.meta["run_name"],
                "arch": self.meta["arch"]["summary"],
                "is_moe": self.meta["arch"]["is_moe"],
                "data": self.meta["data"]["path"],
                "wall_seconds": self.meta["wall_seconds"],
                "summary": self.meta["summary"],
            }
            idx_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = idx_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, idx_path)
        except Exception:  # noqa: BLE001
            pass

    def attach_lm_config(self, lm_config) -> None:
        """训练流程走到「配置组装完成」时补登架构信息。

        ``RunRecorder`` 构造得比 ``lm_config`` 早（要保证任何早期异常都被登记），
        所以架构字段在这里二次填充。解析失败只记一条 note。
        """
        try:
            arch = lm_config.to_arch_config()
            ffn_type = str(arch.type_of("feedforward"))
            n_experts = (int(arch.component("feedforward").get("num_experts", 0))
                         if ffn_type.startswith("moe") else 0)
            self.meta["arch"].update({
                "summary": arch.summary(),
                "dict": arch.to_dict(),
                "n_layers": int(arch.model["num_hidden_layers"]),
                "n_experts": n_experts,
                "is_moe": bool(n_experts),
                "hidden_size": int(arch.model["hidden_size"]),
                "vocab_size": int(arch.model["vocab_size"]),
                "attention": str(arch.type_of("attention")),
                "feedforward": ffn_type,
                "norm": str(arch.type_of("norm")),
                "positional_encoding": str(arch.type_of("positional_encoding")),
            })
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"架构登记失败: {type(exc).__name__}: {exc}")
        self._write()

    def set_model(self, model) -> None:
        """记录真实参数量（MoE 还要区分总参数与激活参数）。"""
        try:
            from ..trainer_utils import unwrap_model as _uw  # noqa: F401
        except Exception:  # noqa: BLE001
            pass
        try:
            raw = model
            for attr in ("module", "_orig_mod"):
                raw = getattr(raw, attr, raw)
            total = sum(p.numel() for p in raw.parameters())
            trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
            self.meta["arch"]["params_total"] = total
            self.meta["arch"]["params_trainable"] = trainable
            # MoE 的「激活参数」——小模型对比里唯一公平的算力口径
            n_routed = int(self.meta["arch"].get("n_experts") or 0)
            if n_routed:
                cfg = getattr(raw, "config", None)
                top_k = int(getattr(cfg, "num_experts_per_tok", 1) or 1)
                expert = sum(p.numel() for n, p in raw.named_parameters() if ".experts.0." in n)
                shared = sum(p.numel() for n, p in raw.named_parameters()
                             if ".shared_experts.0." in n or ".shared_expert." in n)
                n_shared = int(getattr(cfg, "n_shared_experts", 1) or 0) if shared else 0
                base = total - expert * n_routed - shared * n_shared
                self.meta["arch"]["params_activated"] = base + expert * top_k + shared * n_shared
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"参数量登记失败: {type(exc).__name__}: {exc}")
        self._write()

    # ------------------------------------------------------------------ #
    def set(self, **kw) -> None:
        """记录「这一跑的关键事实」（RL 的奖励来源、rollout 后端等）。"""
        self.extra.update({k: _jsonable(v) for k, v in kw.items()})
        self.meta.update(self.extra)
        self._write()

    def note(self, text: str) -> None:
        self.notes.append(text)
        self.meta["notes"] = self.notes
        self._write()

    def attach_artifacts(self, **kw) -> None:
        self.meta["artifacts"].update({k: _jsonable(v) for k, v in kw.items()})
        self._write()

    def finish(self, status: str = "ok", error: Optional[str] = None) -> None:
        if self.status != "running":
            return
        self.status = status
        self.error = error
        self.meta["status"] = status
        self.meta["error"] = error
        self.meta["ended_at"] = datetime.now().isoformat(timespec="seconds")
        self.meta["wall_seconds"] = round(time.time() - self.t0, 1)
        self.meta["summary"] = summarize_run(self.meta)
        self._write()

    def _atexit_finish(self) -> None:
        if self.status == "running":
            self.finish(status="unknown", error="进程退出时未显式收尾（被中断或异常）")


# ---------------------------------------------------------------------- #
def summarize_run(meta: Dict[str, Any]) -> Dict[str, Any]:
    """从指标 CSV 抽最终结果摘要。文件缺失/损坏时返回 {}，不抛。"""
    import csv
    path = (meta.get("artifacts") or {}).get("metrics_csv")
    if not path or not Path(path).is_file():
        return {}
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r]
    except Exception:  # noqa: BLE001
        return {}
    if not rows:
        return {}

    def col(name: str):
        out = []
        for r in rows:
            v = r.get(name)
            if v in (None, ""):
                continue
            try:
                out.append(float(v))
            except ValueError:
                continue
        return out

    def tail(vals, n=50):
        return sum(vals[-n:]) / len(vals[-n:]) if vals else None

    def rnd(x, nd=4):
        return None if x is None else round(x, nd)

    loss = col("loss")
    reward = col("reward")
    val = col("val_loss")
    summary: Dict[str, Any] = {
        "steps": len(rows),
        "loss_first": rnd(loss[0]) if loss else None,
        "loss_last": rnd(loss[-1]) if loss else None,
        "loss_tail": rnd(tail(loss)),
        "loss_min": rnd(min(loss)) if loss else None,
        "val_loss_last": rnd(val[-1]) if val else None,
        "val_loss_min": rnd(min(val)) if val else None,
        "best_val_step": None,
        "reward_tail": rnd(tail(reward)),
        "reward_first": rnd(reward[0]) if reward else None,
        "tokens_per_sec": rnd(tail(col("tokens_per_sec")), 1),
        "gpu_mem_peak_mb": rnd(max(col("gpu_mem_mb") or [0]) or None, 0),
        "grad_norm_max": rnd(max(col("grad_norm") or [0]) or None),
    }
    if val:
        i = val.index(min(val))
        try:
            summary["best_val_step"] = int(float(rows[i].get("step", 0)))
        except ValueError:
            pass
    for k in ("pass_rate", "preference_acc", "kl_ref", "moe_load_cv", "pde"):
        vals = col(k)
        if vals:
            summary[f"{k}_tail"] = rnd(tail(vals))
    return {k: v for k, v in summary.items() if v is not None}


__all__ = ["RunRecorder", "summarize_run", "runs_dir", "artifact_root", "SCHEMA_VERSION"]
