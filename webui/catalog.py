"""配置目录 —— 把「这台机器现在能怎么训」变成结构化数据。

三件事：

1. **配置发现**：扫 ``configs/*.yaml``（含 ``defaults`` 链式继承后的**有效**配置）
   与 ``test/configs/sweep/*.yaml``（架构扫描预设），解析出模型结构 + 数据 + 训练超参，
   并推断出可直接执行的命令行。
2. **资源采样**：``nvidia-smi`` 查设备（型号 / 显存 / 当前占用 / 利用率）、
   ``/proc/meminfo`` 查内存、``torch`` 查 CUDA 版本 —— 前端据此显示「这卡空不空」。
3. **策略枚举**：各算法可选的损失类型、rollout 后端、奖励来源等「训练策略」选项。

**只读**：不写任何文件、不改配置。所有函数都保证不抛异常 —— 采集失败的字段留空，
由前端显示成「—」，绝不能因为某个 yaml 写错就让整个页面打不开。

启动方（训练控制台）也复用这里的 :func:`build_command` 生成命令行，
保证「界面上看到的预览」和「实际执行的命令」是同一个函数拼出来的。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: 与 configs/loader.py 的 ARTIFACT_ROOT、webui/server.py 的 DATA_ROOT 同一口径
ARTIFACT_ROOT = Path(os.environ.get("MINIMIND_ARTIFACT_ROOT")
                     or os.environ.get("MINIMIND_DATA_ROOT")
                     or (REPO_ROOT / "test"))

#: 数据集下载说明里出现的 jsonl 后缀 → 用途标签，用于给扫描到的数据文件分类
_DATA_HINTS = {
    "pretrain": "预训练",
    "sft": "监督微调",
    "lora": "LoRA 垂域",
    "dpo": "偏好对",
    "rlaif": "RLAIF 提示词",
    "agent": "Agent 工具轨迹",
    "eval": "评测",
}

#: 算法 → 中文名 + 一句话说明 + 训练策略的选项
ALGO_INFO: Dict[str, Dict[str, Any]] = {
    "pretrain": {"label": "预训练", "stage": "预训练", "kind": "lm",
                 "desc": "自回归下一 token 预测，无监督，学语言规律"},
    "sft": {"label": "全参 SFT", "stage": "监督微调", "kind": "lm",
            "desc": "只对 assistant 回复算 loss 的指令微调"},
    "lora": {"label": "LoRA", "stage": "监督微调", "kind": "lm",
             "desc": "冻结主体、只训低秩分支，低成本垂域适配"},
    "qlora": {"label": "QLoRA", "stage": "监督微调", "kind": "lm",
               "desc": "NF4 量化基座 + LoRA，进一步降低训练显存"},
    "distill": {"label": "知识蒸馏", "stage": "监督微调", "kind": "lm",
                "desc": "CE + KL 双损失，学生学教师的 token 分布"},
    "dpo": {"label": "DPO", "stage": "偏好优化", "kind": "offline_rl",
            "desc": "离线偏好对，无需 reward model / rollout"},
    "ipo": {"label": "IPO", "stage": "偏好优化", "kind": "offline_rl",
             "desc": "具有有限最优间隔的平方偏好目标"},
    "simpo": {"label": "SimPO", "stage": "偏好优化", "kind": "offline_rl",
               "desc": "长度归一化、无需 reference 的偏好优化"},
    "cpo": {"label": "CPO", "stage": "偏好优化", "kind": "offline_rl",
             "desc": "reference-free 偏好目标 + chosen SFT 约束"},
    "orpo": {"label": "ORPO", "stage": "偏好优化", "kind": "offline_rl",
              "desc": "SFT 与 odds-ratio 偏好目标联合优化"},
    "kto": {"label": "KTO", "stage": "偏好优化", "kind": "offline_rl",
             "desc": "分别优化合意与不合意样本的前景效用"},
    "grpo": {"label": "GRPO / CISPO", "stage": "强化学习", "kind": "online_rl",
             "desc": "组内采样 + 组内相对优势，省掉 critic"},
    "dapo": {"label": "DAPO", "stage": "强化学习", "kind": "online_rl",
              "desc": "Clip-Higher + 动态采样 + token-level 策略梯度"},
    "rloo": {"label": "RLOO", "stage": "强化学习", "kind": "online_rl",
              "desc": "REINFORCE leave-one-out baseline，无需 critic"},
    "ppo": {"label": "PPO", "stage": "强化学习", "kind": "online_rl",
            "desc": "Actor-Critic + GAE + KL 早停"},
    "agent": {"label": "Agentic RL", "stage": "强化学习", "kind": "online_rl",
              "desc": "多轮工具调用，整轮延迟结算奖励"},
}

#: 训练策略的可选项（前端渲染成下拉框）。默认值来自 configs/*.yaml。
STRATEGY_CHOICES: Dict[str, List[Dict[str, str]]] = {
    "dtype": [
        {"value": "bfloat16", "label": "bfloat16（推荐，无需 GradScaler）"},
        {"value": "float16", "label": "float16（需 GradScaler，老卡更稳）"},
    ],
    "loss_type": [
        {"value": "cispo", "label": "CISPO（裁剪式重要性采样，本项目默认）"},
        {"value": "grpo", "label": "GRPO（原始组相对策略梯度）"},
    ],
    "rollout_engine": [
        {"value": "torch", "label": "torch 本地采样（无需额外服务）"},
        {"value": "sglang", "label": "sglang 远端采样（训推分离，吞吐高）"},
    ],
    "device": [],   # 运行时由资源面板填
}


# ---------------------------------------------------------------------- #
# 训练控制台的表单规格
# ---------------------------------------------------------------------- #
#: 每个可调参数的展示信息：中文名、说明、以及它属于哪些算法。
#:
#: 这份表**不是**权限清单（权限在 ``jobs.ALLOWED_FLAGS``），而是「界面怎么解释它」。
#: 两边的键必须一致 —— :func:`field_spec` 会把没在这里登记的参数按原样补出来，
#: 因此新加一个白名单参数不会出现「能传但界面不显示」的情况。
_FIELD_META: Dict[str, Dict[str, Any]] = {
    # 入口
    "run_name":      {"label": "实验名", "group": "运行标识", "ph": "例如 moe-sft-ablation",
                      "help": "只用于日志与登记表标识，不影响训练"},
    "save_weight":   {"label": "权重前缀", "group": "运行标识", "ph": "full_sft",
                      "help": "产物命名为 <前缀>_<hidden>[_moe].pth"},
    "save_dir":      {"label": "产物目录", "group": "运行标识", "ph": "留空 = 配置里的 out",
                      "help": "相对路径会落到 MINIMIND_ARTIFACT_ROOT 下"},
    "from_weight":   {"label": "初始权重", "group": "运行标识", "ph": "none / pretrain / full_sft",
                      "help": "从哪个已有权重继续；distill 用 from_student_weight"},
    # 设备
    "device":        {"label": "设备", "group": "资源", "ph": "cuda:0"},
    "dtype":         {"label": "精度", "group": "资源", "choices": "dtype"},
    "num_workers":   {"label": "DataLoader 进程", "group": "资源"},
    "seed":          {"label": "随机种子", "group": "资源", "help": "对比实验必须固定为同一个值"},
    "use_compile":   {"label": "torch.compile", "group": "资源", "choices": [{"value": "0", "label": "关（首次编译慢，调试期建议关）"}, {"value": "1", "label": "开（长训练可提速）"}]},
    # 数据
    "data_path":     {"label": "数据路径", "group": "数据", "ph": "留空 = 配置里的路径",
                      "help": "相对仓库根，如 dataset/sft/sft_t2t_mini.jsonl"},
    "max_seq_len":   {"label": "最大序列长度", "group": "数据"},
    # 结构
    "hidden_size":   {"label": "hidden_size", "group": "模型结构", "help": "命令行值会覆盖 YAML"},
    "num_hidden_layers": {"label": "层数", "group": "模型结构"},
    "use_moe":       {"label": "强制 MoE", "group": "模型结构",
                      "choices": [{"value": "0", "label": "0 · 沿用配置的选择"}, {"value": "1", "label": "1 · 强制切到普通 top-k MoE"}]},
    # 优化
    "epochs":        {"label": "轮数", "group": "优化"},
    "max_steps":     {"label": "步数上限", "group": "优化", "help": "0 = 不限；等 token 预算对比实验就靠它"},
    "batch_size":    {"label": "batch_size", "group": "优化"},
    "accumulation_steps": {"label": "梯度累积", "group": "优化", "help": "等效 batch = batch_size × 累积步数"},
    "learning_rate": {"label": "学习率", "group": "优化", "step": "any"},
    "critic_learning_rate": {"label": "critic 学习率", "group": "优化", "algos": ["ppo"], "step": "any"},
    "grad_clip":     {"label": "梯度裁剪", "group": "优化", "step": "any"},
    "weight_decay":  {"label": "weight_decay", "group": "优化", "step": "any"},
    "log_interval":  {"label": "日志间隔", "group": "优化"},
    "save_interval": {"label": "存盘间隔", "group": "优化"},
    "from_resume":   {"label": "断点续训", "group": "优化",
                      "choices": [{"value": "0", "label": "0 · 从 from_weight 重新开始"}, {"value": "1", "label": "1 · 从最近断点继续"}]},
    # 记录
    "metrics_detail": {"label": "富指标采集", "group": "实验记录",
                       "choices": [{"value": "1", "label": "1 · 采集参数组范数 / MoE 负载 / 系统资源"}, {"value": "0", "label": "0 · 只记基础 loss"}]},
    "val_interval":  {"label": "验证间隔", "group": "实验记录", "help": "每多少步在留出集上算 val_loss；0 = 不算"},
    "val_samples":   {"label": "留出样本数", "group": "实验记录"},
    "val_batches":   {"label": "验证 batch 数", "group": "实验记录"},
    # LoRA / 蒸馏
    "lora_name":     {"label": "LoRA 适配器名", "group": "LoRA / 蒸馏", "algos": ["lora", "qlora"]},
    "lora_rank":     {"label": "LoRA 秩 r", "group": "LoRA / 蒸馏", "algos": ["lora", "qlora"]},
    "lora_alpha":    {"label": "LoRA alpha", "group": "LoRA / 蒸馏", "algos": ["lora", "qlora"]},
    "lora_dropout":  {"label": "LoRA dropout", "group": "LoRA / 蒸馏", "algos": ["lora", "qlora"]},
    "lora_target_modules": {"label": "LoRA 目标层", "group": "LoRA / 蒸馏", "algos": ["lora", "qlora"]},
    "qlora_quant_type": {"label": "4-bit 类型", "group": "LoRA / 蒸馏", "algos": ["qlora"]},
    "qlora_double_quant": {"label": "Double quant", "group": "LoRA / 蒸馏", "algos": ["qlora"]},
    "qlora_compute_dtype": {"label": "量化计算精度", "group": "LoRA / 蒸馏", "algos": ["qlora"]},
    "qlora_optimizer": {"label": "QLoRA 优化器", "group": "LoRA / 蒸馏", "algos": ["qlora"]},
    "alpha":         {"label": "KL 权重 alpha", "group": "LoRA / 蒸馏", "algos": ["distill"], "step": "any",
                      "help": "总损失 = alpha·CE + (1−alpha)·KL"},
    "temperature":   {"label": "蒸馏温度", "group": "LoRA / 蒸馏", "algos": ["distill"], "step": "any"},
    "student_use_moe": {"label": "学生用 MoE", "group": "LoRA / 蒸馏", "algos": ["distill"],
                        "choices": [{"value": "0", "label": "0 · Dense 学生"}, {"value": "1", "label": "1 · MoE 学生"}]},
    "teacher_use_moe": {"label": "教师用 MoE", "group": "LoRA / 蒸馏", "algos": ["distill"],
                        "choices": [{"value": "0", "label": "0 · Dense 教师"}, {"value": "1", "label": "1 · MoE 教师"}]},
    "from_student_weight": {"label": "学生初始权重", "group": "LoRA / 蒸馏", "algos": ["distill"]},
    "from_teacher_weight": {"label": "教师权重", "group": "LoRA / 蒸馏", "algos": ["distill"]},
    # RL 公共
    "beta":          {"label": "KL/偏好温度 β", "group": "强化学习公共", "algos": ["dpo", "ipo", "simpo", "cpo", "kto", "grpo", "rloo", "ppo", "agent"], "step": "any",
                      "help": "DPO 里是偏好温度，RL 里是相对参考模型的 KL 惩罚"},
    "preference_label_smoothing": {"label": "偏好标签平滑", "group": "离线偏好", "algos": ["dpo"]},
    "simpo_gamma": {"label": "SimPO margin", "group": "离线偏好", "algos": ["simpo"]},
    "cpo_alpha": {"label": "CPO SFT 权重", "group": "离线偏好", "algos": ["cpo"]},
    "orpo_lambda": {"label": "ORPO 偏好权重", "group": "离线偏好", "algos": ["orpo"]},
    "kto_desirable_weight": {"label": "KTO 合意权重", "group": "离线偏好", "algos": ["kto"]},
    "kto_undesirable_weight": {"label": "KTO 不合意权重", "group": "离线偏好", "algos": ["kto"]},
    "num_generations": {"label": "组内采样数", "group": "强化学习公共", "algos": ["grpo", "dapo", "rloo", "ppo", "agent"],
                        "help": "同一 prompt 采样几条算组内相对优势"},
    "loss_type":     {"label": "损失函数", "group": "强化学习公共", "algos": ["grpo", "dapo", "rloo", "agent"], "choices": "loss_type"},
    "epsilon":       {"label": "裁剪下界 ε", "group": "强化学习公共", "algos": ["grpo", "dapo", "rloo", "agent"], "step": "any"},
    "epsilon_high":  {"label": "CISPO 裁剪上限", "group": "强化学习公共", "algos": ["grpo", "rloo", "agent"], "step": "any"},
    "max_gen_len":   {"label": "最大生成长度", "group": "强化学习公共", "algos": ["grpo", "dapo", "rloo", "ppo", "agent"]},
    "max_total_len": {"label": "最大总长度", "group": "强化学习公共", "algos": ["agent"]},
    "thinking_ratio": {"label": "思考链比例", "group": "强化学习公共", "algos": ["grpo", "dapo", "rloo", "ppo", "agent"], "step": "any"},
    "rollout_temperature": {"label": "采样温度", "group": "强化学习公共", "algos": ["grpo", "dapo", "rloo", "ppo", "agent"]},
    "reward_model_path": {"label": "奖励模型路径", "group": "强化学习公共", "algos": ["grpo", "dapo", "rloo", "ppo", "agent"]},
    "kl_coef":       {"label": "KL 惩罚系数", "group": "强化学习公共", "algos": ["ppo", "agent"], "step": "any"},
    "rollout_engine": {"label": "采样后端", "group": "强化学习公共", "algos": ["grpo", "dapo", "rloo", "ppo", "agent"], "choices": "rollout_engine"},
    "sglang_base_url": {"label": "sglang 地址", "group": "强化学习公共", "algos": ["grpo", "dapo", "rloo", "ppo", "agent"]},
    "sglang_shared_path": {"label": "sglang 权重交换目录", "group": "强化学习公共", "algos": ["grpo", "dapo", "rloo", "ppo", "agent"]},
    # DAPO 专属
    "dapo_epsilon_high": {"label": "Clip-Higher 上界", "group": "DAPO 专属", "algos": ["dapo"]},
    "dapo_max_resample": {"label": "动态重采样次数", "group": "DAPO 专属", "algos": ["dapo"]},
    "dapo_reward_std_threshold": {"label": "有效组标准差阈值", "group": "DAPO 专属", "algos": ["dapo"]},
    "dapo_overlong_buffer": {"label": "超长软惩罚区间", "group": "DAPO 专属", "algos": ["dapo"]},
    "dapo_overlong_penalty": {"label": "最大超长惩罚", "group": "DAPO 专属", "algos": ["dapo"]},
    "dapo_kl_coef": {"label": "DAPO KL 系数", "group": "DAPO 专属", "algos": ["dapo"]},
    # PPO 专属
    "clip_epsilon":  {"label": "价值裁剪 ε", "group": "PPO 专属", "algos": ["ppo"], "step": "any"},
    "vf_coef":       {"label": "价值损失权重", "group": "PPO 专属", "algos": ["ppo"], "step": "any"},
    "gamma":         {"label": "折扣因子 γ", "group": "PPO 专属", "algos": ["ppo"], "step": "any"},
    "lam":           {"label": "GAE λ", "group": "PPO 专属", "algos": ["ppo"], "step": "any"},
    "cliprange_value": {"label": "价值裁剪范围", "group": "PPO 专属", "algos": ["ppo"], "step": "any"},
    "ppo_update_iters": {"label": "每批更新轮数", "group": "PPO 专属", "algos": ["ppo"]},
    "early_stop_kl": {"label": "KL 早停阈值", "group": "PPO 专属", "algos": ["ppo"], "step": "any"},
    "mini_batch_size": {"label": "mini-batch", "group": "PPO 专属", "algos": ["ppo"]},
    # wandb
    "use_wandb":     {"label": "启用 wandb", "group": "实验记录", "type": "flag"},
    "wandb_project": {"label": "wandb 项目", "group": "实验记录", "algos": "all"},
}

#: 分组展示顺序
FIELD_GROUPS = ["运行标识", "资源", "数据", "模型结构", "优化", "实验记录",
                "LoRA / 蒸馏", "离线偏好", "强化学习公共", "DAPO 专属", "PPO 专属"]

#: 「算法专属」的分组：非空时只在该算法被选中时展示
_ALGO_GROUPS = {"LoRA / 蒸馏", "离线偏好", "强化学习公共", "DAPO 专属", "PPO 专属"}


def field_spec() -> List[Dict[str, Any]]:
    """训练控制台的表单规格。

    **键来自 ``jobs.ALLOWED_FLAGS``**（唯一权限来源），这里只补展示信息；
    因此不可能出现「界面能填但后端会丢」或反过来的情况。
    """
    import jobs as _jobs  # 延迟导入：jobs 与本模块同为顶层模块（webui/ 无 __init__.py）

    flags = _jobs.ALLOWED_FLAGS

    # 「界面能填的字段」= 「后端允许的字段」，两边共用一个来源
    out: List[Dict[str, Any]] = []
    for key, (kind, choices) in flags.items():
        if key == "config":
            continue                      # 配置文件由上方的选择器负责
        meta = dict(_FIELD_META.get(key) or {})
        if "label" not in meta:
            meta["label"] = key
        if "group" not in meta:
            meta["group"] = "其它"
        if key in STRATEGY_CHOICES and STRATEGY_CHOICES[key]:
            meta.setdefault("choices", key)
        spec = {
            "key": key,
            "label": meta["label"],
            "group": meta["group"],
            "type": "flag" if kind == "flag" else (
                "number" if kind in ("int", "float") else ("choice" if kind == "choice" else "text")),
            "step": meta.get("step", "1" if kind == "int" else "any"),
            "phasize": meta.get("ph", ""),
            "help": meta.get("help", ""),
            "algos": meta.get("algos") or None,
            "choices": [],
        }
        if kind == "choice":
            spec["choices"] = [{"value": v, "label": v} for v in (choices or [])]
        ref = meta.get("choices")
        if isinstance(ref, str):
            spec["choices"] = STRATEGY_CHOICES.get(ref) or []
        elif isinstance(ref, list):
            spec["choices"] = ref
        out.append(spec)
    return out



# ---------------------------------------------------------------------- #
# 资源
# ---------------------------------------------------------------------- #
def _run(cmd: List[str], timeout: float = 6.0) -> Optional[str]:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                       timeout=timeout).decode(errors="replace")
    except Exception:  # noqa: BLE001 采集失败不该影响页面
        return None


def gpu_stats() -> List[Dict[str, Any]]:
    """查 GPU 型号 / 显存 / 当前占用。优先 nvidia-smi，退化到 torch。"""
    out: List[Dict[str, Any]] = []
    txt = _run(["nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,"
                "utilization.gpu,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits"])
    if txt:
        for line in txt.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue

            def num(i):
                try:
                    return float(parts[i])
                except (ValueError, IndexError):
                    return None

            total, used = num(2), num(3)
            out.append({
                "index": int(float(parts[0])) if parts[0].replace(".", "").isdigit() else len(out),
                "name": parts[1],
                "mem_total_mb": total,
                "mem_used_mb": used,
                "mem_free_mb": num(4) if num(4) is not None else (
                    (total - used) if (total is not None and used is not None) else None),
                "util_pct": num(5),
                "temp_c": num(6),
                "power_w": num(7),
                "power_limit_w": num(8),
                "mem_used_pct": round(used / total * 100, 1) if (total and used is not None) else None,
            })
    if not out:
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    out.append({
                        "index": i, "name": torch.cuda.get_device_name(i),
                        "mem_total_mb": round(props.total_memory / 1024 ** 2),
                        "mem_used_mb": None, "mem_free_mb": None, "util_pct": None,
                        "temp_c": None, "power_w": None, "power_limit_w": None,
                        "mem_used_pct": None,
                    })
        except Exception:  # noqa: BLE001
            pass
    return out


def running_jobs() -> List[Dict[str, Any]]:
    """正在跑的训练进程（按命令里是否含 trainer/train.py 判断）。

    用于在界面上提示「现在有任务在跑，显存/算力不是闲的」——提交新任务前
    应当知道这一点，否则两个任务会抢同一张卡。
    """
    txt = _run(["ps", "-eo", "pid,etimes,rss,args", "--no-headers"])
    jobs: List[Dict[str, Any]] = []
    if not txt:
        return jobs
    me = os.getpid()
    for line in txt.splitlines():
        if "train.py" not in line or "trainer" not in line:
            continue
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid_s, etimes, rss, cmd = parts
        if not pid_s.isdigit() or int(pid_s) == me:
            continue
        args = cmd.split()
        def flag(name, default=""):
            return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else default
        jobs.append({
            "pid": int(pid_s),
            "elapsed_sec": int(etimes),
            "rss_mb": round(int(rss) / 1024),
            "algo": flag("--algo", "?"),
            "run_name": flag("--run_name"),
            "save_weight": flag("--save_weight"),
            "device": flag("--device"),
            "cmd": cmd if len(cmd) < 400 else cmd[:400] + "…",
        })
    return jobs


def _mem() -> Dict[str, Any]:
    p = Path("/proc/meminfo")
    if not p.is_file():
        return {}
    try:
        info = {}
        for line in p.read_text().splitlines():
            k, _, v = line.partition(":")
            info[k.strip()] = v.strip()
        total = int(info.get("MemTotal", "0 kB").split()[0]) / 1024 ** 2
        avail = int(info.get("MemAvailable", "0 kB").split()[0]) / 1024 ** 2
        return {"total_gb": round(total, 1),
                "available_gb": round(avail, 1),
                "used_gb": round(total - avail, 1),
                "used_pct": round((total - avail) / total * 100, 1) if total else None}
    except Exception:  # noqa: BLE001
        return {}


def _cpu() -> Dict[str, Any]:
    out: Dict[str, Any] = {"cores": os.cpu_count()}
    try:
        load = os.getloadavg()
        out["load1"], out["load5"] = round(load[0], 2), round(load[1], 2)
        if out.get("cores"):
            out["load_pct"] = round(load[0] / out["cores"] * 100, 1)
    except Exception:  # noqa: BLE001
        pass
    return out


def system_resources() -> Dict[str, Any]:
    """资源总览：GPU / 内存 / CPU / 磁盘 / 关键软件版本。"""
    disk = {}
    try:
        usage = shutil.disk_usage(str(ARTIFACT_ROOT if ARTIFACT_ROOT.is_dir() else REPO_ROOT))
        disk = {"total_gb": round(usage.total / 1024 ** 3, 1),
                "free_gb": round(usage.free / 1024 ** 3, 1),
                "used_pct": round(usage.used / usage.total * 100, 1)}
    except Exception:  # noqa: BLE001
        pass
    software: Dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import torch
        software.update({
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
        })
    except Exception:  # noqa: BLE001
        pass
    return {"gpus": gpu_stats(), "memory": _mem(), "cpu": _cpu(), "disk": disk,
            "software": software, "jobs": running_jobs()}


# ---------------------------------------------------------------------- #
# 配置发现
# ---------------------------------------------------------------------- #
def _flatten_slots(cfg: Dict[str, Any]) -> Dict[str, Any]:
    slots = {}
    for key in ("norm", "positional_encoding", "attention", "feedforward"):
        block = cfg.get(key)
        if isinstance(block, dict) and block.get("type"):
            slots[key] = {k: v for k, v in block.items()}
    return slots


#: 超过这个大小就不逐行数了 —— 目录页不该为了一个「约 8 亿行」的提示卡十几秒。
#: 大文件改按体积估算（行数 ≈ 字节 / 110，见 :func:`_count_rows`）。
_ROWS_SCAN_LIMIT = 256 * 1024 * 1024


def _count_rows(path: Optional[str]) -> Tuple[Optional[int], bool]:
    """数 ``.jsonl`` 行数。返回 ``(行数, 是否精确)``。

    小文件（< ``_ROWS_SCAN_LIMIT``）真实数出来；大文件按平均每行约 110 字节估算，
    并标 ``exact=False``，界面显示成「≈」。数不出来返回 ``(None, False)``。
    """
    if not path:
        return None, False
    p = Path(path)
    if not p.is_file():
        return None, False
    try:
        size = p.stat().st_size
        if size > _ROWS_SCAN_LIMIT:
            return max(1, int(size / 110)), False
        with open(p, "rb") as fh:
            return sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 20), b"")), True
    except Exception:  # noqa: BLE001
        return None, False


def describe_config(path: Path, source: str, preset: bool = False) -> Optional[Dict[str, Any]]:
    """解析一份 YAML 成界面条目。解析失败返回 None（列表里少一项而已）。"""
    from configs import load_config, to_arch_config

    try:
        cfg = load_config(path)
        arch = to_arch_config(cfg)
    except Exception:  # noqa: BLE001 不是模型配置（或缺 model 段）就跳过
        return None

    train = dict(cfg.get("train") or {})
    data = dict(cfg.get("data") or {})
    model = dict(cfg.get("model") or {})
    algo_blocks = cfg.get("algo") or {}

    ffn_type = str(arch.type_of("feedforward"))
    n_experts = int(arch.component("feedforward").get("num_experts", 0)) if ffn_type.startswith("moe") else 0
    is_moe_flag = ffn_type.startswith("moe")

    # 预估参数量的常数项（d=768/L=8/vocab=6400）：Dense 63.9M、MoE 总 214M/激活 79.9M。
    # 只用于「大致多少」的量级提示，不冒充精确值。
    hid = arch.model["hidden_size"]
    layers = arch.model["num_hidden_layers"]

    data_path = data.get("path")
    data_abs = None
    if data_path:
        p = Path(data_path)
        data_abs = str(p if p.is_absolute() else (REPO_ROOT / p))
    rows, rows_exact = _count_rows(data_abs)

    rel = str(path.relative_to(REPO_ROOT)) if str(path).startswith(str(REPO_ROOT)) else str(path)
    return {
        "id": rel,
        "name": path.stem,
        "path": rel,
        "preset": preset,
        "kind": "sweep" if preset else "stage",
        "arch": {
            "summary": arch.summary(),
            "hidden_size": hid,
            "num_hidden_layers": layers,
            "vocab_size": arch.model["vocab_size"],
            "n_experts": n_experts,
            "is_moe": is_moe_flag,
            "slots": _flatten_slots(cfg),
            "tie_word_embeddings": model.get("tie_word_embeddings"),
            "max_position_embeddings": model.get("max_position_embeddings"),
        },
        "data": {
            "path": data_path,
            "resolved": data_abs,
            "exists": bool(data_abs and Path(data_abs).is_file()),
            "rows": rows,
            "rows_exact": rows_exact,
            "max_seq_len": data.get("max_seq_len"),
        },
        "train": {k: v for k, v in train.items() if k not in ("reward_model_path",)},
        "algo_blocks": {k: dict(v) for k, v in algo_blocks.items() if isinstance(v, dict)},
        "label": f"{train.get('save_weight') or path.stem} · {rel}",
    }


def list_configs(root: Optional[Path] = None) -> Dict[str, Any]:
    """列出全部可用配置：阶段配置（configs/） + 扫描预设（test/configs/sweep/）。"""
    root = root or ARTIFACT_ROOT
    stage: List[Dict[str, Any]] = []
    for p in sorted((REPO_ROOT / "configs").glob("*.yaml")):
        d = describe_config(p, "configs")
        if d:
            stage.append(d)
    sweep: List[Dict[str, Any]] = []
    sweep_dir = root / "configs" / "sweep"
    if sweep_dir.is_dir():
        for p in sorted(sweep_dir.glob("*.yaml")):
            if p.stem.startswith("_"):
                continue
            d = describe_config(p, "sweep", preset=True)
            if d:
                sweep.append(d)
    return {"stage": stage, "sweep": sweep}


# ---------------------------------------------------------------------- #
# 数据
# ---------------------------------------------------------------------- #
def list_datasets(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """扫 ``dataset/**/*.jsonl``，给出用途标签与行数（行数按 mtime 缓存到内存）。"""
    root = root or REPO_ROOT
    out: List[Dict[str, Any]] = []
    base = root / "dataset"
    if not base.is_dir():
        return out
    for p in sorted(base.rglob("*.jsonl")):
        name = p.stem.lower()
        tag = ""
        for key, label in _DATA_HINTS.items():
            if key in name:
                tag = label
                break
        rel = str(p.relative_to(REPO_ROOT)) if str(p).startswith(str(REPO_ROOT)) else str(p)
        out.append({
            "path": rel,
            "name": p.name,
            "tag": tag,
            "size_gb": round(p.stat().st_size / 1024 ** 3, 3),
        })
    return out


def dataset_path() -> str:
    """配置文件里写的是相对仓库根的路径（如 ``dataset/sft/...``）。"""
    return "dataset"


__all__ = [
    "ALGO_INFO", "STRATEGY_CHOICES", "FIELD_GROUPS", "field_spec",
    "system_resources", "gpu_stats", "running_jobs",
    "list_configs", "describe_config", "list_datasets", "artifact_root",
]
