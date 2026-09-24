# MiniMind Datasets

数据集按**训练阶段**分目录存放：

```bash
dataset/
├── pretrain/           # 预训练
│   ├── pretrain_t2t.jsonl        (10GB,   847 万条, 约 3.2B token)
│   └── pretrain_t2t_mini.jsonl   (1.2GB,  127 万条, 快速复现用)
├── sft/                # 监督微调（全参 SFT / LoRA / 白盒蒸馏 共用）
│   ├── sft_t2t.jsonl             (14GB,   511 万条, 主线)
│   └── sft_t2t_mini.jsonl        (1.6GB,  90.6 万条, 快速复现用)
├── rl/                 # 强化学习
│   ├── dpo.jsonl                 (53MB,  1.7 万条, 离线偏好优化)
│   ├── rlaif.jsonl               (24MB,  1.95 万条, PPO / GRPO / CISPO)
│   ├── agent_rl.jsonl            (86MB,  4.0 万条, Agentic 多轮工具调用)
│   └── agent_rl_math.jsonl       (18MB,  2.0 万条, 数学 RLVR)
└── external/           # 外部扩展数据集（候选池，与主线分开存放）
    ├── raw/            # 原始下载（约 21GB），保持各数据集原生格式
    ├── sft/            # 转换后：conversations 格式；lora_medical.jsonl 是 LoRA/QLoRA 默认语料
    ├── rl/             # 转换后：chosen/rejected 偏好对，可直接喂 DPODataset
    ├── math/           # 转换后：RLVR 形态（conversations + gt），对齐 agent_rl_math
    ├── agent/          # 转换后：Agentic 多轮（当前为空，见下方说明）
    └── log/            # 下载 / 格式勘察 / 转换 / 校验日志
```

外部数据的转换与校验（转换器按阶段输出，校验用 minimind 原生 Dataset 类实测）：

```bash
python test/storage/download_extra_datasets.py    # 1. 下载（断点续传）
python test/storage/convert_extra_datasets.py     # 2. 转换（--list 查看全部转换器）
python test/storage/verify_extra_datasets.py      # 3. 校验
```

**成套下载**（评测集 + 训练数据一次拉全，见 `test/storage/download_eval_suite.py`）：
它把**评测集**下到 HF 默认 cache、**训练数据**下到 `dataset/external/raw/`
（两者路径互不重叠 —— 评测集落 `dataset/` 会被 `lm_dataset.py` 当成训练数据喂进去）：

```bash
python test/storage/download_eval_suite.py --check              # 先体检，列清单与体积，不下载
python test/storage/download_eval_suite.py --yes                # 必下项（≈1.5 GB）
python test/storage/download_eval_suite.py --yes --optional     # 连可选大件（+LongBench 108MB）
```

新增的转换器（`--list` 里可见）：`perfectblend`（SFT 强基线 136 万条）、
`piqa`（1.6 万条）、`siqa`（3.3 万条）。后两个是**评测集的顺带产出** —— 它们的
主用途在 `eval_suite.py::load_piqa/load_siqa`，落到 `sft/` 只为「成套」时不漏。

> `agent/` 当前为空：`glaive-function-calling-v2` 本质是**工具调用的监督数据**
> （system 内含函数定义原文，用于教模型工具调用格式），而非多轮 agent rollout，
> 因此输出到 `sft/`。`xLAM` 因仓库需授权（GatedRepo 403）未能下载 ——
> **这不妨碍 agentic 评测**：`eval_suite.py` 的 agentic 任务跑的是
> `dataset/rl/agent_rl_math.jsonl`（域外留出集），不依赖 `agent/` 目录。

所有训练脚本的路径已同步指向上述子目录（`configs/*.yaml` 与 `trainer/*.py` 默认值）。
从仓库根目录引用时写 `dataset/<stage>/<file>.jsonl` 即可。

---

Datasets are organized by **training stage**. Place downloaded files into the
matching subdirectory:

- `pretrain/` — pretraining corpora
- `sft/` — supervised fine-tuning data (shared by full SFT / LoRA / distillation)
- `rl/` — RL data (DPO preferences, RLAIF prompts, agentic rollouts)
- `external/` — optional extended dataset pool, kept separate from the mainline
