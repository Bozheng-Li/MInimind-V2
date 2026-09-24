# MiniMind 项目学习指南

> 本文是为学习 MiniMind 项目而写的“陪跑手册”，配合根目录 `README.md` 使用。
> README 侧重**是什么、怎么用**，本指南侧重**按什么顺序学、每一步该读哪段代码、跑哪条命令、怎么验证结果**。
>
> 适用版本：`minimind-3`（结构对齐 Qwen3 / Qwen3-MoE，Dense≈64M，MoE≈198M-A64M）。

---

## 一、这个项目是什么

MiniMind 是一个**从 0 开始、用纯 PyTorch 手写、完整复现大语言模型全流程训练**的开源教程项目。

一句话概括它的野心：**用约 3 块钱的 GPU 租金、约 2 小时的训练时间，在单张 3090 上从零训出一个能对话的 64M 小模型**，并把从预训练到强化学习的每一行核心代码都摊开给你看。

它和市面上大多数“LLM 教程”的关键区别：

- **不是** 用 `transformers` / `trl` / `peft` 的高层 API 拼十几行代码调用别人的模型；
- **而是** 自己实现 Attention、RoPE、RMSNorm、SwiGLU、MoE、LoRA、DPO、PPO、GRPO、CISPO、蒸馏、Agentic RL 的核心逻辑。

覆盖的完整训练链路：

```
Tokenizer 训练
   → Pretrain（预训练，学语言）
      → SFT（指令微调，学对话/工具调用/思考）
         ├── LoRA（低成本垂域微调）
         ├── 知识蒸馏（大模型教小模型）
         ├── DPO（人类偏好对齐，RLHF）
         └── RLAIF：PPO / GRPO / CISPO（AI/规则反馈强化学习）
                → Agentic RL（多轮工具调用的强化学习）
   → 部署（CLI / OpenAI API / Web 控制台 / ollama / vllm / sglang）
```

---

## 二、为什么值得学（设计哲学）

| 特点 | 对学习者的意义 |
|------|----------------|
| 核心算法从 0 用 PyTorch 原生实现 | 能真正看懂 LLM 内部每一步，而不是黑盒调用 |
| 模型极小（64M），词表仅 6400 | 单卡甚至 CPU 可跑，试错成本极低 |
| 结构对齐 Qwen3 | 学到的结构知识可直接迁移到工业级模型 |
| 统一数据格式（jsonl）+ 统一 chat_template | 免去繁琐的数据预处理，专注理解算法 |
| 全阶段脚本参数风格一致 | 学会一个 `train_*.py`，其余触类旁通 |
| 支持断点续训 / 多卡 / wandb | 贴近真实训练工程实践 |

> 核心心法：**“大道至简”**。先把最小闭环跑通（Pretrain + SFT），再逐个模块深入。

---

## 三、代码地图（目录结构）

```
minimind/
├── model/                      # 【模型定义】
│   ├── model_minimind.py       # ★核心：Config + Attention + MoE + 整个 Transformer（约 287 行）
│   ├── model_lora.py           # ★从 0 实现的 LoRA（约 65 行）
│   ├── tokenizer.json          # 分词器（BPE + ByteLevel，词表 6400）
│   └── tokenizer_config.json   # chat_template 模板（含 <tool_call>/<think> 等标记）
│
├── trainer/                    # 【各阶段训练脚本，均需 cd 进来执行】
│   ├── train_pretrain.py       # 预训练（必做）
│   ├── train_full_sft.py       # 全参数监督微调（必做）
│   ├── train_lora.py           # LoRA 微调
│   ├── train_dpo.py            # DPO（RLHF）
│   ├── train_distillation.py   # 白盒知识蒸馏
│   ├── train_grpo.py           # GRPO / CISPO
│   ├── train_ppo.py            # PPO
│   ├── train_agent.py          # Agentic RL（多轮工具调用）
│   ├── train_tokenizer.py      # 训练分词器（一般不需要重训）
│   ├── rollout_engine.py       # RL 采样引擎（torch / sglang 训推分离）
│   └── trainer_utils.py        # 公共工具：学习率、断点、DDP、模型初始化等
│
├── dataset/                    # 【数据加载】
│   ├── lm_dataset.py           # ★各阶段 Dataset：Pretrain/SFT/DPO/RLAIF/AgentRL
│   └── dataset.md              # 数据集下载说明（数据文件需自行下载放这里）
│
├── webui/                      # 【Web 控制台】对话 + 实验台 + 训练控制台（FastAPI，无构建步骤）
│   ├── server.py               # 后端：路由 / SSE 流式 / 工具循环 / 路径防护
│   ├── catalog.py              # 只读扫描：配置、数据集、算法、资源（不写文件）
│   ├── experiments_data.py     # 读 test/ 下的产物，解析成实验数据
│   ├── jobs.py                 # 训练作业：拼 argv（白名单）/ 起停 / 状态
│   └── static/                 # 原生 HTML/CSS/JS（viz.js 是自带的图表库）
│
├── scripts/                    # 【推理 / 服务 / 工具】
│   ├── web_demo.py             # 旧版 Streamlit 聊天页（仅读 transformers 格式目录）
│   ├── serve_openai_api.py     # OpenAI 兼容 API 服务端
│   ├── chat_api.py             # 调用 API 的示例客户端
│   ├── eval_toolcall.py        # 工具调用能力评测
│   └── convert_model.py        # torch <-> transformers 格式转换 & LoRA 合并
│
├── eval_llm.py                 # ★CLI 交互式对话评测（根目录）
├── requirements.txt            # 依赖清单
├── README.md / README_en.md    # 官方完整文档
└── LEARNING_GUIDE.md           # 本文件
```

**建议按重要性精读的 4 个文件**：
1. `model/model_minimind.py` — 模型全貌
2. `dataset/lm_dataset.py` — 各阶段数据长什么样、label/loss_mask 怎么造
3. `trainer/train_pretrain.py` — 最简单的训练循环模板
4. `trainer/trainer_utils.py` — 所有脚本共用的基础设施

---

## 四、模型架构详解（`minimind-3`）

Decoder-Only Transformer，配置对齐 Qwen3，方便导出到 transformers / llama.cpp / ollama / vllm。

| 组件 | 选型 | 代码位置（`model_minimind.py`） |
|------|------|-------------------------------|
| 归一化 | Pre-Norm + **RMSNorm** | `class RMSNorm` |
| 位置编码 | **RoPE**（`rope_theta=1e6`，`max_pos=32768`），支持 **YaRN** 外推 | `precompute_freqs_cis` / `apply_rotary_pos_emb` |
| 注意力 | **GQA** 分组查询注意力（`q_heads=8`，`kv_heads=4`） | `class Attention` / `repeat_kv` |
| 前馈 | **SwiGLU** | `class FeedForward` |
| MoE 前馈 | 4 experts / top-1 routing（去除 shared expert），带负载均衡 aux_loss | `class MOEFeedForward` |
| 主干 | Embedding → N×Block → RMSNorm → lm_head | `MiniMindModel` / `MiniMindForCausalLM` |

**默认配置（在 `MiniMindConfig.__init__` 中）**：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `hidden_size` | 768 | 隐藏维度（d_model） |
| `num_hidden_layers` | 8 | 层数 |
| `vocab_size` | 6400 | 词表大小 |
| `num_attention_heads` | 8 | Q 头数 |
| `num_key_value_heads` | 4 | KV 头数（GQA） |
| `num_experts` | 4 | MoE 专家数 |
| `num_experts_per_tok` | 1 | 每 token 激活专家数（top-1） |
| `use_moe` | False | 是否启用 MoE |

**参数版本对照**：

| Model | params | n_layers | d_model | kv_heads | q_heads | 说明 |
|-------|--------|----------|---------|----------|---------|------|
| minimind-3 | 64M | 8 | 768 | 4 | 8 | Dense（主线） |
| minimind-3-moe | 198M-A64M | 8 | 768 | 4 | 8 | 4 experts / top-1 |

> 训练/推理时若要用 MoE，几乎所有脚本都加 `--use_moe 1` 即可；权重文件会自动带 `_moe` 后缀。
> 为什么选 `dim=768, n_layers=8`？README「模型配置」一节结合 MobileLLM 的结论解释了小模型下“深而窄 vs 宽而浅”的取舍，值得一读。

---

## 五、环境准备

```bash
# 1. 克隆 & 安装依赖
git clone --depth 1 https://github.com/jingyaogong/minimind
cd minimind
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

# 2. 单独安装 PyTorch（requirements 里被注释了，按你的 CUDA 版本装）
#    参考 https://pytorch.org/ ，例如：
pip install torch==2.6.0

# 3. 验证 GPU 可用
python -c "import torch; print(torch.cuda.is_available())"
```

> 参考软硬件：单卡 RTX 3090(24GB)、CUDA 12.2、Python 3.10。CPU / MPS 也能跑但会很慢。
> 国内访问 WandB 困难，项目默认可切 [SwanLab](https://swanlab.cn/)（接口兼容，通常只需把 `import wandb` 换成 `import swanlab as wandb`）。

---

## 六、推荐学习路径（总览）

按下面的顺序走，从“看懂+跑通”到“逐个深入算法”：

```
阶段 0  跑通推理（下载现成模型，先看到效果）           ← 半天
阶段 1  读懂模型与数据代码（model_minimind + lm_dataset）← 1~2 天
阶段 2  预训练 Pretrain（必做，理解“词语接龙”）        ← 跑 ~1.2h
阶段 3  监督微调 SFT（必做，得到能对话的 Zero 模型）    ← 跑 ~1.1h
阶段 4  评测对比（eval_llm / eval_toolcall）
──────────── 以上是最小闭环，务必先完整走一遍 ────────────
阶段 5  LoRA（低成本垂域/自我认知微调）
阶段 6  DPO（人类偏好对齐，RLHF 入门）
阶段 7  知识蒸馏（CE + KL 双损失）
阶段 8  RLAIF：PPO / GRPO / CISPO（策略优化统一视角）
阶段 9  Agentic RL（多轮工具调用 + 训推分离）
阶段 10 部署（OpenAI API / Web 控制台 / 第三方引擎）
```

**数据准备**：从 [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) 或 [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) 下载，放入 `./dataset/`。
最小闭环只需下载两个 `✨` 文件：

```
./dataset/
├── pretrain_t2t_mini.jsonl   (1.2GB, ✨ 预训练-快速复现)
├── sft_t2t_mini.jsonl        (1.6GB, ✨ SFT-快速复现，已混入 tool call 样本)
├── rlaif.jsonl               (24MB,  ✨ RLAIF)
├── dpo.jsonl                 (53MB)
├── agent_rl.jsonl            (86MB)
├── agent_rl_math.jsonl       (18MB)
├── pretrain_t2t.jsonl        (10GB,  完整版)
└── sft_t2t.jsonl             (14GB,  完整版)
```

**权重命名规则**（保存在 `./test/out/`）：`<阶段名>_<hidden_size>[_moe].pth`
例如 `pretrain_768.pth`、`full_sft_768.pth`、`full_sft_768_moe.pth`、`dpo_768.pth`、`grpo_768.pth`、`agent_768.pth`、`lora_medical_768.pth`。

---

## 七、各阶段详解（目标 / 原理 / 代码 / 命令 / 验证）

> 所有 `train_*.py` 都在 `trainer/` 目录下执行，先 `cd trainer`。
> `eval_llm.py` 在**项目根目录**执行。

### 阶段 0：先跑通推理，看到效果

先用作者训练好的模型体验一下，建立“目标感”。

```bash
# 下载现成模型（transformers 格式）
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
# 或：git clone https://huggingface.co/jingyaogong/minimind-3

# CLI 对话（transformers 格式）
python eval_llm.py --load_from ./minimind-3

# 开启“自适应思考”体验
python eval_llm.py --load_from ./minimind-3 --open_thinking 1
```

### 阶段 1：读懂模型与数据代码

**先读模型**（`model/model_minimind.py`，只有 ~287 行，从上往下读）：
- `MiniMindConfig`：所有超参
- `RMSNorm` → `precompute_freqs_cis` / `apply_rotary_pos_emb`（RoPE）→ `repeat_kv`（GQA）
- `Attention.forward`：QKV 投影 → RoPE → KV cache → 缩放点积注意力
- `FeedForward`（SwiGLU）与 `MOEFeedForward`（门控 + top-k + aux_loss）
- `MiniMindBlock` → `MiniMindModel` → `MiniMindForCausalLM`（含 `generate`）

**再读数据**（`dataset/lm_dataset.py`），重点理解每个阶段“喂进去的张量长什么样”：

| Dataset 类 | 用于 | 关键点 |
|-----------|------|--------|
| `PretrainDataset` | 预训练 | 纯文本 → `input_ids`，label 为错位一位的自身 |
| `SFTDataset` | SFT | 套用 chat_template；`generate_labels` 只对 assistant 回复计算 loss（loss_mask） |
| `DPODataset` | DPO | 同时返回 chosen / rejected 及各自 loss_mask |
| `RLAIFDataset` | PPO/GRPO | 只保留 prompt，回复留空供策略模型实时采样；`thinking_ratio` 控制是否注入 `<think>` |
| `AgentRLDataset` | Agentic RL | 解析多轮工具调用轨迹 |

> 关键收获：SFT 阶段**只在 assistant 输出的位置算 loss**（用户输入不参与），这个 loss_mask 机制是理解指令微调的核心。

### 阶段 2：预训练 Pretrain（必做）

- **目标**：让模型“学会高质量词语接龙”，把语言规律和基础知识压进参数。
- **原理**：自回归的 next-token prediction，无监督（交叉熵损失）。
- **训练循环模板**（`train_pretrain.py::train_epoch`，最值得精读）：
  余弦学习率 → `autocast` 混合精度前向 → `loss = res.loss + res.aux_loss` → 梯度累积 → 梯度裁剪 → `scaler` 更新 → 定期保存/断点。

```bash
cd trainer

# 单卡
python train_pretrain.py
# 或用 torchrun（单卡写 1，多卡写 N）
torchrun --nproc_per_node 1 train_pretrain.py

# 常用自定义：开 wandb、指定数据、调长度
python train_pretrain.py --use_wandb --data_path ../dataset/pretrain/pretrain_t2t_mini.jsonl --max_seq_len 768
```

- **产出**：`test/out/pretrain_768.pth`
- **验证**：
```bash
cd ..            # 回根目录
python eval_llm.py --weight pretrain
# 问“为什么天空是蓝色的”——此时模型只会“续写”，还不太会“对话”，属正常
```

关键默认参数：`epochs=2, batch_size=32, lr=5e-4, accumulation_steps=8, max_seq_len=340, from_weight=none`。

### 阶段 3：监督微调 SFT（必做）

- **目标**：让模型适应多轮对话模板（user/assistant/system/tool 角色）、指令跟随，并具备基础工具调用与思考能力（tool call 样本已混入 SFT 数据）。
- **原理**：仍是交叉熵，但只对 assistant 回复位置计算 loss。
- **依赖**：默认 `from_weight=pretrain`，即基于上一步的 `pretrain_768.pth` 继续训。

```bash
cd trainer
python train_full_sft.py
# 或
torchrun --nproc_per_node 1 train_full_sft.py
```

- **产出**：`test/out/full_sft_768.pth` —— 这就是能对话的 **MiniMind Zero**。
- **验证**：
```bash
cd ..
python eval_llm.py --weight full_sft
# 此时应能像样地回答“解释什么是机器学习”“推荐中国美食”等
```

关键默认参数：`epochs=2, batch_size=16, lr=1e-5, max_seq_len=768, from_weight=pretrain`。

> 🎉 走到这里，你已经**从 0 训出了一个会聊天的 LLM**。强烈建议先把阶段 0~4 完整跑通，再进入下面的可选进阶。

### 阶段 4：评测

仓库内置两种“体感评测”：

```bash
# 1. 交互式对话评测（主观感受）
python eval_llm.py --weight full_sft
python eval_llm.py --weight full_sft --temperature 0.85 --top_p 0.95 --historys 2   # 带 2 轮历史

# 2. 工具调用评测（scripts/ 目录）
cd scripts
python eval_toolcall.py --weight full_sft
```

> 标准化 benchmark（C-Eval / C-MMLU / OpenBookQA 等）不在仓库内置脚本中，需要先用 `scripts/convert_model.py` 把 `.pth` 转成 transformers 格式，再接入第三方评测框架（如 OpenCompass / lm-evaluation-harness）。README「评估」一节给出的是主观问答与轻 Agent 任务的横向对比示例。

### 阶段 5：LoRA（参数高效微调）

- **目标**：在冻结主体权重的前提下，只训练低秩增量分支，低成本适配垂直领域或改自我认知。
- **原理**：`W + BA`，只训 `B`、`A`。实现见 `model/model_lora.py`（从 0 手写，不依赖 peft）。
- **依赖**：默认 `from_weight=full_sft`。

```bash
cd trainer
# 医疗垂域（默认数据 lora_medical.jsonl）
python train_lora.py --lora_name lora_medical --data_path ../dataset/sft/lora_medical.jsonl
# 自我认知
python train_lora.py --lora_name lora_identity --data_path ../dataset/lora_identity.jsonl
```

- **产出**：`test/out/lora_medical_768.pth`（LoRA 分支权重，很小）
- **验证**（`--weight` 指基座，`--lora_weight` 指 LoRA 分支）：
```bash
cd ..
python eval_llm.py --weight full_sft --lora_weight lora_medical
python eval_llm.py --weight full_sft --lora_weight lora_identity
```
- **合并导出**（把 LoRA 合回基座成完整权重）：
```bash
cd scripts && python convert_model.py   # 使用其中的 convert_merge_base_lora
```

关键默认参数：`epochs=10, batch_size=32, lr=1e-4, max_seq_len=340, from_weight=full_sft`。LoRA 在 CPU 上也能较轻快地训。

### 阶段 6：DPO（直接偏好优化 / RLHF）

- **目标**：让模型更符合人类偏好——鼓励 chosen 回复、抑制 rejected 回复。
- **原理**：DPO 从带 KL 约束的 PPO 目标解析推导而来，无需单独训练 Reward/Value 模型，只跑 actor + ref 两个模型。数据是静态偏好对（off-policy）。
- **依赖**：默认 `from_weight=full_sft`。**学习率要极小**（默认 `4e-8`）以防遗忘。

```bash
cd trainer
python train_dpo.py
# 或 torchrun --nproc_per_node 1 train_dpo.py
```

- **产出**：`test/out/dpo_768.pth`
- 关键默认参数：`epochs=1, batch_size=4, lr=4e-8, beta=0.15, max_seq_len=1024, data=dpo.jsonl`。

### 阶段 7：知识蒸馏（白盒）

- **目标**：让小学生模型学习大教师模型的 token 分布（不只学答案，还学“倾向”）。
- **原理**：`Loss = α·CE + (1-α)·T²·KL(teacher || student)`。
- **注意默认**：默认 **教师是 MoE、学生是 Dense**（`teacher_use_moe=1, student_use_moe=0`），且两者都从 `full_sft` 权重出发。所以你需要事先准备好 `full_sft_768_moe.pth`（教师）和 `full_sft_768.pth`（学生）。

```bash
cd trainer
python train_distillation.py
# 若想改成 Dense 教 Dense：
python train_distillation.py --teacher_use_moe 0 --temperature 1.5 --alpha 0.5
```

- **产出**：`test/out/full_dist_768.pth`
- 关键默认参数：`epochs=6, batch_size=32, lr=5e-6, alpha=0.5, temperature=1.5`。

### 阶段 8：RLAIF —— PPO / GRPO / CISPO

- **目标**：SFT 之后，用**非人工、可自动获得的反馈信号**（奖励模型 / 规则 / GT 校验 / 环境）继续强化学习（on-policy）。
- **统一视角**（README 精华）：所有 PO 算法都在优化 `J = E[Φ(r_t, A_t) - h(KL_t)]`，区别只在于**策略项 Φ、优势项 A、正则项 h** 的不同实例化。

**准备工作（必须）**：下载奖励模型 `InternLM2-1.8B-Reward`，放到 **minimind 的同级目录**：

```
root/
├── minimind/
└── internlm2-1_8b-reward/     # 与 minimind 平级；脚本默认 ../../internlm2-1_8b-reward
```

```bash
cd trainer

# PPO（需 Actor + Critic + Reward + Ref）
python train_ppo.py
# 或多卡：torchrun --nproc_per_node N train_ppo.py

# GRPO（组内相对优势，省掉 Critic）
#   注意：train_grpo.py 的 loss_type 默认是 cispo！要跑“纯 GRPO”需显式指定：
python train_grpo.py --loss_type grpo
# CISPO（GRPO 的 loss 变体，默认值，直接跑即可）
python train_grpo.py --loss_type cispo
```

- **产出**：`test/out/ppo_actor_768.pth`、`test/out/grpo_768.pth`
- **重要提醒（奖励稀疏）**：MiniMind 只有 0.1B 级别，若用超纲难题 + 二元 0/1 奖励，候选回答几乎全错 → 优势≈0 → 梯度消失。因此项目采用 **model-based 连续奖励**（如 -2.5~+3.0），并建议监控奖励方差 `Var(r)`，接近 0 说明学不动。
- 关键默认参数：`epochs=1, batch_size=2, lr=3e-7, num_generations=6, beta=0.1, data=rlaif.jsonl, from_weight=full_sft`。

### 阶段 9：Agentic RL（多轮工具调用）

- **目标**：在多轮 Tool-Use 场景下做 GRPO/CISPO，奖励是**整轮交互后延迟结算**（如数学题最终答对与否）。
- **亮点**：`rollout_engine.py` 实现了**训推分离**——训练侧更新 policy，rollout 侧高吞吐采样（可选 `torch` 本地或 `sglang` 远端），二者通过轨迹与权重同步衔接（当前为同步模式）。

```bash
cd trainer

# ① 本地 torch rollout（默认）
python train_agent.py --data_path ../dataset/rl/agent_rl.jsonl

# ② 用 sglang 做 rollout（先另起一个服务）
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998
python train_agent.py --rollout_engine sglang --sglang_base_url http://localhost:8998 \
    --sglang_shared_path ./ckpt_mm --data_path ../dataset/rl/agent_rl_math.jsonl --use_wandb
```

- **产出**：`test/out/agent_768.pth`
- **验证**（工具调用能力，agent 应显著优于 full_sft）：
```bash
cd ../scripts && python eval_toolcall.py --weight agent
```
- 关键默认参数：`batch_size=2, lr=3e-7, num_generations=4, loss_type=cispo, data=agent_rl.jsonl`。

> README 实测：在轻量数学 ToolUse 任务上，`agent` 权重 17/20 (85%) vs `full_sft` 12/20 (60%)。但要注意“对齐税”——RL 后工具能力变强，通用问答的事实性可能下降。

### 阶段 10：部署与推理

```bash
# 1. Web 控制台（项目自带，推荐）—— 对话 / 实验台 / 训练控制台三页，直接从 test/ 读产物
python webui/server.py                 # 默认 http://localhost:7860
#   也可以指定配置与权重：--config configs/sft_moe.yaml --weight test/out/xxx.pth --device cuda:0
#   逐页说明见 webui/README.md；本节下面另附「训练指标与可视化」

# 2. OpenAI 兼容 API 服务端
cd scripts && python serve_openai_api.py --load_from ../minimind-3
#   再用 chat_api.py 或任意 OpenAI SDK 调用；支持 reasoning_content / tool_calls / open_thinking

# 3. 旧版 Streamlit 聊天页（仅读 transformers 格式目录）—— 需先把模型拷进 scripts/
cp -r minimind-3 ./scripts/minimind-3
cd scripts && streamlit run web_demo.py

# 4. 第三方推理引擎
ollama run jingyaogong/minimind-3
vllm serve /path/to/model --served-model-name "minimind"
```

OpenAI SDK 开启思考的方式：

```python
response = client.chat.completions.create(
    model="minimind",
    messages=[{"role": "user", "content": "你是谁？"}],
    extra_body={"chat_template_kwargs": {"open_thinking": True}}
)
```

### 附：训练指标与可视化（`/lab` 实验台）

「训练完只看 loss 降没降」是不够的 —— 这个项目在 `trainer/common/metrics.py` 里按四类采集指标，
每一类都在 `/lab` 上有确定落点（`python webui/server.py` 后打开 `/lab`）：

| 家族 | 采集内容 | 诊断什么 | `/lab` 页 |
|------|----------|----------|-----------|
| 优化动力学 | 按 attn / ffn / embed / norm 分组的梯度范数、权重范数、`update_ratio_*` | 哪一部分在学、是否停滞 | `#pretrain`、`#sft` |
| MoE 健康度 | 专家负载 / `load_cv` / `moe_dead_experts` / 路由熵 / 均衡偏置 / `shared_share` | **专家被饿死不会体现在 loss 上** | `#pretrain`「专家负载热力图」 |
| 泛化 | 留出集 `val_loss` | 区分「还在学」与「开始过拟合」 | `#sft`、`#runs` |
| 系统 | GPU 利用率 / 显存峰值 / 进程内存 | 卡是否真在跑 | `#resources`、`#pretrain` |
| 组件中间量 | 逐层 `q/k/v/out_rms`、`gate_mean/std/sat_*`、`hidden_rms` | 门控是否饱和、残差流是否逐层放大 | `#pretrain`「逐层激活量热力图」 |
| 算法专属 | PPO 的 `critic_loss` / `approx_kl`、Agent 的 `pass_rate` / `turns_mean`、DPO 的 `reward_margin` … | 每个算法各看各的 | `#rl` 的专属卡 |

「不同算法记录不同 log」是**有意为之**：各算法 `format_log()` 打出的字段名互不相同
（GRPO 打 `KL_ref`、Agent 打 `KL`、PPO 打 `Critic Loss`），逐 step 落盘的 CSV 列也按 `--algo` 裁剪。
逐字段对照见 [`trainer/README.md`](./trainer/README.md) 的「各算法记录什么」，完整
「指标 → 页面」映射表见 [`webui/README.md`](./webui/README.md)。

另外，`/train` 训练控制台能把**正在跑的那个训练**的 stdout 实时解析成曲线（loss / 奖励 / 梯度范数 /
学习率 / 吞吐 / 显存 / 各算法专属列），不用等它跑完再出报告；`/` 对话页则回答
「当前加载的这份权重是哪次训练产出的、什么架构」。

---

## 八、命令速查表

> 训练脚本均在 `trainer/` 下执行；`eval_llm.py` 在根目录。多卡把 `torchrun --nproc_per_node N` 的 `N` 改成卡数。

| 阶段 | 最简命令 | 产出权重 | 依赖 |
|------|----------|----------|------|
| 预训练 | `python train_pretrain.py` | `pretrain_768.pth` | 无 |
| SFT | `python train_full_sft.py` | `full_sft_768.pth` | pretrain |
| LoRA | `python train_lora.py --lora_name lora_medical` | `lora_medical_768.pth` | full_sft |
| DPO | `python train_dpo.py` | `dpo_768.pth` | full_sft |
| 蒸馏 | `python train_distillation.py` | `full_dist_768.pth` | full_sft(+moe) |
| PPO | `python train_ppo.py` | `ppo_actor_768.pth` | full_sft + reward model |
| GRPO | `python train_grpo.py --loss_type grpo` | `grpo_768.pth` | full_sft + reward model |
| CISPO | `python train_grpo.py --loss_type cispo` | `grpo_768.pth` | full_sft + reward model |
| Agentic RL | `python train_agent.py` | `agent_768.pth` | full_sft + reward model |

**通用开关（几乎所有训练脚本都支持）**：

```bash
--use_moe 1                 # 训练 MoE 版本（权重自动加 _moe 后缀）
--use_wandb                 # 开启训练可视化
--from_resume 1             # 断点续训（自动读 ./test/checkpoints/*_resume.pth）
--from_weight <name>        # 指定基座权重（none 表示从头）
--epochs / --batch_size / --learning_rate / --max_seq_len   # 常调超参
--device cuda:0 / cpu       # 设备
torchrun --nproc_per_node N train_xxx.py                    # 多卡 DDP
```

**推理/评测命令**：

```bash
python eval_llm.py --weight pretrain                          # 测预训练
python eval_llm.py --weight full_sft                          # 测 SFT
python eval_llm.py --weight full_sft --lora_weight lora_medical  # 测 LoRA
python eval_llm.py --load_from ./minimind-3 --open_thinking 1 # 测 transformers 格式 + 思考
python eval_llm.py --weight full_sft --use_moe 1              # 测 MoE
cd scripts && python eval_toolcall.py --weight agent          # 测工具调用
```

---

## 九、关键参数速查

| 参数 | 说明 | 备注 |
|------|------|------|
| `hidden_size` / `num_hidden_layers` | 模型维度/层数 | 默认 768 / 8；改了要在推理时保持一致 |
| `use_moe` | 是否 MoE | 0=Dense，1=MoE |
| `max_seq_len` | 训练截断长度（**token 数，非字符数**） | 中文≈1.5~1.7 字/token；mini 数据建议 ~768，完整版 ~380 |
| `accumulation_steps` | 梯度累积 | 等效放大 batch，显存不够时调大 |
| `from_weight` | 基座权重前缀 | `none`=从头；否则读 `test/out/<from_weight>_<dim>.pth` |
| `from_resume` | 断点续训 | 1 时自动恢复模型/优化器/step/wandb |
| `beta` (DPO/GRPO) | KL/偏好强度 | DPO 默认 0.15，GRPO 默认 0.1 |
| `loss_type` (GRPO) | `grpo` 或 `cispo` | **默认 cispo**，跑纯 GRPO 需显式指定 |
| `num_generations` | 每个 prompt 采样数 | GRPO=6，Agent=4 |
| `thinking_ratio` | 训练时注入 `<think>` 的概率 | 控制“自适应思考”的养成 |
| `open_thinking` (推理) | 是否显式输出思考 | 0=直答，1=先 `<think>` 再答 |
| `inference_rope_scaling` | 推理时 RoPE 外推(YaRN) | 免训练扩展上下文 |

---

## 十、常见问题与建议

1. **显存不足？** 调小 `--batch_size`、调大 `--accumulation_steps`、缩短 `--max_seq_len`；或先只跑 mini 数据集。
2. **RL 学不动 / 奖励几乎为 0？** 这是奖励稀疏——换用连续奖励、降低任务难度、监控 `Var(r)`（见阶段 8）。
3. **推理结果乱码 / 维度报错？** 检查 `eval_llm.py` 的 `--hidden_size/--num_hidden_layers/--use_moe` 是否与训练时一致。
4. **WebUI 报“找不到模型”？** 旧版 Streamlit 页必须先把 transformers 格式模型文件夹拷进 `scripts/` 目录
   （脚本只扫描该目录）；`webui/` 控制台不需要拷贝，它直接按架构组装原生 `.pth` 权重，
   默认扫 `test/out/*.pth` 与 `configs/*.yaml` 配对。
5. **该重训 tokenizer 吗？** 一般**不建议**——重训会破坏与已有权重、数据格式、社区生态的兼容性。`train_tokenizer.py` 仅作示例。
6. **想做垂域模型？** 数据少用 LoRA（阶段 5），数据多可 full_sft，但要混合通用数据防过拟合。
7. **学习顺序建议**：一定先把 **阶段 0→4（推理→模型/数据→预训练→SFT→评测）** 完整跑通再碰 RL；RL 部分先理解 DPO，再按 PPO→GRPO→CISPO→Agent 的顺序推进，配合 README「PO 算法统一视角」一节食用。

---

## 十一、进一步阅读

- 根目录 `README.md`：最权威完整，尤其「模型配置」「RLAIF」「PO 算法统一视角」「评估」几节。
- `model/model_minimind.py`：模型全部实现，反复读。
- `trainer/train_grpo.py` / `train_ppo.py` / `train_agent.py`：对照 README 的损失公式读，理解三类算法差异。
- 官方拓展：MiniMind-V（视觉）、MiniMind-O（多模态 Omni）、MiniMind-dLM（扩散语言模型）、MiniMind-Linear（线性注意力），见项目 Discussions。

> 记住那句 slogan——**“大道至简”**。祝你从 0 拼出属于自己的“那架飞机” ✈️
