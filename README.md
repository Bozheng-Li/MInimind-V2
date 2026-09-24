<div align="center">

![logo](./images/logo.png)

# MiniMind-V2

**可插拔架构 × 可插拔算法** —— 把「换个注意力机制」变成改几行 YAML，
把「换个训练算法」变成换一个 `--algo` 参数。

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab)](https://www.python.org)

</div>

---

## 这是什么

MiniMind-V2 把「从 0 训一个 64M 小模型」这条链路做成了**可插拔的架构 + 可插拔的算法**。
它接手的是一份**写死的架构 + 一摞互相复制的训练脚本**：

- 想试 MLA 注意力？得去改 `model_minimind.py`。
- 想试 DAPO？得复制一份 `train_grpo.py` 再改。
- 修一个训练循环的 bug？八个 `train_*.py` 里各改一遍。

V2 把这两件事都抽成了参数：

```
模型结构  →  arch/      四个槽位 × 18 个已注册组件，YAML 驱动组装
训练算法  →  trainer/algos/   16 个算法单份实现，--algo 切换
```

换来的是**可对比性**：18 组架构扫描、SFT 双臂对比、RL 四算法 × 两基座 ——
所有对比组共用同一套 forward/backward，差异才干净地归因到组件本身，
而不是「这组比那组多了一个我没注意到的实现分歧」。

---

## 目录

- [一、30 秒上手](#一30-秒上手)
- [二、可插拔架构（`arch/`）](#二可插拔架构arch)
- [三、可插拔算法（`trainer/`）](#三可插拔算法trainer)
- [四、配置系统（`configs/`）](#四配置系统configs)
- [五、数据](#五数据)
- [六、Web 控制台（`webui/`）](#六web-控制台webui)
- [七、实验：证据与结论](#七实验证据与结论)
- [八、验证与回归](#八验证与回归)
- [九、部署](#九部署)
- [十、目录结构](#十目录结构)
- [十一、常见坑](#十一常见坑)
- [引用](#引用)
- [致谢 / License](#致谢)

---

## 一、30 秒上手

```bash
git clone https://github.com/Bozheng-Li/MInimind-V2.git
cd MInimind-V2

# 1. 装依赖（torch 单独装，见 requirements.txt 注释）
pip install -r requirements.txt

# 2. 下数据（1.2GB mini 预训练集 + 1.6GB mini SFT 集，够跑通全流程）
#    下载链接与文件清单见 dataset/dataset.md
#    ModelScope: https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
#    HuggingFace: https://huggingface.co/datasets/jingyaogong/minimind_dataset
#    放到 ./dataset/pretrain/ 与 ./dataset/sft/ 下

# 3. 预训练
cd trainer && python train.py --algo pretrain --config configs/pretrain.yaml

# 4. 监督微调
python train.py --algo sft --config configs/sft.yaml

# 5. 对话（架构感知的推理入口）
python eval.py --config configs/sft.yaml --weight full_sft
```

> 第 5 步用的 `trainer/eval.py` 是**新增**的架构感知推理入口。它与 `eval_llm.py`
> 并存，后者是历史脚本，按 `--hidden_size / --num_hidden_layers / --use_moe`
> 三个旋钮构造模型 —— `--use_moe 1` 合成出来的是**普通 top-k MoE**，
> 加载 `gated` / `moe_finegrained` / `mla` 会构建出错误的模型。详见[第九节](#九部署)。

> 第 5 步用的 `trainer/eval.py` 是**新增**的架构感知推理入口。它与 `eval_llm.py`
> 并存，后者是历史脚本，按 `--hidden_size / --num_hidden_layers / --use_moe`
> 三个旋钮构造模型 —— `--use_moe 1` 合成出来的是**普通 top-k MoE**，
> 加载 `gated` / `moe_finegrained` / `mla` 会构建出错误的模型。详见[第九节](#九部署)。

想换成 MoE + 门控注意力的现代架构、并在此基础上做 SFT：

```bash
python train.py --algo pretrain --config configs/pretrain_moe.yaml   # gated + moe_finegrained
python train.py --algo sft      --config configs/sft_moe.yaml        # 结构必须与上面一致
```

**训练产物落在 `test/` 下**（权重、日志、指标 CSV），仓库根只放纯净代码。
想让产物回到仓库根：

```bash
export MINIMIND_ARTIFACT_ROOT=/home/me/minimind
```

---

## 二、可插拔架构（`arch/`）

模型被拆成**四个槽位**，每个槽位都是「`type` + 该组件需要的任意参数」的扁平结构。
新增组件不需要改框架代码 —— 写个文件、加个 `@register` 装饰器、在 YAML 里写名字即可。

```
arch/
├── registry.py       组件注册表（短名 / 完整 import 路径两种定位方式）
├── schema.py         YAML 解析、校验、默认值推导
├── build.py          组件构造与组装
├── block.py          可组合 Transformer Block
├── model.py          ArchModel / ArchForCausalLM
├── norm/             rmsnorm · rmsnorm_zero_centered · layernorm
├── positional/       rope · partial_rope · nope
├── attention/        gqa · gated · sliding_window · mla · compressed · deltanet
└── ffn/              swiglu · geglu · moe · moe_shared · moe_finegrained
```

### 组件清单

| 槽位 | `type` | 来源 | 核心机制 |
|------|--------|------|---------|
| **attention** | `gqa` | Llama / Qwen | 分组查询 + QK-Norm。`num_key_value_heads` = `num_attention_heads` 即退化为 MHA，设为 `1` 即 MQA，无需单独实现 |
| | `gated` | Qwen3-Next | 输出门（sigmoid 门控）+ zero-centered QK-Norm + 支持 partial RoPE |
| | `sliding_window` | Mistral / Gemma | 局部窗口：每个 token 只看前 `window_size` 个 |
| | `mla` | DeepSeek-V2/V3 | KV 低秩压缩到 latent + **解耦 RoPE**，缓存 224B/token（GQA 是 768B，省 3.4×） |
| | `compressed` | DeepSeek-V4 (HCA 思路) | 沿序列维压缩 KV（每 `compress_rate` 个 token 压成 1 项），384B/token |
| | `deltanet` | Qwen3-Next / Kimi Linear | 门控线性注意力，**定长循环状态**替代 KV cache；训练走 chunked 并行 |
| **feedforward** | `swiglu` | Llama | 经典门控 FFN |
| | `geglu` | Gemma | 门控激活换成 GELU |
| | `moe` | 通用 | top-k 路由 + aux loss 负载均衡 |
| | `moe_shared` | DeepSeekMoE | 常驻共享专家 + N 个路由专家 |
| | `moe_finegrained` | DeepSeek-V3 | 细粒度专家 + **aux-loss-free**（可学习 bias 均衡，替代 aux loss） |
| **positional_encoding** | `rope` | 主流 | 旋转位置编码，支持 YaRN 长文本外推（`inference_rope_scaling`） |
| | `partial_rope` | — | 只对前若干维施加旋转 |
| | `nope` | — | 完全无位置编码（消融用） |
| **norm** | `rmsnorm` | Llama | 标准 RMSNorm |
| | `rmsnorm_zero_centered` | Qwen3-Next | 零中心化变体（权重初始为 0，等价于恒等映射起步） |
| | `layernorm` | — | 经典 LayerNorm（消融用） |

### 跨槽依赖自动处理

组件之间不是孤立拼装 —— 框架处理了两处耦合，这是「插拔」能真正成立的前提：

- **注意力 → 位置编码**：位置编码需要知道**旋转维数**，而它不一定是 `head_dim`。
  MLA 只对解耦的那部分维度施加 RoPE（`qk_rope_head_dim`，通常远小于 head_dim）。
  所以注意力组件可以声明 `positional_dim(cfg) -> int`，不声明时退回 `head_dim`。
  见 `arch/build.py::build_positional`。
- **MoE 专家 → 组件自身**：`moe_finegrained` 用 `cfg.override(...)` 派生出窄专家配置，
  所以专家宽度、数量、均衡策略全是槽内参数，不侵入全局 `model` 段。

### 外部组件：写 import 路径即可

`type` 除了短名，还能写**完整 import 路径**：

```yaml
attention:
  type: my_ext.attention.FlashAttention   # 仓库外的实现，无需改框架代码
```

注册表先查短名，查不到就按点号回退到 `importlib`，找不到时抛出的错误会列出该槽位
所有已注册的短名。

---

## 三、可插拔算法（`trainer/`）

历史上有八个 `train_*.py`，各自复制了一套模型、数据、优化器与 RL 公式。
**修一个 bug 要改八处**，也容易出现「统一入口正确、旧脚本仍训练错」的分叉。

现在所有算法只有 `trainer/algos/` 下的**一份实现**；旧脚本退化成兼容壳，
只补上固定的 `--algo` 就转发给统一入口。

```
trainer/
├── train.py              ★ 统一训练入口（换架构 × 换算法）
├── train_*.py            旧命令兼容壳，全部转发
│
├── common/               通用脚手架（算法的「怎么跑」）
│   ├── pipeline.py       run_training：九段式启动流程
│   ├── runner.py         通用训练循环 + TrainContext
│   ├── metrics.py        富指标采集（优化动力学 / MoE 健康度 / 系统）
│   ├── runtime.py        DDP / 种子 / 精度 / compile 包装
│   ├── data.py           DataLoader / DistributedSampler
│   ├── checkpoint.py     存盘与断点续训
│   ├── registry.py       每次训练的结构化 run 登记（meta.json）
│   └── logging.py        Logger + wandb / swanlab
│
├── algos/                算法模块（算法的「优化什么」）
│   ├── base.py           Algorithm 协议
│   ├── pretrain/         预训练（Causal LM）
│   ├── sft/              Full SFT / LoRA / QLoRA / 白盒蒸馏
│   └── rl/
│       ├── preference.py DPO / IPO / SimPO / CPO / ORPO / KTO
│       ├── grpo.py       GRPO / CISPO 公共在线策略目标
│       ├── dapo.py       Clip-Higher + 动态采样 + token-level loss
│       ├── rloo.py       leave-one-out baseline
│       ├── ppo.py        Actor-Critic + GAE
│       └── agent.py      多轮工具调用强化学习
│
├── rollout_engine.py     Torch / SGLang 采样与精确 completion mask
└── eval.py               架构感知的推理入口
```

### 16 个算法

| 类别 | 算法 | 需要 ref / reward / critic | 典型用途 |
|------|------|---------------------------|----------|
| 预训练 | `pretrain` | 无 | 原始文本 next-token 训练 |
| 监督微调 | `sft` | 无 | 全参数指令微调 |
| 参数高效 SFT | `lora` / `qlora` | 无；QLoRA 需 bitsandbytes + CUDA | 小显存垂域微调 |
| 白盒蒸馏 | `distill` | teacher | 分布蒸馏 |
| 离线偏好 | `dpo` / `ipo` / `kto` | reference | 有 chosen/rejected 数据的稳健对齐 |
| Reference-free 偏好 | `simpo` / `cpo` / `orpo` | 无 | 少占一份 reference 显存 |
| 在线无 Critic RL | `grpo` / `rloo` | reward + reference | 可验证奖励任务 |
| 在线无 Critic RL | `dapo` | reward；reference 仅在 KL 系数非零时需要 | 长回答推理与可验证奖励任务 |
| 在线 Actor-Critic | `ppo` | reward + reference + critic | 通用 RLHF |
| Agentic RL | `agent` | reward + reference + 工具环境 | 多轮工具调用 |

**几个实现要点**：

- **CISPO 是默认的在线策略目标**（`--loss_type cispo`）。相比 GRPO，它用
  detach 后的 ratio 做上侧裁剪，避免梯度被裁剪项反复抹掉。
- **DAPO** 实现了 Clip-Higher、动态重采样、token 级归一化与超长惩罚
  （`--dapo_epsilon_high` 是相对 1 的上侧裁剪宽度，不是 ratio 绝对值）。
- **IPO / CPO 的长度口径不同**：IPO 使用每 token 平均后的策略/reference 间隔，
  CPO 的 sigmoid 偏好项使用序列总 log-prob；SimPO 才是 reference-free 的长度归一化目标。
- **QLoRA** 默认 NF4 + double quant + paged AdamW 8-bit。
- **Agentic RL 的工具环境与评测同源**：评测里的 `pass_rate` 与训练控制台打的
  是同一个函数 `trainer/algos/rl/agent_tools.py::calculate_rewards`。

### 加一个新算法

1. 在 `algos/` 下新建模块，继承 `Algorithm`
2. 实现 `build_dataset()` 与 `compute_loss(batch) -> (loss, aux_loss)`
   （`loss` 必须已除以 `accumulation_steps`）
3. 需要额外模型（ref / teacher / critic / reward）就实现 `extra_models()`
4. 在 `algos/__init__.py` 的 `_ALGO_MODULES` 里登记，即可 `--algo <名字>` 调用

可选钩子（都有默认实现）：`configure_model` / `build_optimizer` / `clip_parameters` /
`weight_prefix` / `save_weights` / `checkpoint_extra` / `format_log` / `train_epoch`。

### 统一入口

```bash
# 换算法：同一份配置，不同 --algo
python trainer/train.py --algo sft  --config configs/base.yaml
python trainer/train.py --algo dpo  --config configs/base.yaml
python trainer/train.py --algo grpo --config configs/base.yaml

# 换架构：同一算法，不同配置
python trainer/train.py --algo sft --config configs/sft.yaml
python trainer/train.py --algo sft --config configs/sft_moe.yaml

# 不带 --config 时按算法自动选阶段配置
#   pretrain                      -> configs/pretrain.yaml
#   sft / lora / qlora / distill  -> configs/sft.yaml
#   偏好优化与在线 RL              -> configs/rl.yaml

# 命令行参数永远优先于配置
python trainer/train.py --algo pretrain --epochs 3 --batch_size 16

# 多卡
torchrun --nproc_per_node 4 trainer/train.py --algo grpo
```

---

## 四、配置系统（`configs/`）

```
configs/
├── base.yaml              默认模型结构（被各阶段 defaults 引用）
├── pretrain.yaml          预训练 = base + 数据 + 超参
├── sft.yaml               监督微调（含 algo.lora / algo.qlora / algo.distill 段）
├── rl.yaml                偏好与强化（含 algo.<每个算法> 段）
├── pretrain_moe.yaml      MoE 主线预训练配置（选型依据写在注释里）
├── pretrain_deepseek.yaml MLA + DeepSeekMoE 完整配置（204M / 激活 69.5M）
├── pretrain_moe_mini.yaml 4 小时快速版
├── sft_moe.yaml / sft_moe_full.yaml / sft_dense.yaml    SFT 双臂
├── rl_moe.yaml / sft_moe.yaml
└── loader.py              加载与合并
```

### `--config` 是「叠加变体」，不是整体替换

合成顺序：**阶段配置作基线，`--config` 指定的文件叠加其上**。
因此一份「只换了注意力」的实验配置可以只有几行：

```yaml
# configs/mla.yaml
attention:
  type: mla
  num_attention_heads: 8
  num_key_value_heads: 4
```

数据路径、序列长度、学习率全部自动沿用阶段配置。

### 覆盖规则

**配置只提供默认值，命令行永远优先。**

| 情况 | 结果 |
|------|------|
| 配置写 `epochs: 2`，不传参数 | 用 2 |
| 配置写 `epochs: 2`，传 `--epochs 3` | 用 3 |
| 配置写 `hidden_size: 768`，传 `--hidden_size 512` | 用 512 |

模型结构的三个旧旋钮（`--hidden_size` / `--num_hidden_layers` / `--use_moe`）默认值会
从配置同步，所以它们不会以硬编码默认值压过配置。

> ⚠️ **只把稠密前馈强切成 MoE**：`--use_moe 1` 仅在配置本身是稠密前馈时才生效。
> 若无条件覆写，`moe_shared` / `moe_finegrained` 会被降级成普通 top-k MoE，
> 共享专家与 aux-loss-free 全部丢失，而外面看不出来（参数量只差共享专家那一份）。
> 见 `configs/loader.py::build_lm_config`。

### 两个根目录

| 变量 | 默认值 | 管什么 |
|------|--------|--------|
| `REPO_ROOT` | `configs/` 的上一级 | `data.path`、`configs/` 自身 |
| `ARTIFACT_ROOT` | `<repo>/test` | `train.save_dir`（权重 / 指标 CSV / 续训档） |

配置里的相对路径一律**相对仓库根**解析，与从哪个目录启动脚本无关：
`data.path: dataset/pretrain/x.jsonl` → `<repo>/dataset/pretrain/x.jsonl`。

---

## 五、数据

按**训练阶段**分目录存放（`dataset/dataset.md` 有完整说明）：

```bash
dataset/
├── pretrain/
│   ├── pretrain_t2t.jsonl        (10GB,   847 万条, 约 3.2B token)
│   └── pretrain_t2t_mini.jsonl   (1.2GB,  127 万条, 快速复现)
├── sft/
│   ├── sft_t2t.jsonl             (14GB,   511 万条, 主线，已混入 Tool Call 样本)
│   └── sft_t2t_mini.jsonl        (1.6GB,  90.6 万条, 快速复现)
├── rl/
│   ├── dpo.jsonl                 (53MB,  1.7 万条, 离线偏好优化)
│   ├── rlaif.jsonl               (24MB,  1.95 万条, PPO / GRPO / CISPO)
│   ├── agent_rl.jsonl            (86MB,  4.0 万条, Agentic 多轮工具调用)
│   └── agent_rl_math.jsonl       (18MB,  2.0 万条, 数学 RLVR + 域外留出集)
└── external/                     外部扩展数据集（候选池，与主线分开）
```

下载：[ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) |
[HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)。
无需全部 clone，单独下所需文件即可。

**预训练支持配比混合**（`data.mix`），这是做架构对比实验的基础设施：

```yaml
data:
  path: dataset/pretrain
  mix:
    pretrain_t2t.prose.jsonl: 0.70    # 叙述 / 百科
    pretrain_t2t.qa.jsonl:    0.10    # 指令 / 问答
    pretrain_t2t.en.jsonl:    0.15    # 英文
    pretrain_t2t.code.jsonl:  0.05    # 代码
```

模型看到的比例 = 权重比，与各文件条数无关。想回到「一份混好的文件」就删掉 `mix`。

### 长度口径

`max_seq_len` 是 **token 数**，不是字符数。本项目 tokenizer 中文约
`1.5~1.7 字符/token`，纯英文约 `4~5 字符/token`。
上游文档标注的「最大长度」是字符数，`100/1.5 ≈ 67` 可作粗略换算。

---

## 六、Web 控制台（`webui/`）

一个 FastAPI 后端带三个页面，前端是**原生 HTML/CSS/JS —— 无构建、无 npm、无 CDN**，
断网机器上打开与开发机一致。

```bash
python webui/server.py                    # http://localhost:7860
python webui/server.py --no-load          # 先不加载模型，进界面再选
```

| 路径 | 页面 | 回答什么问题 |
|------|------|-------------|
| `/` | **对话** | 这个权重现在能干什么、它是哪次训练产出的 |
| `/lab` | **实验台**（10 页） | 全流程跑到哪了、各算法/架构/评测怎么比 |
| `/train` | **训练控制台** | 下一步该起哪个训练、用什么配置、正在跑的那个现在怎么样了 |

**为什么不用 Streamlit 那版**：`scripts/web_demo.py` 用
`AutoModelForCausalLM.from_pretrained` 读 transformers 格式目录，而新训的模型
（`gated` + `moe_finegrained`）是原生 torch `.pth`，必须按配置组装。
它还用了 `skip_special_tokens=True` —— 而 `<tool_call>` / `<think>` 都是
**added token，会被一起吃掉**，工具调用实际解析不出来。

WebUI 照搬 `trainer/eval.py` 的架构感知加载路径
（`load_config` → `to_arch_config` → `build_lm_config` → `arch.build_model` → `torch.load`），
因此能加载任意组合的架构。

**实验台**覆盖：MoE 门控健康、专家负载热力图、逐层激活量、奖励分解、
各算法专属指标（PPO 的 critic、Agent 的工具轮次、DPO 的 margin）与
「算法 × 指标家族覆盖矩阵」。

**数据源**：默认从 `test/` 读，`MINIMIND_ARTIFACT_ROOT` 可改。
逐页说明、接口清单与「指标 → 页面」完整映射表见 [`webui/README.md`](webui/README.md)。

---

## 七、实验：证据与结论

> 完整报告：[`test/storage/report/EXPERIMENT_REPORT.md`](test/storage/report/EXPERIMENT_REPORT.md)
> （流程 → 过程 → 结果 → 解读，所有数字都能顺索引回到原始产物）

**实验设计上三个关键决定**，后面所有结论都建立在它们之上：

1. **等 token 预算**（架构扫描）：同数据、同 `max_seq_len`、同 `batch_size`、同 seed、
   同 `max_steps`。差异**只能**归因于架构。
2. **等质量比算力**（每一阶段都算）：MoE 每步激活参数更多，「等 step 比较」是白送它算力。
3. **口径统一**：预训练在全序列算 loss，SFT 只在 assistant 段算，**这两个数永远不直接比**。

### 7.1 架构组件扫描（18 组 × 61M token）

统一口径 step = 15000，loss 取末尾 1500 步窗口均值（末段平台期批间波动 ±0.15，单点不可靠）。

| 配置 | loss@同step | vs 基线 | 吞吐 tok/s | 缓存/token | 总参数 | 激活参数 |
|------|------------|--------|-----------|-----------|--------|---------|
| `t1-01-attn-gated` | 2.5670 | **-0.0548** | 48,115 | 24,576B | 68.6M | 68.6M |
| `t3-01-combo-modern` | 2.5728 | -0.0489 | 47,883 | 24,576B | 68.6M | 68.6M |
| `t1-04-ffn-deepseekmoe` | 2.5794 | -0.0423 | 16,931 | 24,576B | 209.7M | 75.2M |
| `t1-02-ffn-geglu` | 2.6068 | -0.0150 | 52,388 | 24,576B | 63.9M | 63.9M |
| `t1-08-norm-zero-centered` | 2.6146 | -0.0071 | 51,619 | 24,576B | 63.9M | 63.9M |
| `t2-04-attn-mha` | 2.6156 | -0.0061 | 51,781 | 49,152B | 68.6M | 68.6M |
| `t1-00-baseline` | 2.6217 | +0.0000 | 59,559 | 24,576B | 63.9M | 63.9M |
| `t1-09-norm-layernorm` | 2.6238 | +0.0021 | 48,739 | 24,576B | 63.9M | 63.9M |
| `t3-00-combo-deepseek` | 2.6323 | +0.0105 | 16,970 | **7,168B** | 204.0M | 69.5M |
| `t1-05-pos-theta1e4` | 2.6418 | +0.0201 | 53,884 | 24,576B | 63.9M | 63.9M |
| `t1-03-ffn-moe-shared` | 2.6442 | +0.0225 | 28,651 | 24,576B | 243.2M | 108.8M |
| `t2-00-attn-sliding` | 2.6476 | +0.0259 | 50,520 | 24,576B | 63.9M | 63.9M |
| `t1-07-pos-partial-rope` | 2.6625 | +0.0408 | 51,729 | 24,576B | 63.9M | 63.9M |
| `t2-01-attn-mla` | 2.7585 | +0.1367 | 54,194 | 7,168B | 58.2M | 58.2M |
| `t3-02-combo-readme-moe` | 2.7901 | +0.1683 | 31,930 | 24,576B | 198.4M | 63.9M |
| `t2-03-attn-deltanet` | 2.8016 | +0.1799 | 11,496 | 定长 | 68.7M | 68.7M |
| `t2-02-attn-compressed` | 2.8623 | +0.2405 | 46,774 | **384B** | 63.9M | 63.9M |
| `t1-06-pos-nope` | 3.0250 | +0.4032 | 59,641 | 24,576B | 63.9M | 63.9M |

**等效果比算力**（达到 loss ≤ 2.6217 各需多少算力，基线恒为 1.00×）：

| 配置 | 所需 step | 算力倍数 | 结论 |
|------|----------|---------|------|
| `t1-01-attn-gated` | 10,450 | **0.84×** | 更省 |
| `t1-04-ffn-deepseekmoe` | 10,500 | **0.92×** | 更省 |
| `t1-02-ffn-geglu` | 10,600 | **0.79×** | 更省 |
| `t2-04-attn-mha` | 10,600 | **0.85×** | 更省 |
| `t3-01-combo-modern` | 10,600 | **0.85×** | 更省 |
| `t1-00-baseline` | 13,400 | 1.00× | 相当 |
| `t1-03-ffn-moe-shared` | 13,400 | **1.70×** | 更贵 |
| `t3-00-combo-deepseek` | 13,450 | 1.09× | 更贵 |
| `t1-06-pos-nope` / MLA / compressed / deltanet | — | 未达到 | 该预算内做不到 |

### 7.2 结论总表

按「敢不敢写进主线配置」排序：

| # | 结论 | 证据强度 | 是否采用 |
|---|------|---------|---------|
| 1 | **门控注意力（`gated`）是唯一稳健的正收益**：Δloss −0.055，5 点极差 0.007。代价 +7% 参数、吞吐 −19%、缓存不变 | ⭐⭐⭐ 强 | ✅ 已用于 `pretrain_moe.yaml` |
| 2 | **MoE 在 SFT 阶段等算力明确占优**：同质量 0.49×~0.71× 算力，但墙钟要付 1.76×~2.57× | ⭐⭐⭐ 强 | ✅ 已用为主线，部署时权衡墙钟 |
| 3 | **`geglu` ≈ `swiglu`，zero-centered norm 微正** | ⭐⭐ 中 | ⚠️ 均未采用 |
| 4 | **省缓存的注意力在 61M token 预算下全是负收益**（MLA +0.137、compressed +0.241、deltanet +0.180） | ⭐⭐ 中 | ❌ 未采用 |
| 5 | **位置编码不要动**：`rope_theta=1e6` 优于 `1e4`（+0.020），`nope` 灾难性（+0.403） | ⭐⭐ 中 | ✅ 保持 rope(1e6) |
| 6 | **v-norm 无定论**：train loss / val loss 反向，差幅 ~1% | ⭐ 弱 | ❌ 待补实验 |
| 7 | **后训练四算法在 64M 规模的 benchmark 上无分辨力** | ⭐⭐⭐ 强（方向是负面的） | ❌ 需换评测 |

### 7.3 门控注意力：唯一稳健的正收益

18 组等 token 扫描里，`gated` 是唯一在 5 个观察点上极差 ≤0.007 且幅值 >0.04 的组件。
等算力口径下更强：达到基线 loss 只需 10,450 步（**0.84×**），少 22% 训练量。

**解读**：输出门 + zero-centered QK-Norm 提供的是**与容量无关的归纳偏置** ——
它让模型在每个 head 内部重新加权「要听什么」，不需要更多参数。
这解释了为什么它在这个规模就见效，而 MoE（靠容量）还没到时候。

把它和 MoE 组合（`t3-01-combo-modern`，Δloss −0.0489）与单用 gated 几乎同等收益，
说明**二者没有协同也没有冲突** —— 主线可以安全地同时用。

### 7.4 MoE：等 token 吃亏，等算力/等质量占优

这是全套实验里最重要、也最容易被误读的一条。

**等 token 预算下 MoE 微正或持平**（Δloss −0.042），但吞吐只有基线的 28%；
`moe_shared` 更极端，等效果算力 **1.70×**，是全表最贵的配置。

**但拉长到 SFT 阶段，结论完全翻转**。双臂严格同 token 预算 1.39B、
同步数（28,288）、同 seed、同数据，唯一差异是架构：

| 指标 | Dense (gqa+swiglu) | MoE (gated+moe_finegrained) | 差异 |
|------|-------------------|----------------------------|------|
| 总参数 / 激活 | 63.91M / 63.91M | 214.42M / 79.94M | 3.4× / 1.25× |
| token 预算 | 1,390,411,776 | 1,390,411,776 | **相同** |
| **验证 loss（末尾 1000 步窗口）** | 0.6370 | **0.5523** | **−0.0847** |
| 验证 loss（最低） | 0.6348 @28,000 | 0.5496 @27,500 | −0.0852 |
| 训练 loss（末尾） | 1.4541 | 1.3903 | −0.0638 |
| 吞吐 | 124,945 tok/s | 27,816 tok/s | Dense 快 **4.49×** |
| 峰值显存 | 7,027 MB | 12,874 MB | +83% |
| 墙钟 | 1.55 h | 6.94 h | 4.49× |

逐点对照（Δ 全程稳定在 −0.083 ~ −0.089）：

| step | 5,000 | 10,000 | 15,000 | 20,000 | 25,000 |
|------|-------|--------|--------|--------|--------|
| Dense | 0.8082 | 0.7157 | 0.6727 | 0.6494 | 0.6414 |
| MoE | 0.7191 | 0.6296 | 0.5860 | 0.5624 | 0.5582 |
| Δ | −0.0891 | −0.0861 | −0.0867 | −0.0870 | −0.0832 |

**等质量比算力**（5 个阈值全部 MoE 更省）：

| 验证 loss 阈值 | Dense step | MoE step | MoE 算力倍数 | MoE 墙钟倍数 |
|---------------|-----------|----------|-------------|-------------|
| 0.64 | 23,000 | 9,000 | **0.49×** | 1.76× |
| 0.73 | 9,000 | 4,500 | 0.63× | 2.25× |
| 0.86 | 3,500 | 2,000 | 0.71× | 2.57× |
| 0.98 | 2,000 | 1,000 | 0.63× | 2.25× |
| 1.09 | 1,000 | 500 | 0.63× | 2.25× |

**解读**：MoE 的容量优势**需要更多 token 才能兑现**。61M token 对 209.7M 总参数
只够每个参数见 0.29 token，纯属容量浪费；1.39B token 时是 6.5 token/参数，容量开始派上用场。
主线 `pretrain_moe.yaml` 跑 3.2B token（约 15 token/参数），正是基于这个判断。

**这也解释了为什么「等 token」和「等算力」两个口径会打架**：

- 等 token 比「同等数据量下谁的 loss 低」→ 对需要更多步数的 MoE 不利。
- 等算力比「同等计算预算下谁的 loss 低」→ 对激活参数多的 MoE 更公平。
- **等质量比算力**（SFT 报告用的口径）比前两者都更能回答「我要达到某个质量，该选谁」。

### 7.5 训练内部量：MoE 均衡在收敛，V 投影在主动退场

两条 MoE 长训（预训练 132k 步 + SFT mini 28k 步 + SFT full 160k 步）的
138 列指标显示出一致的健康图景：

| 量 | 预训练 首 → 末 | SFT mini 首 → 末 | SFT full 首 → 末 | 判读 |
|----|--------------|-----------------|-----------------|------|
| `moe_load_maxmin` | 2.95 → 1.06 | 1.40 → 1.02 | 1.77 → 1.07 | ✅ 收敛中 |
| `moe_load_cv` | 0.307 → 0.015 | 0.096 → 0.006 | 0.144 → 0.017 | ✅ |
| `moe_dead_experts` | 0 → 0 | 0 → 0 | 0 → 0 | ✅ 16 专家全工作 |
| `moe_entropy_norm` | 0.9994 → 0.9993 | 0.9988 → 0.9992 | 0.9976 → 0.999 | ✅ |
| `moe_bias_std` | 0.008 → **0.025** | 0.025 → 0.044 | 0.025 → 0.040 | ⚠️ 仍在增长 |
| 全局梯度范数 | 3.72 → 0.154 | 0.70 → 0.564 | 0.69 → 0.617 | ✅ 平稳 |

- **`bias_std` 持续增长**说明 aux-loss-free 的纠偏信号还在加大，
  负载均衡尚未真正收敛 —— 不是 bug，但意味着更长训练下 bias 会继续涨。
  该盯的是 bias 与负载是否互相追打；目前 `entropy_norm` 稳在 0.999+，没有追打迹象。
- **V 投影的 RMS 早早见顶后一路下滑，v/q 由 1.31 降到 0.61**，而 Q/K 的 RMS 稳定上升。
  模型在没有任何约束的情况下**主动收回 V 的尺度**。给 V 加 norm 后 train loss 反而变差
  （+0.027，但 val loss −0.011），方向不一致，**目前无定论**。

### 7.6 后训练：四个算法在 benchmark 上无分辨力

Dense 腿的 `full_sft` / `dpo` / `agent` / `grpo` 在 ceval / mmlu / gsm8k 上
**分数完全相同**。这是**评测饱和**，不是「算法无效」：
64M 模型在这些任务上接近随机水平，ceval 的 n=1346 时一题约 0.07 个点。

评测套件分三组，每组对应一个阶段的产出：

| group | 任务 | 对应哪一阶段的产出 |
|-------|------|------------------|
| `core` | ceval / cmmlu / mmlu / arc / openbookqa / hellaswag / piqa / siqa / **bpb** | 基础能力锚点，跨阶段都该看。**bpb 是唯一低方差主指标**（同一段 wikitext-2、同一 token 数，跨权重可直接相减） |
| `sft` | **ifeval** / gsm8k / humaneval | SFT 的主要产出 |
| `rl` | **agentic**（多轮工具调用） | RL 的主要产出 |

三档深度：`quick`（冒烟，不可下结论）/ `standard`（出结论）/ `full`（关键权重跑满）。

```bash
python test/storage/eval_suite.py --list-tasks
python test/storage/eval_suite.py --depth standard
python test/storage/eval_suite.py --depth full --only full_sft_moe,grpo_768_moe
```

---

## 八、验证与回归

「跑起来了」说明不了「比上一版好」，也说明不了「和上一版等价」。
本项目对**新增抽象层与原实现的一致性**做了逐位验证：

| 脚本 | 验证什么 |
|------|---------|
| `test/storage/verify_arch.py` | 模型结构与已训权重的 `strict=True` 加载兼容性 |
| `test/storage/verify_components.py` | 逐个组件：构造 → 前向形状 → 反向非零梯度 → **增量解码与一次性前向一致** |
| `test/storage/verify_trainer.py` | 算法模块可解析 + 配置与 CLI 一致性 |
| `test/storage/verify_extra_datasets.py` | 外部数据集能否被原生 Dataset 消费 |
| `tests/test_training_algorithms.py` | **公式级回归**：离线偏好、在线 RL、Agent 轨迹与 LoRA 的数学边界 |

**兼容性证据**（新增层 vs 重构前）：

| 对比 | 结果 |
|------|------|
| pretrain（新层 vs 原脚本） | 2,954 个 step 的 loss/logits_loss/aux_loss **逐位一致** |
| grpo | 300 个 step 的 7 项指标**逐字符一致** |
| ppo | 310 个 step 的 8 项指标**逐字符一致**（含 Critic Loss 与双学习率） |
| 模型结构（新 vs 重构前） | logits 最大绝对差 **0.000e+00** |

```bash
python test/storage/verify_arch.py
python test/storage/verify_components.py
python test/storage/verify_trainer.py
python -m pytest tests/ -v
```

---

## 九、部署

### 架构感知的推理入口

`trainer/eval.py` 从 `--config` 读架构，因此能加载**任意组合**的模型：

```bash
python trainer/eval.py --config configs/pretrain_moe.yaml --weight pretrain_moe
python trainer/eval.py --config configs/sft_moe.yaml --weight full_sft_moe \
    --prompt "李白是唐代" --prompt "水的化学式是" --max_new_tokens 60
```

> `eval_llm.py` 是历史脚本，只能通过 `--hidden_size / --num_hidden_layers / --use_moe`
> 构造模型，而 `--use_moe 1` 合成出来的是**普通 top-k MoE**。
> 用它加载 `moe_finegrained` / `moe_shared` / `mla` / `gated` 会构建出错误的模型
> —— 参数量对不上，要么报错要么静默错配。
>
> 注意 `eval_llm.py` **没有** `--config` 参数（它不接受 YAML），
> 所以「换架构就必须换配置」这件事，只有 `trainer/eval.py` 做得到。

### OpenAI 兼容服务

```bash
cd scripts && python serve_openai_api.py
```

接入 FastGPT / OpenWebUI / Dify 等第三方 UI。额外支持
`reasoning_content` / `tool_calls` / `open_thinking` 字段。

### 推理引擎

| 引擎 | 转换方式 |
|------|---------|
| **Transformers** | `python scripts/convert_model.py` 做 torch ↔ transformers 转换 |
| **SGLang** | 原生支持，训练侧也支持 `--rollout_engine sglang` 训推分离 |
| **vLLM / llama.cpp / ollama / MNN** | 主线结构对齐 Qwen3 生态，便于导出 |

---

## 十、目录结构

```
minimind/
├── arch/              ★ 可插拔架构（4 槽位 × 18 组件）
├── configs/           ★ 分阶段 YAML 配置 + 加载器
├── trainer/           ★ 统一训练入口 + 16 个算法 + 通用脚手架
├── model/             兼容层（保留全部公开名字与行为）
├── dataset/           数据说明与原生 Dataset 类
├── webui/             FastAPI 控制台（对话 / 实验台 / 训练控制台）
├── scripts/           API 服务、模型转换、工具调用评测
├── tests/             公式级回归测试
├── test/              ✗ 实验产物（日志 / 权重 / 报告，不进 git）
├── images/            文档图片
└── requirements.txt
```

**设计原则**：仓库根只放**纯净项目代码**，能 clone 下来直接训练 / 推理。
实验的一切（几十 GB 的权重与日志）都在 `test/` 下，一刀切开。

`model/model_minimind.py` 是**兼容层**：模型实现已迁移到 `arch/`，
但本文件保留全部公开名字与行为，所以
既有权重仍可 `strict=True` 加载，`MiniMindForCausalLM` 仍可被继承
（如 PPO 的 `CriticModel`）。

---

## 十一、常见坑

1. **PyYAML 浮点陷阱**：指数必须带符号。
   `1.0e6` 会被当成**字符串**，`1.0e+6` 才是 float。

2. **不同架构的权重互不兼容**。MLA 的 `q_proj` 形状与 GQA 不同，
   换架构的实验必须 `--from_weight none` 从零训练。
   SFT 阶段要 `defaults` 引用预训练配置，否则加载失败。

3. **RL 的 rollout 序列是变长的**，需要 `attention.flash_attn_masked: true`。
   不开这个开关，SDPA 会退回 eager 路径并物化 `[B, H, L, L]` 的分数矩阵，
   GRPO 一批 12 条 1792 token 的序列每层就是 GB 级，24G 卡直接 OOM。

4. **`max_steps` 是等 token 预算实验的关键旋钮**。跑满即停，不看 epochs，
   这样不同架构的对比才公平。

5. **`update_ratio` 低于 1e-3 不是 bug**。它是**逐元素**口径 `rms(Δw)/rms(w)`，
   偏低只说明权重尺度大或 lr 小。
   ⚠️ 不能写成 `lr·‖g‖/‖w‖` —— 组级范数相除会漏掉参数量，实测差 5 个数量级。

6. **预训练 loss ≠ SFT loss**。前者全序列算，后者只在 assistant 段算，
   这两个数永远不直接比。

---

## 引用

本项目的模型结构、训练数据与评测口径继承自 MiniMind 系列工作。
如果你在研究中用到了本仓库，除引用本仓库外，也请一并引用上游：

```bibtex
@misc{minimind,
  title = {MiniMind: Train a Tiny LLM from Scratch},
  author = {Jingyao Gong},
  year = {2024},
  url = {https://github.com/jingyaogong/minimind},
  note = {GitHub repository}
}
```

## 致谢

感谢 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 的作者与
上游贡献者，把「从 0 训一个小模型」这条链路完整开源出来 —— 本项目的模型设计、
数据与评测基线均建立在其之上。

架构抽象层（`arch/`）、统一训练入口（`trainer/train.py`）、配置系统（`configs/`）、
Web 控制台（`webui/`）与实验体系为本仓库新增。

## License

[Apache License 2.0](LICENSE)
