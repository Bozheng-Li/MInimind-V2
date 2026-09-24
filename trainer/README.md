# trainer/ —— 训练链路

## 目录结构

```
trainer/
├── train_*.py            旧命令兼容壳，全部转发到统一入口，避免维护两套公式
├── train.py              ★ 统一的可插拔训练入口（换架构 × 换算法）
│
├── common/               通用脚手架（算法的"怎么跑"）
│   ├── runtime.py        DDP / 种子 / 精度 / GradScaler / compile+DDP 包装
│   ├── data.py           DataLoader / DistributedSampler / SkipBatchSampler
│   ├── checkpoint.py     存盘与断点续训（复用 lm_checkpoint，命名语义不变）
│   ├── logging.py        Logger + wandb/swanlab
│   ├── runner.py         通用训练循环 + TrainContext
│   └── pipeline.py       run_training：九段式启动流程
├── algos/                算法模块（算法的"优化什么"）
│   ├── base.py           Algorithm 协议
│   ├── pretrain/         预训练（Causal LM）
│   ├── sft/              监督训练：Full SFT / LoRA / QLoRA / 蒸馏
│   └── rl/               偏好与强化学习
│       ├── preference.py DPO / IPO / SimPO / CPO / ORPO / KTO
│       ├── grpo.py       GRPO / CISPO 公共在线策略目标
│       ├── dapo.py       Clip-Higher + 动态采样 + token-level loss
│       ├── rloo.py       leave-one-out baseline
│       ├── ppo.py        Actor-Critic + GAE
│       └── agent.py      多轮工具调用强化学习
│
├── rollout_engine.py     Torch / SGLang 采样与精确 completion mask
└── trainer_utils.py      模型、checkpoint 等兼容工具
```

所有算法只有 `algos/` 下的一份实现。旧 `train_*.py` 不再复制训练循环，只补固定
`--algo` 后转发到统一入口，因此修复公式、DDP 或 checkpoint 时不会出现新旧分叉。

## 统一入口 `train.py`

```bash
# 换算法：同一份配置，不同 --algo
python trainer/train.py --algo sft --config configs/base.yaml
python trainer/train.py --algo dpo --config configs/base.yaml
python trainer/train.py --algo grpo --config configs/base.yaml
python trainer/train.py --algo dapo                         # DAPO
python trainer/train.py --algo qlora                        # NF4 QLoRA

# 换架构：同一算法，不同配置
python trainer/train.py --algo sft --config configs/base.yaml    # GQA 基线
python trainer/train.py --algo sft --config configs/mla.yaml     # MLA 变体

# 不带 --config 时按算法自动选阶段配置
#   pretrain            -> configs/pretrain.yaml
#   sft / lora / qlora / distill -> configs/sft.yaml
#   偏好优化与在线 RL           -> configs/rl.yaml

# 命令行参数永远优先于配置
python trainer/train.py --algo pretrain --epochs 3 --batch_size 16

# 多卡
torchrun --nproc_per_node 4 trainer/train.py --algo grpo
```

## 算法分类与适用场景

| 类别 | 算法 | 是否需要 reference / reward / critic | 典型用途 |
|------|------|--------------------------------------|----------|
| 预训练 | `pretrain` | 无 | 原始文本 next-token 训练 |
| 监督微调 | `sft` | 无 | 全参数指令微调 |
| 参数高效 SFT | `lora`, `qlora` | 无；QLoRA 需 bitsandbytes/CUDA | 小显存垂域微调 |
| 监督蒸馏 | `distill` | teacher | 白盒分布蒸馏 |
| 离线偏好 | `dpo`, `ipo`, `kto` | reference | 有 chosen/rejected 数据的稳健对齐 |
| Reference-free 偏好 | `simpo`, `cpo`, `orpo` | 无 | 少一份 reference 显存 |
| 在线无 Critic RL | `grpo`, `rloo` | reward + reference | 可在线采样、可验证奖励任务 |
| 在线无 Critic RL | `dapo` | reward；reference 仅在 KL 系数非零时需要 | 长回答推理与可验证奖励任务 |
| 在线 Actor-Critic | `ppo` | reward + reference + critic | 通用 RLHF |
| Agentic RL | `agent` | reward + reference + 工具环境 | 多轮工具调用 |

DAPO 实现了 Clip-Higher、动态重采样、token 级归一化和超长惩罚；RLOO 使用
leave-one-out baseline。QLoRA 默认使用 NF4、double quant 与 paged AdamW 8-bit。
离线偏好算法中，IPO 对策略/reference 的回答 log-prob 做长度归一化，CPO 的
sigmoid 偏好项使用序列总 log-prob；两者不能混用同一个长度口径。

**关于 `--config`**：它是**叠加在阶段配置之上的变体**，不是整体替换。
所以一份"只换了注意力"的实验配置可以只写几行，数据与训练超参自动沿用阶段配置：

```yaml
# configs/mla.yaml —— 只改模型结构
attention:
  type: mla
  num_attention_heads: 8
  num_key_value_heads: 4
```

注意：不同架构的权重**互不兼容**（例如 MLA 的 `q_proj` 形状与 GQA 不同），
换架构的实验要用 `--from_weight none` 从零训练。

## 新增一个算法

1. 在 `algos/` 下新建模块，继承 `Algorithm`
2. 实现 `build_dataset()` 与 `compute_loss(batch) -> (loss, aux_loss)`
   （`loss` 必须已除以 `accumulation_steps`）
3. 需要额外模型（ref / teacher / critic / reward）就实现 `extra_models()`
4. 在 `algos/__init__.py` 的 `_ALGO_MODULES` 里登记，即可用 `--algo <名字>` 调用

可选的钩子（都有默认实现）：

| 钩子 | 用途 |
|------|------|
| `configure_model(model)` | 模型改造（如 LoRA 挂载与冻结），**早于建优化器** |
| `build_optimizer(model)` | 自定义优化器（LoRA 只优化 lora_params） |
| `clip_parameters(model)` | 梯度裁剪作用于哪些参数 |
| `weight_prefix()` / `save_weights()` | 权重命名与保存方式 |
| `checkpoint_extra()` | 额外写进续训档的对象（如 PPO 的 critic） |
| `format_log(...)` | 自定义日志格式（各算法重构前的格式不同） |
| `train_epoch(...)` | RL 类算法覆盖它自建含 rollout 的循环 |

## 各算法记录什么

「针对不同算法记录不同的 log」在本项目分三层落实，**三层的键名互不相同**，界面上也分别展示。

### 第一层：每个 step 打到 stdout 的一行

由 `Algorithm.format_log()` 生成默认格式。蒸馏与离线偏好算法覆盖这个方法；五个
在线 RL 算法覆盖 `train_epoch`，在 rollout 循环里同时输出日志与 CSV 指标：

| 算法族 | 实现 | 主要字段 |
|------|------|----------|
| `pretrain` / `sft` / `lora` / `qlora` | `base.py` 默认格式 | `loss`, `logits_loss`, `aux_loss`, `lr`, `epoch_time` |
| `distill` | `sft/distill.py` | `loss`, `ce`, `distill`, `aux_loss`, `learning_rate` |
| `dpo` / `ipo` / `simpo` / `cpo` / `orpo` / `kto` | `rl/preference.py` | `preference_loss`, `margin`, `pref_acc`, `aux_loss` |
| `grpo` / `dapo` / `rloo` | `rl/grpo.py` | `Reward`, `KL_ref`, `Adv Std`, `Actor Loss`, `Avg Response Len` |
| `ppo` | `rl/ppo.py` | `Reward`, `KL_ref`, `Approx KL`, `ClipFrac`, `Critic Loss`, `Actor LR`, `Critic LR` |
| `agent` | `rl/agent.py` | `Reward`, `KL`, `GrpStd`, `AdvStd`, `Loss`, `AvgLen`, `AdvMean` |

> stdout 为兼容历史脚本保留了少量不同名字：GRPO 打 `KL_ref`、Agent 打 `KL`、
> GRPO 打 `Adv Std`、Agent 打 `AdvStd`、PPO 打 `Actor LR` 而 GRPO 打 `Learning Rate`。
> 就算名字几乎一样，口径也未必相同 —— GRPO 的 `KL_ref` 与 Agent 的 `KL` 算式形状一致
> （`Σ(ref−π) / Σ 掩码 token`，见 `grpo.py:143` 与 `agent.py:160`），但 Agent 的 `completion_mask`
> 来自多轮 rollout（`full_response_masks`，含工具返回段），与 GRPO 的单轮 completion 不是同一个 token 集。
> 所以 `/train` 控制台的实时曲线解析器**只归一化真正同源的键**（`tokens_s` → `tokens_per_sec`、
> `learning_rate` → `lr`），其余一律按原样各画一条 —— 硬把 `KL` 并到 `KL_ref` 就是在编造口径。

在线 RL 另有 `wandb.log(...)` 的并行一路（键名与 stdout 又不同，如 Agent 的 `KL` 落进
`kl_ref`），以及 `--max_steps` 到点时的一行「已达到 max_steps=…，停止训练」。

### 第二层：逐 step 落盘的指标 CSV

`common/runner.py::build_metric_columns()` 生成列清单。**列必须预先声明**（`DictWriter`
遇到表头外的字段直接报错，追加写模式下改表头要重写整个文件），所以是「按模型结构 + 按算法」
拼出来的三段：

| 段 | 内容 |
|----|------|
| `BASE_METRIC_COLUMNS`（所有算法） | loss / logits_loss / aux_loss / lr / 分组梯度范数与权重范数 / `update_ratio_{attn,ffn,embed,norm}` / `val_loss` / 吞吐 / 显存与利用率 / 内存 / MoE 健康度 |
| 结构相关（按模型补列） | `moe_load_e0..e{n-1}`（逐专家）；`{q,k,v,out,hidden,routed,shared}_rms_L0..L{n-1}` 与 `gate_{mean,std}_L*`（逐层） |
| 算法相关（按 `--algo` 补列） | RL 共性 `_RL_ONLINE_COLUMNS`：`reward` / `rew_{len,think_len,think_close,rep,rm}` / `kl_ref` / `perplexity` / `eos_rate` / `trunc_rate` / `adv_zero_frac` … |

RL 算法的**专属列**（缺的列留空，不影响别的算法）：

| 算法 | 专属列 |
|------|--------|
| `grpo` / `dapo` / `rloo` | `group_reward_std`, `group_reward_zero_std`, `ratio_mean`, `clipfrac` |
| `ppo` | `critic_loss`, `value_loss`, `approx_kl`, `clipfrac`, `adv_raw_mean`, `adv_raw_std`, `kl_early_stop`, `actor_lr`, `critic_lr` |
| `agent` | `group_reward_std`, `pass_rate`, `unfinished_rate`, `tool_calls_mean`, `valid_call_rate`, `tool_gap_mean`, `turns_mean` |
| `dpo` | `preference_loss`, `dpo_loss`（兼容别名）, `reward_margin`, `preference_acc` |
| `ipo` / `simpo` / `cpo` / `orpo` / `kto` | `preference_loss`, `reward_margin`, `preference_acc` |

### 第三层：`common/metrics.py` 的富采集

逐 step 的列里绝大部分由这里的 `MetricsCollector.snapshot()` 产出，出处与诊断用途见该文件头部：

| 家族 | 采集内容 | 诊断什么 |
|------|----------|----------|
| 优化动力学 | 按 `attn` / `ffn` / `embed` / `norm` 分组的梯度范数与权重范数；`update_ratio_*` | 哪一部分在学、是否停滞 |
| MoE 健康度 | 专家负载 `per` / `moe_load_cv` / `moe_load_maxmin` / `moe_dead_experts`、路由熵、均衡偏置 `moe_bias_*`、`shared_share` | **专家被饿死不会体现在 loss 上**，aux-loss-free 的均衡全靠 bias 反馈回路 |
| 泛化 | 留出集 `val_loss`（`evaluate_loss`，元组与 dict 两种 batch 形状都认） | 区分「还在学」与「开始过拟合」 |
| 系统 | `GpuSampler` 后台轮询（利用率均值 / 最低值 / 显存峰值）+ torch 显存 + 进程内存 | 卡是否真在跑、有没有被别的进程挤 |
| 组件中间量 | 逐层 `q/k/v/out_rms`、`gate_{mean,std,sat_lo,sat_hi}`、`hidden_rms`、`routed/shared_rms` | 门控是否饱和、残差流是否逐层放大 |

> `update_ratio` 是**逐元素**口径 `rms(Δw)/rms(w)`。⚠️ 不能写成 `lr·‖g‖/‖w‖` ——
> 组级范数相除会漏掉参数量，实测差 5 个数量级，会得出「FFN 几乎没在学」的错误结论
> （完整推导见 `metrics.py` 中该行注释）。

### 在界面上看

| 想看的 | 去 `/lab` 哪一页 |
|--------|------------------|
| 各算法的日志字段差异（stdout 那一层） | `/train` 作业卡的「本次算法声明的指标 N 个」 |
| loss / 吞吐 / 梯度范数 | `#pretrain`、`#sft`；`/train` 实时曲线 |
| 优化动力学、门控健康、逐层激活量、专家负载热力图 | `#pretrain` |
| 分算法的专属指标（PPO 的 critic、Agent 的工具轮次、DPO 的 margin） | `#rl` 的「PPO 专属」/「Agent 专属」/「DPO 专属」卡 |
| 「谁记了哪些列」的横向对照 | `#rl` 页底的「算法 × 指标家族覆盖矩阵」 |
| 三类指标落在哪张 CSV 的哪一列 | `#runs` 展开某次训练；原始列在 `#logs` 与 `test/out/*_metrics.csv` |

完整映射表见 [`webui/README.md`](../webui/README.md) 的「指标 → 页面」一节。

### 怎么证明有效

训练日志能说明「跑起来了、没炸」，说明不了「比上一版好」。判断「这次改动的效果」
要另跑 `test/storage/eval_suite.py` —— 它按 `group` 把任务分成三组，**每一组对应
一个阶段的产出**，所以「哪个算法改进了什么」是有明确对照的：

| group | 任务 | 对应哪个阶段的产出 |
|---|---|---|
| `core` | ceval / cmmlu / mmlu / arc / openbookqa / hellaswag / piqa / siqa / **bpb** | 基础能力锚点，**跨阶段都该看**。bpb 是唯一低方差主指标（同一段 wikitext-2、同一 token 数，跨权重可直接相减） |
| `sft` | **ifeval**（指令跟随）/ gsm8k / humaneval | SFT 的主要产出。`ifeval` 的 prompt 级严格通过率就是「指令跟随」的量化 |
| `rl` | **agentic**（多轮工具调用） | RL 的主要产出。`pass_rate` / `valid_call_rate` / `turns_mean` |

**agentic 评测与训练同源**：它不另写一套 rollout，而是直接调
`trainer/algos/rl/agent_tools.py` 的 `rollout_single` + `calculate_rewards(return_stats=True)`
——所以评测里看到的 `pass_rate` 与训练控制台打的那个 `pass_rate` 是**同一个函数**算的。
这正是本文件开头「口径不同不能硬合」原则的正向应用：能同源的就别再造第二套。

默认跑 `dataset/rl/agent_rl_math.jsonl` 作**域外留出集**（实测它与训练默认用的
`agent_rl.jsonl` user 文本交集为 0）；`--agentic_indomain` 换成训练集子集作对照，
那一栏只能说明「记住了没」，不能当泛化证据。

```bash
python test/storage/eval_suite.py --list-tasks        # 看任务表与三档配额
python test/storage/eval_suite.py --depth standard    # 出结论（quick 档只能冒烟）
bash test/storage/run_eval_suite.sh                   # 13 权重 × 5 卡并行
```

## 回归验证

```bash
python test/storage/verify_arch.py        # 模型结构与已训权重兼容性
python test/storage/verify_components.py  # 逐个模型组件的可用性
python test/storage/verify_trainer.py     # 算法模块可解析 + 配置与 CLI 一致性
```

**兼容性证据**（新增层与原实现的一致性）：

| 对比 | 结果 |
|------|------|
| pretrain（新层 vs 原脚本） | 2954 个 step 的 loss/logits_loss/aux_loss **逐位一致** |
| grpo | 300 个 step 的 7 项指标**逐字符一致** |
| ppo | 310 个 step 的 8 项指标**逐字符一致**（含 Critic Loss 与双学习率） |
| 模型结构（新 vs 重构前） | logits 最大绝对差 **0.000e+00** |
