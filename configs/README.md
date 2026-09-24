# configs/ —— 分阶段配置

按训练阶段拆分的配置，**模型结构与训练超参放在一起**，改配置即可换架构或调超参。

```
configs/
├── base.yaml      默认模型结构（被各阶段通过 defaults 引用）
├── pretrain.yaml  预训练
├── sft.yaml       监督微调（Full SFT / LoRA / QLoRA / 蒸馏）
├── rl.yaml        偏好与强化学习（DPO 系 / GRPO / DAPO / RLOO / PPO / Agent）
└── loader.py      加载与合并
```

## 用法

```bash
# 统一入口：--config 选架构、--algo 选算法
python trainer/train.py --algo sft  --config configs/base.yaml
python trainer/train.py --algo sft  --config configs/mla.yaml    # 换架构
python trainer/train.py --algo grpo --config configs/base.yaml   # 换算法

# 不带 --config 时按算法自动选阶段配置
python trainer/train.py --algo pretrain

# 命令行参数永远优先于配置
python trainer/train.py --algo pretrain --epochs 3 --batch_size 16
```

在代码里用：

```python
from configs import load_config, to_arch_config, train_options

cfg = load_config("configs/base.yaml")       # 已处理 defaults 链式合并
arch = to_arch_config(cfg)                   # -> ArchConfig，交给 arch 组装模型
opts = train_options(cfg)                    # -> dict，用作 argparse 默认值
```

### `--config` 是「叠加变体」，不是整体替换

配置的合成顺序是：**阶段配置作基线，`--config` 指定的文件叠加其上**。

因此一份"只换了注意力"的实验配置可以只有几行 —— 数据路径、序列长度、
学习率等自动沿用阶段配置：

```yaml
# configs/mla.yaml
attention:
  type: mla
  num_attention_heads: 8
  num_key_value_heads: 4
```

若变体里也写了 `data` / `train`，则按字段覆盖基线。

> 注意：不同架构的权重互不兼容（例如 MLA 的 `q_proj` 形状与 GQA 不同），
> 换架构的实验需要用 `--from_weight none` 从零训练。

## 覆盖规则（重要）

**配置只提供默认值，命令行永远优先。** 具体地：

| 情况 | 结果 |
|------|------|
| 配置里写 `epochs: 2`，不传参数 | 用 2 |
| 配置里写 `epochs: 2`，传 `--epochs 3` | 用 3 |
| 配置里写 `hidden_size: 768`，传 `--hidden_size 512` | 用 512 |

模型结构里的三个旧旋钮（`--hidden_size` / `--num_hidden_layers` / `--use_moe`）
默认值会**从配置同步**，所以它们不会以硬编码默认值压过配置。其中 `--use_moe 1`
可强制切到 MoE；`--use_moe 0` 表示「沿用配置的选择」（想强制稠密请改配置的
`feedforward.type`）。

## 相对路径

配置里的相对路径一律**相对仓库根目录**解析，与从哪个目录启动脚本无关：

```yaml
data:
  path: dataset/pretrain/pretrain_t2t.jsonl     # -> <repo>/dataset/pretrain/pretrain_t2t.jsonl
train:
  save_dir: out                        # -> <repo>/test/out
```

（原来 argparse 里的 `../dataset/...`、`../out` 是从 `trainer/` 启动时的写法，
指向同一批目录；配置里改成了更稳的仓库根相对路径。）

### 两个根目录

`data.path` 相对**仓库根**解析（数据集跟着代码走，在仓库里），而 `train.save_dir`
相对**产物根**解析：

| 变量 | 默认值 | 管什么 |
| --- | --- | --- |
| `REPO_ROOT` | `configs/` 的上一级 | `data.path`、`configs/` 自身 |
| `ARTIFACT_ROOT` | `<repo>/test` | `train.save_dir`（权重 / 指标 CSV） |

产物默认进 `test/`，这样仓库根永远是纯净代码（见 [`test/README.md`](../test/README.md)）。
想让产物回到仓库根：

```bash
MINIMIND_ARTIFACT_ROOT=/home/me/minimind python trainer/train.py ...
```

同一个环境变量还被 `trainer/common/checkpoint.py`（续训档）、
`trainer/trainer_utils.py::init_model`（`from_weight` 读哪里）、
`webui/server.py`（UI 读哪里）读取，所以「训练写哪里」和「UI 读哪里」永远一致。
绝对路径（如 `--save_dir /data/x`）不受影响，原样使用。

## 字段说明

### `model` / 四个槽位

与 `arch/` 的组件一一对应，详见 [`arch/README.md`](../arch/README.md)。

```yaml
model:      { hidden_size, num_hidden_layers, vocab_size, ... }
norm:       { type: rmsnorm | rmsnorm_zero_centered | layernorm }
positional_encoding: { type: rope | partial_rope | nope, ... }
attention:  { type: gqa | gated | sliding_window | mla | compressed | deltanet, ... }
feedforward:{ type: swiglu | geglu | moe | moe_shared | moe_finegrained, ... }
```

### `data` / `train`

```yaml
data:
  path: dataset/...          # 训练数据
  max_seq_len: 768           # 截断长度（token 数，不是字符数）
train:
  save_dir / save_weight / epochs / batch_size / learning_rate /
  accumulation_steps / grad_clip / log_interval / save_interval /
  num_workers / dtype / seed / from_weight
```

### `algo.<名字>`

同一份配置服务多个算法时，用 `algo` 段区分各自的专属超参；
该段同时可以覆盖 `data` 与 `train` 里的任意项。

```bash
python train_lora.py        # 读 configs/sft.yaml 的 algo.lora
python train_dpo.py         # 读 configs/rl.yaml 的 algo.dpo
python train_grpo.py        # 读 configs/rl.yaml 的 algo.grpo
```

## 一个 YAML 的坑

PyYAML 判定浮点时要求**指数必须带符号**：

```yaml
rope_theta: 1.0e6     # ✗ 被解析成字符串 "1.0e6"（指数没带符号）
rope_theta: 1.0e+6    # ✓ float 1000000.0
rope_theta: 1.0e-6    # ✓ float（负指数自带符号）
learning_rate: 5.0e-4 # ✓
```

组件里对数值字段做了 `float()` 兜底，但写配置时建议统一带上 `+`。
