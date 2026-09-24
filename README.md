<div align="center">

<img src="./images/logo.png" alt="MiniMind-V2" width="120">

# MiniMind-V2

**换一种注意力，改几行 YAML。换一种训练算法，换一个 `--algo`。**

把「从 0 训一个小模型」做成可插拔的架构，加上可插拔的算法。
四槽位 × 18 个组件，16 种算法共用同一套 forward / backward。

[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org)

</div>

---

## 为什么是 V2

接手的是一份写死的架构，外加八个互相复制的 `train_*.py`。
想试 MLA，得改 `model_minimind.py`；想试 DAPO，得复制一份 `train_grpo.py` 再改；
训练循环里修一个 bug，八个脚本各改一遍。

V2 把这两件事都收成参数。所有对比组走同一套前向与反向，
差异只来自你换上的那个组件，而不是「这组比那组多了一处没注意到的实现分歧」。

```
模型结构   arch/            四个槽位 × 18 个已注册组件，YAML 组装
训练算法   trainer/algos/   16 个算法各一份实现，--algo 切换
```

---

## 30 秒上手

```bash
git clone https://github.com/Bozheng-Li/MInimind-V2.git
cd MInimind-V2
pip install -r requirements.txt          # torch 单独装，见文件内注释
```

数据不进仓库。下 [mini 预训练集 1.2GB](https://huggingface.co/datasets/jingyaogong/minimind_dataset) 与
[mini SFT 集 1.6GB](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)，
放到 `dataset/pretrain/` 与 `dataset/sft/`。完整清单见 [`dataset/dataset.md`](dataset/dataset.md)。

```bash
cd trainer
python train.py --algo pretrain --config configs/pretrain.yaml
python train.py --algo sft      --config configs/sft.yaml
python eval.py  --config configs/sft.yaml --weight full_sft
```

换成门控注意力 + 细粒度 MoE，配置与权重必须成对：

```bash
python train.py --algo pretrain --config configs/pretrain_moe.yaml
python train.py --algo sft      --config configs/sft_moe.yaml
```

权重、日志、指标 CSV 默认落在 `test/`。想改落点：`export MINIMIND_ARTIFACT_ROOT=/path/to/artifacts`。

---

## 可插拔架构

模型拆成四个槽位。每个槽位是 `type` 加该组件自己的参数。
新增组件：写一个文件，加 `@register`，在 YAML 里写名字。不用改框架。

```
arch/
├── attention/     gqa · gated · sliding_window · mla · compressed · deltanet
├── ffn/           swiglu · geglu · moe · moe_shared · moe_finegrained
├── positional/    rope · partial_rope · nope
└── norm/          rmsnorm · rmsnorm_zero_centered · layernorm
```

| 槽位 | `type` | 来自 | 它做的事 |
|------|--------|------|----------|
| attention | `gqa` | Llama / Qwen | 分组查询 + QK-Norm。KV 头数等于查询头数即 MHA，设为 1 即 MQA |
| | `gated` | Qwen3-Next | 输出门 + 零中心 QK-Norm，可配 partial RoPE |
| | `sliding_window` | Mistral / Gemma | 每个 token 只看前 `window_size` 个 |
| | `mla` | DeepSeek-V2/V3 | KV 压进 latent，RoPE 只打在解耦的那几维。缓存 224B/token，GQA 是 768B |
| | `compressed` | DeepSeek-V4 | 沿序列维压缩，每 `compress_rate` 个 token 收成一项，384B/token |
| | `deltanet` | Qwen3-Next | 门控线性注意力，定长循环状态替代 KV cache |
| feedforward | `swiglu` / `geglu` | Llama / Gemma | 门控 FFN，激活分别是 SiLU 与 GELU |
| | `moe` | — | top-k 路由 + aux loss |
| | `moe_shared` | DeepSeekMoE | 常驻共享专家 + N 个路由专家 |
| | `moe_finegrained` | DeepSeek-V3 | 细粒度专家，可学习 bias 做均衡，不靠 aux loss |
| positional | `rope` | — | 旋转位置编码，`inference_rope_scaling` 开 YaRN |
| | `partial_rope` / `nope` | — | 只旋转前若干维 / 完全不加位置编码 |
| norm | `rmsnorm` | Llama | 标准 RMSNorm |
| | `rmsnorm_zero_centered` | Qwen3-Next | 权重从 0 起步，等价于恒等映射 |
| | `layernorm` | — | 经典 LayerNorm，留给消融 |

插拔能成立，靠的是框架把两处耦合收进了组件内部，而不是让你在 YAML 里手工对齐：

- **注意力决定旋转维数。** MLA 的 RoPE 只作用在 `qk_rope_head_dim` 上，远小于 `head_dim`。
  组件可声明 `positional_dim(cfg)`，不声明就退回 `head_dim`。
- **MoE 专家是槽内参数。** `moe_finegrained` 用 `cfg.override(...)` 派生窄专家，
  专家宽度、数量、均衡策略都不进全局 `model` 段。

`type` 也可以写仓库外的完整路径，注册表查不到短名就按 `importlib` 回退：

```yaml
attention:
  type: my_ext.attention.FlashAttention
```

---

## 可插拔算法

每个算法在 `trainer/algos/` 里只有一份实现。旧的 `train_*.py` 退化成兼容壳，补上固定的 `--algo` 后转发给统一入口。

```bash
python trainer/train.py --algo sft  --config configs/base.yaml
python trainer/train.py --algo dpo  --config configs/base.yaml
python trainer/train.py --algo grpo --config configs/base.yaml

torchrun --nproc_per_node 4 trainer/train.py --algo grpo
```

不带 `--config` 时按算法选阶段配置：`pretrain` → `pretrain.yaml`，
`sft` / `lora` / `qlora` / `distill` → `sft.yaml`，偏好优化与在线 RL → `rl.yaml`。
命令行永远压过配置。

| | 算法 | 额外依赖 | 用在 |
|---|------|----------|------|
| 预训练 | `pretrain` | — | 原始文本 next-token |
| 监督微调 | `sft` | — | 全参数指令微调 |
| 参数高效 | `lora` · `qlora` | QLoRA 需 bitsandbytes + CUDA | 小显存垂域微调，默认 NF4 + double quant |
| 蒸馏 | `distill` | teacher | 白盒分布蒸馏 |
| 离线偏好 | `dpo` · `ipo` · `kto` | reference | 有 chosen / rejected |
| 无 reference | `simpo` · `cpo` · `orpo` | — | 少占一份 reference 显存 |
| 在线，无 Critic | `grpo` · `rloo` | reward + reference | 可验证奖励 |
| | `dapo` | reward | Clip-Higher、动态采样、token 级 loss |
| Actor-Critic | `ppo` | reward + reference + critic | 通用 RLHF |
| 多轮工具 | `agent` | reward + reference + 工具环境 | 工具调用强化学习 |

三个容易踩错的口径：

- **CISPO 是默认的在线策略目标**（`--loss_type cispo`）。它用 detach 后的 ratio 做上侧裁剪，梯度不会被裁剪项反复抹掉。
- **IPO 与 CPO 的长度口径不同。** IPO 用每 token 平均后的间隔；CPO 的 sigmoid 项用序列总 log-prob。SimPO 才是无 reference 的长度归一化目标。
- **Agent 的训练奖励与评测 `pass_rate` 是同一个函数**，`trainer/algos/rl/agent_tools.py::calculate_rewards`。

加一个算法：继承 `Algorithm`，实现 `build_dataset()` 与 `compute_loss(batch) -> (loss, aux_loss)`
（`loss` 须已除以 `accumulation_steps`），需要 ref / teacher / critic / reward 就实现 `extra_models()`，
然后在 `algos/__init__.py` 的 `_ALGO_MODULES` 登记。

---

## 配置

`--config` 是叠加，不是替换。阶段配置打底，你的文件盖在上面，所以一份实验配置可以只有几行：

```yaml
# configs/mla.yaml
attention:
  type: mla
  num_attention_heads: 8
  num_key_value_heads: 4
```

数据路径、序列长度、学习率沿用阶段配置。相对路径一律相对仓库根解析，与启动目录无关。

| 变量 | 默认 | 管什么 |
|------|------|--------|
| `REPO_ROOT` | `configs/` 的上一级 | `data.path`、配置自身 |
| `ARTIFACT_ROOT` | `<repo>/test` | 权重、指标 CSV、续训档 |

两条必须知道的约束：

- **YAML 里的指数要带符号。** `1.0e6` 会被 PyYAML 读成字符串，`1.0e+6` 才是 float。
- **`--use_moe 1` 只在前馈本身是稠密时生效。** 无条件覆写会把 `moe_shared` / `moe_finegrained`
  降级成普通 top-k MoE，共享专家和 aux-loss-free 一起丢掉，而参数量只差共享专家那一份，外面看不出来。

---

## 数据

```
dataset/
├── pretrain/   pretrain_t2t.jsonl          10GB · 847 万条 · ~3.2B token
│               pretrain_t2t_mini.jsonl      1.2GB · 127 万条
├── sft/        sft_t2t.jsonl               14GB · 511 万条（已混入 Tool Call）
│               sft_t2t_mini.jsonl           1.6GB · 90.6 万条
└── rl/         dpo.jsonl · rlaif.jsonl · agent_rl.jsonl · agent_rl_math.jsonl
```

下载：[HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) ·
[ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)。按需单文件下载即可。

预训练支持按权重配比，模型看到的比例等于权重比，与各文件条数无关：

```yaml
data:
  path: dataset/pretrain
  mix:
    pretrain_t2t.prose.jsonl: 0.70
    pretrain_t2t.qa.jsonl:    0.10
    pretrain_t2t.en.jsonl:    0.15
    pretrain_t2t.code.jsonl:  0.05
```

`max_seq_len` 是 token 数。本仓库 tokenizer 下，中文约 1.5–1.7 字符/token，英文约 4–5。
上游文档里的「最大长度」标的是字符数。

---

## Web 控制台

FastAPI 后端，三个页面。前端是原生 HTML / CSS / JS，无构建、无 npm、无 CDN。

```bash
python webui/server.py            # http://localhost:7860
python webui/server.py --no-load  # 进界面再选模型
```

| 路径 | 页面 | 它回答 |
|------|------|--------|
| `/` | 对话 | 这个权重现在能干什么，它是哪次训练产出的 |
| `/lab` | 实验台 | 各算法、架构、评测怎么比 |
| `/train` | 训练控制台 | 下一步该起哪个训练，正在跑的那个怎么样了 |

它走 `trainer/eval.py` 的加载路径，所以能打开任意架构组合的原生 `.pth`。
逐页说明与指标映射见 [`webui/README.md`](webui/README.md)。

`scripts/web_demo.py` 是历史的 Streamlit 入口。它读 transformers 格式目录，
且 `skip_special_tokens=True` 会把 `<tool_call>`、`<think>` 这类 added token 一起吃掉。
新训的模型请用上面这条。

---

## 实验里站住的结论

18 组架构 × 61M token 等预算扫描，外加 SFT 双臂（同 1.39B token、同 seed、同数据，唯一差异是架构）。
完整数字与回溯索引在 [`test/storage/report/EXPERIMENT_REPORT.md`](test/storage/report/EXPERIMENT_REPORT.md)。

| 结论 | 证据 | 主线 |
|------|------|------|
| **门控注意力是唯一稳健的正收益** | Δloss −0.055，达到基线只需 0.84× 算力；代价是参数 +7%、吞吐 −19% | 已用于 `pretrain_moe.yaml` |
| **MoE 要等算力口径才公平** | 61M token 下微正但吞吐仅基线 28%；SFT 等质量只需 0.49×–0.71× 算力，墙钟要付 1.76×–2.57× | 已作主线，部署时权衡墙钟 |
| **省缓存的注意力在这个预算下全是负的** | MLA +0.137、compressed +0.241、deltanet +0.180 | 未采用 |
| **位置编码不要动** | `rope_theta=1e6` 优于 `1e4`；`nope` 为 +0.403 | 保持 RoPE |
| **后训练四算法在 64M 的 benchmark 上没有分辨力** | dense 的 `full_sft` / `dpo` / `agent` / `grpo` 分数相同，接近随机 | 换评测，不换算法 |

SFT 双臂的核心数字：同样 1.39B token，dense（gqa + swiglu，63.9M）验证 loss 0.6370，
MoE（gated + moe_finegrained，214M / 激活 80M）0.5523，差 −0.0847，全程稳定在 −0.083 到 −0.089。
dense 吞吐是 MoE 的 4.49 倍。

评测套件分三组：`core`（含 bpb，唯一低方差、可直接相减的主指标）、`sft`（ifeval / gsm8k / humaneval）、
`rl`（多轮工具调用）。深度三档 `quick` / `standard` / `full`，`quick` 只用于冒烟。

```bash
python test/storage/eval_suite.py --depth standard
```

---

## 验证

新增抽象层相对原实现做过逐位对照，而不是「跑起来了」。

| | 结果 |
|---|------|
| pretrain，新层 vs 原脚本 | 2,954 step 的 loss / logits_loss / aux_loss 逐位一致 |
| grpo | 300 step、7 项指标逐字符一致 |
| ppo | 310 step、8 项指标逐字符一致 |
| 重构前后 logits | 最大绝对差 0 |

```bash
python test/storage/verify_arch.py
python test/storage/verify_components.py
python test/storage/verify_trainer.py
python -m pytest tests/ -v
```

---

## 推理与部署

`trainer/eval.py` 从 `--config` 读架构，能加载任意组合：

```bash
python trainer/eval.py --config configs/sft_moe.yaml --weight full_sft_moe \
    --prompt "水的化学式是" --max_new_tokens 60
```

`eval_llm.py` 是历史脚本。它只认 `--hidden_size` / `--num_hidden_layers` / `--use_moe`，
而 `--use_moe 1` 拼出来的是普通 top-k MoE。用它加载 `gated`、`moe_finegrained`、`mla` 会得到错误的模型。

```bash
cd scripts && python serve_openai_api.py     # OpenAI 兼容，含 reasoning_content / tool_calls
python scripts/convert_model.py              # torch ↔ transformers
```

vLLM、llama.cpp、ollama、MNN 走主线结构与 Qwen3 生态的对齐。训练侧也可以 `--rollout_engine sglang`。

---

## 目录

```
arch/        四槽位 × 18 组件
configs/     分阶段 YAML 与加载器
trainer/     统一入口、16 个算法、训练脚手架
model/       兼容层：公开名字与行为保留，既有权重仍可 strict=True 加载
dataset/     数据说明与 Dataset 类
webui/       对话 / 实验台 / 训练控制台
scripts/     API 服务、模型转换、工具调用评测
tests/       公式级回归
test/        实验产物，不进 git
```

仓库根只放能 clone 下来直接训练的代码。权重与日志在 `test/`，一刀切开。

---

## 踩过的坑

1. **不同架构的权重不能互载。** MLA 的 `q_proj` 形状就和 GQA 不同，换架构必须 `--from_weight none`。
   SFT 要 `defaults` 引用对应的预训练配置，否则加载直接失败。
2. **RL 的 rollout 序列是变长的**，要开 `attention.flash_attn_masked: true`。
   否则 SDPA 退回 eager，物化 `[B, H, L, L]`，一批 12 条 1792 token 的序列就能在 24G 卡上 OOM。
3. **`max_steps` 是等 token 预算实验的旋钮。** 跑满即停，不看 epochs，不同架构才可比。
4. **`update_ratio` 低于 1e-3 不是 bug。** 它是逐元素的 `rms(Δw) / rms(w)`。
   写成 `lr·‖g‖ / ‖w‖` 会漏掉参数量，实测差五个数量级。
5. **预训练 loss 与 SFT loss 不能直接比。** 前者全序列，后者只在 assistant 段。

---

## 引用

模型结构、训练数据与评测口径继承自 MiniMind。研究中使用本仓库时，请一并引用上游：

```bibtex
@misc{minimind,
  title  = {MiniMind: Train a Tiny LLM from Scratch},
  author = {Jingyao Gong},
  year   = {2024},
  url    = {https://github.com/jingyaogong/minimind}
}
```

## 致谢

感谢 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 把「从 0 训一个小模型」完整开源。
本仓库在其上新增了 `arch/`、`trainer/train.py`、`configs/`、`webui/` 与实验体系。

[Apache License 2.0](LICENSE)
