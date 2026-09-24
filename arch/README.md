# arch/ —— 可组合模型架构

用一个 YAML 文件自由组合模型结构：换注意力机制、换 MoE 架构、换位置编码，
**都不需要改 Python 代码**。

> 配置统一放在 [`configs/`](../configs/)：`base.yaml` 是默认模型结构，
> `pretrain.yaml` / `sft.yaml` / `rl.yaml` 各自 `defaults` 引用它并叠加训练超参。

```bash
# 最直观的用法：改 configs/base.yaml（或某个阶段配置），然后照常训练
cd trainer && python train_pretrain.py
```

---

## 目录结构

```
arch/
├── registry.py         组件注册表（短名 / import 路径两种定位方式）
├── schema.py           YAML 解析、校验、默认值推导、旧参数合成
├── config_class.py     ArchConfig -> PretrainedConfig 的包装
├── build.py            组件构造与组装入口
├── block.py            可组合 Transformer Block
├── model.py            ArchModel（主干）/ ArchForCausalLM（完整模型）
├── norm/               归一化：rmsnorm / rmsnorm_zero_centered / layernorm
├── positional/         位置编码：rope / partial_rope / nope
├── attention/          注意力：gqa / gated / sliding_window / mla / compressed / deltanet
└── ffn/                前馈：swiglu / geglu / moe / moe_shared / moe_finegrained
```

---

## 组件清单

### attention

| type | 来源 | 核心机制 |
|------|------|---------|
| `gqa` | Llama / Qwen | 分组查询 + QK-Norm。**设 `num_key_value_heads` = `num_attention_heads` 即退化为 MHA，设为 1 即 MQA**，无需单独实现 |
| `gated` | Qwen3-Next | 输出门（sigmoid 门控注意输出）+ zero-centered QK-Norm + 支持 partial RoPE |
| `sliding_window` | Mistral / Gemma | 局部窗口：每个 token 只看前 `window_size` 个 |
| `mla` | DeepSeek-V2/V3/V4 | KV 低秩压缩到 latent + **解耦 RoPE**；缓存的是压缩形式而非完整 K/V |
| `compressed` | DeepSeek-V4 (HCA 思路) | 沿序列维压缩 KV（每 `compress_rate` 个 token 压成 1 项） |
| `deltanet` | Qwen3-Next / Kimi Linear | 门控线性注意力，**定长循环状态**替代 KV cache；训练走 chunked 并行 |

### feedforward

| type | 来源 | 核心机制 |
|------|------|---------|
| `swiglu` | Llama | 经典门控 FFN |
| `geglu` | Gemma | 门控激活换成 GELU |
| `moe` | 通用 | top-k 路由 + aux loss 负载均衡 |
| `moe_shared` | DeepSeekMoE / Qwen3-Next | 常驻共享专家（`shared_experts`）+ N 个路由专家 |
| `moe_finegrained` | DeepSeek-V3 | 细粒度专家 + **aux-loss-free**（可学习 bias 均衡，替代 aux loss） |

### positional_encoding

| type | 来源 | 机制 |
|------|------|------|
| `rope` | RoPE / YaRN | 标准旋转位置编码，`inference_rope_scaling: true` 启用 YaRN 外推 |
| `partial_rope` | Qwen3-Next | 只对 head_dim 的前 `rotary_dim` 维施加旋转 |
| `nope` | 近两年研究 | 完全不施加位置编码，仅靠因果掩码 |

### norm

| type | 来源 | 机制 |
|------|------|------|
| `rmsnorm` | — | 标准 RMSNorm |
| `rmsnorm_zero_centered` | Qwen3-Next | 缩放系数用 `(1 + w)`，且 `w` 零初始化 |
| `layernorm` | 经典 | 减均值 + 缩放平移（比 RMSNorm 多一个 bias） |

---

## 三种用法

```python
# 1. 只改 YAML，跑现有训练脚本
#    cd trainer && python train_pretrain.py

# 2. 在代码里用配置
from configs import load_config, to_arch_config
from arch import build_model
arch = to_arch_config(load_config("configs/pretrain.yaml"))
model = build_model(arch)

# 3. 沿用旧参数（等价于重构前的结构，可加载旧权重）
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
```

---

## 新增一个组件

### 方式 A：放进本仓库

新建 `arch/attention/my_attn.py`：

```python
from ..registry import register

@register("attention", "my_attn")           # <- 注册后 YAML 里就能写 type: my_attn
class MyAttention(nn.Module):
    def __init__(self, cfg, positional=None):
        super().__init__()
        self.window = cfg.get("window_size", 512)    # 读 YAML 里的自定义参数
        self.positional = positional                 # 位置编码组件，由框架注入
        ...                                          # 你的实现

    def forward(self, x, position_embeddings, past_key_value=None,
                use_cache=False, attention_mask=None):
        ...                                          # 返回 (output, present_state)

    @staticmethod
    def state_seq_len(state) -> int:                 # 见下方「状态接口」
        return 0 if state is None else int(state[0].shape[1])
```

然后在 `arch/attention/__init__.py` 里 import 它（**必须导入，装饰器才会执行**），
YAML 里写：

```yaml
attention:
  type: my_attn
  window_size: 1024      # 自定义参数，框架不关心，组件自己读
```

### 方式 B：放在仓库外（不改本仓库）

`type` 支持完整 import 路径，注册表会自动回退到 import：

```yaml
attention:
  type: my_experiment.always_attend.HardAttention
```

---

## 组件接口约定

组件构造时收到的是「全局参数 + 槽内参数」合并后的配置对象，
用 `cfg.xxx` 读取、`cfg.get("xxx", 默认值)` 兜底。

| 槽位 | 约定 |
|------|------|
| `norm` | `cls(dim, eps=...)` |
| `positional_encoding` | `cls(cfg)`；`build_buffers()` 返回 `(cos, sin)` 或 `None`；`buffer_names` 指定 buffer 名（`None` 表示不需要）；`apply(q, k, cos, sin) -> (q, k)` |
| `attention` | `cls(cfg, positional=None)`；`forward(x, position_embeddings, past_key_value, use_cache, attention_mask) -> (output, present_state)`；可选 `state_seq_len(state) -> int` |
| `feedforward` | `cls(cfg)`；`forward(x) -> Tensor`。若为 MoE，额外暴露 `aux_loss`（模型会自动汇总所有带该属性的层） |

### 状态接口（线性注意力的关键）

注意力返回的 `present_state` 是**不透明的**：

- softmax 注意力（gqa / sliding_window / mla / compressed）返回 `(k, v)` 元组
- **线性注意力（deltanet）返回定长循环状态矩阵** —— 没有序列维

模型本身不解析状态内容，只在需要「已处理多少 token」时调用
`state_seq_len(state)`（见 `arch/attention/__init__.py::get_state_seq_len`）。
所以新增线性注意力时，`state_seq_len` **必须返回 0**，
否则模型算 `start_pos` 会出错。

### 位置编码注入

注意力层**不硬编码 RoPE**。模型先把位置编码组件构造好（`self.positional`），
再注入每个注意力层；注意力在 forward 里调 `self.positional.apply(...)` 委托给它。
因此 `nope` / `partial_rope` 才能真正即插即用。

---

## 兼容性保证

重构经过**位级数值对拍**验证：

- 参数量 63,912,192，`state_dict` 91 个键，与重构前逐一对应
- `load_state_dict(out/pretrain_768.pth, strict=True)` 零 missing / 零 unexpected
- 与重构前实现对比同一输入的 logits：**最大绝对差 0.000e+00**
- 旧公开名字全部保留：`MiniMindConfig` / `MiniMindForCausalLM` / `MiniMindModel` /
  `MiniMindBlock` / `RMSNorm` / `precompute_freqs_cis` / `apply_rotary_pos_emb` / `repeat_kv`

回归脚本：

```bash
python test/storage/verify_arch.py        # 与重构前实现对拍 + 权重可加载
python test/storage/verify_components.py  # 逐个组件的可用性 + 增量解码一致性
```

---

## 已知限制

1. **权重文件名不含架构信息**。命名仍是 `{阶段}_{hidden_size}{_moe}.pth`，
   只看 `hidden_size` 和 `use_moe`。若用配置换了注意力/层数/位置编码但这两个不变，
   新旧权重会重名而无法自动区分。
2. **位置编码改动不会报错**。RoPE 的 cos/sin 是非持久化 buffer，不进 `state_dict`，
   所以改了 `rope_theta` 之类，旧权重照常加载但数值行为已变。
3. **checkpoint 不保存架构配置**。`lm_checkpoint` 只存模型/优化器/进度，
   架构靠运行时重建，`--from_resume 1` 用的是**当前配置**而非训练时的配置。
4. **跨层 KV 共享（Gemma 4 风格）尚未接线**。它需要「层间调度」能力
   （让某层复用另一层的 KV），属于下一步的组装层设计；
   当前 attention 组件已具备暴露可复用 KV 的形态，缺的是调度侧接线。
5. **混合层间调度（如 3:1 局部的全局注意力）尚未支持**。当前所有层共用同一个
   attention 组件。这是 2025–2026 主流模型的标配模式，需要给配置加「按层/按周期」
   的调度概念，属于框架级改动。
