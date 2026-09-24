<div align="center">

<img src="./images/logo.png" alt="MiniMind-V2" width="600">

### Minimalist · Modular · End-to-End Tiny LLM Framework

**Swap attention with a few lines of YAML ｜ Switch training algorithms with a single `--algo`**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Datasets-ffd21e.svg?style=flat-square)](https://huggingface.co/datasets/jingyaogong/minimind_dataset)
[![ModelScope](https://img.shields.io/badge/ModelScope-Datasets-624aff.svg?style=flat-square)](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=flat-square)](https://github.com/Bozheng-Li/MInimind-V2/pulls)

<p align="center">
  <a href="#quickstart">⚡ Quick Start</a> •
  <a href="#architecture">🧩 Architecture</a> •
  <a href="#algorithms">🚀 16 Algorithms</a> •
  <a href="#experiments">📊 Experiments</a> •
  <a href="#webui">🖥️ WebUI</a> •
  <a href="#datasets">📦 Datasets</a> •
  <a href="./README.md">🇨🇳 中文版</a>
</p>

<!-- Live Demo -->
<img src="./images/minimind-3.gif" alt="MiniMind Live Chat Demo" width="100%">

</div>

---

### ✨ Key Features

<table align="center" width="100%">
  <tr>
    <td width="50%">
      <h4>🧩 4-Slot Modular Architecture</h4>
      <p>Decoupled Attention · FFN · Positional · Norm slots. 18 registered components freely assembled via YAML (GQA, MLA, SwiGLU, DeepSeekMoE, etc.). Register new components with a single <code>@register</code> decorator.</p>
    </td>
    <td width="50%">
      <h4>🚀 16 Unified Training Algorithms</h4>
      <p>Pretrain · SFT · LoRA/QLoRA · Distill · DPO/KTO/SimPO · PPO/GRPO/DAPO · Agent RL. Switched seamlessly via <code>--algo</code> with unified optimization pipelines.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h4>📊 Rigorous Iso-FLOPs Benchmarks</h4>
      <p>18 architectural sweeps under identical token budgets (61M tokens & 1.39B SFT arms). Real Pareto frontiers and KV Cache efficiency without hidden implementation bias.</p>
    </td>
    <td width="50%">
      <h4>🖥️ Zero-Dependency Native WebUI</h4>
      <p>Pure FastAPI + vanilla JS/CSS without npm/node or CDN dependencies. Unified Chat terminal (<code>/</code>), Lab comparison matrix (<code>/lab</code>), and live Training dashboard (<code>/train</code>).</p>
    </td>
  </tr>
</table>

---

<a id="quickstart"></a>
## ⚡ Quick Start in 30 Seconds

### 1. Installation

```bash
git clone https://github.com/Bozheng-Li/MInimind-V2.git
cd MInimind-V2
pip install -r requirements.txt  # Install PyTorch matching your CUDA setup
```

### 2. Prepare Datasets

Download the [mini pretrain set (1.2GB)](https://huggingface.co/datasets/jingyaogong/minimind_dataset) and [mini SFT set (1.6GB)](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files):

```bash
dataset/pretrain/pretrain_t2t_mini.jsonl
dataset/sft/sft_t2t_mini.jsonl
```

### 3. Train & Evaluate

```bash
# 1. Pretraining
python trainer/train.py --algo pretrain --config configs/pretrain.yaml

# 2. Supervised Fine-Tuning (SFT)
python trainer/train.py --algo sft --config configs/sft.yaml

# 3. Interactive CLI Inference
python trainer/eval.py --config configs/sft.yaml --weight full_sft
```

> 💡 **Switching to Gated Attention + Fine-grained MoE**:
> ```bash
> python trainer/train.py --algo pretrain --config configs/pretrain_moe.yaml
> python trainer/train.py --algo sft      --config configs/sft_moe.yaml
> ```

---

<a id="architecture"></a>
## 🧩 Modular 4-Slot Architecture

The model is partitioned into four standard slots. All components reside under `arch/` and are configured via declarative YAML:

<table align="center" width="100%">
  <tr>
    <td align="center" width="50%"><b>Dense Model (GQA + SwiGLU)</b></td>
    <td align="center" width="50%"><b>MoE Model (Gated + Finegrained MoE)</b></td>
  </tr>
  <tr>
    <td align="center"><img src="./images/LLM-structure.jpg" width="100%" alt="Dense Architecture"></td>
    <td align="center"><img src="./images/LLM-structure-moe.jpg" width="100%" alt="MoE Architecture"></td>
  </tr>
</table>

### 18 Registered Components

| Slot | `type` | Provenance | Description |
|---|---|---|---|
| **Attention** | `gqa` | Llama / Qwen | Grouped-Query Attention with QK-Norm; KV heads = 1 yields MQA |
| | `gated` | Qwen3-Next | Output gating + zero-centered QK-Norm; supports partial RoPE |
| | `sliding_window` | Mistral / Gemma | Local sliding window attention limited by `window_size` |
| | `mla` | DeepSeek-V2/V3 | Multi-Head Latent Attention (224B/token vs GQA 768B cache) |
| | `compressed` | DeepSeek-V4 | Strided token compression along sequence dimension (384B/token) |
| | `deltanet` | Qwen3-Next | Linear attention with constant-size recurrent state |
| **FFN** | `swiglu` / `geglu` | Llama / Gemma | Gated FFN activated by SiLU or GELU |
| | `moe` | Standard MoE | Top-k routing with auxiliary balance loss |
| | `moe_shared` | DeepSeekMoE | Shared permanent expert + routed sparse experts |
| | `moe_finegrained` | DeepSeek-V3 | Fine-grained experts with learnable bias balancing (aux-free) |
| **Positional** | `rope` | RoFormer | Rotary Positional Embedding with YaRN dynamic context scaling |
| | `partial_rope` / `nope` | — | RoPE on leading dims / No Positional Embedding |
| **Norm** | `rmsnorm` | Llama | Standard Root Mean Square Normalization |
| | `rmsnorm_zero_centered`| Qwen3-Next | Zero-initialized weight equivalent to residual identity |
| | `layernorm` | Transformer | Standard LayerNorm baseline |

### Architecture Efficiency: KV Cache vs. Parameters

<div align="center">
  <img src="./images/arch_cache_params.png" width="95%" alt="Cache and Parameters Comparison">
</div>

```yaml
# configs/experiments/mla_finegrained.yaml
attention:
  type: mla
  num_attention_heads: 8
  num_key_value_heads: 4
ffn:
  type: moe_finegrained
  num_experts: 8
  num_experts_per_tok: 2
```

---

<a id="algorithms"></a>
## 🚀 16 Unified Training Algorithms

All algorithms are implemented under `trainer/algos/`, spanning next-token pretraining, preference alignment, and multi-turn tool-calling reinforcement learning.

<div align="center">
  <img src="./images/rl-structure.jpg" width="95%" alt="MiniMind RL Training System">
</div>

### Algorithm Matrix

| Phase | Algorithm | Extra Dependencies | Target Scenarios |
|---|---|---|---|
| **Pretraining** | `pretrain` | — | Self-supervised next-token language modeling |
| **Supervised SFT** | `sft` | — | Full-parameter instruction fine-tuning |
| **Parameter-Efficient** | `lora` · `qlora` | bitsandbytes | Low memory footprint with NF4 double quantization |
| **Distillation** | `distill` | Teacher model | White-box distribution distillation matching teacher logits |
| **Offline Alignment** | `dpo` · `ipo` · `kto` | Reference model | Pairwise preference optimization (chosen / rejected) |
| **Reference-Free** | `simpo` · `cpo` · `orpo` | — | Saves 50% model memory via direct margin loss |
| **Online RL (Critic-Free)** | `grpo` · `rloo` | Reward + Reference | Rule-verifiable tasks (math, code), low memory overhead |
| | `dapo` | Reward | Clip-Higher with adaptive token-level loss objective |
| **Actor-Critic RL** | `ppo` | Reward + Ref + Critic | Standard generalized PPO reinforcement learning |
| **Agent Tool RL** | `agent` | Reward + Tool Env | Multi-turn Tool-Call reinforcement learning |

> 📌 **Key Notes**:
> - **CISPO is the default online policy objective** (`--loss_type cispo`), eliminating gradient cancellation from symmetric clipping.
> - **Distributed multi-GPU training**:
>   ```bash
>   torchrun --nproc_per_node 4 trainer/train.py --algo grpo --config configs/rl.yaml
>   ```

---

<a id="experiments"></a>
## 📊 Empirical Findings & Loss Gallery

Evaluated across 18 distinct architectural configurations under identical token budgets (61M tokens) with identical random seeds and data distributions.

<table align="center" width="100%">
  <tr>
    <td align="center" width="52%"><b>Benchmark Radar Matrix</b></td>
    <td align="center" width="48%"><b>Compute vs. Loss Pareto Frontier</b></td>
  </tr>
  <tr>
    <td align="center"><img src="./images/benchmark_radar.jpg" width="100%" alt="Benchmark Radar"></td>
    <td align="center"><img src="./images/arch_pareto.png" width="100%" alt="Pareto Frontier"></td>
  </tr>
</table>

### Full Lifecycle Training Loss Gallery

<table align="center" width="100%">
  <tr>
    <td align="center" width="50%"><b>① Pretrain Convergence Curve</b><br><img src="./images/pretrain_loss.jpg" width="100%" alt="Pretrain Loss"></td>
    <td align="center" width="50%"><b>② SFT Instruction Tuning Curve</b><br><img src="./images/sft_loss.jpg" width="100%" alt="SFT Loss"></td>
  </tr>
  <tr>
    <td align="center" width="50%"><b>③ GRPO Online RL Convergence</b><br><img src="./images/grpo_loss.jpg" width="100%" alt="GRPO Loss"></td>
    <td align="center" width="50%"><b>④ Agent Multi-Turn Tool RL Curve</b><br><img src="./images/agent_rl_loss.jpg" width="100%" alt="Agent RL Loss"></td>
  </tr>
</table>

<details>
<summary>🔍 Click to expand: PPO Loss Curves & RoPE Context Extrapolation</summary>

<table align="center" width="100%">
  <tr>
    <td align="center" width="50%"><b>PPO Multi-Metric Convergence</b><br><img src="./images/ppo_loss.jpg" width="100%" alt="PPO Loss"></td>
    <td align="center" width="50%"><b>RoPE Context Scaling vs. PPL</b><br><img src="./images/rope_ppl.png" width="100%" alt="RoPE PPL"></td>
  </tr>
</table>

</details>

### Top 5 Empirical Takeaways

1. **Gated Attention is a robust win**: Consistently yields Δloss −0.055 gain, reaching baseline loss with 0.84× steps (adopted in main config).
2. **MoE requires iso-compute accounting**: In SFT, MoE (214M / 80M active) reduces validation loss from 0.6370 to 0.5523 (−0.0847) vs Dense (64M), with 22% wall-clock throughput.
3. **Avoid aggressive cache compression on tiny models**: At 64M scale, MLA and compressed attention reduce parameter capacity; best suited for multi-billion parameter models.
4. **RoPE remains the gold standard**: `rope_theta=1.0e+6` is optimal; eliminating positional encodings (`nope`) severely hurts accuracy.
5. **Standard benchmark saturation**: Post-training algorithms on 64M models require multi-turn tool calling and specialized environments to demonstrate separation.

---

<a id="webui"></a>
## 🖥️ Native Full-Stack WebUI

Built with zero frontend build dependencies (FastAPI + native HTML/JS/CSS). **No node/npm, no bundlers, no CDN requirement**.

<table align="center" width="100%">
  <tr>
    <td align="center" width="50%"><b>Chat & Tool-Call Terminal (<code>/</code>)</b><br><img src="./images/agent_webui.jpg" width="100%" alt="Chat & Tool Call WebUI"></td>
    <td align="center" width="50%"><b>Model Evaluation Matrix (<code>/lab</code>)</b><br><img src="./webui/screenshots/lab_eval_viz.png" width="100%" alt="Lab Evaluation Matrix"></td>
  </tr>
</table>

```bash
# Launch server (default port 7860)
python webui/server.py

# Launch and load weights interactively inside the interface
python webui/server.py --no-load
```

| Route | Role | Capabilities |
|---|---|---|
| `http://localhost:7860/` | **Chat Terminal** | Load any native `.pth` architecture, streaming text, `<think>` reasoning display, tool-call execution |
| `http://localhost:7860/lab` | **Model Lab** | Cross-architecture comparison, 13 benchmark radar charts, Pareto efficiency scatter plots |
| `http://localhost:7860/train` | **Train Monitor** | Live loss, token throughput, and GPU VRAM monitoring |

---

<a id="datasets"></a>
## 📦 Datasets & Downloads

<div align="center">
  <img src="./images/dataset.jpg" width="100%" alt="Dataset Distribution">
</div>

<p align="center">
  <a href="https://huggingface.co/datasets/jingyaogong/minimind_dataset"><img src="./images/with_huggingface.png" height="40" alt="HuggingFace"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files"><img src="./images/with_modelscope.png" height="40" alt="ModelScope"></a>
</p>

| Stage | File | Size | Samples / Description |
|---|---|---|---|
| **Pretraining** | `pretrain_t2t.jsonl` | 10.0 GB | 8.47M samples, ~3.2B tokens uncurated corpus |
| | `pretrain_t2t_mini.jsonl` | 1.2 GB | 1.27M samples, starter pretraining corpus |
| **SFT** | `sft_t2t.jsonl` | 14.0 GB | 5.11M samples covering general conversation, QA, code, and tool calls |
| | `sft_t2t_mini.jsonl` | 1.6 GB | 906K samples, rapid verification instruction set |
| **RLHF** | `dpo.jsonl` · `rlaif.jsonl` | ~1.5 GB | Pairwise preference alignment datasets |
| | `agent_rl.jsonl` · `agent_rl_math.jsonl` | ~200 MB | Multi-turn environment tool-calling and math reasoning |

Arbitrary dataset mixtures are supported in pretraining YAML:
```yaml
data:
  path: dataset/pretrain
  mix:
    pretrain_t2t.prose.jsonl: 0.70
    pretrain_t2t.qa.jsonl:    0.10
    pretrain_t2t.en.jsonl:    0.15
    pretrain_t2t.code.jsonl:  0.05
```

---

## ⚡ Inference, Serving & Ecosystem

```bash
# 1. Start an OpenAI-compatible API server (with reasoning_content and tool_calls)
python scripts/serve_openai_api.py --config configs/sft.yaml --weight full_sft

# 2. Export native PyTorch checkpoint to HuggingFace Transformers format
python scripts/convert_model.py --torch_path test/full_sft.pth --output_dir ./minimind-hf

# 3. Run automated multi-turn tool-calling benchmarks
python scripts/agent_eval.py
```

Seamlessly exports to **vLLM**, **llama.cpp**, and **Ollama** inference runtimes.

---

<details>
<summary>📂 <b>Click to expand: Repository Structure & Troubleshooting</b></summary>

### Repository Layout

```text
arch/        Four slots × 18 components (attention / ffn / positional / norm)
configs/     Hierarchical YAML configs and inheritance resolver
trainer/     Unified training engine, 16 algorithm modules, scaffolding
model/       Legacy compatibility layer for strict checkpoint loading
dataset/     Dataset loading abstractions and metadata
webui/       Native zero-dependency WebUI (Chat / Lab / Train Monitor)
scripts/     API server, checkpoint converters, evaluation suites
tests/       Exact numerical regression and unit tests
test/        Local artifacts (weights, logs, reports; git-ignored)
```

### Tips & Pitfalls

1. **Checkpoints cannot cross incompatible architectures**: MLA and GQA have different projection shapes; set `--from_weight none` when changing architecture.
2. **Variable-length RL rollouts require FlashAttention**: Set `attention.flash_attn_masked: true` to avoid PyTorch SDPA falling back to eager quadratic memory.
3. **Scientific notation in YAML must include signs**: Write `1.0e+6` rather than `1.0e6` to avoid string parsing.
4. **`update_ratio` magnitude**: Parameter delta metric `rms(Δw) / rms(w)` normally ranges between 1e-4 and 1e-3.

</details>

---

## 🙏 Acknowledgements & Citations

MiniMind-V2 builds upon and extends the foundational work of [MiniMind](https://github.com/jingyaogong/minimind). We express deep gratitude to **Jingyao Gong** for open-sourcing the original design.

If you use MiniMind-V2 in your research or project, please cite:

```bibtex
@misc{minimind,
  title  = {MiniMind: Train a Tiny LLM from Scratch},
  author = {Jingyao Gong},
  year   = {2024},
  url    = {https://github.com/jingyaogong/minimind}
}
```

## 📄 License

This repository is licensed under the [Apache 2.0 License](LICENSE).
