"""按 ArchConfig 组装出的模型。

``ArchModel``       —— Transformer 主干（embedding + N×Block + final norm + 位置编码）
``ArchForCausalLM`` —— 加上 lm_head / 权重绑定 / loss / generate 的完整语言模型

本文件不依赖 ``model/`` 目录，因此 ``arch/`` 可以独立使用::

    from arch import build_model
    model = build_model(yaml_path="configs/base.yaml")
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from .attention import get_state_seq_len
from .build import build_block, build_norm_layer, build_positional
from .config_class import ArchPretrainedConfig
from .norm.rmsnorm_zero_centered import RMSNormZeroCentered
from .schema import ArchConfig, arch_config_from


def causal_lm_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """计算 next-token 交叉熵，并安全处理整批没有监督 token 的情况。

    SFT/LoRA 数据在很短的 ``max_seq_len`` 下可能把 assistant 回复完整截掉，
    此时 shift 后的标签全是 ``-100``。PyTorch 对“所有元素都被 ignore”的均值
    会返回 NaN；这里显式筛出有效位置，空集合则返回与 logits 计算图相连的 0，
    使 DDP/GradScaler 仍能完成一次合法的零梯度反向。
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid = shift_labels.ne(-100)
    if valid.any():
        return F.cross_entropy(shift_logits[valid], shift_labels[valid])
    return shift_logits.sum() * 0.0


class ArchModel(nn.Module):
    """Transformer 主干。子模块（norm / 位置编码 / 注意力 / 前馈）全部按配置构造。"""

    def __init__(self, config: Any = None):
        super().__init__()
        arch: ArchConfig = arch_config_from(config)
        self.arch_config = arch
        self.config = config

        self.vocab_size = arch.model["vocab_size"]
        self.num_hidden_layers = arch.model["num_hidden_layers"]
        hidden = arch.model["hidden_size"]

        self.embed_tokens = nn.Embedding(self.vocab_size, hidden)
        self.dropout = nn.Dropout(arch.model["dropout"])
        self.norm = build_norm_layer(arch, hidden)

        # ---- 位置编码（必须先于各层构建，因为它要注入每个注意力层）----
        self.positional = build_positional(arch)
        buffers = self.positional.build_buffers() if hasattr(self.positional, "build_buffers") else None
        buffer_names = getattr(self.positional, "buffer_names", ("freqs_cos", "freqs_sin"))
        if buffers is not None:
            for name, value in zip(buffer_names, buffers):
                # persistent=False：与重构前一致，不进 state_dict
                self.register_buffer(name, value, persistent=False)

        self.layers = nn.ModuleList(
            [build_block(arch, positional=self.positional) for _ in range(self.num_hidden_layers)]
        )

    # ------------------------------------------------------------------ #
    def _position_embeddings(self, hidden_states, start_pos: int, seq_length: int):
        buffer_names = getattr(self.positional, "buffer_names", None)
        if buffer_names is None:
            # 非 buffer 型位置编码：直接向组件索取当前需要的张量
            return self.positional.build_buffers()

        cos, sin = getattr(self, buffer_names[0]), getattr(self, buffer_names[1])
        # transformers>=5 的 meta-device 初始化会丢掉 buffer，这里按需重算
        if cos[0, 0] == 0:
            cos, sin = self.positional.build_buffers()
            cos, sin = cos.to(hidden_states.device), sin.to(hidden_states.device)
            setattr(self, buffer_names[0], cos)
            setattr(self, buffer_names[1], sin)
        return cos[start_pos:start_pos + seq_length], sin[start_pos:start_pos + seq_length]

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        # 通过组件的 state_seq_len 询问已缓存长度，而非硬读 KV 形状 ——
        # 线性注意力的状态是定长矩阵、没有序列维
        start_pos = get_state_seq_len(self.layers[0].self_attn, past_key_values[0])
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = self._position_embeddings(hidden_states, start_pos, seq_length)

        presents = []
        #: 逐层残差流的 RMS（训练时记录，用于观察深层是否数值退化/爆炸）
        self.last_hidden_rms = [0.0] * len(self.layers) if self.training else None
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)
            if self.training:
                with torch.no_grad():
                    self.last_hidden_rms[i] = float(hidden_states.detach().float().pow(2).mean().sqrt())

        hidden_states = self.norm(hidden_states)
        # 任何暴露 aux_loss 的层（如 MoE）都参与汇总；非 MoE 时为零张量
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if hasattr(l.mlp, "aux_loss")],
            hidden_states.new_zeros(1).squeeze(),
        )
        return hidden_states, presents, aux_loss


class ArchForCausalLM(PreTrainedModel, GenerationMixin):
    """加上 lm_head / 权重绑定 / 交叉熵 loss / 自研 generate 的完整模型。

    子类可覆盖 ``config_class``（例如
    ``MiniMindForCausalLM.config_class = MiniMindConfig``）。
    """

    config_class = ArchPretrainedConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def _init_weights(self, module):
        """跳过零中心 RMSNorm 的默认初始化。

        transformers 的默认 ``_init_weights`` 按**类名**匹配，把类名含 "RMSNorm"
        的模块权重一律 ``fill_(1.0)``。我们的 ``RMSNorm`` 恰好要 1.0 所以无碍，
        但 ``RMSNormZeroCentered`` 的语义是 ``(1 + w)``，必须是 0.0 ——
        被填成 1.0 后缩放变成 2.0，静默破坏该组件的设计。

        其余模块仍走父类默认逻辑（保持与重构前完全一致的初始化行为）。
        """
        if isinstance(module, RMSNormZeroCentered):
            module._is_hf_initialized = True
            return
        super()._init_weights(module)

    def __init__(self, config: Any = None):
        # 允许直接塞 ArchConfig / dict / 旧参数对象，这里统一包装成 PretrainedConfig
        if not isinstance(config, PretrainedConfig):
            config = ArchPretrainedConfig(config)
        self.config = config
        super().__init__(config)
        self.model = ArchModel(config)
        self.lm_head = nn.Linear(
            self.model.arch_config.model["hidden_size"],
            self.model.arch_config.model["vocab_size"],
            bias=False,
        )
        if self.model.arch_config.model["tie_word_embeddings"]:
            # 与 embedding 共享同一张量（state_dict 里两个键都会出现）
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = causal_lm_cross_entropy(logits, labels)
        return MoeCausalLMOutputWithPast(
            loss=loss, aux_loss=aux_loss, logits=logits,
            past_key_values=past_key_values, hidden_states=hidden_states,
        )

    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer:
            streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = get_state_seq_len(self.model.layers[0].self_attn, past_key_values[0]) if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i])
                    score = logits[i, seen]
                    logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None:
                next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer:
                streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break
        if streamer:
            streamer.end()
        if kwargs.get("return_kv"):
            return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
