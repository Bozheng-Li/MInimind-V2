"""把 ``ArchConfig`` 包装成 transformers 能接受的 ``PretrainedConfig``。

``PreTrainedModel.__init__`` 强制要求 config 是 ``PretrainedConfig`` 实例，
所以 ``arch/`` 独立使用时需要一个能承载 ``ArchConfig`` 的 config 类。

同时把常用标量（hidden_size / vocab_size / num_hidden_layers ...）**镜像**成
普通属性，让 transformers 的通用机制（save_pretrained、GenerationMixin 等）
以及仓库里读取这些属性的脚本都能正常工作。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from transformers import PretrainedConfig

from .schema import ArchConfig, arch_config_from


class ArchPretrainedConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(self, arch: Optional[Any] = None, **kwargs: Any):
        arch = arch_config_from(arch)
        m = arch.model
        attn = arch.component("attention")
        ffn = arch.component("feedforward")

        # 先铺 derive 出来的标量，再让显式 kwargs 覆盖 —— 这样 from_pretrained
        # 回灌 to_dict() 的结果时不会出现「重复关键字」错误
        scalars = dict(
            vocab_size=m["vocab_size"],
            hidden_size=m["hidden_size"],
            num_hidden_layers=m["num_hidden_layers"],
            num_attention_heads=attn.num_attention_heads,
            num_key_value_heads=attn.get("num_key_value_heads"),
            head_dim=attn.head_dim,
            intermediate_size=ffn.get("intermediate_size"),
            max_position_embeddings=m["max_position_embeddings"],
            rms_norm_eps=m["rms_norm_eps"],
            tie_word_embeddings=bool(m["tie_word_embeddings"]),
        )
        scalars.update(kwargs)
        super().__init__(**scalars)

        # 放在 super().__init__ 之后，避免被 PretrainedConfig 的属性处理覆盖
        self.arch = arch

        # ---- 兼容旧脚本读取的属性（如 moe_suffix 逻辑、MoE 相关统计）----
        self.use_moe = arch.type_of("feedforward") == "moe"
        self.num_experts = ffn.get("num_experts")
        self.num_experts_per_tok = ffn.get("num_experts_per_tok")
        self.moe_intermediate_size = ffn.get("moe_intermediate_size")
        self.norm_topk_prob = ffn.get("norm_topk_prob")
        self.router_aux_loss_coef = ffn.get("router_aux_loss_coef")
        self.hidden_act = m.get("hidden_act", "silu")
        self.dropout = m.get("dropout", 0.0)
        self.rope_theta = arch.component("positional_encoding").get("rope_theta")
        self.inference_rope_scaling = bool(
            arch.component("positional_encoding").get("inference_rope_scaling", False)
        )

    # ------------------------------------------------------------------ #
    def to_arch_config(self) -> ArchConfig:
        return self.arch

    def to_dict(self) -> Dict[str, Any]:
        """把嵌套的架构配置一并序列化，便于 ``save_pretrained`` 与实验复现。"""
        output = super().to_dict()
        output["arch"] = self.arch.to_dict()
        return output
