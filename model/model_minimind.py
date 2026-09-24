"""MiniMind 模型的公开接口（兼容层）。

模型的实际实现已迁移到 ``arch/`` 包 —— 那里按「归一化 / 位置编码 / 注意力 / 前馈」
四个可插拔槽位组织，并由 ``configs/base.yaml`` 驱动组装。

本文件保留全部公开名字与行为，因此：
- 现有训练脚本（``trainer/train_*.py``）与推理脚本**无需改动**
- 已训练的 ``test/out/pretrain_768.pth`` 等权重仍可 ``strict=True`` 加载
- ``MiniMindForCausalLM`` 仍可被继承（如 ``train_ppo.py`` 的 ``CriticModel``）

新增架构实验请优先改 ``configs/base.yaml``，或参考 ``arch/README.md`` 添加组件；
只有需要新的对外配置项时才动本文件。
"""
import math

from transformers import PretrainedConfig

from arch.attention.gqa import repeat_kv
from arch.model import ArchForCausalLM, ArchModel
from arch.block import ArchBlock
from arch.norm.rmsnorm import RMSNorm
from arch.positional.rope import apply_rotary_pos_emb, precompute_freqs_cis
from arch.schema import (
    ArchConfig,
    arch_config_from,
    load_arch_config,
    synthesize_arch_config,
)

__all__ = [
    "MiniMindConfig",
    "MiniMindForCausalLM",
    "MiniMindModel",
    "MiniMindBlock",
    "RMSNorm",
    "precompute_freqs_cis",
    "apply_rotary_pos_emb",
    "repeat_kv",
]


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

    # ------------------------------------------------------------------ #
    #                 可组合架构（arch/）接口
    # ------------------------------------------------------------------ #
    def to_arch_config(self) -> ArchConfig:
        """把本 config 翻译成 ``ArchConfig``，供 ``arch/`` 组装模型。

        若本对象来自 ``from_yaml``（带有完整槽位信息），直接返回原配置，
        避免把 YAML 里的自定义槽位参数丢失。
        """
        bound = getattr(self, "_arch_config", None)
        if isinstance(bound, ArchConfig):
            return bound
        return synthesize_arch_config(self)

    @classmethod
    def from_arch_config(cls, arch: ArchConfig) -> "MiniMindConfig":
        """用 ``ArchConfig`` 构造一个 MiniMindConfig。

        标量字段会被**镜像**成旧属性，让依赖它们的代码（如 ``moe_suffix`` 逻辑、
        ``convert_model.py``）继续正常工作；完整槽位则留存在 ``_arch_config``。
        """
        model = arch.model
        attn = arch.component("attention")
        ffn = arch.component("feedforward")
        pe = arch.component("positional_encoding")

        cfg = cls(
            hidden_size=model["hidden_size"],
            num_hidden_layers=model["num_hidden_layers"],
            # 只要是 MoE 家族（moe / moe_shared / moe_finegrained）就置位 ——
            # 这个标志决定权重文件名是否带 `_moe` 后缀，写死 == "moe" 会让
            # moe_finegrained 的权重被存成非 MoE 的名字
            use_moe=arch.type_of("feedforward").startswith("moe"),
            vocab_size=model["vocab_size"],
            max_position_embeddings=model["max_position_embeddings"],
            rms_norm_eps=model["rms_norm_eps"],
            dropout=model.get("dropout", 0.0),
            hidden_act=model.get("hidden_act", "silu"),
            tie_word_embeddings=model.get("tie_word_embeddings", True),
            bos_token_id=model.get("bos_token_id", 1),
            eos_token_id=model.get("eos_token_id", 2),
            num_attention_heads=attn.num_attention_heads,
            num_key_value_heads=attn.get("num_key_value_heads"),
            head_dim=attn.head_dim,
            intermediate_size=ffn.get("intermediate_size"),
            num_experts=ffn.get("num_experts"),
            num_experts_per_tok=ffn.get("num_experts_per_tok"),
            moe_intermediate_size=ffn.get("moe_intermediate_size"),
            norm_topk_prob=ffn.get("norm_topk_prob"),
            router_aux_loss_coef=ffn.get("router_aux_loss_coef"),
            rope_theta=pe.get("rope_theta", 1e6),
            inference_rope_scaling=bool(pe.get("inference_rope_scaling", False)),
        )
        cfg._arch_config = arch
        return cfg

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MiniMindConfig":
        """从 ``config.yaml`` 构造 config，即可用新架构跑现有训练脚本::

            lm_config = MiniMindConfig.from_yaml('configs/base.yaml')
        """
        return cls.from_arch_config(load_arch_config(yaml_path))

    def to_dict(self):
        """覆写父类：把内部持有的 ``ArchConfig`` 转成可 JSON 序列化的嵌套 dict。

        ``PretrainedConfig.to_dict()`` 会被 ``save_pretrained``（如
        ``scripts/convert_model.py``）调用；若不处理，``_arch_config`` 这个
        自定义对象会导致 ``TypeError: not JSON serializable``。
        """
        output = super().to_dict()
        arch = output.pop("_arch_config", None)
        if isinstance(arch, ArchConfig):
            output["arch"] = arch.to_dict()
        return output


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindForCausalLM(ArchForCausalLM):
    """MiniMind 因果语言模型。实现见 ``arch/model.py::ArchForCausalLM``。"""

    config_class = MiniMindConfig


# 保持旧名字可用（``ArchModel`` / ``ArchBlock`` 接受同样的 config 对象）
MiniMindModel = ArchModel
MiniMindBlock = ArchBlock
