"""QLoRA：4-bit NF4 量化基座 + 可训练 LoRA。

QLoRA 的关键不是“LoRA 前加一个 q”这么简单，而是：

- 冻结基座的线性层以 NF4 4-bit 保存，显著降低权重显存；
- 可选 double quantization，继续压缩量化常数；
- LoRA 分支保持浮点可训练，优化器只接收适配器参数；
- ``lm_head`` 与 embedding 的共享权重不量化，避免破坏 tied weights。

当前实现先按普通方式加载基座权重，再原地量化，因此训练期显存低，但模型初始化
瞬间仍需容纳一份浮点权重。若要加载百亿级外部模型，应改用分片量化加载器。
"""
from __future__ import annotations

import torch

from trainer.trainer_utils import Logger

from .lora import LoRAAlgorithm


class QLoRAAlgorithm(LoRAAlgorithm):
    """标准 QLoRA 监督微调。"""

    name = "qlora"

    def configure_model(self, model):
        from model.model_lora import quantize_model_4bit

        if "cuda" not in str(self.runtime.device):
            raise RuntimeError("QLoRA 的 NF4 训练需要 CUDA；CPU 上请改用 --algo lora")
        dtype = torch.bfloat16 if self.args.qlora_compute_dtype == "bfloat16" else torch.float16
        replaced = quantize_model_4bit(
            model,
            compute_dtype=dtype,
            quant_type=self.args.qlora_quant_type,
            use_double_quant=bool(self.args.qlora_double_quant),
        )
        Logger(f"QLoRA 已量化 {len(replaced)} 个线性层；量化类型={self.args.qlora_quant_type}，"
               f"计算精度={self.args.qlora_compute_dtype}，double_quant={bool(self.args.qlora_double_quant)}")
        # 量化必须发生在挂 LoRA 之前，否则会把适配器内部的 A/B 也错误量化。
        return super().configure_model(model)

    def build_optimizer(self, model):
        if self.args.qlora_optimizer == "paged_adamw_8bit":
            try:
                import bitsandbytes as bnb
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("paged_adamw_8bit 需要 bitsandbytes") from exc
            return bnb.optim.PagedAdamW8bit(self.lora_params, lr=self.args.learning_rate)
        return super().build_optimizer(model)

    def restore_model_state(self, model, state_dict):
        """QLoRA 续训只恢复适配器。

        4-bit 基座冻结不变，启动时已从 ``from_weight`` 重新量化。bitsandbytes 在
        state_dict 中附加的 absmax/quant_map 不是普通参数，直接 strict load 会被
        PyTorch 判为 unexpected keys；只恢复 LoRA 也避免量化后端版本差异。
        """
        adapter_state = {key: value for key, value in state_dict.items() if ".lora." in key}
        incompatible = model.load_state_dict(adapter_state, strict=False)
        missing_adapter = [key for key in incompatible.missing_keys if ".lora." in key]
        if missing_adapter:
            raise RuntimeError(f"QLoRA 续训档缺少适配器参数: {missing_adapter[:5]}")
