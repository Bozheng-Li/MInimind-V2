"""LoRA（Low-Rank Adaptation，低秩适配）。

只训练注意力与前馈投影上的低秩增量分支，主体权重全部冻结。因此：
- 优化器只吃 ``lora_params``，梯度裁剪也只作用于它们
- 存盘只存 LoRA 分支（几十 KB 而非 132 MB）
- ``torch.compile`` 默认关闭（运行时绑定的 LoRA forward 不适合重复图捕获）

标准公式为 ``y = Wx + (alpha/r) * BAx``。旧实现既没有 ``alpha/r`` 缩放，
又用“输入输出维度相等”筛层，会漏掉 GQA 的 K/V 和 FFN 投影；现在统一由
``model.model_lora`` 按模块名选择目标层。
"""
from __future__ import annotations

from torch import optim

from dataset.lm_dataset import SFTDataset

from ..base import Algorithm


class LoRAAlgorithm(Algorithm):
    name = "lora"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.lora_params = []
        if getattr(self.args, "use_compile", 0) == 1:
            self.args.use_compile = 0
            from trainer.trainer_utils import Logger
            Logger('LoRA训练时禁用torch.compile加速（monkey-patch 的 forward 与其不兼容）')

    def build_dataset(self):
        return SFTDataset(self.args.data_path, self.tokenizer, max_length=self.args.max_seq_len)

    def configure_model(self, model):
        """挂载 LoRA 并冻结非 LoRA 参数。必须早于 build_optimizer。"""
        from model.model_lora import apply_lora, iter_lora_parameters
        from trainer.trainer_utils import Logger

        matched = apply_lora(
            model,
            rank=self.args.lora_rank,
            alpha=self.args.lora_alpha,
            dropout=self.args.lora_dropout,
            target_modules=self.args.lora_target_modules,
        )

        total_params = sum(p.numel() for p in model.parameters())
        lora_count = sum(p.numel() for p in iter_lora_parameters(model))
        Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
        Logger(f"LoRA 参数量: {lora_count / 1e6:.3f} M")
        Logger(f"LoRA 参数占比: {lora_count / total_params * 100:.2f}%")
        Logger(f"LoRA 目标层: {len(matched)} 个；示例: {', '.join(matched[:6])}")

        for name, param in model.named_parameters():
            if '.lora.' in name:
                param.requires_grad = True
                self.lora_params.append(param)
            else:
                param.requires_grad = False
        return model

    def build_optimizer(self, model):
        return optim.AdamW(self.lora_params, lr=self.args.learning_rate)

    def clip_parameters(self, model):
        return self.lora_params

    def weight_prefix(self) -> str:
        return self.args.lora_name

    def save_weights(self, ctx, epoch, step):
        """只导出 LoRA 分支权重。"""
        from model.model_lora import save_lora

        moe_suffix = '_moe' if ctx.lm_config.use_moe else ''
        path = f'{ctx.args.save_dir}/{self.args.lora_name}_{ctx.lm_config.hidden_size}{moe_suffix}.pth'
        save_lora(ctx.model, path)

    def compute_loss(self, batch):
        input_ids = batch[0].to(self.runtime.device)
        labels = batch[1].to(self.runtime.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        res = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = (res.loss + res.aux_loss) / self.args.accumulation_steps
        return loss, res.aux_loss
