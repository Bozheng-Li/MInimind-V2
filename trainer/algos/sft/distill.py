"""知识蒸馏（白盒 KD）。

学生不只学教师的最终输出，还拟合教师的 token 分布：

    Loss = alpha * CE + (1 - alpha) * T^2 * KL(teacher^T || student^T)

教师默认是 MoE、学生是 Dense（见 ``--teacher_use_moe`` / ``--student_use_moe``）。
loss 与日志格式均与重构前的 ``train_distillation.py`` 保持一致。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from dataset.lm_dataset import SFTDataset
from model.model_minimind import MiniMindConfig

from ..base import Algorithm


def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return (temperature ** 2) * kl


class DistillAlgorithm(Algorithm):
    name = "distill"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.teacher_model = None
        self._ce_raw = 0.0
        self._distill = 0.0

    def build_dataset(self):
        return SFTDataset(self.args.data_path, self.tokenizer, max_length=self.args.max_seq_len)

    def extra_models(self):
        """教师模型：结构与权重都由 ``--teacher_*`` 参数决定。"""
        from trainer.trainer_utils import Logger, init_model

        lm_config_teacher = MiniMindConfig(hidden_size=self.args.teacher_hidden_size,
                                           num_hidden_layers=self.args.teacher_num_layers,
                                           use_moe=bool(self.args.teacher_use_moe))
        teacher_model, _ = init_model(lm_config_teacher, self.args.from_teacher_weight,
                                      save_dir=self.args.save_dir, device=self.runtime.device)
        if self.args.from_teacher_weight == "none":
            # 主模型稍后由 DDP 同步，但 teacher 不会进入 DDP；从零蒸馏时必须显式
            # 广播，否则每张卡会用不同的随机教师产生互相矛盾的 KL 目标。
            from trainer.common.runtime import synchronize_module
            synchronize_module(teacher_model)
        teacher_model.eval()
        teacher_model.requires_grad_(False)
        Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')
        return {"teacher_model": teacher_model}

    def on_epoch_start(self, loader, iters):
        if self.teacher_model is not None:
            self.teacher_model.eval()
            self.teacher_model.requires_grad_(False)

    def compute_loss(self, batch):
        dev = self.runtime.device
        input_ids = batch[0].to(dev)
        labels = batch[1].to(dev)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        # loss mask：只在 assistant 回复的位置上计算（SFT 数据集已把其余位置置为 -100）
        loss_mask = (labels[..., 1:] != -100).float()

        # --- 学生前向 ---
        res = self.model(input_ids, attention_mask=attention_mask)
        student_logits = res.logits[..., :-1, :].contiguous()

        # --- 教师前向（冻结、no_grad，并按学生词表截断）---
        teacher_logits = None
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_logits = self.teacher_model(
                    input_ids, attention_mask=attention_mask
                ).logits[..., :-1, :].contiguous()
                teacher_logits = teacher_logits[..., :student_logits.size(-1)]

        # --- 1) Ground-Truth 交叉熵 ---
        shift_labels = labels[..., 1:].contiguous()
        loss_mask_flat = loss_mask.view(-1).bool()
        student_flat = student_logits.view(-1, student_logits.size(-1))
        labels_flat = shift_labels.view(-1)
        if loss_mask_flat.any():
            ce_loss_raw = F.cross_entropy(student_flat[loss_mask_flat],
                                          labels_flat[loss_mask_flat])
        else:
            # 极端截断样本可能没有 assistant token。返回与学生图相连的 0，避免
            # cross_entropy(all ignore_index) 产生 NaN 并污染整个优化器状态。
            ce_loss_raw = student_flat.sum() * 0.0
        # 学生的 MoE 负载均衡项也计入 CE 侧（与重构前一致）
        ce_loss = ce_loss_raw + res.aux_loss if self.lm_config.use_moe else ce_loss_raw

        # --- 2) 白盒蒸馏 ---
        if teacher_logits is not None and loss_mask_flat.any():
            distill_loss = distillation_loss(
                student_flat[loss_mask_flat],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat],
                temperature=self.args.temperature,
            )
        else:
            distill_loss = torch.tensor(0.0, device=dev)

        # --- 3) 总损失 = alpha * CE + (1 - alpha) * Distill ---
        loss = (self.args.alpha * ce_loss + (1 - self.args.alpha) * distill_loss) / self.args.accumulation_steps

        self._ce_raw = ce_loss_raw.item()
        self._distill = distill_loss.item()
        aux = res.aux_loss if self.lm_config.use_moe else None
        return loss, aux

    def format_log(self, *, epoch, step, iters, start_step, loss_scaled, aux_loss, lr, spend_time):
        """重构前蒸馏脚本的日志：单独打 ``ce`` 与 ``distill``，eta 保留 3 位。"""
        total_loss = loss_scaled * self.args.accumulation_steps
        aux = float(aux_loss) if aux_loss is not None else 0.0
        eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
        line = (f'Epoch:[{epoch + 1}/{self.args.epochs}]({step}/{iters}), '
                f'loss: {total_loss:.4f}, ce: {self._ce_raw:.4f}, aux_loss: {aux:.4f}, '
                f'distill: {self._distill:.4f}, learning_rate: {lr:.8f}, epoch_time: {eta_min:.3f}min')
        metrics = {"loss": total_loss, "ce_loss": self._ce_raw, "aux_loss": aux,
                   "distill_loss": self._distill, "learning_rate": lr, "epoch_time": eta_min}
        return line, metrics
