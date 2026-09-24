"""离线偏好优化算法族：DPO、IPO、SimPO、CPO、ORPO 与 KTO。

这些算法都读取同一份 chosen/rejected 偏好对，差别集中在目标函数：

- DPO：策略相对 reference 的偏好间隔做二分类；
- IPO：把 DPO 的 logistic 目标改为有有限最优点的平方损失；
- SimPO：不需要 reference，用长度归一化 log-prob 与目标 margin；
- CPO：reference-free 偏好损失加 chosen 的 SFT 约束；
- ORPO：chosen SFT 损失加序列 odds-ratio 偏好约束；
- KTO：分别优化“合意/不合意”样本的前景理论效用，允许二者权重不同。

统一实现的好处是 mask、序列 log-prob 和指标口径只有一份，不会出现某个算法
漏掉 padding、另一个算法又多做一次 shift 的隐蔽差异。
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from dataset.lm_dataset import DPODataset

from ..base import Algorithm


def token_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """取每个目标 token 的 log-prob，形状 ``[B,T,V] -> [B,T]``。

    ``DPODataset`` 已经把 x/y 做过一次 next-token 位移，所以这里直接按 y gather，
    不能再裁掉 logits 的最后一位。
    """
    return F.log_softmax(logits.float(), dim=-1).gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)


def sequence_log_probs(per_token: torch.Tensor, mask: torch.Tensor):
    """同时返回序列总 log-prob 与长度归一化 log-prob。"""
    mask = mask.to(per_token.dtype)
    total = (per_token * mask).sum(dim=-1)
    average = total / mask.sum(dim=-1).clamp(min=1)
    return total, average


def _log1mexp(log_p: torch.Tensor) -> torch.Tensor:
    """稳定计算 ``log(1-exp(log_p))``；输入应为不大于 0 的 log 概率。"""
    log_p = log_p.clamp(max=-1e-7)
    split = -0.6931471805599453  # log(0.5)
    return torch.where(log_p < split, torch.log1p(-torch.exp(log_p)),
                       torch.log(-torch.expm1(log_p)))


def preference_objective(
    loss_type: str,
    policy_chosen_sum: torch.Tensor,
    policy_rejected_sum: torch.Tensor,
    policy_chosen_avg: torch.Tensor,
    policy_rejected_avg: torch.Tensor,
    *,
    ref_chosen_sum: torch.Tensor | None = None,
    ref_rejected_sum: torch.Tensor | None = None,
    ref_chosen_avg: torch.Tensor | None = None,
    ref_rejected_avg: torch.Tensor | None = None,
    chosen_nll: torch.Tensor | None = None,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    simpo_gamma: float = 0.5,
    sft_weight: float = 1.0,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
    kto_chosen_kl: torch.Tensor | None = None,
    kto_rejected_kl: torch.Tensor | None = None,
):
    """计算偏好损失，返回 ``(标量损失, 每对偏好 logit)``。

    该函数不依赖模型，便于用手工张量做公式级单元测试。返回的 ``logits`` 越大，
    表示模型越偏向 chosen；它只用于诊断，不一定是每种 loss 的直接输入。
    """
    name = loss_type.lower()

    if name in {"dpo", "ipo", "kto"}:
        if ref_chosen_sum is None or ref_rejected_sum is None:
            raise ValueError(f"{name.upper()} 必须提供 reference log-prob")
        chosen_ratio = policy_chosen_sum - ref_chosen_sum
        rejected_ratio = policy_rejected_sum - ref_rejected_sum
        logits = chosen_ratio - rejected_ratio

        if name == "dpo":
            positive = -F.logsigmoid(beta * logits)
            negative = -F.logsigmoid(-beta * logits)
            loss = ((1.0 - label_smoothing) * positive + label_smoothing * negative).mean()
            return loss, logits

        if name == "ipo":
            if ref_chosen_avg is None or ref_rejected_avg is None:
                raise ValueError("IPO 必须提供长度归一化的 reference log-prob")
            # IPO 的平方目标会直接受序列长度缩放。公开实现采用每 token 平均
            # log-prob，使不同长度回答共享同一个有限最优间隔 1/(2*beta)。
            logits = ((policy_chosen_avg - ref_chosen_avg)
                      - (policy_rejected_avg - ref_rejected_avg))
            return ((logits - 1.0 / (2.0 * beta)) ** 2).mean(), logits

        # Paired-KTO：用另一类样本的批级 KL 作为基线，并 detach，防止模型通过
        # 操纵 KL 基线而不是改善样本效用来降低损失。
        chosen_kl = (chosen_ratio.mean().clamp(min=0).detach()
                     if kto_chosen_kl is None else kto_chosen_kl.detach())
        rejected_kl = (rejected_ratio.mean().clamp(min=0).detach()
                       if kto_rejected_kl is None else kto_rejected_kl.detach())
        chosen_loss = 1.0 - torch.sigmoid(beta * (chosen_ratio - rejected_kl))
        rejected_loss = 1.0 - torch.sigmoid(beta * (chosen_kl - rejected_ratio))
        weighted = torch.cat((desirable_weight * chosen_loss,
                              undesirable_weight * rejected_loss))
        return weighted.mean(), logits

    if name == "simpo":
        logits = policy_chosen_avg - policy_rejected_avg
        return -F.logsigmoid(beta * logits - simpo_gamma).mean(), logits

    if name == "cpo":
        if chosen_nll is None:
            raise ValueError("CPO 必须提供 chosen 的 SFT 损失")
        # 标准 CPO 的 sigmoid 偏好项使用回答序列的总 log-prob；长度归一化
        # 是 SimPO（以及 IPO 工程实现）的定义，不能在这里混用。
        logits = policy_chosen_sum - policy_rejected_sum
        pref = -F.logsigmoid(beta * logits).mean()
        return pref + sft_weight * chosen_nll, logits

    if name == "orpo":
        if chosen_nll is None:
            raise ValueError("ORPO 必须提供 chosen 的 SFT 损失")
        # 序列平均 log-prob 保持在可表示范围内，再换算为 log odds。
        chosen_odds = policy_chosen_avg - _log1mexp(policy_chosen_avg)
        rejected_odds = policy_rejected_avg - _log1mexp(policy_rejected_avg)
        logits = chosen_odds - rejected_odds
        return chosen_nll + sft_weight * (-F.logsigmoid(logits).mean()), logits

    raise ValueError(f"未知偏好损失 {loss_type!r}")


class PreferenceAlgorithm(Algorithm):
    """偏好优化公共训练逻辑；子类只声明 ``name``。"""

    name = ""
    reference_based = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_pref = 0.0
        self._last_margin = 0.0
        self._last_acc = 0.0

    def build_dataset(self):
        return DPODataset(self.args.data_path, self.tokenizer,
                          max_length=self.args.max_seq_len)

    def extra_models(self):
        if not self.reference_based:
            return {}
        import copy
        from trainer.trainer_utils import Logger

        # 必须复制初始策略，尤其是 from_weight=none 时不能重新随机初始化。
        ref_model = copy.deepcopy(self.model)
        ref_model.eval().requires_grad_(False)
        Logger(f"参考模型参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M")
        return {"ref_model": ref_model}

    @staticmethod
    def _distributed_nonnegative_mean(values: torch.Tensor) -> torch.Tensor:
        """计算跨 rank 的非负均值，并把结果从计算图中分离。

        Paired-KTO 把批级 KL 当作常数基线。如果每个 rank 使用自己的局部均值，
        DDP 最后平均的是多个不同目标的梯度，world size 一变目标函数也会变。
        这里归约 ``sum + count``，即使尾批样本数不同也能得到真正的全局均值。
        """
        stats = torch.stack((values.detach().float().sum(),
                             values.new_tensor(values.numel(), dtype=torch.float32)))
        if dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return (stats[0] / stats[1].clamp(min=1)).clamp(min=0).to(values.dtype)

    def compute_loss(self, batch):
        device = self.runtime.device
        x = torch.cat((batch["x_chosen"], batch["x_rejected"])).to(device)
        y = torch.cat((batch["y_chosen"], batch["y_rejected"])).to(device)
        mask = torch.cat((batch["mask_chosen"], batch["mask_rejected"])).to(device)
        attention_mask = x.ne(self.tokenizer.pad_token_id).long()

        outputs = self.model(x, attention_mask=attention_mask)
        policy_token = token_log_probs(outputs.logits, y)
        policy_sum, policy_avg = sequence_log_probs(policy_token, mask)
        policy_chosen_sum, policy_rejected_sum = policy_sum.chunk(2)
        policy_chosen_avg, policy_rejected_avg = policy_avg.chunk(2)

        ref_chosen_sum = ref_rejected_sum = None
        ref_chosen_avg = ref_rejected_avg = None
        if self.reference_based:
            with torch.no_grad():
                ref_logits = self.ref_model(x, attention_mask=attention_mask).logits
                ref_sum, ref_avg = sequence_log_probs(token_log_probs(ref_logits, y), mask)
                ref_chosen_sum, ref_rejected_sum = ref_sum.chunk(2)
                ref_chosen_avg, ref_rejected_avg = ref_avg.chunk(2)

        chosen_mask = mask[:mask.size(0) // 2].to(policy_token.dtype)
        chosen_nll = -(policy_token[:policy_token.size(0) // 2] * chosen_mask).sum()
        chosen_nll = chosen_nll / chosen_mask.sum().clamp(min=1)

        kto_chosen_kl = kto_rejected_kl = None
        if self.name == "kto":
            # KTO 的基线不参与反向，只负责给前景效用提供参考点；多卡必须先
            # 聚合为全局值，不能让每张卡各自优化一个不同的局部目标。
            kto_chosen_kl = self._distributed_nonnegative_mean(
                policy_chosen_sum - ref_chosen_sum
            )
            kto_rejected_kl = self._distributed_nonnegative_mean(
                policy_rejected_sum - ref_rejected_sum
            )

        pref_loss, pref_logits = preference_objective(
            self.name,
            policy_chosen_sum,
            policy_rejected_sum,
            policy_chosen_avg,
            policy_rejected_avg,
            ref_chosen_sum=ref_chosen_sum,
            ref_rejected_sum=ref_rejected_sum,
            ref_chosen_avg=ref_chosen_avg,
            ref_rejected_avg=ref_rejected_avg,
            chosen_nll=chosen_nll,
            beta=self.args.beta,
            label_smoothing=self.args.preference_label_smoothing,
            simpo_gamma=self.args.simpo_gamma,
            sft_weight=(self.args.orpo_lambda if self.name == "orpo" else self.args.cpo_alpha),
            desirable_weight=self.args.kto_desirable_weight,
            undesirable_weight=self.args.kto_undesirable_weight,
            kto_chosen_kl=kto_chosen_kl,
            kto_rejected_kl=kto_rejected_kl,
        )
        loss = (pref_loss + outputs.aux_loss) / self.args.accumulation_steps

        with torch.no_grad():
            self._last_pref = pref_loss.item()
            self._last_margin = pref_logits.mean().item()
            self._last_acc = (pref_logits > 0).float().mean().item()
        return loss, outputs.aux_loss

    def format_log(self, *, epoch, step, iters, start_step, loss_scaled,
                   aux_loss, lr, spend_time):
        total_loss = loss_scaled * self.args.accumulation_steps
        aux = float(aux_loss) if aux_loss is not None else 0.0
        eta_min = spend_time / max(step - start_step, 1) * (iters - step) / 60
        line = (f"Epoch:[{epoch + 1}/{self.args.epochs}]({step}/{iters}), "
                f"loss: {total_loss:.4f}, {self.name}_loss: {self._last_pref:.4f}, "
                f"margin: {self._last_margin:.4f}, pref_acc: {self._last_acc:.3f}, "
                f"aux_loss: {aux:.4f}, learning_rate: {lr:.8f}, epoch_time: {eta_min:.2f}min")
        metrics = {
            "loss": total_loss,
            "logits_loss": self._last_pref,
            "preference_loss": self._last_pref,
            "reward_margin": self._last_margin,
            "preference_acc": self._last_acc,
            "aux_loss": aux,
            "learning_rate": lr,
            "epoch_time": eta_min,
        }
        if self.name == "dpo":
            # 兼容旧版 CSV/WebUI 的字段名；统一的新字段是 preference_loss。
            metrics["dpo_loss"] = self._last_pref
        return line, metrics


class DPOAlgorithm(PreferenceAlgorithm):
    name = "dpo"
    reference_based = True


class IPOAlgorithm(PreferenceAlgorithm):
    name = "ipo"
    reference_based = True


class SimPOAlgorithm(PreferenceAlgorithm):
    name = "simpo"


class CPOAlgorithm(PreferenceAlgorithm):
    name = "cpo"


class ORPOAlgorithm(PreferenceAlgorithm):
    name = "orpo"


class KTOAlgorithm(PreferenceAlgorithm):
    name = "kto"
    reference_based = True
