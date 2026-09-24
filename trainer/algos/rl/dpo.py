"""DPO 兼容导入。

偏好算法已统一到 :mod:`trainer.algos.rl.preference`；保留本模块是为了不破坏
历史代码中的 ``from trainer.algos.rl.dpo import DPOAlgorithm``。
"""

from .preference import (DPOAlgorithm, preference_objective,
                         sequence_log_probs, token_log_probs)

# 旧版公开函数名兼容。
logits_to_log_probs = token_log_probs


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """兼容旧版的 DPO 纯函数接口。"""
    ref_sum, _ = sequence_log_probs(ref_log_probs, mask)
    policy_sum, policy_avg = sequence_log_probs(policy_log_probs, mask)
    ref_chosen, ref_rejected = ref_sum.chunk(2)
    policy_chosen, policy_rejected = policy_sum.chunk(2)
    avg_chosen, avg_rejected = policy_avg.chunk(2)
    loss, _ = preference_objective(
        "dpo", policy_chosen, policy_rejected, avg_chosen, avg_rejected,
        ref_chosen_sum=ref_chosen, ref_rejected_sum=ref_rejected, beta=beta,
    )
    return loss


__all__ = ["DPOAlgorithm", "logits_to_log_probs", "dpo_loss"]
