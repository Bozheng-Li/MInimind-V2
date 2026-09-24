"""DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）。

在 GRPO 基础上实现 DAPO 论文/公开配方中的四个关键改动：

1. Clip-Higher：负优势下界用 ``1-epsilon``，正优势上界放宽到
   ``1+dapo_epsilon_high``；
2. Dynamic Sampling：若一个 prompt 的整组 reward 完全相同，重新采样回答，
   最多尝试 ``dapo_max_resample`` 次，并过滤仍无训练信号的组；
3. Token-level Policy Gradient：全批有效 token 统一归一化，而非先逐回答平均；
4. Overlong Reward Shaping：接近最大生成长度时施加平滑惩罚，降低截断回答比例。

DAPO 默认不加 reference KL（``dapo_kl_coef=0``），但保留显式开关方便做消融。
"""
from __future__ import annotations

from ...trainer_utils import Logger, is_main_process
from .grpo import GRPOAlgorithm


class DAPOAlgorithm(GRPOAlgorithm):
    """带动态采样和非对称裁剪的组相对策略优化。"""

    name = "dapo"
    token_level_loss = True

    def uses_reference_model(self):
        """DAPO 默认无 KL；系数为 0 时不额外占一份 reference 模型显存。"""
        return float(self.args.dapo_kl_coef) != 0.0

    def _collect_rollout(self, prompts, prompt_inputs, rollout_engine, reward_model, device):
        """对零方差 reward 组动态重采样。

        每次候选 rollout 都覆盖完整 prompt batch，保留“有效组最多”的那一批。
        这样所有张量仍是规则矩阵，兼容 torch/SGLang 两种后端；最终依旧零方差的
        组会在 ``_valid_groups`` 中被 mask，不会产生虚假的零优势更新。
        """
        best = None
        best_valid = -1
        attempts = max(1, int(self.args.dapo_max_resample))
        for attempt in range(attempts):
            candidate = super()._collect_rollout(
                prompts, prompt_inputs, rollout_engine, reward_model, device
            )
            result, rewards, _ = candidate
            # 动态采样必须按“最终用于优化的 reward”判断是否有信号。若忽略超长
            # shaping，原始 RM 同分但长度不同的组会被误判成零方差并白白重采样。
            shaped = self._shape_rewards(rewards, result.completion_mask.to(device))
            std = shaped.view(-1, self.args.num_generations).std(dim=1, unbiased=False)
            valid = int((std > self.args.dapo_reward_std_threshold).sum().item())
            if valid > best_valid:
                best, best_valid = candidate, valid
            if valid == len(prompts):
                break
        if attempts > 1 and best_valid < len(prompts) and is_main_process():
            Logger(f"[DAPO] 动态采样后仍有 {len(prompts) - best_valid}/{len(prompts)} 个"
                   "零方差 prompt，本步将过滤这些组")
        return best

    def _valid_groups(self, group_std):
        return group_std > self.args.dapo_reward_std_threshold

    def _shape_rewards(self, rewards, completion_mask):
        """对超过软上限的回答施加线性、封顶的超长惩罚。"""
        # 当调试配置的 max_gen_len 小于默认 buffer 时，buffer 必须同步缩短；否则
        # 满长度回答也只能吃到 max_gen_len/default_buffer 的一小部分惩罚。
        buffer = max(1, min(int(self.args.dapo_overlong_buffer),
                            int(self.args.max_gen_len)))
        soft_limit = max(0, int(self.args.max_gen_len) - buffer)
        lengths = completion_mask.sum(dim=1).to(rewards.dtype)
        excess_ratio = ((lengths - soft_limit) / buffer).clamp(min=0.0, max=1.0)
        return rewards - self.args.dapo_overlong_penalty * excess_ratio

    def _clip_high(self):
        return self.args.dapo_epsilon_high

    def _kl_beta(self):
        return self.args.dapo_kl_coef
