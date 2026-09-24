"""RLOO（REINFORCE Leave-One-Out）在线强化学习。

同一 prompt 采样 K 条回答，第 i 条用其余 K-1 条 reward 均值作为 baseline：

``A_i = r_i - mean(r_j, j != i)``

它不需要 critic，比 PPO 简洁；又避免了普通 REINFORCE 的高方差 baseline。
训练、KL 与 rollout 基础设施复用 GRPO，但优势不做组内标准差归一化。
"""
from __future__ import annotations

from .grpo import GRPOAlgorithm


class RLOOAlgorithm(GRPOAlgorithm):
    name = "rloo"
    advantage_mode = "rloo"
