"""预训练算法。

预训练与 SFT 都使用 next-token 交叉熵，但数据标签语义不同，因此分目录存放，
避免把“语言模型预训练”和“指令监督微调”混成同一类算法。
"""

from .causal_lm import PretrainAlgorithm

__all__ = ["PretrainAlgorithm"]
