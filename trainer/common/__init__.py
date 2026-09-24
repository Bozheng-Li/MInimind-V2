"""训练公共脚手架。

分工：

- ``trainer/trainer_utils.py`` —— 底层工具函数（Logger / lm_checkpoint / init_model /
  SkipBatchSampler / LMForRewardModel 等），保持原样不动
- ``trainer/common/`` —— 在其之上的**编排层**：初始化运行时、装配数据、跑训练循环、存盘

算法模块（``trainer/algos/``）只描述「优化什么」，其余交给这里。
"""
from .checkpoint import load_resume, save_inference_weights, save_resume
from .data import build_loader, build_sampler
from .logging import init_wandb, log_metrics
from .pipeline import run_training
from .runner import TrainContext, run_epochs
from .runtime import Runtime, build_optimizer, build_scaler, init_runtime, unwrap_model

__all__ = [
    "Runtime", "init_runtime", "build_optimizer", "build_scaler", "unwrap_model",
    "build_loader", "build_sampler",
    "save_inference_weights", "save_resume", "load_resume",
    "init_wandb", "log_metrics",
    "TrainContext", "run_epochs",
    "run_training",
]
