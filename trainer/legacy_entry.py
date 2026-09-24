"""旧训练脚本的统一兼容入口。

历史上的 ``train_*.py`` 各自复制了一套模型、数据、优化器与 RL 公式，修一个 bug
必须改八处，也容易出现“统一入口正确、旧脚本仍训练错”的分叉。现在旧文件只负责
补上固定的 ``--algo``，其余参数、配置与执行全部转交 ``trainer/train.py``。
"""
from __future__ import annotations

import sys

from configs import apply_to_parser, stage_of
from trainer.common import run_training
from trainer.train import build_parser


def run_legacy(algo: str) -> None:
    """以指定算法运行统一训练入口，同时保留旧脚本的命令行调用方式。"""
    argv = ["--algo", algo, *sys.argv[1:]]
    parser = build_parser()
    cfg = apply_to_parser(parser, stage_of(algo), algo=algo, argv=argv)
    args = parser.parse_args(argv)
    run_training(args, algo, cfg, config_path=args.config)
