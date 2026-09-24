"""兼容入口：预训练实现位于 ``trainer.algos.pretrain``。"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trainer.legacy_entry import run_legacy

if __name__ == "__main__":
    run_legacy("pretrain")
