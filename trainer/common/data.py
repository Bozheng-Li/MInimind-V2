"""数据装配：DistributedSampler + SkipBatchSampler + DataLoader。

重构前这段逻辑在 8 个脚本里逐字重复（"段5 + 段8" 的一部分），现在只写一次。

**注意**：``setup_seed(args.seed + epoch)`` 与随后的 ``torch.randperm``
顺序不能变 —— 它决定了每个 epoch 的数据顺序，改变顺序会让「同一 seed
跑出同一 loss」这一兼容性保证失效。
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from ..trainer_utils import SkipBatchSampler, setup_seed


def build_sampler(dataset):
    """有 DDP 就用 DistributedSampler，否则返回 None（用随机索引）。"""
    return DistributedSampler(dataset) if dist.is_initialized() else None


def build_loader(dataset, args, epoch: int, sampler=None, start_step: int = 0, start_epoch: int = 0):
    """构造某个 epoch 的 DataLoader。

    返回 ``(loader, skip)``；``skip`` 是该 epoch 因断点续训要跳过的 step 数。
    """
    if sampler is not None:
        sampler.set_epoch(epoch)
    # 每个 epoch 重新洗牌：seed 与 epoch 绑定，保证可复现
    # （蒸馏脚本历史上无 --seed，用 getattr 兜底为 42）
    setup_seed(getattr(args, "seed", 42) + epoch)
    indices = torch.randperm(len(dataset)).tolist()
    skip = start_step if (epoch == start_epoch and start_step > 0) else 0
    batch_sampler = SkipBatchSampler(sampler or indices, args.batch_size, skip)
    loader = DataLoader(dataset, batch_sampler=batch_sampler,
                        num_workers=args.num_workers, pin_memory=True)
    return loader, skip


class PrefixDataset(Dataset):
    """只取基数据集的前 ``n`` 条 —— 把尾部留作验证集。

    用「长度截断」而不是「索引列表」，对 800 万级数据集也不会额外占内存。
    不修改原始 ``dataset/lm_dataset.py``。
    """

    def __init__(self, base, n: int):
        self.base, self.n = base, int(n)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.base[i]


class RangeDataset(Dataset):
    """只取基数据集的 ``[start, end)`` 区间（验证集用）。"""

    def __init__(self, base, start: int, end: int):
        self.base, self.start, self.end = base, int(start), int(end)

    def __len__(self):
        return max(0, self.end - self.start)

    def __getitem__(self, i):
        return self.base[self.start + i]


def split_holdout(dataset, val_samples: int):
    """把数据集尾部 ``val_samples`` 条留作验证，返回 ``(train_ds, val_ds)``。

    ``val_samples<=0`` 时不切分，返回 ``(dataset, None)``。
    留出集固定（尾部），因此跨 step 可比。
    """
    if not val_samples or val_samples <= 0:
        return dataset, None
    n = len(dataset)
    n_val = min(int(val_samples), max(n // 100, 1))     # 至少留 1 条，且不超过 1%
    train_ds = PrefixDataset(dataset, n - n_val)
    val_ds = RangeDataset(dataset, n - n_val, n)
    return train_ds, val_ds
