"""标准因果语言模型预训练。"""
from __future__ import annotations

import os

from dataset.lm_dataset import MixedPretrainDataset, PretrainDataset

from ..base import Algorithm


class PretrainAlgorithm(Algorithm):
    """在普通文本上训练 next-token prediction。

    ``PretrainDataset`` 会保留正文与 EOS 的标签、屏蔽 padding；模型内部完成
    ``logits[..., :-1]`` 与 ``labels[..., 1:]`` 的错位对齐。
    """

    name = "pretrain"

    def build_dataset(self):
        mix = getattr(self.args, "data_mix", None)
        if mix:
            # 配比模式：data_path 是目录，data_mix 是 {文件名: 权重}
            base = self.args.data_path
            sources = [(os.path.join(base, name), float(w)) for name, w in mix.items()]
            ds = MixedPretrainDataset(sources, self.tokenizer,
                                      max_length=self.args.max_seq_len,
                                      seed=getattr(self.args, "seed", 42))
            if self.runtime.is_main:
                from ...trainer_utils import Logger
                total = sum(ds.weights)
                parts = "，".join(f"{n} {w / total:.0%}" for n, w in zip(ds.names, ds.weights))
                sizes = "，".join(f"{n.split('.')[-2] if '.' in n else n} {len(s):,} 条"
                                  for n, s in zip(ds.names, ds.samples))
                Logger(f"[data] 预训练配比：{parts}")
                Logger(f"[data] 各类存量：{sizes}（配比超过存量的类会被重复采样）")
            return ds
        return PretrainDataset(self.args.data_path, self.tokenizer,
                               max_length=self.args.max_seq_len)

    def compute_loss(self, batch):
        input_ids = batch[0].to(self.runtime.device)
        labels = batch[1].to(self.runtime.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss + outputs.aux_loss
        return loss / self.args.accumulation_steps, outputs.aux_loss
