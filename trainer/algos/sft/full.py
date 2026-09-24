"""全参数监督微调（Full SFT）。

``SFTDataset`` 已经把用户、系统和工具消息对应的标签置为 ``-100``，因此模型的
交叉熵只监督 assistant 回复。模型内部负责 next-token 位移，这里不能再次手工
shift，否则会造成标签错位。
"""
from __future__ import annotations

from dataset.lm_dataset import SFTDataset

from ..base import Algorithm


class SFTAlgorithm(Algorithm):
    """全参数 SFT：更新模型的全部可训练参数。"""

    name = "sft"

    def build_dataset(self):
        return SFTDataset(self.args.data_path, self.tokenizer,
                          max_length=self.args.max_seq_len)

    def compute_loss(self, batch):
        input_ids = batch[0].to(self.runtime.device)
        labels = batch[1].to(self.runtime.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        res = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = res.loss + res.aux_loss
        return loss / self.args.accumulation_steps, res.aux_loss
