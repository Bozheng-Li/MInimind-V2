"""MoE 前馈网络（top-k 路由 + 负载均衡）—— 与重构前 ``MOEFeedForward`` 逐字一致。

关键约束：
- 专家必须挂在 ``self.experts``（``nn.ModuleList``）下，模块路径保持
  ``mlp.experts.{i}.{gate,up,down}_proj.weight``；
  ``trainer_utils.get_model_params`` 与 ``scripts/convert_model.py`` 都硬编码了这个路径。
- 未被分到 token 的专家会执行一个 ``0 * sum(params)`` 的“悬挂”操作，
  让它参与计算图，避免 DDP 报 unused parameters。
- 训练时产出 ``aux_loss``；评估时为零张量。
"""
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from ..registry import register, resolve


@register("feedforward", "moe")
class MoEFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)

        # 专家实现本身也是可插拔的：默认复用同一槽位下的 swiglu
        expert_type = cfg.get("expert_type", "swiglu")
        expert_cls = resolve("feedforward", expert_type)
        expert_cfg = cfg.override(intermediate_size=cfg.moe_intermediate_size)
        self.experts = nn.ModuleList([expert_cls(expert_cfg) for _ in range(cfg.num_experts)])

        self.act_fn = ACT2FN[cfg.hidden_act]

    def forward(self, x, token_mask=None):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 让未激活专家的参数留在计算图上，避免 DDP unused-parameter 报错
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if token_mask is not None:
            valid = token_mask[:, -seq_len:].reshape(-1).bool()
        else:
            valid = torch.ones(x_flat.size(0), dtype=torch.bool, device=x.device)
        if self.training and self.config.router_aux_loss_coef > 0 and valid.any():
            # padding token 不属于训练分布，计入路由负载会把大量尾部 padding 误当
            # 真实 token，尤其在短样本 batch 中会严重扭曲辅助损失。
            valid_idx, valid_scores = topk_idx[valid], scores[valid]
            load = F.one_hot(valid_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * valid_scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)
