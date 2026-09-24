"""细粒度专家 + 共享专家 + aux-loss-free 负载均衡的 MoE（DeepSeekMoE / DeepSeek-V3）。

本组件是 DeepSeekMoE 的完整三件套，缺一不可：

- **细粒度专家**：把 N 个中等大小的专家切成更多、更小的专家，同时提高每个
  token 激活的专家数（``num_experts`` 大、``moe_intermediate_size`` 小、
  ``num_experts_per_tok`` 大），在**总激活参数量不变**的前提下组合出更多知识
  通路。这部分不增加任何结构，纯靠配置驱动。
- **共享专家**（``num_shared_experts``，默认 1）：所有 token 常驻激活的稠密
  通路，输出与路由专家结果相加。它承担通用知识，路由专家只需专门化，从而
  缓解「每个专家都各自学一遍公共知识」的冗余。它**不参与路由**：不进 gate
  的输出维度、不进 bias 更新、也不计入负载统计。
- **aux-loss-free 负载均衡**：不碰损失函数，改在路由打分上加一个与梯度无关
  的偏置来调均衡。细粒度专家越多，路由坍塌越容易发生，而传统 aux loss 会
  和语言建模目标互相拉扯，所以这里用无损失的方式做。

## 什么是 aux-loss-free 负载均衡

传统做法（``moe.py``）是在主损失上加一项辅助损失，惩罚「路由分布偏离均匀」：

    L = L_task + alpha * sum_i(f_i * P_i)      f_i 实际负载, P_i 平均路由概率

它有效，但有明显副作用：辅助损失和语言建模目标是**互相拉扯**的——为了压均衡，
模型被迫牺牲一部分预测质量，alpha 越大均衡越好但 loss 越高，需要仔细调参。

DeepSeek-V3 的做法是**不碰损失函数**，改在路由打分上加一个与梯度无关的偏置：

    s_{i,t} = sigmoid(gate(x_t))_i          原始分数（V3 用 sigmoid，不是 softmax）
    s'_{i,t} = s_{i,t} + b_i                选专家时用「加了偏置」的分数
    weight  = s_{i,t}                        但权重仍用原始分数

也就是说 b 只影响「选谁」，不影响「选上之后权重多大」，因此不会扭曲模型的
预测目标。训练过程中根据每个专家的实际负载**用固定步长手工更新** b：

    b_i <- b_i + gamma * (target_load - actual_load_i)

过载的专家（actual > target）b 变小、以后更难被选中；欠载的 b 变大、更容易被
选中，几轮之后负载自然收敛到均匀。b 是 buffer（``persistent=True``）而不是
参数，不参与反向传播，也不产生任何损失项——这就是 "auxiliary-loss-free"。

代价是均衡靠显式的反馈回路而非梯度，因此需要能随 checkpoint 存取 b，且
只在训练时更新。评估时不更新、也不产生 aux_loss。
"""
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from ..registry import register, resolve


@register("feedforward", "moe_finegrained")
class MoEFineGrainedFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.num_experts = int(cfg.num_experts)
        self.top_k = int(cfg.num_experts_per_tok)
        self.gate = nn.Linear(cfg.hidden_size, self.num_experts, bias=False)

        # 专家实现本身也是可插拔的：默认复用同一槽位下的 swiglu
        expert_type = cfg.get("expert_type", "swiglu")
        expert_cls = resolve("feedforward", expert_type)
        expert_cfg = cfg.override(intermediate_size=cfg.moe_intermediate_size)
        self.experts = nn.ModuleList([expert_cls(expert_cfg) for _ in range(self.num_experts)])

        # 共享专家：常驻激活的稠密通路，不参与路由（不进 gate 的输出维度、
        # 不进 bias 更新、不计入负载统计）。即使只有 1 个也包成 ModuleList，
        # 保持 `mlp.shared_experts.0.` 的模块路径——trainer/trainer_utils.py 的
        # get_model_params 按这个字符串统计参数量。
        # 宽度默认与路由专家一致，可用 shared_intermediate_size 单独指定；
        # 设为 0 时挂一个空列表，state_dict 与不带共享专家的版本逐键一致。
        self.num_shared_experts = int(cfg.get("num_shared_experts", cfg.get("n_shared_experts", 1)))
        shared_cfg = cfg.override(
            intermediate_size=cfg.get("shared_intermediate_size", cfg.moe_intermediate_size)
        )
        self.shared_experts = nn.ModuleList(
            [expert_cls(shared_cfg) for _ in range(self.num_shared_experts)]
        )

        # aux-loss-free 开关与超参
        self.aux_loss_free = bool(cfg.get("aux_loss_free_balance", True))
        self.bias_update_rate = float(cfg.get("bias_update_rate", 1e-3))
        self.seq_aux = bool(cfg.get("seq_aux", False))
        # V3 对归一化后的权重再乘一个固定放大系数（论文用 2.5）；默认 1.0 不改幅度
        self.routed_scaling_factor = float(cfg.get("routed_scaling_factor", 1.0))

        # 负载均衡偏置：不是参数、不需要梯度，但要随断点续训一起存取，
        # 所以注册成 persistent buffer（DDP 每个 forward 会广播 buffer，
        # 各 rank 上的 b 会自动保持一致）
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(self.num_experts), persistent=True
        )

        self.act_fn = ACT2FN[cfg.hidden_act]

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _update_bias(self, topk_idx):
        """按实际负载手工调整路由偏置：过载减、欠载加。

        目标负载是均匀分布 ``top_k / num_experts``；用 ``(target - actual)``
        的连续方向即可（V3 论文写成 sign 形式，连续形式是它的平滑版，
        差别只体现在等效步长上）。整个更新不参与计算图。

        注意 ``actual_load`` 的口径是「分到该专家的 token 占比」（按 token 数
        而非按 top-k 次选中数归一），这样均匀路由时它的均值恰好等于
        ``top_k / num_experts``；两个口径不一致会让所有偏置同步漂移。
        """
        num_tokens = topk_idx.shape[0]
        actual = F.one_hot(topk_idx, self.num_experts).float().sum(dim=(0, 1)) / num_tokens
        target = self.top_k / self.num_experts
        delta = (target - actual).to(self.e_score_correction_bias.dtype)
        self.e_score_correction_bias.add_(self.bias_update_rate * delta)

    def _seq_aux_loss(self, scores, topk_idx, batch_size, seq_len, valid):
        """DeepSeek-V3 的序列级辅助损失（可选补充项，默认关闭）：

            L = coef * sum_s sum_i f_{i,s} * P_{i,s}
            f_{i,s} = N_r / (T * K) * 序列 s 中选中专家 i 的 token 数
            P_{i,s} = 序列 s 内所有 token 对专家 i 的平均打分

        它逐条序列地拉均衡，比批级 aux loss 更细粒度；V3 里只作为
        aux-loss-free 的补充，系数取很小。
        """
        n_routed, top_k = self.num_experts, self.top_k
        scores_s = scores.view(batch_size, seq_len, n_routed)
        selected = F.one_hot(topk_idx.view(batch_size, seq_len, top_k), n_routed).float().sum(dim=2)
        valid_s = valid.view(batch_size, seq_len, 1).to(scores.dtype)
        token_count = valid_s.sum(dim=1).clamp(min=1)
        f = (selected * valid_s).sum(dim=1) * (n_routed / (token_count * top_k))
        p = (scores_s * valid_s).sum(dim=1) / token_count
        return (f * p).sum() * self.config.router_aux_loss_coef

    def _compute_aux_loss(self, scores, topk_idx, batch_size, seq_len, valid):
        """aux_loss 的取值规则。

        - 默认（aux-loss-free 打开、seq_aux 关闭）：恒为零张量，均衡完全交给 bias；
        - ``seq_aux: true``：返回序列级辅助损失（V3 的可选补充项）；
        - ``aux_loss_free_balance: false``：退回 ``moe.py`` 的传统批级 aux loss；
        - 评估模式或 ``router_aux_loss_coef <= 0``：恒为零张量。
        """
        zero = scores.new_zeros(1).squeeze()
        if not self.training or self.config.router_aux_loss_coef <= 0 or not valid.any():
            return zero
        if self.seq_aux:
            return self._seq_aux_loss(scores, topk_idx, batch_size, seq_len, valid)
        if self.aux_loss_free:
            return zero
        valid_idx, valid_scores = topk_idx[valid], scores[valid]
        load = F.one_hot(valid_idx, self.num_experts).float().mean(0)
        return (load * valid_scores.mean(0)).sum() * self.num_experts * self.config.router_aux_loss_coef

    # ------------------------------------------------------------------ #
    def forward(self, x, token_mask=None):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        # V3 用 sigmoid 给每个专家独立打分（不是 softmax），专家之间不互相竞争
        scores = torch.sigmoid(self.gate(x_flat))
        # 偏置只参与「选谁」；权重取原始分数，保证不扭曲模型目标
        bias = self.e_score_correction_bias.to(scores.dtype)
        topk_idx = torch.topk(scores + bias, k=self.top_k, dim=-1, sorted=False).indices
        topk_weight = scores.gather(dim=-1, index=topk_idx)
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weight = topk_weight * self.routed_scaling_factor
        if token_mask is not None:
            valid = token_mask[:, -seq_len:].reshape(-1).bool()
        else:
            valid = torch.ones(x_flat.size(0), dtype=torch.bool, device=x.device)

        # aux-loss-free：训练时按负载手工更新偏置，不产生任何损失项
        if self.training and self.aux_loss_free and valid.any():
            self._update_bias(topk_idx[valid])

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

        # 共享专家：所有 token 常驻生效，输出直接相加（不乘门控权重、
        # 也不参与任何负载统计）。num_shared_experts 为 0 时这里是空循环。
        if self.training:
            with torch.no_grad():
                self.last_routed_rms = float(y.detach().float().pow(2).mean().sqrt())
        for shared in self.shared_experts:
            sh = shared(x_flat)
            y = y + sh
            if self.training:
                with torch.no_grad():
                    # 共享专家输出的量级 —— 用来判断「共享专家是否主导了 FFN 的贡献」
                    self.last_shared_rms = float(sh.detach().float().pow(2).mean().sqrt())
        if self.training and not self.shared_experts:
            self.last_shared_rms = 0.0

        self.aux_loss = self._compute_aux_loss(scores, topk_idx, batch_size, seq_len, valid)

        # ---- 暴露本步的负载统计给外部指标采集（不参与计算图、不改任何数值）----
        # aux-loss-free 的均衡完全靠 bias 的反馈回路，因此「专家是否被均匀使用」
        # 是这个模型最重要的健康指标，而它不会体现在 loss 上。
        if self.training:
            with torch.no_grad():
                # 每个专家被多少个 token 选中（按 top-k 计，累加所有头）
                self.last_load = F.one_hot(topk_idx[valid], self.num_experts).float().sum(dim=(0, 1)).detach()
                # 路由打分的分布（未加 bias）——用于算路由熵，判断是否坍缩到少数专家
                self.last_scores_mean = (scores[valid].mean(dim=0).detach()
                                         if valid.any() else scores.new_zeros(self.num_experts))
        return y.view(batch_size, seq_len, hidden_dim)
