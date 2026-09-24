"""GRPO / DAPO / RLOO / PPO / Agentic RL 共用的骨架。

五个在线 RL 算法共享「rollout → reward → advantage」链路，区别在优势估计与
loss 形式。旧 ``train_grpo.py`` / ``train_ppo.py`` / ``train_agent.py`` 里的
``rep_penalty`` / ``calculate_rewards`` / 优化器与调度器搭建原本都是重复的，
这里收拢成一份。

这里集中维护共同语义，改动会同时影响五个在线 RL 算法。

关于续训：``common.checkpoint.restore`` 只负责 model / optimizer / scaler，
算法特有的调度器与 critic 状态由本模块的 :meth:`RLAlgorithm._restore_extra`
自行读回（``checkpoint_extra`` 负责写）。
"""
from __future__ import annotations

import copy
import math
import re
from collections import defaultdict

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ...common.checkpoint import load_resume
from ...common.runtime import build_optimizer as _build_adamw
from ...rollout_engine import create_rollout_engine
from ...trainer_utils import LMForRewardModel, init_model
from ..base import Algorithm


# --------------------------------------------------------------------------- #
# 奖励整形（GRPO / PPO / Agent 共用）
# --------------------------------------------------------------------------- #
def rep_penalty(text, n=3, cap=0.5):
    """重复 n-gram 惩罚：0.5 = 1.0 * 0.5。"""
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model, num_generations, device,
                      return_breakdown=False):
    """基于奖励模型的打分 + 长度 / 思考格式 / 重复惩罚。

    ``responses`` 按 ``prompts × num_generations`` 展平（PPO 传 num_generations=1，
    此时与重构前 ``zip(prompts, responses)`` 的逐样本循环等价）。

    ``return_breakdown=True`` 时额外返回各分项的**均值**（长度分 / 思考长度分 /
    思考闭合分 / 重复惩罚 / RM 分）。奖励是一堆分项加出来的，只看总分无法诊断
    到底是哪一项在驱动策略 —— 分项不参与任何数值计算，只用于记录。
    """
    rewards = torch.zeros(len(responses), device=device)
    parts = defaultdict(float)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        for i in range(batch_size):
            for j in range(num_generations):
                response_idx = i * num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response
                length_term = 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                rewards[response_idx] += length_term
                parts["rew_len"] += length_term
                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    think_term = 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    close_term = 0.25 if response.count('</think>') == 1 else -0.25
                    rewards[response_idx] += think_term
                    rewards[response_idx] += close_term
                    parts["rew_think_len"] += think_term
                    parts["rew_think_close"] += close_term
                    answer = answer_content.strip()
                penalty = rep_penalty(answer)
                rewards[response_idx] -= penalty
                parts["rew_rep"] -= penalty

                score = reward_model.get_score(messages, answer)
                parts["rew_rm"] += score
                reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=device)
        rewards += reward_model_scores

    if return_breakdown:
        n = max(len(responses), 1)
        return rewards, {k: v / n for k, v in parts.items()}
    return rewards


# --------------------------------------------------------------------------- #
# 额外模型 / rollout 引擎构造
# --------------------------------------------------------------------------- #
def build_reference_model(args, lm_config, runtime, policy_model=None):
    """冻结的 reference 模型（KL 基准），必须与初始策略逐参数相同。

    优先深拷贝已经构造好的策略。若 ``from_weight=none``，重新调用初始化函数会
    消耗新的随机数并得到另一组参数，使训练从第一步就承受虚假的 KL 惩罚。
    """
    if policy_model is not None:
        ref_model = copy.deepcopy(policy_model)
    else:
        ref_model, _ = init_model(lm_config, args.from_weight, save_dir=args.save_dir,
                                  device=runtime.device)
    return ref_model.eval().requires_grad_(False)


def build_reward_model(args, runtime):
    return LMForRewardModel(args.reward_model_path, device=runtime.device, dtype=torch.float16)


def build_rollout_engine(args, policy_model, tokenizer, runtime):
    """按 ``--rollout_engine`` 造引擎；只负责 policy 推理。"""
    return create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=policy_model,
        tokenizer=tokenizer,
        device=runtime.device,
        autocast_ctx=runtime.autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )


# --------------------------------------------------------------------------- #
# RL 算法公共骨架
# --------------------------------------------------------------------------- #
class RLAlgorithm(Algorithm):
    """RL 型算法基类：优化器 / 调度器 / rollout 引擎 / 续训。

    子类覆盖 :meth:`train_epoch` 自建循环（rollout → reward → advantage）；
    调度器的 ``T_max``、续训时要灌回的额外状态、额外优化器与额外模型的包装
    分别通过 :meth:`_total_optimizer_steps` / :meth:`_load_extra_state` /
    :meth:`_build_extra_optimizers` / :meth:`_bind_models` 定制（PPO 全用上了）。
    """

    def build_optimizer(self, model):
        """显式建 AdamW，让算法自己拿着它去建余弦调度器。"""
        self.optimizer = _build_adamw(model, self.args.learning_rate)
        return self.optimizer

    def uses_reference_model(self):
        """是否需要冻结 reference；DAPO 在 KL 系数为 0 时会覆盖为 False。"""
        return True

    def extra_models(self):
        models = {
            "reward_model": build_reward_model(self.args, self.runtime),
        }
        if self.uses_reference_model():
            models["ref_model"] = build_reference_model(
                self.args, self.lm_config, self.runtime, policy_model=self.model
            )
        return models

    def checkpoint_extra(self):
        return {"scheduler": self.scheduler}

    # ---------------- 调度器 ---------------- #
    def _budget_steps(self, iters):
        """整个训练任务会消费的 rollout batch 数，``--max_steps`` 是全局上限。

        以前这里直接用 ``iters``（= 一个 epoch 的长度），于是 ``--max_steps 1000``
        对 GRPO / PPO / Agent **完全无效** —— 它们覆盖了 train_epoch，压根不读这个参数，
        设了 1000 步会一路跑到 epoch 结束（实测 PPO 跑了 9654 步 / 10.5 小时，Agent
        按 epoch 长度要跑 19988 步 / 84 小时）。修完两件事一起对：循环按时停，
        余弦调度也按同一个总步数衰减。
        """
        epoch_budget = int(iters) * int(self.args.epochs)
        max_steps = int(getattr(self.args, "max_steps", 0) or 0)
        return min(epoch_budget, max_steps) if max_steps else epoch_budget

    def _total_optimizer_steps(self, iters):
        """余弦调度的 ``T_max``。默认对应 GRPO / Agent 的公式；PPO 覆盖。"""
        return math.ceil(self._budget_steps(iters) / self.args.accumulation_steps)

    def _make_schedulers(self, iters):
        """建调度器。PPO 需要 actor + critic 两个，故覆盖此方法。"""
        total_steps = self._total_optimizer_steps(iters)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(total_steps, 1),
                                           eta_min=self.args.learning_rate / 10)

    # ---------------- 首次进入训练循环前的装配 ---------------- #
    def _count_iters(self, ctx):
        """一个 epoch 的 batch 数 —— 与重构前 ``len(loader_for_count)`` 同构。"""
        return len(DataLoader(self.dataset, batch_size=self.args.batch_size, sampler=ctx.sampler))

    def _restore_extra(self, ctx):
        """把调度器 / critic 等额外状态从续训档读回。

        ``common.checkpoint.restore`` 只处理 model / optimizer / scaler，
        所以这里按同样的路径约定再读一次续训档，只取额外的键。
        """
        if self.args.from_resume != 1:
            return
        ckp_data = load_resume(self.args, ctx.lm_config, self.weight_prefix())
        if ckp_data:
            self._load_extra_state(ckp_data)

    def _load_extra_state(self, ckp_data):
        """默认只恢复余弦调度器（GRPO / Agent）；PPO 覆盖以带上 critic 的状态。"""
        if 'scheduler' in ckp_data and getattr(self, "scheduler", None) is not None:
            self.scheduler.load_state_dict(ckp_data['scheduler'])

    def _build_extra_optimizers(self):
        """子类按需覆盖（PPO 的 critic 优化器）。"""

    def _bind_models(self, ctx):
        """子类按需覆盖（PPO 的 critic 需要 DDP 包装）。"""

    def _ensure_setup(self, ctx):
        """在第一个 train_epoch 开头做一次性的装配。

        此时 pipeline 已完成 DDP / compile 包装，因此把 ``ctx.model``（最终形态）
        交给 rollout 引擎，等价于重构前「包装后再 ``update_policy``」。
        """
        if getattr(self, "_rl_ready", False):
            return
        self._rl_ready = True
        self._build_extra_optimizers()
        self._make_schedulers(self._count_iters(ctx))
        self._restore_extra(ctx)
        self._bind_models(ctx)
        self.rollout_engine = build_rollout_engine(self.args, ctx.model, self.tokenizer, self.runtime)
        # Torch 后端持有当前模型引用；SGLang 后端则可能还加载着上一次实验的权重。
        # 在第一次采样前强制同步一次，确保 old log-prob 确实来自当前初始策略。
        self.rollout_engine.update_policy(ctx.model)


__all__ = [
    "rep_penalty", "calculate_rewards",
    "build_reference_model", "build_reward_model", "build_rollout_engine",
    "RLAlgorithm",
]
