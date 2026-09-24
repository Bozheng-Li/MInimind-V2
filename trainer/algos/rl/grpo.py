"""GRPO / CISPO：组内相对优势 + KL 正则。

对应旧 ``trainer/train_grpo.py`` 的核心目标：

- 组内优势 ``adv = (r - mean_g) / (std_g + 1e-4)``
- 每 token KL 惩罚 ``exp(kl) - kl - 1``
- 两种 loss：``cispo``（clamp 后 detach 的比率）与 ``grpo``（双侧 clip 取 min）
- fp16 使用 GradScaler；bf16 下 scaler 自动退化为无操作

与 PPO / Agent 的区别只在优势估计与 loss 形式，rollout → reward 链路见
``common.py``。
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from dataset.lm_dataset import RLAIFDataset

from ...common.checkpoint import save_resume
from ...common.runner import record_rl_step
from ...rollout_engine import build_full_attention_mask
from ...trainer_utils import Logger, is_main_process
from .common import RLAlgorithm, calculate_rewards


def grouped_advantages(rewards: torch.Tensor, num_generations: int,
                       mode: str = "grpo") -> tuple[torch.Tensor, torch.Tensor]:
    """计算组内优势，并返回每个组的 reward 标准差。

    ``grpo`` 使用组内 z-score；``rloo`` 使用 leave-one-out baseline。RLOO 的
    第 i 条优势是 ``r_i - mean(r_{-i})``，因此至少需要两条生成。
    """
    grouped = rewards.view(-1, num_generations)
    group_std = grouped.std(dim=1, unbiased=False)
    if mode == "rloo":
        if num_generations < 2:
            raise ValueError("RLOO 的 num_generations 必须 >= 2")
        other_mean = (grouped.sum(dim=1, keepdim=True) - grouped) / (num_generations - 1)
        advantages = grouped - other_mean
    else:
        mean = grouped.mean(dim=1, keepdim=True)
        advantages = (grouped - mean) / (group_std.unsqueeze(1) + 1e-4)
    return advantages.reshape(-1), group_std


def clipped_policy_loss(per_token_logps: torch.Tensor,
                        old_per_token_logps: torch.Tensor,
                        ref_per_token_logps: torch.Tensor,
                        advantages: torch.Tensor,
                        completion_mask: torch.Tensor,
                        *, loss_type: str, epsilon_low: float,
                        epsilon_high: float, kl_beta: float,
                        token_level: bool = False):
    """GRPO/CISPO/DAPO 共用的 token 级策略目标。

    DAPO 的 ``epsilon_high`` 表示上侧相对裁剪宽度（例如 0.28 -> 上界 1.28）；
    CISPO 沿用其论文定义，把它解释为 ratio 的绝对上限。返回 loss 之外也返回
    ratio 和每 token KL，供日志诊断。
    """
    log_ratio = per_token_logps - old_per_token_logps
    ratio = torch.exp(log_ratio)
    kl_delta = ref_per_token_logps - per_token_logps
    per_token_kl = torch.exp(kl_delta) - kl_delta - 1.0
    advantage = advantages.unsqueeze(1)

    if loss_type == "cispo":
        # CISPO 的裁剪权重必须 detach：它只控制采样 token 的更新强度，不应让梯度
        # 再穿过 ratio 形成额外的二阶形态。
        weight = ratio.clamp(max=epsilon_high).detach()
        per_token_loss = -(weight * advantage * per_token_logps - kl_beta * per_token_kl)
    else:
        clipped = ratio.clamp(1.0 - epsilon_low, 1.0 + epsilon_high)
        surrogate = torch.minimum(ratio * advantage, clipped * advantage)
        per_token_loss = -(surrogate - kl_beta * per_token_kl)

    mask = completion_mask.to(per_token_loss.dtype)
    if token_level:
        # DAPO 的 token-level loss：整批有效 token 只做一次归一化，长回答按 token
        # 数自然获得更大权重；GRPO 则先逐回答平均，再对回答平均。
        loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1)
    else:
        per_sample = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        valid = mask.sum(dim=1) > 0
        loss = per_sample[valid].mean() if valid.any() else per_token_loss.sum() * 0.0
    return loss, ratio, per_token_kl


class GRPOAlgorithm(RLAlgorithm):
    name = "grpo"
    advantage_mode = "grpo"
    token_level_loss = False

    def _collect_rollout(self, prompts, prompt_inputs, rollout_engine, reward_model, device):
        """生成一组回答并打分；DAPO 会覆盖它以实现动态重采样。"""
        result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=self.args.num_generations,
            max_new_tokens=self.args.max_gen_len,
            temperature=self.args.rollout_temperature,
        )
        rewards, parts = calculate_rewards(
            prompts, result.completions, reward_model,
            self.args.num_generations, device, return_breakdown=True,
        )
        return result, rewards.to(device), parts

    def _shape_rewards(self, rewards, completion_mask):
        """奖励整形钩子；标准 GRPO 不额外改奖励。"""
        return rewards

    def _valid_groups(self, group_std):
        """哪些组参与训练；标准 GRPO 保留所有组（零方差组的优势自然为零）。"""
        return torch.ones_like(group_std, dtype=torch.bool)

    def _clip_high(self):
        return self.args.epsilon if self.args.loss_type == "grpo" else self.args.epsilon_high

    def _kl_beta(self):
        return self.args.beta

    def build_dataset(self):
        self.dataset = RLAIFDataset(self.args.data_path, self.tokenizer,
                                    max_length=self.args.max_seq_len + self.args.max_gen_len,
                                    thinking_ratio=self.args.thinking_ratio)
        return self.dataset

    def train_epoch(self, ctx, epoch, loader, iters, start_step=0):
        self._ensure_setup(ctx)
        args, device = self.args, ctx.device
        model, tokenizer = ctx.model, self.tokenizer
        optimizer, scheduler = self.optimizer, self.scheduler
        rollout_engine = self.rollout_engine
        ref_model = getattr(self, "ref_model", None)
        reward_model = self.reward_model
        wandb = ctx.wandb
        step = start_step
        start_time = time.time()
        max_steps = int(getattr(args, "max_steps", 0) or 0)
        pending_micro_steps = 0
        save_requested = False
        hit_limit = False

        def optimizer_step(micro_steps):
            """提交策略梯度；尾组不足时恢复为对实际 micro-batch 求平均。"""
            ctx.scaler.unscale_(optimizer)
            correction = args.accumulation_steps / max(micro_steps, 1)
            if correction != 1.0:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            if ctx.metrics is not None:
                # 先记录再裁剪，才能观察优化器真正收到的全局梯度。
                ctx.metrics.record_grads()
            max_norm = args.grad_clip if args.grad_clip > 0 else float("inf")
            ctx.last_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            )
            old_scale = ctx.scaler.get_scale()
            ctx.scaler.step(optimizer)
            ctx.scaler.update()
            if ctx.scaler.get_scale() >= old_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader, start=start_step + 1):
            prompts = batch['prompt']  # list[str], length B
            prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                      padding_side="left", add_special_tokens=False).to(device)
            if args.max_seq_len:
                prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
                prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

            rollout_result, rewards, rew_parts = self._collect_rollout(
                prompts, prompt_inputs, rollout_engine, reward_model, device
            )
            outputs = rollout_result.output_ids
            completion_ids = rollout_result.completion_ids
            completions = rollout_result.completions
            old_per_token_logps = rollout_result.per_token_logps.to(device).detach()
            prompt_lens = rollout_result.prompt_lens.to(device)
            logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=device).unsqueeze(0)
            completion_pad_mask = rollout_result.completion_mask.to(device).bool()
            full_mask = build_full_attention_mask(
                outputs, prompt_lens, completion_pad_mask, tokenizer.pad_token_id
            )

            with ctx.autocast_ctx:
                # 训练前向必须走 DDP 包装后的模型；直接调用 ``model.module`` 会绕过
                # reducer 的 forward 准备阶段，导致多卡梯度不同步。
                res = model(outputs, attention_mask=full_mask)
                aux_loss = res.aux_loss if ctx.lm_config.use_moe else torch.tensor(0.0, device=device)
                per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
            if ctx.metrics is not None:
                # 放在训练前向之后；rollout 的 eval 路由统计不代表实际梯度负载。
                ctx.metrics.record_step()

            if ref_model is None:
                # KL 系数为 0 的 DAPO 无需 reference；占位张量只为复用同一损失函数，
                # detach 后不会引入额外梯度，也不会分配另一份模型。
                ref_per_token_logps = per_token_logps.detach()
            else:
                with torch.no_grad():
                    ref_per_token_logps = F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

            if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
                for i in range(len(prompts)):
                    Logger(f"[DEBUG] step={step}, sample[{i}]")
                    Logger('-'*100)
                    Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                    Logger(prompts[i])
                    Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                    for j in range(args.num_generations):
                        idx = i * args.num_generations + j
                        Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                        Logger(completions[idx])
                        Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                        Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                    Logger('='*100)

            is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask  # [B*num_gen, R]
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            completion_mask = ((torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()  # [B*num_gen, R]
            rewards = self._shape_rewards(rewards, completion_mask)
            advantages, group_std = grouped_advantages(
                rewards, args.num_generations, mode=self.advantage_mode
            )
            valid_groups = self._valid_groups(group_std)
            valid_rows = valid_groups.repeat_interleave(args.num_generations)
            completion_mask = completion_mask * valid_rows.unsqueeze(1)
            grouped_rewards = rewards.view(-1, args.num_generations)

            policy_loss, ratio, _ = clipped_policy_loss(
                per_token_logps, old_per_token_logps, ref_per_token_logps,
                advantages, completion_mask,
                loss_type=args.loss_type,
                epsilon_low=args.epsilon,
                epsilon_high=self._clip_high(),
                kl_beta=self._kl_beta(),
                token_level=self.token_level_loss,
            )
            loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
            ctx.scaler.scale(loss).backward()
            pending_micro_steps += 1

            did_optimizer_step = False
            if pending_micro_steps == args.accumulation_steps:
                optimizer_step(pending_micro_steps)
                pending_micro_steps = 0
                did_optimizer_step = True

            hit_limit = bool(max_steps and epoch * iters + step >= max_steps)
            terminal = step == iters or hit_limit
            if terminal and pending_micro_steps:
                optimizer_step(pending_micro_steps)
                pending_micro_steps = 0
                did_optimizer_step = True

            # Torch 引擎引用同一模型，调用只是更新引用；SGLang 则必须在每次真实
            # optimizer.step 后同步，否则下一批 rollout 来自陈旧策略。
            if did_optimizer_step:
                rollout_engine.update_policy(model)

            if step % args.log_interval == 0 or terminal:
                policy_loss_val = policy_loss.item()
                current_aux_loss = aux_loss.item()
                avg_reward_val = rewards.mean().item()
                avg_len_val = completion_mask.sum(dim=1).float().mean().item()
                kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(completion_mask.sum().item(), 1)
                advantages_mean_val = advantages.mean().item()
                advantages_std_val = advantages.std(unbiased=False).item()
                current_lr = optimizer.param_groups[0]['lr']

                Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                       f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                       f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                       f'Actor Loss: {policy_loss_val:.4f}, Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

                if wandb and is_main_process():
                    wandb.log({
                        "reward": avg_reward_val,
                        "kl_ref": kl_ref_val,
                        "advantages_std": advantages_std_val,
                        "advantages_mean": advantages_mean_val,
                        "policy_loss": policy_loss_val,
                        "avg_response_len": avg_len_val,
                        "learning_rate": current_lr
                    })

                # 逐 step 落盘：与 SFT / 预训练同一张表，并补上 GRPO 的诊断量
                cm = completion_mask.bool()
                with torch.no_grad():
                    if not cm.any():
                        clipfrac_val = ratio_mean_val = perplexity_val = 0.0
                    elif args.loss_type == "cispo":
                        clipfrac_val = (ratio[cm] > args.epsilon_high).float().mean().item()
                        ratio_mean_val = ratio[cm].mean().item()
                        perplexity_val = float(torch.exp(-per_token_logps[cm].mean()))
                    else:
                        clipped = ((ratio[cm] < 1.0 - args.epsilon)
                                   | (ratio[cm] > 1.0 + self._clip_high()))
                        clipfrac_val = clipped.float().mean().item()
                        ratio_mean_val = ratio[cm].mean().item()
                        perplexity_val = float(torch.exp(-per_token_logps[cm].mean()))
                    record_rl_step(ctx, epoch=epoch, step=step, iters=iters,
                                   start_step=start_step, spend_time=time.time() - start_time,
                                   metrics={
                                       "loss": policy_loss_val,
                                       "logits_loss": policy_loss_val,
                                       "aux_loss": current_aux_loss,
                                       "reward": avg_reward_val,
                                       "reward_std": rewards.std(unbiased=False).item(),
                                       "group_reward_std": group_std.mean().item(),
                                       # 组内 reward 全同 -> advantage 恒为 0 -> 这一步白跑
                                       "group_reward_zero_std": (group_std < 1e-6).float().mean().item(),
                                       "adv_zero_frac": (advantages.abs() < 1e-6).float().mean().item(),
                                       "policy_loss": policy_loss_val,
                                       "kl_ref": kl_ref_val,
                                       "perplexity": perplexity_val,
                                       "ratio_mean": ratio_mean_val,
                                       "clipfrac": clipfrac_val,
                                       "advantages_mean": advantages_mean_val,
                                       "advantages_std": advantages_std_val,
                                       "avg_response_len": avg_len_val,
                                       "gen_tokens": int(cm.sum().item()),
                                       "eos_rate": is_eos.any(dim=1).float().mean().item(),
                                       "trunc_rate": 1.0 - is_eos.any(dim=1).float().mean().item(),
                                       **rew_parts,
                                   })

            save_requested = save_requested or step % args.save_interval == 0
            if (terminal or (save_requested and pending_micro_steps == 0)) and is_main_process():
                model.eval()
                self.save_weights(ctx, epoch, step)
                save_resume(ctx.lm_config, self.weight_prefix(), model, optimizer,
                            scaler=ctx.scaler, epoch=epoch, step=step, wandb=wandb,
                            **self.checkpoint_extra())
                model.train()
                save_requested = False

            del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
            del completions, rewards, grouped_rewards, group_std, advantages, completion_mask, completion_pad_mask, prompt_lens, logp_pos

            # --max_steps 预算到点即停（与 runner.run_one_epoch 同口径）
            if hit_limit:
                Logger(f'已达到 max_steps={max_steps}，停止训练')
                break
        return hit_limit
