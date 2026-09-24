"""PPO：actor-critic + GAE + KL 早停。

对应重构前的 ``trainer/train_ppo.py``（第 35-306 行）。数值逻辑逐字保留：

- ``CriticModel`` 在本模块定义（继承 ``MiniMindForCausalLM``，加 ``value_head``）
- GAE 逆序递推，外部奖励只加在每条 response 的最后一个有效 token 上，优势白化
- 双层循环：``ppo_update_iters`` × minibatch，逐 minibatch 双 clip 更新
- ``approx_kl`` 超过阈值即早停；早停后仍走 ``loss * 0.0`` 的 forward-backward
  以保持 DDP 通信闭环
- actor / critic 双优化器双调度器，续训时经 ``checkpoint_extra`` 一起写回

与 GRPO / Agent 的差别只在优势估计与 loss 形式，rollout → reward 链路见
``common.py``。
"""
from __future__ import annotations

import math
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.lm_dataset import RLAIFDataset
from model.model_minimind import MiniMindForCausalLM

from ...common.checkpoint import save_resume
from ...common.runner import record_rl_step
from ...common.runtime import build_optimizer as build_adamw
from ...common.runtime import wrap_model
from ...rollout_engine import build_full_attention_mask
from ...trainer_utils import Logger, is_main_process
from .common import RLAlgorithm, calculate_rewards


# 自定义的Critic模型，继承自MiniMindLM
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        # 删除语言模型头：它与 embedding 共享参数，直接“冻结 lm_head”会连带冻结
        # embedding；删除别名既保留 embedding 可训练，又避免 DDP 看到未使用参数。
        del self.lm_head
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 使用基础模型获取隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # ArchModel.forward 返回的 hidden state 已经经过 final norm，不能再归一化一次。
        hidden_states = outputs[0]
        # 使用value_head获取价值估计
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def compute_gae(token_rewards: torch.Tensor, values: torch.Tensor,
                mask: torch.Tensor, gamma: float, lam: float):
    """按有效 response token 计算 GAE 与 return。

    ``mask`` 为 0 的位置既不产生 advantage，也会切断向左的递推。这一点对变长
    batch 很关键：短回答右侧 padding 的 critic 值不能泄漏回 EOS 前的 token。
    """
    batch_size, gen_len = values.shape
    last_gae = values.new_zeros(batch_size)
    reversed_advantages = []
    for t in reversed(range(gen_len)):
        current_mask = mask[:, t]
        if t < gen_len - 1:
            next_value = values[:, t + 1]
            next_nonterminal = mask[:, t + 1]
        else:
            next_value = 0.0
            next_nonterminal = 0.0
        delta = (token_rewards[:, t] + gamma * next_value * next_nonterminal
                 - values[:, t]) * current_mask
        last_gae = (delta + gamma * lam * next_nonterminal * last_gae) * current_mask
        reversed_advantages.append(last_gae)
    advantages = torch.stack(reversed_advantages[::-1], dim=1)
    return advantages, advantages + values * mask


class PPOAlgorithm(RLAlgorithm):
    name = "ppo"

    def build_dataset(self):
        self.dataset = RLAIFDataset(self.args.data_path, self.tokenizer,
                                    max_length=(self.args.max_seq_len + self.args.max_gen_len),
                                    thinking_ratio=self.args.thinking_ratio)
        return self.dataset

    def extra_models(self):
        models = super().extra_models()
        models["critic_model"] = self._build_critic()
        return models

    def _build_critic(self):
        """Critic 复用初始 actor 主干，并新增随机初始化的 value head。

        直接复制当前 actor 的 state_dict，既保证逐参数同源，也支持
        ``--from_weight none``；旧实现硬读权重文件会在从零调试时直接失败。
        """
        critic_model = CriticModel(self.lm_config)
        critic_model.load_state_dict(self.model.state_dict(), strict=False)
        return critic_model.to(self.runtime.device)

    # ---------------- 优化器 / 调度器 / 续训 ---------------- #
    def _build_extra_optimizers(self):
        self.critic_optimizer = build_adamw(self.critic_model, self.args.critic_learning_rate)

    def _total_optimizer_steps(self, iters):
        mb_factor = max(1, math.ceil(self.args.batch_size / self.args.mini_batch_size))
        return math.ceil(self._budget_steps(iters) * self.args.ppo_update_iters
                         * mb_factor / self.args.accumulation_steps)

    def _make_schedulers(self, iters):
        total_steps = self._total_optimizer_steps(iters)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(total_steps, 1),
                                           eta_min=self.args.learning_rate / 10)
        self.critic_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=max(total_steps, 1),
                                                  eta_min=self.args.critic_learning_rate / 10)

    def _bind_models(self, ctx):
        self.critic_model = wrap_model(self.critic_model, self.runtime)

    def _load_extra_state(self, ckp_data):
        for key, obj in (('critic_model', self.critic_model),
                         ('critic_optimizer', self.critic_optimizer),
                         ('scheduler', self.scheduler),
                         ('critic_scheduler', self.critic_scheduler)):
            if key in ckp_data:
                obj.load_state_dict(ckp_data[key])

    def checkpoint_extra(self):
        return {"scheduler": self.scheduler,
                "critic_model": self.critic_model,
                "critic_optimizer": self.critic_optimizer,
                "critic_scheduler": self.critic_scheduler}

    # ---------------- 训练循环 ---------------- #
    def train_epoch(self, ctx, epoch, loader, iters, start_step=0):
        self._ensure_setup(ctx)
        args, device = self.args, ctx.device
        actor_model, critic_model = ctx.model, self.critic_model
        tokenizer = self.tokenizer
        actor_optimizer, critic_optimizer = self.optimizer, self.critic_optimizer
        actor_scheduler, critic_scheduler = self.scheduler, self.critic_scheduler
        rollout_engine = self.rollout_engine
        ref_model, reward_model = self.ref_model, self.reward_model
        wandb = ctx.wandb

        actor_model.train()
        critic_model.train()
        pending_micro_steps = 0
        hit_limit = False
        save_requested = False
        start_time = time.time()
        max_steps = int(getattr(args, "max_steps", 0) or 0)

        def optimizer_step(micro_steps):
            """原子地提交 actor/critic 梯度，保持两个优化器与调度器同步。"""
            ctx.scaler.unscale_(actor_optimizer)
            ctx.scaler.unscale_(critic_optimizer)
            correction = args.accumulation_steps / max(micro_steps, 1)
            if correction != 1.0:
                for parameter in list(actor_model.parameters()) + list(critic_model.parameters()):
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            if ctx.metrics is not None:
                ctx.metrics.record_grads()
            max_norm = args.grad_clip if args.grad_clip > 0 else float("inf")
            ctx.last_grad_norm = float(clip_grad_norm_(actor_model.parameters(), max_norm))
            clip_grad_norm_(critic_model.parameters(), max_norm)
            old_scale = ctx.scaler.get_scale()
            ctx.scaler.step(actor_optimizer)
            ctx.scaler.step(critic_optimizer)
            ctx.scaler.update()
            if ctx.scaler.get_scale() >= old_scale:
                actor_scheduler.step()
                critic_scheduler.step()
            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader, start=start_step + 1):
            prompts = batch["prompt"]  # list[str], length B
            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len,
                            padding_side="left").to(device)  # input_ids: [B, P], attention_mask: [B, P]

            rollout_result = rollout_engine.rollout(
                prompt_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                num_generations=1,
                max_new_tokens=args.max_gen_len,
                # 与 GRPO / DAPO / RLOO 共用同一个采样温度开关。这里若写死
                # 0.8，用户调低温度做稳定性实验时 PPO 会静默忽略配置，采样
                # 分布、old log-prob 和最终 importance ratio 都会偏离预期。
                temperature=args.rollout_temperature,
            )
            gen_out = rollout_result.output_ids
            completion_ids = rollout_result.completion_ids
            prompt_lens = rollout_result.prompt_lens.to(device)
            responses_text = rollout_result.completions
            old_resp_logp = rollout_result.per_token_logps.to(device)
            rewards, rew_parts = calculate_rewards(prompts, responses_text, reward_model, 1, device,
                                                   return_breakdown=True)  # [B]
            if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
                for i in range(len(prompts)):
                    Logger(f"[DEBUG] step={step}, sample[{i}]")
                    Logger('-'*100)
                    Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                    Logger(prompts[i])
                    Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                    Logger(f"[DEBUG] prompt_len={prompt_lens[i].item()}, response_len={len(responses_text[i])}")
                    Logger(f"{'=' * 28} [DEBUG] sample[{i}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(responses_text[i])
                    Logger(f"{'=' * 29} [DEBUG] sample[{i}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] reward={rewards[i].item():.4f}")
                    Logger('='*100)

            labels = gen_out[:, 1:].clone()  # [B, P+R-1]
            B = len(prompts)
            resp_labels = completion_ids
            resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)
            logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
            resp_pad_mask = rollout_result.completion_mask.to(device).bool()
            full_mask = build_full_attention_mask(
                gen_out, prompt_lens, resp_pad_mask, tokenizer.pad_token_id
            )
            resp_lengths = resp_pad_mask.sum(dim=1); valid_resp = resp_lengths > 0; eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask
            has_eos = eos_mask.any(dim=1); eos_pos = torch.argmax(eos_mask.int(), dim=1)
            resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)
            resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()
            resp_value_mask = resp_policy_mask.clone()

            # old value 与 rollout log-prob 一样属于行为策略快照。Critic 若保持 train
            # 模式，dropout 会让尚未更新的同一模型也产生不同 value，破坏 value clip。
            critic_was_training = critic_model.training
            critic_model.eval()
            try:
                with torch.no_grad():
                    critic_for_rollout = (critic_model.module
                                          if isinstance(critic_model, DistributedDataParallel)
                                          else critic_model)
                    values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
                    old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask

                    ref_resp_logp = F.log_softmax(ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
                    token_rewards = torch.zeros_like(old_resp_logp)
                    last_idx = resp_lengths - 1  # [B]
                    token_rewards[torch.arange(B, device=device)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]  # 末尾加外部奖励

                    advantages, returns = compute_gae(
                        token_rewards, old_resp_values, resp_value_mask, args.gamma, args.lam
                    )

                    valid_token_count = resp_policy_mask.sum().clamp(min=1)
                    adv_raw_mean = (advantages * resp_policy_mask).sum() / valid_token_count
                    adv_raw_var = (((advantages - adv_raw_mean) ** 2 * resp_policy_mask).sum()
                                   / valid_token_count)
                    advantages = ((advantages - adv_raw_mean)
                                  * torch.rsqrt(adv_raw_var + 1e-8) * resp_policy_mask)
                    adv_mean = (advantages * resp_policy_mask).sum() / valid_token_count
                    adv_var = (((advantages - adv_mean) ** 2 * resp_policy_mask).sum()
                               / valid_token_count)
            finally:
                if critic_was_training:
                    critic_model.train()

            mb_size = max(1, min(args.mini_batch_size, B))
            stop_ppo = False
            policy_loss_sum = 0.0
            value_loss_sum = 0.0
            kl_sum = 0.0
            kl_ref_sum = 0.0
            clipfrac_sum = 0.0
            aux_loss_sum = 0.0
            log_count = 0
            actor_updated = False
            for ppo_epoch in range(args.ppo_update_iters):
                if stop_ppo:
                    break
                b_inds = torch.randperm(B, device=device)
                for i in range(0, B, mb_size):
                    inds = b_inds[i:i + mb_size]

                    with ctx.autocast_ctx:
                        # 两个训练前向都必须经过 DDP wrapper，不能调用 ``.module``。
                        mb_values_seq = critic_model(input_ids=gen_out[inds],
                                                     attention_mask=full_mask[inds])
                        mb_resp_values = mb_values_seq.gather(1, logp_pos[inds])
                        res = actor_model(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                        aux_loss = res.aux_loss if ctx.lm_config.use_moe else torch.tensor(0.0, device=device)
                        # 在 autocast 内计算 log_softmax，避免直接对 fp16/bf16 logits
                        # 计算造成额外数值偏差。
                        mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1).gather(2, labels[inds].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos[inds])

                    log_ratio = mb_resp_logp - old_resp_logp[inds]

                    # 可开关的诊断：观察首轮首个 minibatch 的 mb 与 old logp 差异。
                    if args.debug_log_ratio and ppo_epoch == 0 and i == 0 and is_main_process():
                        _lr = log_ratio.detach()
                        _m = resp_policy_mask[inds].bool()
                        if _m.any():
                            _lrv = _lr[_m]
                            Logger(f"[DBG log_ratio] step={step} max|lr|={_lrv.abs().max().item():.6e} "
                                   f"mean|lr|={_lrv.abs().mean().item():.6e} "
                                   f"ratio_max={torch.exp(_lrv).max().item():.6f} "
                                   f"ratio_min={torch.exp(_lrv).min().item():.6f} "
                                   f"dropout={getattr(ctx.lm_config, 'dropout', None)} "
                                   f"training={actor_model.training}")
                    approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)

                    # 同步各卡的 approx_kl，防止某卡 break 而其它卡继续导致 DDP 死锁
                    approx_kl_val = approx_kl.detach().clone()
                    if dist.is_initialized():
                        dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)

                    if approx_kl_val > args.early_stop_kl:
                        stop_ppo = True

                    ratio = torch.exp(log_ratio)
                    clipfrac = ((((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[inds]).sum()
                                / resp_policy_mask[inds].sum().clamp(min=1))
                    kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp) - (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
                                      * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                    policy_loss = ((torch.max(-advantages[inds] * ratio,
                                              -advantages[inds] * torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon))
                                   * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                                   + args.kl_coef * kl_ref_penalty)
                    value_loss = 0.5 * (torch.max((mb_resp_values - returns[inds]) ** 2,
                                                  (torch.clamp(mb_resp_values, old_resp_values[inds] - args.cliprange_value,
                                                               old_resp_values[inds] + args.cliprange_value) - returns[inds]) ** 2)
                                        * resp_value_mask[inds]).sum() / resp_value_mask[inds].sum().clamp(min=1)

                    kl = approx_kl_val
                    kl_ref = kl_ref_penalty.detach()

                    # 早停时必须保证 forward-backward 闭环，故只截断 loss 不中断 DDP 通信
                    if stop_ppo:
                        loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0
                    else:
                        loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps

                    ctx.scaler.scale(loss).backward()

                    policy_loss_sum += policy_loss.item()
                    value_loss_sum += value_loss.item()
                    kl_sum += kl.item()
                    kl_ref_sum += kl_ref.item()
                    clipfrac_sum += clipfrac.item()
                    aux_loss_sum += aux_loss.item()
                    log_count += 1

                    # 超过 KL 阈值的 minibatch 只做一次零梯度 backward 来闭合 DDP
                    # 通信，不应算进梯度累积的分母，也不应触发一次空 optimizer.step。
                    if not stop_ppo:
                        pending_micro_steps += 1
                        if pending_micro_steps == args.accumulation_steps:
                            optimizer_step(pending_micro_steps)
                            pending_micro_steps = 0
                            actor_updated = True
                    else:
                        break

            if ctx.metrics is not None:
                # PPO 每批会做多次 actor 前向；读取最后一次训练前向留下的路由统计。
                ctx.metrics.record_step()

            hit_limit = bool(max_steps and epoch * iters + step >= max_steps)
            terminal = step == iters or hit_limit
            if terminal and pending_micro_steps:
                optimizer_step(pending_micro_steps)
                pending_micro_steps = 0
                actor_updated = True

            if actor_updated:
                rollout_engine.update_policy(actor_model)

            # 指标在所有 rank 上都要算：record_rl_step 内部的 snapshot 会做跨卡
            # all_reduce，只在 rank0 调会让其余卡卡在集合通信上（Logger/wandb 仍只 rank0）
            critic_loss_val = value_loss_sum / max(log_count, 1)
            reward_val = rewards.mean().item()
            approx_kl_val = kl_sum / max(log_count, 1)
            kl_ref_val = kl_ref_sum / max(log_count, 1)
            clipfrac_val = clipfrac_sum / max(log_count, 1)
            policy_loss_val = policy_loss_sum / max(log_count, 1)
            avg_len_val = resp_lengths.float().mean().item()
            actor_lr, critic_lr = actor_optimizer.param_groups[0]['lr'], critic_optimizer.param_groups[0]['lr']

            with torch.no_grad():
                cm = resp_policy_mask.bool()
                perplexity = (float(torch.exp(-old_resp_logp[cm].mean()))
                              if cm.any() else 0.0)
                record_rl_step(ctx, epoch=epoch, step=step, iters=iters,
                               start_step=start_step, spend_time=time.time() - start_time,
                               metrics={
                                   "loss": policy_loss_val,
                                   "logits_loss": policy_loss_val,
                                   "aux_loss": aux_loss_sum / max(log_count, 1),
                                   "reward": reward_val,
                                   "reward_std": rewards.std(unbiased=False).item(),
                                   "policy_loss": policy_loss_val,
                                   "critic_loss": critic_loss_val,
                                   "value_loss": value_loss_sum / max(log_count, 1),
                                   "kl_ref": kl_ref_val,
                                   "approx_kl": approx_kl_val,
                                   "perplexity": perplexity,
                                   "clipfrac": clipfrac_val,
                                   "advantages_mean": adv_mean.item(),
                                   "advantages_std": adv_var.sqrt().item(),
                                   "adv_raw_mean": adv_raw_mean.item(),
                                   "adv_raw_std": adv_raw_var.sqrt().item(),
                                   "avg_response_len": avg_len_val,
                                   "gen_tokens": int(cm.sum().item()),
                                   # 早停触发次数：PPO 独有，以前完全没记录
                                   "kl_early_stop": int(stop_ppo),
                                   "actor_lr": actor_lr,
                                   "critic_lr": critic_lr,
                                   **rew_parts,
                               })

            if is_main_process():
                if wandb is not None:
                    wandb.log({
                        "reward": reward_val,
                        "kl_ref": kl_ref_val,
                        "approx_kl": approx_kl_val,
                        "clipfrac": clipfrac_val,
                        "critic_loss": critic_loss_val,
                        "avg_response_len": avg_len_val,
                        "actor_lr": actor_lr,
                        "critic_lr": critic_lr,
                    })

                Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                       f"Reward: {reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, Approx KL: {approx_kl_val:.4f}, "
                       f"ClipFrac: {clipfrac_val:.4f}, Critic Loss: {critic_loss_val:.4f}, "
                       f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

            save_requested = save_requested or step % args.save_interval == 0
            if (terminal or (save_requested and pending_micro_steps == 0)) and is_main_process():
                actor_model.eval()
                self.save_weights(ctx, epoch, step)
                # 续训档里额外带上 critic 的状态
                save_resume(ctx.lm_config, self.weight_prefix(), actor_model, actor_optimizer,
                            scaler=ctx.scaler, epoch=epoch, step=step, wandb=wandb,
                            **self.checkpoint_extra())
                actor_model.train()
                save_requested = False

            # --max_steps 是跨 epoch 的全局预算；先保存最后一步，再退出。
            if hit_limit:
                Logger(f'已达到 max_steps={max_steps}，停止训练')
                break

            del enc, gen_out, completion_ids, responses_text, rewards, full_mask, values_seq, advantages
            del labels, resp_labels, resp_idx, resp_pad_mask, valid_resp, eos_mask, has_eos, eos_pos, resp_lengths, resp_policy_mask, resp_value_mask, old_resp_logp, ref_resp_logp
            del kl, kl_ref, policy_loss, value_loss, loss, token_rewards, returns, old_resp_values, prompt_lens, logp_pos
        return hit_limit
