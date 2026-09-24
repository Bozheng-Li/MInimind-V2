"""Agentic RL：多轮工具调用 + 整轮延迟结算奖励。

主体是 :func:`agent_tools.rollout_single` 的多轮 rollout、变长序列打包，以及只在
模型自己生成的 token 上计算策略梯度；工具返回 token 只作为后续上下文，mask 为 0。
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.lm_dataset import AgentRLDataset

from ...common.checkpoint import save_resume
from ...common.runner import record_rl_step
from ...rollout_engine import compute_per_token_logps
from ...trainer_utils import Logger, is_main_process
from .agent_tools import (
    calculate_rewards,
    collate_agent_batch,
    pack_agent_trajectory,
    rollout_batch,
)
from .common import RLAlgorithm
from .grpo import clipped_policy_loss, grouped_advantages


class AgentAlgorithm(RLAlgorithm):
    name = "agent"

    def build_dataset(self):
        self.dataset = AgentRLDataset(self.args.data_path, self.tokenizer,
                                      max_length=self.args.max_seq_len + self.args.max_gen_len)
        return self.dataset

    def extra_models(self):
        models = super().extra_models()
        Logger(f'Loaded reward model from {self.args.reward_model_path}')
        return models

    def train_epoch(self, ctx, epoch, loader, iters, start_step=0):
        self._ensure_setup(ctx)
        args, device = self.args, ctx.device
        model, tokenizer = ctx.model, self.tokenizer
        optimizer, scheduler = self.optimizer, self.scheduler
        rollout_engine, ref_model, reward_model = self.rollout_engine, self.ref_model, self.reward_model
        wandb = ctx.wandb

        # 嵌套样本需要自定义 collate，沿用框架已建好的 batch_sampler 重建 loader
        loader = DataLoader(loader.dataset, batch_sampler=loader.batch_sampler,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_agent_batch)

        start_time = time.time()
        max_steps = int(getattr(args, "max_steps", 0) or 0)
        pending_micro_steps = 0
        save_requested = False
        hit_limit = False

        def optimizer_step(micro_steps):
            """提交策略梯度；不足一个累积周期的尾组按实际数量重新归一化。"""
            ctx.scaler.unscale_(optimizer)
            correction = args.accumulation_steps / max(micro_steps, 1)
            if correction != 1.0:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            if ctx.metrics is not None:
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
            messages_batch = batch['messages']
            tools_batch = batch['tools']
            gt_batch = batch['gt']
            with torch.no_grad():
                completions, contexts, prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch, turn_outputs_batch, unfinished_batch = rollout_batch(
                    rollout_engine,
                    tokenizer,
                    messages_batch,
                    tools_batch,
                    args.num_generations,
                    max_turns=3,
                    max_new_tokens=args.max_gen_len,
                    thinking_ratio=args.thinking_ratio,
                    temperature=args.rollout_temperature,
                    device=device,
                )

            prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, tools=t) for m, t in zip(messages_batch, tools_batch)]
            packed_samples = []
            for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch):
                packed_samples.append(pack_agent_trajectory(
                    p, r, m, old_lp, args.max_total_len
                ))
            seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed_samples], device=device)
            max_len = seq_lens.max().item()
            input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _, _ in packed_samples], device=device)
            prompt_lens = torch.tensor([prompt_len for _, _, prompt_len, _ in packed_samples], device=device)
            full_response_masks = torch.tensor([mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed_samples], device=device, dtype=torch.float32)
            old_per_token_logps = torch.tensor([old_logps + [0.0] * ((max_len - 1) - len(old_logps)) for _, _, _, old_logps in packed_samples], device=device, dtype=torch.float32)
            full_mask = (input_ids != tokenizer.pad_token_id).long()

            rewards, agent_stats = calculate_rewards(prompts, completions, gt_batch, tools_batch, args.num_generations, reward_model, device=device, turn_outputs_batch=turn_outputs_batch, unfinished_batch=unfinished_batch, return_stats=True)
            with ctx.autocast_ctx:
                # 训练前向必须经过 DDP 包装层，绕过 ``model.module`` 会使梯度不同步。
                res = model(input_ids, attention_mask=full_mask)
                aux_loss = res.aux_loss if ctx.lm_config.use_moe else torch.tensor(0.0, device=device)
                logits = res.logits[:, :-1, :]
                per_token_logps = F.log_softmax(logits, dim=-1).gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            if ctx.metrics is not None:
                # 只记录参与反向的策略前向；工具 rollout 的 eval 统计不代表梯度负载。
                ctx.metrics.record_step()

            with torch.no_grad():
                ref_per_token_logps = compute_per_token_logps(ref_model, input_ids, input_ids.size(1) - 1, attention_mask=full_mask)

            completion_mask = full_response_masks[:, 1:]
            is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()
            eos_idx = torch.full((completion_mask.size(0),), completion_mask.size(1) - 1, device=device, dtype=torch.long)
            has_eos = is_eos.any(dim=1)
            eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
            pos = torch.arange(completion_mask.size(1), device=device).unsqueeze(0)
            completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
            token_counts = completion_mask.sum(dim=1)

            if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
                for i in range(len(messages_batch)):
                    Logger(f"[DEBUG] step={step}, gt[{i}]: {repr(gt_batch[i])}")
                    Logger('-'*100)
                    for j in range(args.num_generations):
                        idx = i * args.num_generations + j
                        plen, slen = prompt_lens[idx].item(), seq_lens[idx].item()
                        Logger(f"{'=' * 30} [DEBUG] gen[{i}][{j}] CONTEXT_BEGIN {'=' * 30}")
                        Logger(contexts[idx])
                        Logger(f"{'=' * 31} [DEBUG] gen[{i}][{j}] CONTEXT_END {'=' * 31}")
                        Logger(f"[DEBUG] gen[{i}][{j}] prompt_len={plen}, seq_len={slen}")
                        tokens = input_ids[idx, plen:slen].tolist()
                        text = tokenizer.decode(tokens, skip_special_tokens=False)
                        Logger(f"{'=' * 28} [DEBUG] gen[{i}][{j}] COMPLETION_BEGIN [{plen}:{slen}] {'=' * 28}")
                        Logger(text)
                        Logger(f"{'=' * 29} [DEBUG] gen[{i}][{j}] COMPLETION_END {'=' * 29}")
                        Logger(f"[DEBUG] gen[{i}][{j}] reward={rewards[idx].item():.4f}")
                        Logger('='*100)

            grouped_rewards = rewards.view(-1, args.num_generations)
            advantages, group_std = grouped_advantages(
                rewards, args.num_generations, mode="grpo"
            )
            policy_loss, _, _ = clipped_policy_loss(
                per_token_logps, old_per_token_logps, ref_per_token_logps,
                advantages, completion_mask,
                loss_type=args.loss_type,
                epsilon_low=args.epsilon,
                epsilon_high=(args.epsilon if args.loss_type == "grpo" else args.epsilon_high),
                kl_beta=args.beta,
                token_level=False,
            )
            loss = (policy_loss + aux_loss) / args.accumulation_steps
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
            if did_optimizer_step:
                rollout_engine.update_policy(model)

            if step % args.log_interval == 0 or terminal:
                pl = policy_loss.item()
                ar = rewards.mean().item()
                al = token_counts.float().mean().item()
                kl = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(token_counts.sum().item(), 1)
                gs = group_std.mean().item()
                am, ast = advantages.mean().item(), advantages.std(unbiased=False).item()
                lr = optimizer.param_groups[0]['lr']
                Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), Reward:{ar:.4f}, KL:{kl:.4f}, GrpStd:{gs:.4f}, AdvStd:{ast:.4f}, Loss:{pl:.4f}, AvgLen:{al:.2f}, AdvMean:{am:.4f}, LR:{lr:.8f}')
                if wandb and is_main_process():
                    wandb.log({"reward":ar,"kl_ref":kl,"group_reward_std":gs,"advantages_std":ast,"policy_loss":pl,"avg_response_len":al,"advantages_mean":am,"learning_rate":lr})

                # 逐 step 落盘：成功率 / 工具调用这些 agentic RL 最该看的量以前算完就丢
                cm = completion_mask.bool()
                with torch.no_grad():
                    perplexity = (float(torch.exp(-per_token_logps[cm].mean()))
                                  if cm.any() else 0.0)
                    record_rl_step(ctx, epoch=epoch, step=step, iters=iters,
                                   start_step=start_step, spend_time=time.time() - start_time,
                                   metrics={
                                       "loss": pl,
                                       "logits_loss": pl,
                                       "aux_loss": aux_loss.item(),
                                       "reward": ar,
                                       "reward_std": rewards.std(unbiased=False).item(),
                                       "group_reward_std": gs,
                                       "group_reward_zero_std": (group_std < 1e-6).float().mean().item(),
                                       "adv_zero_frac": (advantages.abs() < 1e-6).float().mean().item(),
                                       "policy_loss": pl,
                                       "kl_ref": kl,
                                       "perplexity": perplexity,
                                       "advantages_mean": am,
                                       "advantages_std": ast,
                                       "avg_response_len": al,
                                       "gen_tokens": int(cm.sum().item()),
                                       **agent_stats,
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

            del per_token_logps, ref_per_token_logps
            del completions, rewards, grouped_rewards, advantages, completion_mask

            # --max_steps 预算到点即停（与 runner.run_one_epoch 同口径）
            if hit_limit:
                Logger(f'已达到 max_steps={max_steps}，停止训练')
                break
        return hit_limit
