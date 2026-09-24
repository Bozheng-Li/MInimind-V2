"""训练算法的公式级回归测试。

这些测试只使用很小的手工张量，不需要数据集、奖励模型或 GPU。目的不是验证“代码
能 import”，而是固定 SFT、离线偏好、在线 RL、Agent 轨迹与 LoRA 的数学边界。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from arch.model import causal_lm_cross_entropy
from model.model_lora import apply_lora, apply_lora_from_checkpoint, save_lora
from trainer.algos.rl.agent_tools import pack_agent_trajectory, rollout_single
from trainer.algos.rl.grpo import clipped_policy_loss, grouped_advantages
from trainer.algos.rl.ppo import compute_gae
from trainer.algos.rl.preference import preference_objective
from trainer.rollout_engine import RolloutResult, build_full_attention_mask


class PreferenceLossTests(unittest.TestCase):
    def setUp(self):
        self.pc = torch.tensor([-2.0, -1.0])
        self.pr = torch.tensor([-3.0, -2.0])
        self.pca = self.pc / 2
        self.pra = self.pr / 2
        self.rc = torch.tensor([-2.5, -1.5])
        self.rr = torch.tensor([-2.5, -1.5])

    def test_dpo_matches_closed_form(self):
        loss, logits = preference_objective(
            "dpo", self.pc, self.pr, self.pca, self.pra,
            ref_chosen_sum=self.rc, ref_rejected_sum=self.rr, beta=0.2,
        )
        expected_logits = (self.pc - self.rc) - (self.pr - self.rr)
        expected = -torch.nn.functional.logsigmoid(0.2 * expected_logits).mean()
        torch.testing.assert_close(logits, expected_logits)
        torch.testing.assert_close(loss, expected)

    def test_ipo_has_finite_target(self):
        beta = 0.5
        target = 1.0 / (2.0 * beta)
        # 序列总 log-prob 的间隔故意不等于 target；每 token 平均间隔恰好
        # 等于 target，以此锁定 IPO 必须采用长度归一化口径。
        loss, logits = preference_objective(
            "ipo", torch.tensor([target * 4]), torch.tensor([0.0]),
            torch.tensor([target]), torch.tensor([0.0]),
            ref_chosen_sum=torch.tensor([0.0]), ref_rejected_sum=torch.tensor([0.0]),
            ref_chosen_avg=torch.tensor([0.0]), ref_rejected_avg=torch.tensor([0.0]),
            beta=beta,
        )
        torch.testing.assert_close(logits, torch.tensor([target]))
        self.assertAlmostEqual(loss.item(), 0.0, places=7)

    def test_cpo_uses_sequence_sum_not_token_average(self):
        loss, logits = preference_objective(
            "cpo",
            policy_chosen_sum=torch.tensor([-2.0]),
            policy_rejected_sum=torch.tensor([-6.0]),
            policy_chosen_avg=torch.tensor([-1.0]),
            policy_rejected_avg=torch.tensor([-1.5]),
            chosen_nll=torch.tensor(1.25),
            beta=0.2,
            sft_weight=0.0,
        )
        expected_logits = torch.tensor([4.0])
        expected_loss = -torch.nn.functional.logsigmoid(0.2 * expected_logits).mean()
        torch.testing.assert_close(logits, expected_logits)
        torch.testing.assert_close(loss, expected_loss)

    def test_reference_free_losses_are_finite(self):
        for name in ("simpo", "cpo", "orpo"):
            loss, logits = preference_objective(
                name, self.pc, self.pr, self.pca, self.pra,
                chosen_nll=torch.tensor(1.25), beta=0.2,
            )
            self.assertTrue(torch.isfinite(loss), name)
            self.assertTrue(torch.isfinite(logits).all(), name)

    def test_kto_has_gradients_for_both_classes(self):
        pc = self.pc.clone().requires_grad_()
        pr = self.pr.clone().requires_grad_()
        loss, _ = preference_objective(
            "kto", pc, pr, self.pca, self.pra,
            ref_chosen_sum=self.rc, ref_rejected_sum=self.rr, beta=0.2,
        )
        loss.backward()
        self.assertGreater(pc.grad.abs().sum().item(), 0)
        self.assertGreater(pr.grad.abs().sum().item(), 0)

    def test_kto_accepts_detached_global_kl_baselines(self):
        beta = 0.2
        chosen_kl = torch.tensor(0.3)
        rejected_kl = torch.tensor(0.4)
        loss, _ = preference_objective(
            "kto", self.pc, self.pr, self.pca, self.pra,
            ref_chosen_sum=self.rc,
            ref_rejected_sum=self.rr,
            beta=beta,
            kto_chosen_kl=chosen_kl,
            kto_rejected_kl=rejected_kl,
        )
        chosen_ratio = self.pc - self.rc
        rejected_ratio = self.pr - self.rr
        expected = torch.cat((
            1.0 - torch.sigmoid(beta * (chosen_ratio - rejected_kl)),
            1.0 - torch.sigmoid(beta * (chosen_kl - rejected_ratio)),
        )).mean()
        torch.testing.assert_close(loss, expected)


class SupervisedLossTests(unittest.TestCase):
    def test_all_ignored_sft_batch_returns_connected_zero(self):
        logits = torch.randn(2, 4, 7, requires_grad=True)
        labels = torch.full((2, 4), -100, dtype=torch.long)
        loss = causal_lm_cross_entropy(logits, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.abs().sum().item(), 0.0)


class OnlineRLMathTests(unittest.TestCase):
    def test_agent_rollout_forwards_configured_temperature(self):
        class TinyEncoding(dict):
            """只实现 rollout_single 所需的 BatchEncoding 最小接口。"""

            def to(self, device):
                return TinyEncoding({key: value.to(device) for key, value in self.items()})

        class TinyTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def apply_chat_template(self, *args, **kwargs):
                return "prompt"

            def __call__(self, *args, **kwargs):
                return TinyEncoding({
                    "input_ids": torch.tensor([[1, 3]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                })

        class RecordingEngine:
            def __init__(self):
                self.temperature = None

            def rollout(self, prompt_ids, attention_mask, num_generations,
                        max_new_tokens, temperature):
                self.temperature = temperature
                return RolloutResult(
                    output_ids=torch.tensor([[1, 3, 4]]),
                    completion_ids=torch.tensor([[4]]),
                    per_token_logps=torch.tensor([[-0.5]]),
                    completions=["回答"],
                    prompt_lens=torch.tensor([2]),
                    completion_mask=torch.tensor([[1]]),
                )

        engine = RecordingEngine()
        rollout_single(
            engine,
            TinyTokenizer(),
            messages=[{"role": "user", "content": "测试"}],
            tools=[],
            temperature=0.37,
            device="cpu",
        )
        self.assertEqual(engine.temperature, 0.37)

    def test_agent_trajectory_truncation_keeps_logps_aligned(self):
        ids, mask, prompt_len, old_logps = pack_agent_trajectory(
            prompt_ids=[10, 11, 12],
            response_ids=[20, 21, 30, 31],
            response_mask=[1, 1, 0, 1],
            response_old_logps=[0.2, 0.3, 0.0, 0.5],
            max_total_len=5,
        )
        self.assertEqual(ids, [12, 20, 21, 30, 31])
        self.assertEqual(mask, [0, 1, 1, 0, 1])
        self.assertEqual(prompt_len, 1)
        self.assertEqual(len(old_logps), len(ids) - 1)
        torch.testing.assert_close(torch.tensor(old_logps),
                                   torch.tensor([0.2, 0.3, 0.0, 0.5]))

    def test_rollout_mask_removes_repeated_eos(self):
        output_ids = torch.tensor([[0, 5, 6, 2, 2, 2]])
        mask = build_full_attention_mask(
            output_ids,
            prompt_lens=torch.tensor([3]),
            completion_mask=torch.tensor([[1, 0, 0]]),
            pad_token_id=0,
        )
        torch.testing.assert_close(mask, torch.tensor([[0, 1, 1, 1, 0, 0]]))

    def test_grpo_group_normalization(self):
        advantages, std = grouped_advantages(torch.tensor([1.0, 2.0, 3.0, 7.0]), 2, "grpo")
        grouped = advantages.view(2, 2)
        torch.testing.assert_close(grouped.mean(dim=1), torch.zeros(2), atol=1e-6, rtol=0)
        torch.testing.assert_close(std, torch.tensor([0.5, 2.0]))

    def test_rloo_leave_one_out_baseline(self):
        advantages, _ = grouped_advantages(torch.tensor([1.0, 2.0, 4.0]), 3, "rloo")
        torch.testing.assert_close(advantages, torch.tensor([-2.0, -0.5, 2.5]))

    def test_dapo_asymmetric_clipping(self):
        ratio = torch.tensor([[1.25], [0.75]])
        new_logp = ratio.log().requires_grad_()
        old_logp = torch.zeros_like(new_logp)
        advantages = torch.tensor([1.0, -1.0])
        mask = torch.ones_like(new_logp)
        loss, measured_ratio, _ = clipped_policy_loss(
            new_logp, old_logp, new_logp.detach(), advantages, mask,
            loss_type="grpo", epsilon_low=0.2, epsilon_high=0.2,
            kl_beta=0.0, token_level=True,
        )
        # 正优势上界裁成 1.2 -> loss=-1.2；负优势下界裁成 0.8 -> loss=+0.8。
        self.assertAlmostEqual(loss.item(), -0.2, places=6)
        torch.testing.assert_close(measured_ratio, ratio)

    def test_gae_stops_at_padding_boundary(self):
        rewards = torch.tensor([[0.0, 1.0, 99.0]])
        values = torch.tensor([[0.2, 0.3, 50.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        advantages, returns = compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)
        # padding 位无论 reward/value 多大都必须为 0，且不能影响左侧两个有效 token。
        self.assertEqual(advantages[0, 2].item(), 0.0)
        self.assertEqual(returns[0, 2].item(), 0.0)
        torch.testing.assert_close(advantages[0, :2], torch.tensor([0.8, 0.7]))


class LoRATests(unittest.TestCase):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(3, 4, bias=False)

    def test_zero_init_preserves_output_and_scaling(self):
        model = self.Tiny()
        x = torch.ones(2, 3)
        before = model.q_proj(x).detach()
        apply_lora(model, rank=2, alpha=4, dropout=0, target_modules="q_proj")
        torch.testing.assert_close(model.q_proj(x), before)

        nn.init.ones_(model.q_proj.lora.A.weight)
        nn.init.ones_(model.q_proj.lora.B.weight)
        delta = model.q_proj(x) - before
        # A: 每个秩通道为 3；B: 两个通道相加为 6；alpha/r=2，最终为 12。
        torch.testing.assert_close(delta, torch.full_like(delta, 12.0))

    def test_checkpoint_restores_rank_alpha_and_targets(self):
        source = self.Tiny()
        apply_lora(source, rank=2, alpha=6, dropout=0, target_modules="q_proj")
        nn.init.ones_(source.q_proj.lora.A.weight)
        nn.init.ones_(source.q_proj.lora.B.weight)

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "adapter.pth")
            save_lora(source, path)
            target = self.Tiny()
            target.q_proj.weight.data.copy_(source.q_proj.weight.data)
            apply_lora_from_checkpoint(target, path)

        self.assertEqual(target.q_proj.lora.rank, 2)
        self.assertEqual(target.q_proj.lora.alpha, 6.0)
        self.assertEqual(target.q_proj.lora.scaling, 3.0)
        torch.testing.assert_close(target.q_proj(torch.ones(1, 3)),
                                   source.q_proj(torch.ones(1, 3)))


if __name__ == "__main__":
    unittest.main()
