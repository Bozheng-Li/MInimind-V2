"""训练指标的富采集。

普通的 loss / lr 不足以诊断一次几十小时的训练。这里补充四类关键信息：

1. **优化动力学**：参数按组（attn / ffn / embed / norm）拆分的梯度范数、权重范数，
   以及 update-to-weight ratio（``lr·‖g‖ / ‖w‖``）——判断哪部分在学、是否停滞。
2. **MoE 健康度**（本项目的重点）：专家负载分布、路由熵、负载均衡偏置的演化。
   aux-loss-free 的均衡完全靠 bias 反馈回路，**专家被饿死不会体现在 loss 上**，
   这是最典型的隐性故障。
3. **泛化**：固定留出集上的验证 loss —— 区分「还在学」与「开始过拟合」。
4. **系统**：GPU 利用率采样、显存峰值。

采集本身不修改任何数值，也不参与计算图。
"""
from __future__ import annotations

import os
import subprocess
import threading
from collections import defaultdict

import torch
import torch.distributed as dist


def _group_of(name: str) -> str:
    """把参数名归到 attn / ffn / embed / norm 四组。"""
    if 'self_attn' in name:
        return 'attn'
    if '.mlp.' in name:
        return 'ffn'
    if 'embed_tokens' in name or 'lm_head' in name:
        return 'embed'
    return 'norm'


class GpuSampler:
    """后台采样 GPU 利用率（nvidia-smi 轮询，放在单独线程，不阻塞训练）。"""

    def __init__(self, gpu_ids, interval=5.0):
        self.gpu_ids = list(gpu_ids)
        self.interval = interval
        self.samples = []          # [(util%, mem_mb), ...]
        self._stop = threading.Event()
        self._th = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                     '--format=csv,noheader,nounits'],
                    timeout=5, stderr=subprocess.DEVNULL).decode()
                for line in out.strip().splitlines():
                    idx, util, mem = [x.strip() for x in line.split(',')]
                    if int(idx) in self.gpu_ids:
                        self.samples.append((float(util), float(mem)))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval)

    def start(self):
        if self._th is None:
            self._th = threading.Thread(target=self._loop, daemon=True)
            self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join(timeout=2)

    def snapshot_and_reset(self):
        """返回窗口内的 (利用率均值, 利用率最低值, 显存峰值MB) 并清空。"""
        if not self.samples:
            return None, None, None
        utils = [s[0] for s in self.samples]
        mems = [s[1] for s in self.samples]
        out = (sum(utils) / len(utils), min(utils), max(mems))
        self.samples = []
        return out


class MetricsCollector:
    """持有跨 step 的累积状态，并在每个 log 点产出完整指标。"""

    def __init__(self, model, runtime, hidden_size: int, use_gpu_sampler=True):
        self.model = model
        # 剥掉 DDP / compile 包装，便于按层访问子模块
        raw = model.module if hasattr(model, 'module') else model
        self._raw = getattr(raw, '_orig_mod', raw)
        self.runtime = runtime
        self.hidden = hidden_size

        # MoE 层：以 e_score_correction_bias 为标志（在 __init__ 里就已存在，无需前向）
        self.moe_modules = [m for m in model.modules() if hasattr(m, 'e_score_correction_bias')]

        self._load_sum = None      # [num_experts] 累积负载
        self._scores_sum = None
        self._moe_steps = 0
        self._grad_by_group = {}
        self._last_step = {}

        gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        ids = [int(x) for x in gpus.split(',') if x.strip().isdigit()]
        self.gpu_sampler = GpuSampler(ids) if (use_gpu_sampler and ids) else None
        if self.gpu_sampler:
            self.gpu_sampler.start()

    # ------------------------------------------------------------------ #
    def record_grads(self):
        """在梯度裁剪前调用（此刻 grad 仍在且已被 DDP 平均过，是全局梯度）。"""
        acc = defaultdict(float)
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                acc[_group_of(n)] += float(p.grad.detach().float().pow(2).sum())
        self._grad_by_group = {k: v ** 0.5 for k, v in acc.items()}

    def record_step(self):
        """每个训练步调用：累积 MoE 负载（本地累积，到 log 点再跨卡汇总）。"""
        if not self.moe_modules:
            return
        with torch.no_grad():
            for m in self.moe_modules:
                load = getattr(m, 'last_load', None)
                if load is None:
                    continue
                if self._load_sum is None:
                    self._load_sum = torch.zeros_like(load, dtype=torch.float32)
                    self._scores_sum = torch.zeros_like(load, dtype=torch.float32)
                self._load_sum += load.float()
                sm = getattr(m, 'last_scores_mean', None)
                if sm is not None:
                    self._scores_sum += sm.float()
            self._moe_steps += 1

    # ------------------------------------------------------------------ #
    def _allreduce(self, t):
        if t is not None and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t

    # ------------------------------------------------------------------ #
    def _component_stats(self) -> dict:
        """汇总各组件在最近一次前向里记录的中间统计（逐层 + 跨层均值）。

        这些字段由组件自己在 forward 里写入（``last_*``），框架只负责读取。
        """
        attn_keys = ('last_q_rms', 'last_k_rms', 'last_v_rms', 'last_out_rms',
                     'last_gate_mean', 'last_gate_std', 'last_gate_sat_lo', 'last_gate_sat_hi')
        moe_keys = ('last_routed_rms', 'last_shared_rms')
        acc = defaultdict(list)
        for blk in getattr(self._raw, 'model', self._raw).layers:
            attn = blk.self_attn
            for k in attn_keys:
                v = getattr(attn, k, None)
                if v is not None:
                    acc[k].append(float(v))
            for k in moe_keys:
                v = getattr(blk.mlp, k, None)
                if v is not None:
                    acc[k].append(float(v))

        out = {}
        for k, vals in acc.items():
            name = k.replace('last_', '')
            out[name] = round(sum(vals) / len(vals), 6)
            # 逐层也记一份（列名 _L0..），用于画「深度方向」的热力图
            if len(vals) > 1:
                for i, x in enumerate(vals):
                    out[f'{name}_L{i}'] = round(x, 6)

        # 共享专家占总 FFN 输出的比例（判断共享专家是否主导）
        r, s = out.get('routed_rms'), out.get('shared_rms')
        if r is not None and s is not None and (r + s) > 0:
            out['shared_share'] = round(s / (r + s), 4)

        # 残差流逐层 RMS
        hid = getattr(getattr(self._raw, 'model', self._raw), 'last_hidden_rms', None)
        if hid:
            out['hidden_rms'] = round(sum(hid) / len(hid), 6)
            for i, x in enumerate(hid):
                out[f'hidden_rms_L{i}'] = round(x, 6)
        return out

    @staticmethod
    def _ram_mb():
        try:
            import psutil
            vm = psutil.virtual_memory()
            return round(vm.used / 1024 ** 2, 0), round(vm.percent, 1)
        except Exception:  # noqa: BLE001
            return None, None

    def _torch_mem_mb(self):
        try:
            if 'cuda' in str(self.runtime.device):
                return (round(torch.cuda.memory_allocated() / 1024 ** 2, 1),
                        round(torch.cuda.memory_reserved() / 1024 ** 2, 1),
                        round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1))
        except Exception:  # noqa: BLE001
            pass
        return None, None, None

    def snapshot(self) -> dict:
        """产出本窗口的完整指标（并清空窗口累积）。"""
        out = {}

        # ---- 优化动力学 ----
        wnorm = defaultdict(float)
        wcount = defaultdict(int)
        for n, p in self.model.named_parameters():
            g = _group_of(n)
            wnorm[g] += float(p.detach().float().pow(2).sum())
            wcount[g] += p.numel()
        wnorm = {k: v ** 0.5 for k, v in wnorm.items()}
        out['weight_norm'] = sum(v * v for v in wnorm.values()) ** 0.5
        for g in ('attn', 'ffn', 'embed'):
            out[f'weight_norm_{g}'] = round(wnorm.get(g, 0.0), 4)

        total_grad = 0.0
        for g in ('attn', 'ffn', 'embed', 'norm'):
            gn = self._grad_by_group.get(g, 0.0)
            out[f'grad_norm_{g}'] = round(gn, 4)
            total_grad += gn * gn
            # update-to-weight ratio：**逐元素**口径 rms(Δw)/rms(w)。
            # Adam 的逐元素更新幅度 ≈ lr，故 rms(Δw) ≈ lr，rms(w) = ‖w‖/√参数数。
            # ⚠️ 不能写成 lr·‖g‖/‖w‖ —— 两个组级范数相除会漏掉参数量，
            # 结果比正确值小 √N 量级（实测差 5 个数量级，会得出「FFN 几乎没在学」的错误结论）。
            cnt = wcount.get(g, 0)
            wn = wnorm.get(g, 0.0)
            if cnt > 0 and wn > 0:
                rms_w = wn / (cnt ** 0.5)
                out[f'update_ratio_{g}'] = float(
                    f"{self._last_step.get('lr', 0.0) / rms_w:.3e}")
        out['grad_norm_sum_groups'] = round(total_grad ** 0.5, 4)

        # ---- MoE ----
        if self._load_sum is not None and self._moe_steps:
            load = self._allreduce(self._load_sum.clone())
            scores = self._allreduce(self._scores_sum.clone())
            n_exp = load.numel()
            n_layer_steps = self._moe_steps * len(self.moe_modules)
            per = (load / max(n_layer_steps, 1)).float()          # 每个专家平均被选次数/层/step
            mean = float(per.mean())
            std = float(per.std(unbiased=False))
            out['moe_load_mean'] = round(mean, 3)
            out['moe_load_cv'] = round(std / mean, 4) if mean > 0 else None
            out['moe_load_maxmin'] = round(float(per.max() / (per.min() + 1e-9)), 3)
            # 「饿死」定义：负载不到平均值的 1/10
            out['moe_dead_experts'] = int((per < mean * 0.1).sum().item())
            # 路由熵（归一化到 [0,1]，1 = 完全均匀）
            p = scores.float()
            p = p / (p.sum() + 1e-12)
            ent = float(-(p * (p + 1e-12).log()).sum())
            out['moe_entropy_norm'] = round(ent / torch.log(torch.tensor(float(n_exp))).item(), 4)
            # 每个专家的归一化负载（供画热力图）—— 16 列
            for i in range(n_exp):
                out[f'moe_load_e{i}'] = round(float(per[i]), 4)
            # 负载均衡偏置
            b = self.moe_modules[0].e_score_correction_bias.detach().float()
            out['moe_bias_std'] = round(float(b.std(unbiased=False)), 5)
            out['moe_bias_absmean'] = round(float(b.abs().mean()), 5)
            out['moe_bias_nonzero'] = int((b.abs() > 1e-9).sum().item())

        # ---- 系统 ----
        if self.gpu_sampler:
            util, util_min, mem_peak = self.gpu_sampler.snapshot_and_reset()
            out['gpu_util_mean'] = round(util, 1) if util is not None else None
            out['gpu_util_min'] = util_min
            out['gpu_mem_peak_mb'] = mem_peak
        alloc, reserved, tpeak = self._torch_mem_mb()
        out['gpu_mem_alloc_mb'] = alloc
        out['gpu_mem_reserved_mb'] = reserved
        out['gpu_mem_peak_torch_mb'] = tpeak
        ram, ram_pct = self._ram_mb()
        out['ram_used_mb'] = ram
        out['ram_pct'] = ram_pct

        # ---- 组件中间量（QKV / 门控 / 共享专家 / 残差流，逐层 + 跨层均值）----
        out.update(self._component_stats())

        # 清空窗口
        self._load_sum, self._scores_sum, self._moe_steps = None, None, 0
        self._grad_by_group = {}
        return out

    def set_step_info(self, lr: float):
        self._last_step['lr'] = lr

    def close(self):
        if self.gpu_sampler:
            self.gpu_sampler.stop()


# ---------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_loss(model, loader, device, autocast_ctx, max_batches: int = None,
                  algorithm=None) -> float:
    """在留出集上算平均 loss（模型切 eval，不产生梯度）。

    批次有两种形状，都要认：
      - 元组 ``(ids, labels)``：预训练 / SFT 的 Dataset；
      - dict ``{'x', 'y'}``：DPO 这类要同时喂 chosen/rejected 的 Dataset。
    以前只写了元组分支，DPO 跑到 epoch 末尾触发验证时直接 ``KeyError: 0`` 崩掉
    —— 而 DPO 恰恰是唯一走默认循环、真的会去验证的 RL 算法。
    """
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if algorithm is not None:
            # 用算法自己的目标函数验证：DPO/ORPO 等不能退化成普通 causal CE，
            # 蒸馏也必须包含教师 KL。compute_loss 已除 accumulation_steps，日志口径
            # 需要乘回去。no_grad 由本函数统一包住调用方循环。
            with torch.no_grad(), autocast_ctx:
                loss, _ = algorithm.compute_loss(batch)
                loss = loss * algorithm.args.accumulation_steps
            total += float(loss.item())
            n += 1
            continue
        if isinstance(batch, dict):
            if "x_chosen" in batch:
                # DPO：一个样本同时带 chosen / rejected，按训练时的做法拼成一个 batch。
                # 两个坑，都踩过：
                #  1) ``y`` 是**未掩码**的原始 label（答案段掩码单独放在 ``mask_*`` 里），
                #     不套 mask 会把 prompt 和 padding 一起算进 loss；
                #  2) ``x``/``y`` 已经做过一次位移（``x=input_ids[:-1]``、``y=input_ids[1:]``），
                #     而模型内部还会再移一位 —— 直接喂 ``y`` 当 labels 会**双重位移**，
                #     拿错位的 token 对齐，loss 虚高到 10+（实测 10.8，修正后 2.4）。
                #     这里先把完整 input_ids 还原出来，让内部位移刚好对上。
                x = torch.cat([batch["x_chosen"], batch["x_rejected"]], dim=0).to(device)
                y = torch.cat([batch["y_chosen"], batch["y_rejected"]], dim=0).to(device)
                keep = torch.cat([batch["mask_chosen"], batch["mask_rejected"]], dim=0).to(device)
                ids = torch.cat([x[:, :1], y], dim=1)
                labels = ids.clone()
                labels[:, 0] = -100                                  # 首位会被内部位移丢掉
                labels[:, 1:] = labels[:, 1:].masked_fill(keep == 0, -100)
            else:
                ids, labels = batch["x"].to(device), batch["y"].to(device)
        else:
            ids = batch[0].to(device)
            labels = batch[1].to(device)
        with autocast_ctx:
            res = model(ids, labels=labels)
            loss = res.loss + res.aux_loss
        total += float(loss.item())
        n += 1
    if was_training:
        model.train()
    return total / max(n, 1)
