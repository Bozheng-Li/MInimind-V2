"""算法协议。

设计原则：**算法只描述「优化什么」，脚手架负责「怎么跑」。**

批训练算法（pretrain / SFT / 参数高效微调 / 离线偏好）只需实现 ``compute_loss``；
在线 RL（GRPO / DAPO / RLOO / PPO / Agent）可以覆盖 ``train_epoch`` 自建循环（它们需要
rollout → reward → advantage 的阶段）。
"""
from __future__ import annotations

from ..common.runner import run_batch_epoch


class Algorithm:
    #: 算法名，同时也是 ``algos.get_algorithm`` 的键
    name = ""

    def __init__(self, args, lm_config, tokenizer, runtime):
        self.args = args
        self.lm_config = lm_config
        self.tokenizer = tokenizer
        self.runtime = runtime
        #: 由 pipeline 在模型构造后注入，供 compute_loss 使用
        self.model = None

    # ------------------------------------------------------------------ #
    # 子类必须实现
    # ------------------------------------------------------------------ #
    def build_dataset(self):
        """返回训练用的 Dataset。"""
        raise NotImplementedError

    def compute_loss(self, batch):
        """返回 ``(loss, aux_loss)``。

        ``loss`` **必须已经除以** ``args.accumulation_steps``（与原实现一致）。
        ``aux_loss`` 仅用于日志，无 MoE 时返回 None。
        """
        raise NotImplementedError

    def format_log(self, *, epoch, step, iters, start_step, loss_scaled, aux_loss, lr, spend_time):
        """把一步的指标格式化成 ``(日志文本, wandb metrics)``。

        默认是因果语言模型训练格式；偏好优化、蒸馏等需要额外指标时覆盖本方法。
        """
        total_loss = loss_scaled * self.args.accumulation_steps
        aux = float(aux_loss) if aux_loss is not None else 0.0
        eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
        line = (f'Epoch:[{epoch + 1}/{self.args.epochs}]({step}/{iters}), '
                f'loss: {total_loss:.4f}, logits_loss: {total_loss - aux:.4f}, '
                f'aux_loss: {aux:.4f}, lr: {lr:.8f}, epoch_time: {eta_min:.1f}min')
        metrics = {"loss": total_loss, "logits_loss": total_loss - aux, "aux_loss": aux,
                   "learning_rate": lr, "epoch_time": eta_min}
        return line, metrics

    # ------------------------------------------------------------------ #
    # 可选钩子
    # ------------------------------------------------------------------ #
    def configure_model(self, model):
        """模型构造后、DDP/compile 包装前调用（如 LoRA 挂载与冻结）。"""
        return model

    def build_optimizer(self, model):
        """返回 None 表示用默认的 ``AdamW(model.parameters(), lr)``。

        LoRA 需要只优化 ``lora_params``，因此覆盖此方法。
        """
        return None

    def clip_parameters(self, model):
        """梯度裁剪作用于哪些参数。LoRA 只裁 ``lora_params``。"""
        return model.parameters()

    def weight_prefix(self) -> str:
        """权重文件的前缀名。LoRA 用的是 ``lora_name`` 而非 ``save_weight``。"""
        from ..common.checkpoint import resolve_weight_prefix
        return resolve_weight_prefix(self.args)

    def save_weights(self, ctx, epoch: int, step: int):
        """保存推理权重。默认存整模型；LoRA 覆盖为「只存 LoRA 分支」。"""
        from ..common.checkpoint import save_inference_weights
        save_inference_weights(ctx.model, ctx.args.save_dir, self.weight_prefix(), ctx.lm_config)

    def extra_models(self):
        """构造额外模型（ref / teacher / critic / reward）。

        返回 ``{属性名: 模型对象}``，pipeline 会挂到 ``self`` 上。
        """
        return {}

    def checkpoint_extra(self):
        """额外写进续训档的对象（如 PPO 的 critic 状态）。"""
        return {}

    def restore_model_state(self, model, state_dict):
        """恢复续训模型状态；特殊存储格式（如 QLoRA）可覆盖。"""
        model.load_state_dict(state_dict)

    def on_epoch_start(self, loader, iters):
        """每个 epoch 开始时的钩子（RL 型算法用来更新 rollout 策略）。"""

    # ------------------------------------------------------------------ #
    def train_epoch(self, ctx, epoch, loader, iters, start_step) -> bool:
        """默认实现：每个 batch 一次前向 + 一次反向。

        返回 True 表示触达 ``--max_steps``，应停止后续 epoch。
        """
        return run_batch_epoch(ctx, self, epoch, loader, iters, start_step)
