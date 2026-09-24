"""架构感知的推理入口（**新增**，不修改原有的 ``eval_llm.py``）。

为什么需要它：``eval_llm.py`` 只能通过 ``--hidden_size / --num_hidden_layers / --use_moe``
构造模型，而 ``--use_moe 1`` 合成出来的是**普通 top-k MoE**。用它加载
``moe_finegrained`` / ``moe_shared`` / ``mla`` / ``gated`` 等新架构会构建出错误的模型
（参数量对不上，要么报错要么静默错配）。本脚本改为从 ``--config`` 读架构，因此能
加载任意组合。

用法::

    # 交互式对话
    python trainer/eval.py --config configs/pretrain_moe.yaml --weight pretrain_moe

    # 跑一批固定 prompt（非交互，便于脚本化对比）
    python trainer/eval.py --config configs/pretrain_moe.yaml --weight pretrain_moe \
        --prompt "李白是唐代" --prompt "水的化学式是" --max_new_tokens 60

    # 对比不同架构的输出（同一批 prompt）
    python trainer/eval.py --config configs/base.yaml --weight pretrain --prompt "..." --no_chat
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import random
import time
import warnings

import torch

from configs import load_config, build_lm_config, to_arch_config

warnings.filterwarnings('ignore')

TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model')


def build_parser():
    p = argparse.ArgumentParser(description="MiniMind 架构感知推理")
    p.add_argument('--config', required=True,
                   help="模型结构配置（与训练时用的**必须一致**，否则权重形状对不上）")
    p.add_argument('--weight', default='pretrain_moe', help="权重前缀（默认在 test/out 下找）")
    p.add_argument('--save_dir', default='../test/out', help="权重目录（默认 ../test/out：产物在 test/ 下）")
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--prompt', action='append', default=None,
                   help="要测试的 prompt，可重复多次；不给则进入交互模式")
    p.add_argument('--no_chat', action='store_true',
                   help="不走 chat 模板（评测**预训练**模型时用，它只会续写）")
    p.add_argument('--max_new_tokens', type=int, default=80)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top_p', type=float, default=0.9)
    p.add_argument('--repetition_penalty', type=float, default=1.2,
                   help="小模型容易复读，建议 >1")
    p.add_argument('--seed', type=int, default=42)
    return p


def weight_path(save_dir, prefix, lm_config):
    moe = '_moe' if lm_config.use_moe else ''
    return f'{save_dir}/{prefix}_{lm_config.hidden_size}{moe}.pth'


def main():
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    arch = to_arch_config(cfg)

    # 从配置构造一个只用于「拿标量 + 决定文件名」的 MiniMindConfig
    class _Ns:
        hidden_size = arch.model["hidden_size"]
        num_hidden_layers = arch.model["num_hidden_layers"]
        use_moe = 1 if str(arch.type_of("feedforward")).startswith("moe") else 0
        student_hidden_size = hidden_size
        student_num_layers = num_hidden_layers
        student_use_moe = use_moe
    lm_config = build_lm_config(_Ns(), cfg)

    # 用 arch 直接组装模型（与训练完全同一条路径）
    from arch import build_model
    model = build_model(arch)

    ckp = weight_path(args.save_dir, args.weight, lm_config)
    if not os.path.isfile(ckp):
        print(f"❌ 找不到权重: {ckp}\n   （检查 --weight / --save_dir；文件名会带 _moe 后缀与否取决于前馈是否为 MoE）")
        return 1

    state = torch.load(ckp, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_missing, n_unexpected = len(missing), len(unexpected)
    if n_missing or n_unexpected:
        print(f"⚠️  权重未完全匹配：missing={n_missing} unexpected={n_unexpected}")
        for k in (missing[:3] + unexpected[:3]):
            print(f"     {k}")
    else:
        print("✅ 权重完全匹配")

    model = model.half().eval().to(args.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    print(f"   架构: {arch.summary()}")
    print(f"   权重: {ckp}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    prompts = args.prompt or []
    interactive = not prompts
    if interactive:
        print("\n输入 prompt 回车，空行退出\n")

    while True:
        if interactive:
            try:
                q = input('💬: ').strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            prompts = [q]

        for q in prompts:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            if args.no_chat:
                text = tok.bos_token + q          # 预训练模型：加 BOS 做续写
            else:
                text = tok.apply_chat_template([{"role": "user", "content": q}],
                                               tokenize=False, add_generation_prompt=True)
            inp = tok(text, return_tensors='pt').to(args.device)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(inp.input_ids, attention_mask=inp.attention_mask,
                                     max_new_tokens=args.max_new_tokens, do_sample=True,
                                     temperature=args.temperature, top_p=args.top_p,
                                     repetition_penalty=args.repetition_penalty,
                                     pad_token_id=tok.pad_token_id,
                                     eos_token_id=tok.eos_token_id)
            resp = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
            dt = time.time() - t0
            n_new = out.shape[1] - inp.input_ids.shape[1]
            print(f"💬: {q}\n🧠: {resp.strip()}")
            print(f"    [{n_new} tokens, {n_new/max(dt,1e-9):.1f} tok/s]\n")

        if not interactive:
            break        # 非交互模式：跑完给定的 prompt 就退出，不要循环重跑

    return 0


if __name__ == "__main__":
    sys.exit(main())
