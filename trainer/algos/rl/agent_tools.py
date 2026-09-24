"""Agentic RL 的工具集与多轮 rollout。

对应重构前 ``train_agent.py`` 第 32-239 行：工具定义 / 模拟执行 / 多轮 rollout /
整轮延迟结算的 reward。重复惩罚复用 ``common.py`` 的同一份实现。

这里的函数都不持有模型状态，``rollout_engine`` / ``tokenizer`` 由调用方传入，
因此可以独立测试。
"""
from __future__ import annotations

import json
import math
import random
import re
import signal

import torch

from ...trainer_utils import Logger
from .common import rep_penalty

# ======== 工具定义 ========
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位换算", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
]

# ======== 模拟数据 ========
WEATHER_DATA = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"), "深圳": ("30°C", "晴"), "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"), "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"), "西安": ("16°C", "大风"), "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"), "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗")}
TIME_DATA = {"Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00", "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00", "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00"}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12, ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79, ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72}
TRANSLATE_DATA = {("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好", ("今天天气真好", "english"): "The weather is nice today", ("I love programming", "chinese"): "我喜欢编程", ("机器学习很有趣", "english"): "Machine learning is interesting", ("Happy birthday", "chinese"): "生日快乐"}
UNIT_DATA = {"km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462, "pounds_kg": 0.453592, "meters_feet": 3.28084, "feet_meters": 0.3048, "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556}

# ======== 模拟执行 ========
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(eval(str(args.get("expression", "0")).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-").replace("（", "(").replace("）", ")"), {"__builtins__": {}, "math": math}))},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * UNIT_DATA.get(f"{args.get('from_unit', '').lower()}_{args.get('to_unit', '').lower()}", 1), 4)},
    "get_current_weather": lambda args: (lambda w: {"city": args.get("location"), "temperature": w[0], "humidity": "65%", "condition": w[1]})(WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda args: {"datetime": TIME_DATA.get(args.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency"), "to": args.get("to_currency"), "rate": EXCHANGE_DATA.get((args.get("from_currency"), args.get("to_currency")), 1.0)},
    "translate_text": lambda args: {"translated_text": TRANSLATE_DATA.get((args.get("text"), args.get("target_language")), args.get("text", ""))},
}

# ======== 参数校验 ========
CHECK_ARGS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_current_time": lambda a: True,
    "get_exchange_rate": lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text": lambda a: bool(a.get("text")) and bool(a.get("target_language")),
}

# ======== 工具调用解析与执行 ========
def parse_tool_calls(text):
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try:
            calls.append(json.loads(m.strip()))
        except (json.JSONDecodeError, TypeError):
            # 模型输出不受信任；单个坏 JSON 只视作无效调用，不能中断整个 batch。
            continue
    return calls

def execute_tool(name, args):
    fn = MOCK_RESULTS.get(name)
    if not fn: return None
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(1)
        return fn(args)
    except Exception:  # noqa: BLE001 模拟工具的任意解析/执行错误都转换成调用失败
        return None
    finally:
        try:
            signal.alarm(0)
        except Exception:  # noqa: BLE001 Windows/非主线程可能不支持 SIGALRM
            pass

# ======== 多轮 Rollout ========
def rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=3,
                   max_new_tokens=256, thinking_ratio=0.5,
                   temperature=0.8, device="cuda"):
    """采样一条多轮工具轨迹，并保存行为策略对每个生成 token 的 log-prob。

    ``temperature`` 必须原样传给 rollout 引擎。策略梯度中的 old log-prob
    对应的正是这个采样分布，因此 Agent 不能在此处偷偷使用固定温度。
    """
    all_outputs = []
    prompt_ids = None
    response_ids = []
    response_mask = []
    response_old_logps = []
    final_context = ""
    unfinished = False
    open_thinking = random.random() < thinking_ratio
    for turn in range(max_turns):
        context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
        inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(device)
        context_ids = inputs["input_ids"][0].tolist()
        if prompt_ids is None:
            prompt_ids = context_ids
        rollout_result = rollout_engine.rollout(
            prompt_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        new_ids = rollout_result.completion_ids[0].tolist()
        new_logps = rollout_result.per_token_logps[0].tolist()
        if len(new_ids) != len(new_logps): Logger(f"rollout token/logprob length mismatch: {len(new_ids)} vs {len(new_logps)}")
        pairs = [(t, lp) for t, lp in zip(new_ids, new_logps) if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
        new_ids = [t for t, _ in pairs]
        new_logps = [lp for _, lp in pairs]
        new_text = rollout_result.completions[0]
        all_outputs.append(new_text)
        response_ids.extend(new_ids)
        response_mask.extend([1] * len(new_ids))
        response_old_logps.extend(new_logps)
        final_context = context + new_text
        calls = parse_tool_calls(new_text)
        if not calls:
            break
        unfinished = turn == max_turns - 1
        messages.append({"role": "assistant", "content": new_text})
        for call in calls:
            name, raw = call.get("name", ""), call.get("arguments", {})
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    raw = {}
            result = execute_tool(name, raw)
            result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]  # 防止天文数字撑爆tokenizer
            messages.append({"role": "tool", "content": result_str})

        observe_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
        observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        current_len = len(prompt_ids) + len(response_ids)
        obs_delta = observe_ids[current_len:]
        response_ids.extend(obs_delta)
        response_mask.extend([0] * len(obs_delta))
        response_old_logps.extend([0.0] * len(obs_delta))
        final_context = observe_context

    final_output = all_outputs[-1] if all_outputs else ""
    prompt_ids = prompt_ids or []
    return final_output, final_context, prompt_ids, response_ids, response_mask, response_old_logps, list(all_outputs), unfinished

def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, num_gen,
                  max_turns=3, max_new_tokens=256, thinking_ratio=0.5,
                  temperature=0.8, device="cuda"):
    """按 ``样本 × num_gen`` 展开并采样多轮轨迹。"""
    all_completions = []
    all_contexts = []
    all_prompt_ids = []
    all_response_ids = []
    all_response_masks = []
    all_response_old_logps = []
    all_turn_outputs = []
    all_unfinished = []
    for messages, tools in zip(messages_batch, tools_batch):
        for _ in range(num_gen):
            msgs_copy = [dict(m) for m in messages]
            completion, context, prompt_ids, response_ids, response_mask, response_old_logps, turn_outputs, unfinished = rollout_single(
                rollout_engine, tokenizer, msgs_copy, tools,
                max_turns=max_turns,
                max_new_tokens=max_new_tokens,
                thinking_ratio=thinking_ratio,
                temperature=temperature,
                device=device,
            )
            all_completions.append(completion)
            all_contexts.append(context)
            all_prompt_ids.append(prompt_ids)
            all_response_ids.append(response_ids)
            all_response_masks.append(response_mask)
            all_response_old_logps.append(response_old_logps)
            all_turn_outputs.append(turn_outputs)
            all_unfinished.append(unfinished)
    return all_completions, all_contexts, all_prompt_ids, all_response_ids, all_response_masks, all_response_old_logps, all_turn_outputs, all_unfinished


def pack_agent_trajectory(prompt_ids, response_ids, response_mask,
                          response_old_logps, max_total_len):
    """把一条多轮轨迹整理成训练所需的四个对齐序列。

    ``response_mask`` 中模型生成 token 为 1，工具观察/模板 token 为 0；
    ``response_old_logps`` 与 response token 一一对应，观察 token 的值应为 0。
    next-token log-prob 比 input ids 少一项，所以 prompt 前缀只补 ``P-1`` 个 0。
    左截断后必须保留最后 ``len(ids)-1`` 个 log-prob，不能按 token 长度截，否则
    所有策略比率都会错开一位。
    """
    if not (len(response_ids) == len(response_mask) == len(response_old_logps)):
        raise ValueError(
            "Agent 轨迹长度不一致：response_ids / response_mask / old_logps 必须等长"
        )
    if max_total_len <= 1:
        raise ValueError("max_total_len 必须大于 1")

    ids = list(prompt_ids) + list(response_ids)
    mask = [0] * len(prompt_ids) + list(response_mask)
    old_logps = [0.0] * max(len(prompt_ids) - 1, 0) + list(response_old_logps)
    if len(ids) > max_total_len:
        ids = ids[-max_total_len:]
        mask = mask[-max_total_len:]
        keep_logps = max(len(ids) - 1, 0)
        old_logps = old_logps[-keep_logps:] if keep_logps else []

    expected = max(len(ids) - 1, 0)
    if len(old_logps) != expected:
        raise RuntimeError(
            f"Agent 轨迹 log-prob 对齐失败：ids={len(ids)}，old_logps={len(old_logps)}"
        )
    prompt_len = next((i for i, value in enumerate(mask) if value == 1), len(mask))
    return ids, mask, prompt_len, old_logps

# ======== Reward 计算 ========
def validate_gt_in_text(text, gt_list):
    text, text_num = str(text), str(text).replace(',', '')
    nums = [float(x) for x in re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])', text_num)]
    return {g for g in gt_list if ((s := str(g).strip()) and s.lower() in text.lower()) or (re.fullmatch(r'[-+]?\d+(?:\.\d+)?', str(g).strip().replace(',', '')) and any(abs(float(str(g).strip().replace(',', '')) - n) < 1e-6 for n in nums))}

def calculate_rewards(prompts, completions, gt_batch, tools_batch, num_gen, reward_model=None, device="cuda", turn_outputs_batch=None, unfinished_batch=None, return_stats=False):
    """多轮工具调用的整轮延迟结算。

    ``return_stats=True`` 时额外返回整轮的诊断统计（成功率 / 工具调用 / 轮数），
    这些量本来就在函数里算出来了，只是以前算完就丢 —— 而它们恰恰是判断 agentic RL
    到底学没学会用工具的最直接指标。统计不参与任何数值计算。
    """
    rewards = torch.zeros(len(completions), device=device)
    st = {"n": 0, "pass_sum": 0.0, "pass_n": 0, "unfinished": 0,
          "calls": 0, "valid": 0, "calls_n": 0, "tool_gap": 0.0, "gap_n": 0, "turns": 0}
    for idx, response in enumerate(completions):
        reward, answer = 0.0, response
        sample_idx = idx // num_gen
        tools = tools_batch[sample_idx]
        turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch is not None else [response]
        unfinished = unfinished_batch[idx] if unfinished_batch is not None else False
        turn_answers = [turn.split('</think>', 1)[-1].strip() if '</think>' in turn else turn.strip() for turn in turn_outputs]
        answer = turn_answers[-1] if turn_answers else response.strip()
        valid_names = {t['function']['name'] for t in tools} if tools else set()
        tool_calls = []
        for turn_answer in turn_answers: tool_calls.extend(parse_tool_calls(turn_answer))  # 解析tool调用
        reward -= 0.5 * sum(abs(turn.count('<tool_call>') - turn.count('</tool_call>')) for turn in turn_answers)  # 标签扣分
        st["n"] += 1
        st["calls"] += len(tool_calls)
        st["calls_n"] += 1
        st["turns"] += len(turn_outputs)
        st["unfinished"] += int(bool(unfinished))
        # -------- 无工具调用：格式+reward奖励 --------
        if not tool_calls:
            reward += 0.5 if 5 <= len(response.strip()) <= 800 else -0.5  # 长度分
            if '</think>' in response:
                think, answer = response.split('</think>', 1)
                reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5  # 思考长度分
                reward += 0.25 if response.count('</think>') == 1 else -0.25  # 思考闭合分
                answer = answer.strip()
            if reward_model is not None:
                prompt = prompts[sample_idx]
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                score = reward_model.get_score(messages, answer)
                reward += score  # RM分
            reward -= rep_penalty(answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)  # 总分Clip
        # -------- 有工具调用：执行结果奖励 --------
        else:
            gt = gt_batch[sample_idx]
            valid_call_count = 0
            for tool_call in tool_calls:
                name, raw = tool_call.get("name", ""), tool_call.get("arguments", {})
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except json.JSONDecodeError:
                        raw = {}
                check = CHECK_ARGS.get(name)
                # arguments 可能是 list / str / None（模型吐什么都有可能），
                # 参数校验一律假设 dict —— 不挡住就会 AttributeError 整个 run 挂掉
                # （实测 MoE 那轮跑了 8.4 小时后才崩在这）。
                ok_args = isinstance(raw, dict) and bool(check) and bool(check(raw))
                valid_call_count += int(bool(name in valid_names) and ok_args)
            st["valid"] += valid_call_count
            tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)  # tool数差值
            st["tool_gap"] += tool_gap
            st["gap_n"] += 1
            reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap  # tool对齐分

            final_text = "" if unfinished else (answer.split('</tool_call>')[-1] if '</tool_call>' in answer else answer)
            verified = validate_gt_in_text(final_text, gt) if gt else set()
            if gt:
                reward += 2.5 * len(verified) / len(gt)  # GT分
                st["pass_sum"] += len(verified) / len(gt)
                st["pass_n"] += 1
            if unfinished: reward -= 0.5  # 未完成扣分
            reward -= rep_penalty(final_text if final_text else answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)  # 总分Clip
    if not return_stats:
        return rewards
    n = max(st["n"], 1)
    stats = {
        "pass_rate": st["pass_sum"] / max(st["pass_n"], 1),          # 有 GT 的样本里答对的比例
        "unfinished_rate": st["unfinished"] / n,                      # 轮数用完还没收敛
        "tool_calls_mean": st["calls"] / max(st["calls_n"], 1),       # 每条回答平均发起几次调用
        "valid_call_rate": st["valid"] / max(st["calls"], 1),         # 发起的调用里参数合法的比例
        "tool_gap_mean": st["tool_gap"] / max(st["gap_n"], 1),        # 调用数与 GT 的偏离
        "turns_mean": st["turns"] / n,                                # 平均实际轮数
    }
    return rewards, stats


__all__ = [
    "TOOLS", "parse_tool_calls", "execute_tool", "rollout_single", "rollout_batch",
    "pack_agent_trajectory", "validate_gt_in_text", "calculate_rewards",
    "collate_agent_batch",
]


def collate_agent_batch(batch):
    """AgentRLDataset 的样本是嵌套结构，必须自己拼 batch（原脚本的 collate_fn）。"""
    return {'messages': [b['messages'] for b in batch],
            'tools': [b['tools'] for b in batch],
            'gt': [b['gt'] for b in batch]}
