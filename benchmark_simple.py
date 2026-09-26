"""
推理性能基准测试

测量：TFTT / QPS / RPS / Tokens-per-sec
方法：逐 token 生成，精确计时
"""

import time
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

MODEL_PATH = "./model/Qwen3-4B-merged"
NUM_RUNS = 3
MAX_NEW_TOKENS = 128

TEST_QUESTIONS = [
    "比亚迪商用车的质保期是多久？",
    "发动机故障灯亮了怎么办？",
    "新能源商用车续航里程是多少？",
    "我的比亚迪T5出现了发动机故障灯亮、动力不足，故障码P0101，该怎么办？",
    "ABS故障灯亮了还能开吗？",
    "水温报警怎么办？",
    "维修需要多长时间？",
    "路上抛锚怎么办？",
    "多久保养一次？",
    "保养需要做什么项目？",
]

SYSTEM_PROMPT = "你是比亚迪商用车智能售后助手，精通车辆故障诊断、维修方案、质保政策和保养知识。"

print("=" * 50, flush=True)
print("推理性能基准测试", flush=True)
print("=" * 50, flush=True)
print(f"\n模型：{MODEL_PATH}", flush=True)
print(f"每题测试 {NUM_RUNS} 轮，max_new_tokens={MAX_NEW_TOKENS}", flush=True)

print("\n加载模型...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto", device_map="auto")
print("模型加载完成！\n", flush=True)


def measure_single(question, max_new_tokens=MAX_NEW_TOKENS):
    """
    逐 token 生成，精确测量 TFTT 和吞吐量。
    返回: tftt_sec, total_sec, token_count, tokens_per_sec
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    past_kv = None
    first_token_time = None
    token_count = 0

    start = time.perf_counter()

    with torch.no_grad():
        # Prefill
        out = model(input_ids=input_ids)
        past_kv = out.past_key_values
        logits = out.logits[:, -1, :]

        for _ in range(max_new_tokens):
            # Greedy decode
            next_token = logits.argmax(dim=-1, keepdim=True)

            if first_token_time is None:
                first_token_time = time.perf_counter() - start

            token_count += 1

            if next_token.item() == tokenizer.eos_token_id:
                break

            out = model(input_ids=next_token, past_key_values=past_kv)
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]

    total_time = time.perf_counter() - start

    return {
        "tftt": first_token_time,
        "total_time": total_time,
        "tokens": token_count,
        "tok_per_sec": token_count / total_time if total_time > 0 else 0,
    }


# ============================================================
# 运行
# ============================================================
all_results = []

for qi, question in enumerate(TEST_QUESTIONS):
    # 预热 1 轮
    measure_single(question, max_new_tokens=4)

    runs = []
    for r in range(NUM_RUNS):
        res = measure_single(question)
        runs.append(res)
        print(f"  Q{qi+1:2d} run {r+1}: TFTT={res['tftt']*1000:7.1f}ms  total={res['total_time']:5.2f}s  tok={res['tokens']:3d}  speed={res['tok_per_sec']:5.1f} tok/s", flush=True)

    avg_tftt = np.mean([r["tftt"] for r in runs])
    avg_total = np.mean([r["total_time"] for r in runs])
    avg_tokens = np.mean([r["tokens"] for r in runs])
    avg_speed = np.mean([r["tok_per_sec"] for r in runs])

    all_results.append({
        "q": qi + 1,
        "tftt_ms": round(avg_tftt * 1000, 2),
        "total_s": round(avg_total, 3),
        "tokens": round(avg_tokens, 1),
        "tok_per_sec": round(avg_speed, 1),
        "qps": round(1.0 / avg_total, 3),
    })

# ============================================================
# 报告
# ============================================================
print("\n" + "=" * 50, flush=True)
print("📊 推理性能基准测试报告", flush=True)
print("=" * 50, flush=True)

tftts = [r["tftt_ms"] for r in all_results]
qps_vals = [r["qps"] for r in all_results]
tps_vals = [r["tok_per_sec"] for r in all_results]

print(f"\n🕐 TFTT（Time to First Token）：", flush=True)
print(f"  平均：{np.mean(tftts):.1f} ms", flush=True)
print(f"  P50： {np.percentile(tftts, 50):.1f} ms", flush=True)
print(f"  P95： {np.percentile(tftts, 95):.1f} ms", flush=True)
print(f"  范围：{min(tftts):.1f} ~ {max(tftts):.1f} ms", flush=True)

print(f"\n⚡ QPS（Queries Per Second）：", flush=True)
print(f"  平均：{np.mean(qps_vals):.3f} req/s", flush=True)
print(f"  范围：{min(qps_vals):.3f} ~ {max(qps_vals):.3f} req/s", flush=True)

print(f"\n🚀 RPS（Requests Per Second）：", flush=True)
print(f"  平均：{np.mean(qps_vals):.3f} req/s", flush=True)

print(f"\n📈 Tokens Per Second：", flush=True)
print(f"  平均：{np.mean(tps_vals):.1f} tok/s", flush=True)
print(f"  范围：{min(tps_vals):.1f} ~ {max(tps_vals):.1f} tok/s", flush=True)

print(f"\n📋 逐题详情：", flush=True)
print(f"  {'No':<4} {'TFTT(ms)':<10} {'QPS':<8} {'tok/s':<8} {'tokens':<8}", flush=True)
print(f"  {'─'*38}", flush=True)
for r in all_results:
    print(f"  {r['q']:<4} {r['tftt_ms']:<10.1f} {r['qps']:<8.3f} {r['tok_per_sec']:<8.1f} {r['tokens']:<8.0f}", flush=True)

# 保存
eval_path = "./eval_report.json"
try:
    with open(eval_path, "r", encoding="utf-8") as f:
        report = json.load(f)
except:
    report = {}

report["inference_metrics"] = {
    "TFTT": {"avg_ms": round(np.mean(tftts), 2), "p50_ms": round(np.percentile(tftts, 50), 2), "p95_ms": round(np.percentile(tftts, 95), 2)},
    "QPS": {"avg": round(np.mean(qps_vals), 3)},
    "RPS": {"avg": round(np.mean(qps_vals), 3)},
    "tokens_per_sec": {"avg": round(np.mean(tps_vals), 1)},
    "per_question": all_results,
}

with open(eval_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n报告已保存：{eval_path}", flush=True)
print("=" * 50, flush=True)
