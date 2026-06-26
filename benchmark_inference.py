"""
推理性能基准测试脚本

测量指标：
- TFTT (Time to First Token): 首 token 延迟
- QPS  (Queries Per Second):  每秒处理请求数
- RPS  (Requests Per Second): 每秒完成请求数

运行：python benchmark_inference.py
"""

import time
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from datetime import datetime

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "./model/Qwen3-4B-merged"
NUM_RUNS = 10               # 每个问题跑几轮取平均
WARMUP_RUNS = 2             # 预热轮数（不计入统计）

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


# ============================================================
# 加载模型
# ============================================================
print("=" * 50)
print("推理性能基准测试")
print("=" * 50)
print(f"\n模型：{MODEL_PATH}")
print(f"测试轮数：{NUM_RUNS}（预热 {WARMUP_RUNS} 轮）")
print(f"测试问题数：{len(TEST_QUESTIONS)}")

print("\n加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)
print("模型加载完成！")


# ============================================================
# 测量函数
# ============================================================

def measure_tftt(question, max_new_tokens=256):
    """
    测量 Time to First Token (TFTT)

    方法：用 streamer 逐 token 接收，记录第一个 token 到达的时间。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 在子线程中生成，主线程计时
    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    start_time = time.perf_counter()
    thread.start()

    first_token_time = None
    total_tokens = 0
    full_response = ""

    for new_text in streamer:
        if first_token_time is None:
            first_token_time = time.perf_counter() - start_time
        full_response += new_text
        total_tokens += 1  # 粗略计数

    thread.join()
    total_time = time.perf_counter() - start_time

    return {
        "tftt": first_token_time,
        "total_time": total_time,
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
    }


def measure_standard(question, max_new_tokens=256):
    """
    标准推理测量（用于 QPS/RPS 计算）
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    end_time = time.perf_counter()

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    num_tokens = len(generated_ids)
    total_time = end_time - start_time

    return {
        "total_time": total_time,
        "output_tokens": num_tokens,
        "tokens_per_sec": num_tokens / total_time if total_time > 0 else 0,
    }


# ============================================================
# 运行基准测试
# ============================================================
print("\n" + "=" * 50)
print("开始测试...")
print("=" * 50)

# --- TFTT 测试 ---
print("\n📊 [1/3] 测量 TFTT（Time to First Token）...")
tftt_results = []

for qi, question in enumerate(TEST_QUESTIONS):
    # 预热
    for _ in range(WARMUP_RUNS):
        measure_tftt(question, max_new_tokens=32)

    # 正式测量
    tftts = []
    for run in range(NUM_RUNS):
        result = measure_tftt(question, max_new_tokens=128)
        tftts.append(result["tftt"])
        print(f"  Q{qi+1} run {run+1}: TFTT={result['tftt']*1000:.1f}ms")

    avg_tftt = np.mean(tftts)
    p50 = np.percentile(tftts, 50)
    p95 = np.percentile(tftts, 95)
    p99 = np.percentile(tftts, 99)

    tftt_results.append({
        "question_idx": qi + 1,
        "question": question[:30] + "...",
        "avg_ms": round(avg_tftt * 1000, 2),
        "p50_ms": round(p50 * 1000, 2),
        "p95_ms": round(p95 * 1000, 2),
        "p99_ms": round(p99 * 1000, 2),
        "min_ms": round(min(tftts) * 1000, 2),
        "max_ms": round(max(tftts) * 1000, 2),
    })

# --- QPS/RPS 测试 ---
print("\n📊 [2/3] 测量 QPS/RPS（吞吐量）...")
throughput_results = []

for qi, question in enumerate(TEST_QUESTIONS):
    # 预热
    for _ in range(WARMUP_RUNS):
        measure_standard(question, max_new_tokens=32)

    # 正式测量
    times = []
    token_counts = []
    for run in range(NUM_RUNS):
        result = measure_standard(question, max_new_tokens=128)
        times.append(result["total_time"])
        token_counts.append(result["output_tokens"])
        print(f"  Q{qi+1} run {run+1}: {result['total_time']:.2f}s, {result['output_tokens']} tokens, {result['tokens_per_sec']:.1f} tok/s")

    avg_time = np.mean(times)
    avg_tokens = np.mean(token_counts)

    throughput_results.append({
        "question_idx": qi + 1,
        "avg_time_s": round(avg_time, 3),
        "avg_tokens": round(avg_tokens, 1),
        "qps": round(1.0 / avg_time, 3),       # 每秒完成请求数
        "rps": round(1.0 / avg_time, 3),        # 同 QPS（单请求场景）
        "tokens_per_sec": round(avg_tokens / avg_time, 1),
    })

# --- 汇总 ---
print("\n📊 [3/3] 汇总统计...")

all_tftt = [r["avg_ms"] for r in tftt_results]
all_qps = [r["qps"] for r in throughput_results]
all_tps = [r["tokens_per_sec"] for r in throughput_results]

summary = {
    "tftt": {
        "avg_ms": round(np.mean(all_tftt), 2),
        "p50_ms": round(np.percentile(all_tftt, 50), 2),
        "p95_ms": round(np.percentile(all_tftt, 95), 2),
        "min_ms": round(min(all_tftt), 2),
        "max_ms": round(max(all_tftt), 2),
    },
    "qps": {
        "avg": round(np.mean(all_qps), 3),
        "min": round(min(all_qps), 3),
        "max": round(max(all_qps), 3),
    },
    "rps": {
        "avg": round(np.mean(all_qps), 3),  # 单请求 = QPS
    },
    "tokens_per_sec": {
        "avg": round(np.mean(all_tps), 1),
        "min": round(min(all_tps), 1),
        "max": round(max(all_tps), 1),
    },
}

# ============================================================
# 输出报告
# ============================================================
print("\n" + "=" * 50)
print("📊 推理性能基准测试报告")
print("=" * 50)

print(f"\n🕐 TFTT（Time to First Token）：")
print(f"  平均：{summary['tftt']['avg_ms']:.1f} ms")
print(f"  P50： {summary['tftt']['p50_ms']:.1f} ms")
print(f"  P95： {summary['tftt']['p95_ms']:.1f} ms")
print(f"  范围：{summary['tftt']['min_ms']:.1f} ~ {summary['tftt']['max_ms']:.1f} ms")

print(f"\n⚡ QPS（Queries Per Second）：")
print(f"  平均：{summary['qps']['avg']:.3f} req/s")
print(f"  范围：{summary['qps']['min']:.3f} ~ {summary['qps']['max']:.3f} req/s")

print(f"\n🚀 RPS（Requests Per Second）：")
print(f"  平均：{summary['rps']['avg']:.3f} req/s")

print(f"\n📈 吞吐量（Tokens Per Second）：")
print(f"  平均：{summary['tokens_per_sec']['avg']:.1f} tok/s")
print(f"  范围：{summary['tokens_per_sec']['min']:.1f} ~ {summary['tokens_per_sec']['max']:.1f} tok/s")

print(f"\n📋 逐题详情：")
print(f"  {'No.':<5} {'问题':<35} {'TFTT(ms)':<12} {'QPS':<10} {'tok/s':<10}")
print(f"  {'─'*72}")
for i, (t, q) in enumerate(zip(tftt_results, throughput_results)):
    print(f"  {t['question_idx']:<5} {t['question']:<35} {t['avg_ms']:<12.1f} {q['qps']:<10.3f} {q['tokens_per_sec']:<10.1f}")

# ============================================================
# 保存完整报告
# ============================================================
report = {
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_PATH,
    "device": str(model.device),
    "dtype": str(model.dtype),
    "num_runs": NUM_RUNS,
    "warmup_runs": WARMUP_RUNS,
    "summary": summary,
    "tftt_details": tftt_results,
    "throughput_details": throughput_results,
}

# 读取已有 eval_report.json 并合并
eval_report_path = "./eval_report.json"
try:
    with open(eval_report_path, "r", encoding="utf-8") as f:
        eval_report = json.load(f)
except:
    eval_report = {}

eval_report["inference_metrics"] = {
    "TFTT": summary["tftt"],
    "QPS": summary["qps"],
    "RPS": summary["rps"],
    "tokens_per_sec": summary["tokens_per_sec"],
    "details": {
        "tftt_by_question": tftt_results,
        "throughput_by_question": throughput_results,
    }
}

with open(eval_report_path, "w", encoding="utf-8") as f:
    json.dump(eval_report, f, ensure_ascii=False, indent=2)

print(f"\n完整报告已保存：{eval_report_path}")
print("=" * 50)
