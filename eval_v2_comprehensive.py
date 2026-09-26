"""
Comprehensive evaluation for v2 finetuned model
Adapted from 13.6 项目部 6-step RAGAS flow for pure LLM finetune
Steps: init -> load test -> run pipeline (via API) -> build dataset -> evaluate -> export bad cases
"""
import json
import csv
import re
import time
import requests
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "data" / "byd_test_v3.jsonl"
API_URL = "http://localhost:8100/v1/chat/completions"
MODEL_NAME = "qwen3-crm"
OUTPUT_CSV = PROJECT_DIR / "eval_v2_results.csv"
BAD_CASES_CSV = PROJECT_DIR / "eval_v2_bad_cases.csv"
REPORT_JSON = PROJECT_DIR / "eval_v2_report.json"
TRAINER_STATE = PROJECT_DIR / "finetuned" / "QLoRA_CRM_BYD_v2" / "checkpoint-800" / "trainer_state.json"

# fallback if checkpoint-800 not exists, try adapter
if not TRAINER_STATE.exists():
    TRAINER_STATE = PROJECT_DIR / "finetuned" / "QLoRA_CRM_BYD_v2" / "trainer_state.json"

MAX_SAMPLES = 80  # evaluate 80 samples for speed
TIMEOUT = 60

SYSTEM_PROMPT = "你是比亚迪商用车智能售后助手，精通车辆故障诊断、维修方案、质保政策、索赔管理、CRM系统操作和保养知识。请用专业、简洁、有条理的方式回答用户问题。"

JUNK_PATTERNS = [
    r"共\d+页",
    r"第\d+页",
    r"__.*ERROR__",
    r"DataValidation",
    r"第\s*\d+\s*页,\s*共",
    r"https?://",
    r"申请编号",
    r"纸单线上审批",
]

def call_api(question):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        "max_tokens": 256,
        "temperature": 0.2
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[ERROR] {e}"

def token_overlap(a, b):
    """simple token-level F1"""
    a_tokens = set(re.findall(r'[\w\u4e00-\u9fff]+', a.lower()))
    b_tokens = set(re.findall(r'[\w\u4e00-\u9fff]+', b.lower()))
    if not a_tokens or not b_tokens:
        return 0
    inter = len(a_tokens & b_tokens)
    prec = inter / len(a_tokens) if a_tokens else 0
    rec = inter / len(b_tokens) if b_tokens else 0
    if prec + rec == 0:
        return 0
    return 2 * prec * rec / (prec + rec)

def detect_issues(answer, ground_truth):
    issues = []
    for pat in JUNK_PATTERNS:
        if re.search(pat, answer):
            issues.append(f"junk:{pat[:10]}")
    if len(answer.strip()) < 10:
        issues.append("too_short")
    if len(answer.strip()) > 800:
        issues.append("too_long")
    # check if answer is mostly page refs
    if re.search(r"(共\d+页|第\d+页)", answer) and len(answer) < 200:
        issues.append("page_ref_only")
    # check off-topic: if ground truth contains keyword but answer doesn't
    # simple: if answer contains system doc fragment not related to question
    if "售后服务工程通" in answer and "工程通" not in ground_truth:
        issues.append("off_topic_doc")
    return issues

def main():
    print("="*60)
    print("Step 1: 初始化 - 检查API连通性")
    print("="*60)
    try:
        r = requests.get("http://localhost:8100/v1/models", timeout=5)
        print(f"  API OK: {r.json()}")
    except Exception as e:
        print(f"  API 未就绪: {e}")
        print("  请确保 serve_openai.py 正在运行 (port 8100)")
        return

    print("\nStep 2: 加载测试集")
    samples = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"  总测试集: {len(samples)} 条")
    # sample evenly
    step = max(1, len(samples) // MAX_SAMPLES)
    selected = samples[::step][:MAX_SAMPLES]
    print(f"  本次评估抽样: {len(selected)} 条 (每{step}条抽1)")

    print("\nStep 3: 运行 Pipeline (调用微调模型API)")
    results = []
    for idx, s in enumerate(selected):
        msgs = s["messages"]
        q = next((m["content"] for m in msgs if m["role"] == "user"), "")
        gt = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        print(f"  [{idx+1}/{len(selected)}] Q: {q[:30]}...", end=" ", flush=True)
        ans = call_api(q)
        # strip thinking tags
        ans_clean = re.sub(r"<think>.*?</think>", "", ans, flags=re.DOTALL).strip()
        score = token_overlap(ans_clean, gt)
        issues = detect_issues(ans_clean, gt)
        # heuristic correctness: high overlap = correct, junk = incorrect
        correctness = score
        if issues:
            correctness = min(correctness, 0.4)  # penalize junk
        is_bad = correctness < 0.5 or len(issues) > 0
        results.append({
            "question": q,
            "ground_truth": gt[:200],
            "answer": ans_clean[:500],
            "full_answer": ans_clean,
            "token_overlap": round(score, 3),
            "correctness": round(correctness, 3),
            "issues": ";".join(issues) if issues else "none",
            "is_bad": is_bad,
            "answer_len": len(ans_clean)
        })
        print(f"-> overlap={score:.2f} issues={issues if issues else 'none'} {'[BAD]' if is_bad else ''}")
        time.sleep(0.3)

    print("\nStep 4: 构建数据集 & Step 5: 执行评估")
    avg_overlap = sum(r["token_overlap"] for r in results) / len(results)
    avg_correct = sum(r["correctness"] for r in results) / len(results)
    bad_count = sum(1 for r in results if r["is_bad"])
    good_count = len(results) - bad_count

    # issue distribution
    all_issues = []
    for r in results:
        if r["issues"] != "none":
            all_issues.extend(r["issues"].split(";"))
    issue_counter = Counter(all_issues)

    print(f"  平均 token_overlap: {avg_overlap:.3f}")
    print(f"  平均 correctness: {avg_correct:.3f}")
    print(f"  Good: {good_count}/{len(results)} ({good_count/len(results)*100:.1f}%)")
    print(f"  Bad: {bad_count}/{len(results)} ({bad_count/len(results)*100:.1f}%)")
    print(f"  问题分布: {dict(issue_counter)}")

    print("\nStep 6: 导出结果 + Bad Cases")
    # CSV 9 cols: question, ground_truth, answer, correctness, token_overlap, issues, is_bad, answer_len
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["question","ground_truth","answer","token_overlap","correctness","issues","is_bad","answer_len"])
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in ["question","ground_truth","answer","token_overlap","correctness","issues","is_bad","answer_len"]})
    print(f"  已导出: {OUTPUT_CSV}")

    with open(BAD_CASES_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["question","ground_truth","answer","token_overlap","correctness","issues"])
        w.writeheader()
        for r in results:
            if r["is_bad"]:
                w.writerow({k: r[k] for k in ["question","ground_truth","answer","token_overlap","correctness","issues"]})
    print(f"  Bad Cases: {BAD_CASES_CSV} ({bad_count}条)")

    # training metrics
    train_metrics = {}
    if TRAINER_STATE.exists():
        try:
            with open(TRAINER_STATE) as f:
                state = json.load(f)
            logs = state.get("log_history", [])
            eval_logs = [l for l in logs if "eval_loss" in l]
            train_logs = [l for l in logs if "loss" in l and "eval_loss" not in l]
            if eval_logs:
                train_metrics["final_eval_loss"] = eval_logs[-1]["eval_loss"]
                train_metrics["final_eval_acc"] = eval_logs[-1].get("eval_mean_token_accuracy")
            if train_logs:
                train_metrics["final_train_loss"] = train_logs[-1].get("loss")
                train_metrics["final_train_acc"] = train_logs[-1].get("mean_token_accuracy")
        except Exception as e:
            print(f"  读取trainer_state失败: {e}")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Qwen3-4B-QLoRA-BYD-v2-merged",
        "training_metrics": train_metrics,
        "eval_summary": {
            "samples": len(results),
            "avg_token_overlap": round(avg_overlap, 3),
            "avg_correctness": round(avg_correct, 3),
            "good": good_count,
            "bad": bad_count,
            "good_rate": round(good_count/len(results), 3),
            "issue_distribution": dict(issue_counter)
        },
        "threshold": 0.5,
        "bad_case_definition": "correctness <0.5 or junk/off_topic detected"
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_JSON}")
    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
