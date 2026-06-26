"""
训练监控 + 自动验证脚本

功能：
1. 监控训练是否完成
2. 训练完成后自动合并模型
3. 自动运行推理验证
4. 记录评估指标

运行：python monitor_and_eval.py
"""

import os
import time
import subprocess
import json
from datetime import datetime

# ============================================================
# 配置
# ============================================================
FINETUNED_DIR = "./finetuned/LoRA_CRM"
LOG_DIR = "./logs/LoRA_CRM"
CHECK_INTERVAL = 60  # 每 60 秒检查一次


def check_training_done():
    """检查训练是否完成（通过检查 finetuned 目录是否有文件）"""
    if not os.path.exists(FINETUNED_DIR):
        return False
    files = os.listdir(FINETUNED_DIR)
    # 检查是否有 adapter_model.safetensors（LoRA 适配器）
    return any("adapter" in f for f in files)


def run_merge():
    """运行模型合并"""
    print("\n" + "=" * 50)
    print("训练完成！开始合并模型...")
    print("=" * 50)
    result = subprocess.run(
        ["python", "merge_model.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("合并失败:", result.stderr)
        return False
    return True


def run_inference():
    """运行推理验证"""
    print("\n" + "=" * 50)
    print("开始推理验证...")
    print("=" * 50)
    result = subprocess.run(
        ["python", "inference.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("推理失败:", result.stderr)
        return False
    return True


def parse_training_logs():
    """解析训练日志，提取关键指标"""
    log_files = []
    if os.path.exists(LOG_DIR):
        for root, dirs, files in os.walk(LOG_DIR):
            for f in files:
                if f.endswith(".json") or f == "trainer_state.json":
                    log_files.append(os.path.join(root, f))

    metrics = {
        "final_loss": None,
        "final_accuracy": None,
        "eval_loss": None,
        "total_steps": None,
        "training_time": None,
    }

    # 尝试读取 trainer_state.json
    for log_file in log_files:
        if "trainer_state.json" in log_file:
            try:
                with open(log_file, "r") as f:
                    state = json.load(f)
                    log_history = state.get("log_history", [])
                    if log_history:
                        last_log = log_history[-1]
                        metrics["final_loss"] = last_log.get("loss")
                        metrics["final_accuracy"] = last_log.get("mean_token_accuracy")
                        metrics["total_steps"] = last_log.get("step")
                        # 找 eval_loss
                        for log in reversed(log_history):
                            if "eval_loss" in log:
                                metrics["eval_loss"] = log["eval_loss"]
                                break
            except Exception as e:
                print(f"解析日志失败: {e}")

    return metrics


def run_eval_metrics():
    """计算评估指标"""
    print("\n" + "=" * 50)
    print("计算评估指标...")
    print("=" * 50)

    metrics = parse_training_logs()

    # 基础指标
    print(f"\n📊 训练指标：")
    print(f"  最终 Loss：{metrics['final_loss']}")
    print(f"  最终准确率：{metrics['final_accuracy']}")
    print(f"  验证 Loss：{metrics['eval_loss']}")
    print(f"  总步数：{metrics['total_steps']}")

    # 推理性能指标（需要实际运行推理）
    print(f"\n📊 推理性能指标（待测量）：")
    print(f"  TFTT（Time to First Token）：首 token 延迟")
    print(f"  QPS（Queries Per Second）：每秒处理请求数")
    print(f"  RPS（Requests Per Second）：每秒完成请求数")

    # 保存指标
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": "Qwen3-4B",
        "method": "LoRA",
        "training_metrics": metrics,
        "inference_metrics": {
            "TFTT": "待测量",
            "QPS": "待测量",
            "RPS": "待测量",
        }
    }

    report_path = "./eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n评估报告已保存：{report_path}")
    return report


def main():
    print("=" * 50)
    print("训练监控脚本")
    print(f"检查间隔：{CHECK_INTERVAL} 秒")
    print("=" * 50)

    while True:
        if check_training_done():
            print(f"\n✅ 检测到训练完成！")

            # 1. 合并模型
            if run_merge():
                # 2. 推理验证
                run_inference()

                # 3. 计算评估指标
                run_eval_metrics()

                print("\n" + "=" * 50)
                print("全部完成！")
                print("=" * 50)
                break
            else:
                print("合并失败，请手动检查")
                break
        else:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] 训练中... 等待 {CHECK_INTERVAL} 秒后重新检查")
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
