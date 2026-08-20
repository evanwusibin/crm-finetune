"""
数据清洗脚本 v3：过滤掉有问题的样本
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# 质量规则
MIN_ANSWER_LEN = 10
MAX_ANSWER_LEN = 2000

BAD_PATTERNS = [
    r"__.*ERROR__",
    r"DataValidation",
    r"got an unexpected keyword",
    r"Traceback \(most recent",
    r"null\s+None",
    r"undefined",
]

JUNK_PATTERNS = [
    r"编制部门[：:]\s*\S+",
    r"拟\s*稿\s*人[：:]\s*\S+",
    r"受控状态[：:]",
    r"秘密等级[：:]",
    r"版本号\s+生效日期",
    r"电话\d{8,}",
    r"邮箱\S+@\S+",
    r"申请编号\d+",
    r"申请人工号\d+",
    r"纸单线上审批单",
    r"基本信息\d+",
    r"^\d+[\s\-]\s*N/A\s*N/A",
    r"第\s*\d+\s*页,\s*共\s*\d+\s*页",
    r"https?://\S+",
    r"详见[：:]\s*https",
]


def is_valid(sample):
    """检查样本是否有效"""
    messages = sample.get("messages", [])
    if len(messages) < 2:
        return False

    answer = ""
    for msg in messages:
        if msg["role"] == "assistant":
            answer = msg["content"]
            break

    if not answer:
        return False

    if len(answer) < MIN_ANSWER_LEN:
        return False

    if len(answer) > MAX_ANSWER_LEN:
        return False

    for pattern in BAD_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            return False

    for pattern in JUNK_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            return False

    text_only = re.sub(r'[\d\s\W]', '', answer)
    if len(text_only) < len(answer) * 0.3:
        return False

    if answer.count(answer[:20]) > 3:
        return False

    return True


def main():
    print("=" * 60)
    print("数据清洗 v3")
    print("=" * 60)

    for split, out_name in [
        ("byd_train_v2.jsonl", "byd_train_v3.jsonl"),
        ("byd_test_v2.jsonl", "byd_test_v3.jsonl"),
    ]:
        filepath = DATA_DIR / split
        outpath = DATA_DIR / out_name

        if not filepath.exists():
            print(f"\n❌ 文件不存在: {filepath}")
            continue

        total = 0
        kept = 0

        with open(filepath, "r", encoding="utf-8") as fin, \
             open(outpath, "w", encoding="utf-8") as fout:
            for line in fin:
                total += 1
                try:
                    sample = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                if is_valid(sample):
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    kept += 1

        removed = total - kept
        print(f"\n📄 {split} → {out_name}")
        print(f"  原始: {total} 条")
        print(f"  保留: {kept} 条")
        print(f"  过滤: {removed} 条 ({removed/total*100:.1f}%)")


if __name__ == "__main__":
    main()
