"""
数据质量检查脚本
检查训练数据中的质量问题：错误内容、过短/过长、格式异常
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# 质量规则
MIN_ANSWER_LEN = 10      # 回答至少10字
MAX_ANSWER_LEN = 2000    # 回答最多2000字
BAD_PATTERNS = [
    r"__.*ERROR__",                    # 处理错误
    r"DataValidation",                  # pandas错误
    r"got an unexpected keyword",       # 参数错误
    r"Traceback \(most recent",         # Python traceback
    r"null\s+None",                     # null值
    r"undefined",                       # undefined
]

# 无意义内容模式（文档元数据、表单等）
JUNK_PATTERNS = [
    r"编制部门[：:]\s*\S+",             # 文档页眉
    r"拟\s*稿\s*人[：:]\s*\S+",        # 拟稿人
    r"受控状态[：:]",                   # 受控状态
    r"秘密等级[：:]",                   # 密级
    r"版本号\s+生效日期",               # 版本履历
    r"电话\d{8,}",                      # 电话号码
    r"邮箱\S+@\S+",                     # 邮箱
    r"申请编号\d+",                     # 审批单号
    r"申请人工号\d+",                   # 工号
    r"纸单线上审批单",                  # 审批单
    r"基本信息\d+",                     # 审批单字段
    r"^\d+[\s\-]\s*N/A\s*N/A",         # N/A表格行
    r"第\s*\d+\s*页,\s*共\s*\d+\s*页", # 页码
    r"https?://\S+",                    # URL
    r"详见[：:]\s*https",               # 链接引用
]


def check_sample(sample, idx):
    """检查单条样本的质量问题"""
    issues = []
    messages = sample.get("messages", [])

    # 检查消息结构
    if len(messages) < 2:
        issues.append("消息数不足")

    # 获取回答
    answer = ""
    for msg in messages:
        if msg["role"] == "assistant":
            answer = msg["content"]
            break

    if not answer:
        issues.append("无assistant回答")
        return issues

    # 长度检查
    if len(answer) < MIN_ANSWER_LEN:
        issues.append(f"回答过短({len(answer)}字)")

    if len(answer) > MAX_ANSWER_LEN:
        issues.append(f"回答过长({len(answer)}字)")

    # 错误模式检查
    for pattern in BAD_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            issues.append(f"包含错误模式: {pattern}")

    # 垃圾内容检查
    for pattern in JUNK_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            issues.append(f"包含垃圾内容: {pattern[:20]}...")
            break

    # 检查回答是否主要是符号/数字
    text_only = re.sub(r'[\d\s\W]', '', answer)
    if len(text_only) < len(answer) * 0.3:
        issues.append("回答主要是符号/数字")

    # 检查重复内容
    if answer.count(answer[:20]) > 3:
        issues.append("重复内容")

    return issues


def main():
    print("=" * 60)
    print("数据质量检查")
    print("=" * 60)

    for split in ["byd_train_v2.jsonl", "byd_test_v2.jsonl"]:
        filepath = DATA_DIR / split
        if not filepath.exists():
            print(f"\n❌ 文件不存在: {filepath}")
            continue

        print(f"\n📄 {split}")
        print("-" * 40)

        total = 0
        issues_count = 0
        issue_types = {}

        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                total += 1
                try:
                    sample = json.loads(line.strip())
                except json.JSONDecodeError:
                    print(f"  Line {line_num}: JSON解析失败")
                    issues_count += 1
                    continue

                issues = check_sample(sample, line_num)
                if issues:
                    issues_count += 1
                    # 只打印前5个有问题的样本
                    if issues_count <= 5:
                        answer = sample["messages"][-1]["content"][:50]
                        print(f"  Line {line_num}: {issues}")
                        print(f"    内容: {answer}...")
                    for issue in issues:
                        key = issue.split(":")[0] if ":" in issue else issue
                        issue_types[key] = issue_types.get(key, 0) + 1

        print(f"\n  总计: {total} 条")
        print(f"  有问题: {issues_count} 条 ({issues_count/total*100:.1f}%)")
        print(f"  质量通过: {total - issues_count} 条")
        print(f"\n  问题分布:")
        for issue_type, count in sorted(issue_types.items(), key=lambda x: -x[1]):
            print(f"    {issue_type}: {count}")


if __name__ == "__main__":
    main()
