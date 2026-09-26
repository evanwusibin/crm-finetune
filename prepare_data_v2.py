"""
数据清洗脚本 v2：清洗 BYD_excerpt 文档，生成干净的 QA 训练数据

核心改进：
1. 去除分页标记、页眉页脚、目录点号
2. 去除表格格式（管道符行）
3. 提取关键信息作为 answer（而非整段 dump）
4. 生成更自然的 user 问题
"""

import json
import os
import random
import re
from pathlib import Path

# ============================================================
# 配置
# ============================================================
BYD_EXCERPT_DIR = r"D:\heimaAI\PytorchSDXX\CRMProject\CRMProject_c\BYD_excerpt"
OUTPUT_DIR = "./data"
TRAIN_RATIO = 0.8

SYSTEM_PROMPT = "你是比亚迪商用车智能售后助手，精通车辆故障诊断、维修方案、质保政策、索赔管理、CRM系统操作和保养知识。请用专业、简洁、有条理的方式回答用户问题。"

# 最小有效内容长度
MIN_CONTENT_LEN = 50
# answer 最大长度
MAX_ANSWER_LEN = 600
# answer 最小长度
MIN_ANSWER_LEN = 30


# ============================================================
# 1. 清洗函数
# ============================================================
def clean_text(text):
    """清洗文档文本，去除噪声"""
    lines = text.split("\n")
    cleaned = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # 跳过空行（保留单个空行）
        if not line_stripped:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        
        # 跳过分页标记 === PAGE N === / === SLIDE N === / === SHEET N ===
        if re.match(r"^=== (PAGE|SLIDE|SHEET) \d+ ===$", line_stripped):
            continue
        
        # 跳过 ★秘密★
        if line_stripped == "★秘密★":
            continue
        
        # 跳过文件头信息
        if re.match(r"^文件名称\s+", line_stripped):
            continue
        if re.match(r"^文件层级\s+", line_stripped):
            continue
        if "Copyright © BYD" in line_stripped:
            continue
        
        # 跳过目录行（包含大量点号 ...）
        if line_stripped.count(".") > 20:
            continue
        
        # 跳过纯数字行（页码）
        if re.match(r"^\d+$", line_stripped):
            continue
        
        # 跳过纯符号行
        if re.match(r"^[=\-_*#]+$", line_stripped):
            continue
        
        # 跳过页码格式 "页 次 X / Y"
        if re.match(r"^页\s+次\s+\d+", line_stripped):
            continue
        
        # 跳过章节号+页码表格行（如 "1 目的 4"）
        if re.match(r"^\d+\s+\S+\s+\d+$", line_stripped) and len(line_stripped) < 30:
            continue
        
        cleaned.append(line_stripped)
    
    # 合并连续空行
    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def is_table_line(line):
    """判断是否是表格行（包含 3 个以上管道符）"""
    return line.count("|") >= 3


def extract_key_content(text):
    """从清洗后的文本中提取关键信息"""
    lines = text.split("\n")
    key_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 跳过表格行
        if is_table_line(line):
            continue
        
        # 跳过纯编号行（如 "a)" "b)" "1." "2." 单独出现）
        if re.match(r"^[a-z]\)$", line) or re.match(r"^\d+\)$", line):
            continue
        
        # 保留有意义的内容行
        if len(line) > 5:
            key_lines.append(line)
    
    return "\n".join(key_lines)


def summarize_content(text, max_len=MAX_ANSWER_LEN):
    """将内容摘要为简洁的回答"""
    # 提取关键内容
    key_content = extract_key_content(text)
    
    # 按段落分割
    paragraphs = [p.strip() for p in key_content.split("\n\n") if p.strip()]
    
    # 如果段落少，直接返回
    if len(paragraphs) <= 2:
        result = "\n\n".join(paragraphs)
        return result[:max_len] if len(result) > max_len else result
    
    # 提取前 3 个最有信息量的段落
    scored = []
    for p in paragraphs:
        # 评分：包含关键词的段落得分高
        score = 0
        if any(kw in p for kw in ["规定", "要求", "标准", "流程", "步骤", "定义", "目的"]):
            score += 2
        if any(kw in p for kw in ["必须", "应当", "需要", "可以", "禁止"]):
            score += 1
        if len(p) > 30:
            score += 1
        scored.append((score, p))
    
    # 按分数排序，取前 3 个
    scored.sort(key=lambda x: -x[0])
    selected = [p for _, p in scored[:3]]
    
    result = "\n\n".join(selected)
    return result[:max_len] if len(result) > max_len else result


# ============================================================
# 2. 问题生成
# ============================================================
def generate_questions(text, filename):
    """从内容生成多样化的问题"""
    questions = []
    content_lower = text.lower()
    name_lower = filename.lower()
    
    # 根据文档类型和内容生成问题
    if "索赔" in name_lower or "索赔" in content_lower[:500]:
        questions.extend([
            "索赔流程是怎样的？",
            "什么情况下可以申请索赔？",
            "索赔单怎么提交？",
            "索赔审核的标准是什么？",
            "索赔被驳回了怎么办？",
        ])
    
    if "质保" in name_lower or "保修" in name_lower or "质保" in content_lower[:500]:
        questions.extend([
            "质保政策是什么？",
            "质保期是多久？",
            "哪些情况不在质保范围内？",
            "质保期内维修怎么申请？",
        ])
    
    if "保养" in name_lower or "保养" in content_lower[:500]:
        questions.extend([
            "保养周期是多久？",
            "首保需要做什么？",
            "保养项目有哪些？",
            "保养费用是多少？",
        ])
    
    if "crm" in name_lower or "系统" in content_lower[:500]:
        questions.extend([
            "CRM系统怎么操作？",
            "怎么在系统里创建工单？",
            "系统有哪些功能？",
        ])
    
    if "故障" in content_lower[:500] or "维修" in content_lower[:500]:
        questions.extend([
            "这个故障怎么处理？",
            "维修步骤是什么？",
            "故障诊断的流程是什么？",
        ])
    
    if "稽查" in name_lower or "考评" in name_lower or "考核" in content_lower[:500]:
        questions.extend([
            "稽查/考评的标准是什么？",
            "考核不合格会有什么后果？",
            "考评周期是多久？",
        ])
    
    if "备件" in name_lower or "备件" in content_lower[:500]:
        questions.extend([
            "备件管理有什么规定？",
            "备件怎么申请？",
            "备件库存怎么管理？",
        ])
    
    if "会议" in name_lower or "纪要" in name_lower:
        questions.extend([
            "这次会议讨论了什么？",
            "会议中有哪些待办事项？",
        ])
    
    # 通用问题（如果上面没匹配到）
    if not questions:
        questions.extend([
            "这个文档的主要内容是什么？",
            "请介绍一下相关规定",
            "有什么需要注意的事项？",
        ])
    
    # 去重并限制数量
    questions = list(dict.fromkeys(questions))
    return questions[:5]


# ============================================================
# 3. 从文档生成 QA 对
# ============================================================
def process_file(filepath):
    """处理单个文件，返回 QA 对列表"""
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  跳过 {filepath.name}: {e}")
        return []
    
    # 清洗
    cleaned = clean_text(content)
    if len(cleaned) < MIN_CONTENT_LEN:
        return []
    
    # 按章节切分
    sections = split_by_sections(cleaned)
    
    samples = []
    for section in sections:
        if len(section) < MIN_CONTENT_LEN:
            continue
        
        # 提取关键内容作为 answer
        answer = summarize_content(section)
        if len(answer) < MIN_ANSWER_LEN:
            continue
        
        # 检查 answer 质量
        if answer.count("|") > 5:  # 管道符太多，跳过
            continue
        
        # 生成问题
        questions = generate_questions(section, filepath.name)
        
        for q in questions:
            samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": answer}
                ]
            })
    
    return samples


def split_by_sections(text):
    """按章节标题切分"""
    # 匹配：1. 标题 / 6.1.1 标题 / 一、标题
    pattern = r"\n(?=(?:\d+\.?\d*\.?\d*\s+[^\d]|[一二三四五六七八九十]+、))"
    
    parts = re.split(pattern, text)
    sections = [p.strip() for p in parts if p.strip() and len(p.strip()) > MIN_CONTENT_LEN]
    
    # 如果切分太少，按段落切分
    if len(sections) <= 1:
        paragraphs = text.split("\n\n")
        sections = []
        current = ""
        for p in paragraphs:
            if len(current) + len(p) > 1500:
                if current and len(current) > MIN_CONTENT_LEN:
                    sections.append(current)
                current = p
            else:
                current += "\n\n" + p if current else p
        if current and len(current) > MIN_CONTENT_LEN:
            sections.append(current)
    
    return sections


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("BYD 数据清洗脚本 v2")
    print("=" * 60)
    
    # 1. 读取所有文件
    print("\n[1/3] 读取 BYD_excerpt 文件...")
    files = []
    for f in Path(BYD_EXCERPT_DIR).glob("*.txt"):
        if f.name.startswith("~$"):
            continue
        files.append(f)
    print(f"  找到 {len(files)} 个文件")
    
    # 2. 处理文件
    print("\n[2/3] 清洗并生成 QA 对...")
    all_samples = []
    for i, f in enumerate(files):
        samples = process_file(f)
        all_samples.extend(samples)
        if (i + 1) % 20 == 0:
            print(f"  已处理 {i+1}/{len(files)} 文件，累计 {len(all_samples)} 条样本")
    
    print(f"  → 共生成 {len(all_samples)} 条样本")
    
    # 3. 打乱、划分、保存
    print("\n[3/3] 保存数据...")
    random.seed(42)
    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * TRAIN_RATIO)
    train_data = all_samples[:split_idx]
    test_data = all_samples[split_idx:]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_path = os.path.join(OUTPUT_DIR, "byd_train_v2.jsonl")
    test_path = os.path.join(OUTPUT_DIR, "byd_test_v2.jsonl")
    
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(test_path, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print("\n" + "=" * 60)
    print("数据清洗完成！")
    print(f"  训练集：{len(train_data)} 条 → {train_path}")
    print(f"  测试集：{len(test_data)} 条 → {test_path}")
    print("=" * 60)
    
    # 打印示例
    if train_data:
        print("\n示例样本：")
        print(json.dumps(train_data[0], ensure_ascii=False, indent=2)[:800])


if __name__ == "__main__":
    main()
