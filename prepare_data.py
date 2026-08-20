"""
数据转换脚本：把 BYD_excerpt 的比亚迪业务文档转成微调格式

输入：CRMProject_c/BYD_excerpt/ 下的 200 个 .txt 文件
输出：data/crm_train.jsonl + data/crm_test.jsonl

数据格式（ShareGPT / OpenAI messages）：
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

原版 prepare_data.py 读取 auto-carcrm 的 JSON 数据（65 条）
本版读取 BYD_excerpt 的 .txt 文件（200 个文件，数千页）
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
TRAIN_RATIO = 0.8  # 80% 训练，20% 测试

SYSTEM_PROMPT = "你是比亚迪商用车智能售后助手，精通车辆故障诊断、维修方案、质保政策、索赔管理、CRM系统操作和保养知识。请用专业、简洁、有条理的方式回答用户问题。"

# 每个文档切分的最大字符数（防止过长）
CHUNK_MAX_CHARS = 2000
# 每个 chunk 最少字符数（太短的丢弃）
CHUNK_MIN_CHARS = 100


# ============================================================
# 1. 读取所有 .txt 文件
# ============================================================
def load_all_txt_files():
    """读取 BYD_excerpt 下所有 .txt 文件"""
    files = []
    for f in Path(BYD_EXCERPT_DIR).glob("*.txt"):
        # 跳过临时文件（~$ 开头）
        if f.name.startswith("~$"):
            continue
        files.append(f)
    print(f"  找到 {len(files)} 个 .txt 文件")
    return files


# ============================================================
# 2. 按文档类型切分内容
# ============================================================
def split_by_pages(content, filename):
    """按 PAGE/SLIDE/SHEET 标记切分"""
    # PDF 提取格式：=== PAGE 1 ===
    # PPTX 提取格式：=== SLIDE 1 ===
    # XLSX 提取格式：=== SHEET Sheet1 ===
    pattern = r"=== (PAGE|SLIDE|SHEET) .+? ==="
    
    chunks = []
    parts = re.split(pattern, content)
    
    # parts[0] 是第一个标记之前的内容（通常是封面/目录）
    if parts[0].strip() and len(parts[0].strip()) > CHUNK_MIN_CHARS:
        chunks.append(parts[0].strip())
    
    # 每个标记后的内容
    for i in range(1, len(parts), 2):
        chunk = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if chunk and len(chunk) > CHUNK_MIN_CHARS:
            chunks.append(chunk)
    
    return chunks


def split_by_sections(content):
    """按章节标题切分（无标记的纯文本）"""
    # 匹配：1. 标题 / 一、标题 / 第一章 / ## 标题
    pattern = r"\n(?=(?:\d+\.|[一二三四五六七八九十]+、|第[一二三四五六七八九十]+章|#{1,3} ))"
    
    chunks = re.split(pattern, content)
    chunks = [c.strip() for c in chunks if c.strip() and len(c.strip()) > CHUNK_MIN_CHARS]
    
    return chunks


def chunk_document(content, filename):
    """智能切分文档"""
    # 先尝试按 PAGE/SLIDE/SHEET 切分
    chunks = split_by_pages(content, filename)
    
    # 如果切分结果太少，按章节切分
    if len(chunks) <= 1:
        chunks = split_by_sections(content)
    
    # 如果还是太少，按固定长度切分
    if len(chunks) <= 1 and len(content) > CHUNK_MAX_CHARS:
        chunks = []
        for i in range(0, len(content), CHUNK_MAX_CHARS):
            chunk = content[i:i + CHUNK_MAX_CHARS]
            if len(chunk) > CHUNK_MIN_CHARS:
                chunks.append(chunk)
    
    # 对过长的 chunk 再切分
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > CHUNK_MAX_CHARS:
            # 按段落切分
            paragraphs = chunk.split("\n\n")
            current = ""
            for para in paragraphs:
                if len(current) + len(para) > CHUNK_MAX_CHARS:
                    if current and len(current) > CHUNK_MIN_CHARS:
                        final_chunks.append(current)
                    current = para
                else:
                    current += "\n\n" + para if current else para
            if current and len(current) > CHUNK_MIN_CHARS:
                final_chunks.append(current)
        else:
            final_chunks.append(chunk)
    
    return final_chunks


# ============================================================
# 3. 从文档内容生成 Q&A 对
# ============================================================
def generate_qa_from_chunk(chunk, filename):
    """从文档片段生成多个 Q&A 对"""
    samples = []
    
    # 清理内容
    chunk = chunk.strip()
    if len(chunk) < CHUNK_MIN_CHARS:
        return samples
    
    # 从文件名推断文档类型
    doc_type = infer_doc_type(filename, chunk)
    
    # 策略 1：基于文档类型生成特定问题
    qa_pairs = generate_typed_qa(chunk, filename, doc_type)
    samples.extend(qa_pairs)
    
    # 策略 2：基于章节标题生成问题
    title_qa = generate_title_qa(chunk, filename)
    samples.extend(title_qa)
    
    # 策略 3：基于关键词生成问题
    keyword_qa = generate_keyword_qa(chunk, doc_type)
    samples.extend(keyword_qa)
    
    return samples


def infer_doc_type(filename, content):
    """推断文档类型"""
    name_lower = filename.lower()
    content_lower = content.lower()
    
    if "质保" in name_lower or "保修" in name_lower or "保养" in name_lower:
        return "warranty"
    elif "索赔" in name_lower or "审核" in name_lower:
        return "claim"
    elif "crm" in name_lower or "系统" in name_lower or "操作" in name_lower:
        return "crm_system"
    elif "培训" in name_lower or "ppt" in name_lower:
        return "training"
    elif "会议" in name_lower or "纪要" in name_lower:
        return "meeting"
    elif "稽查" in name_lower or "考评" in name_lower:
        return "audit"
    elif "故障" in content_lower or "维修" in content_lower:
        return "repair"
    elif "备件" in name_lower or "零件" in name_lower:
        return "parts"
    else:
        return "general"


def generate_typed_qa(chunk, filename, doc_type):
    """根据文档类型生成特定 Q&A"""
    samples = []
    
    # 提取文档标题（第一行非空文本）
    lines = [l.strip() for l in chunk.split("\n") if l.strip()]
    doc_title = lines[0] if lines else filename.replace(".txt", "")
    
    type_templates = {
        "warranty": [
            (f"比亚迪商用车的质保政策是什么？", f"根据{doc_title}：\n\n{chunk[:1500]}"),
            (f"质保期内哪些情况可以免费维修？", f"根据{doc_title}：\n\n{chunk[:1500]}"),
            (f"质保期是多久？怎么计算？", f"根据{doc_title}：\n\n{chunk[:1500]}"),
        ],
        "claim": [
            (f"索赔流程是怎样的？", f"根据{doc_title}：\n\n{chunk[:1500]}"),
            (f"索赔单怎么提交？需要哪些材料？", f"根据{doc_title}：\n\n{chunk[:1500]}"),
            (f"什么情况下会被驳回？", f"根据{doc_title}：\n\n{chunk[:1500]}"),
        ],
        "crm_system": [
            (f"纷享销客CRM系统怎么操作？", f"根据{doc_title}：\n\n{chunk[:1500]}"),
            (f"CRM系统有哪些功能模块？", f"根据{doc_title}：\n\n{chunk[:1500]}"),
            (f"怎么在CRM里创建工单？", f"根据{doc_title}：\n\n{chunk[:1500]}"),
        ],
        "training": [
            (f"这个培训的主要内容是什么？", f"根据{doc_title}：\n\n{chunk[:1500]}"),
            (f"培训中提到了哪些操作步骤？", f"根据{doc_title}：\n\n{chunk[:1500]}"),
        ],
        "meeting": [
            (f"这次会议讨论了什么？", f"根据{doc_title}：\n\n{chunk[:1500]}"),
            (f"会议中提到了哪些待办事项？", f"根据{doc_title}：\n\n{chunk[:1500]}"),
        ],
        "audit": [
            (f"稽查/考评的标准是什么？", f"根据{doc_title}：\n\n{chunk[:1500]}"),
            (f"考评不合格会有什么后果？", f"根据{doc_title}：\n\n{chunk[:1500]}"),
        ],
        "repair": [
            (f"这个故障怎么处理？", f"根据{doc_title}：\n\n{chunk[:1500]}"),
            (f"维修步骤是什么？", f"根据{doc_title}：\n\n{chunk[:1500]}"),
        ],
        "parts": [
            (f"备件管理有什么规定？", f"根据{doc_title}：\n\n{chunk[:1500]}"),
            (f"备件怎么申请？", f"根据{doc_title}：\n\n{chunk[:1500]}"),
        ],
        "general": [
            (f"请介绍一下这个文档的主要内容", f"根据{doc_title}：\n\n{chunk[:1500]}"),
        ],
    }
    
    templates = type_templates.get(doc_type, type_templates["general"])
    for question, answer in templates:
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        })
    
    return samples


def generate_title_qa(chunk, filename):
    """从章节标题生成 Q&A"""
    samples = []
    
    # 匹配章节标题
    title_patterns = [
        r"^(#{1,3})\s+(.+)$",           # Markdown: # Title
        r"^(\d+\.?\d*\.?\d*)\s+(.+)$",   # 数字编号: 1.2.3 Title
        r"^([一二三四五六七八九十]+、)\s*(.+)$",  # 中文编号
    ]
    
    titles = []
    for line in chunk.split("\n"):
        line = line.strip()
        for pattern in title_patterns:
            match = re.match(pattern, line)
            if match:
                title = match.group(2).strip()
                if len(title) > 2 and len(title) < 50:
                    titles.append(title)
                break
    
    # 为每个标题生成 Q&A
    for title in titles[:3]:  # 最多 3 个
        question = f"关于「{title}」，有什么规定？"
        answer = f"根据文档内容：\n\n{chunk[:1500]}"
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        })
    
    return samples


def generate_keyword_qa(chunk, doc_type):
    """基于关键词生成 Q&A"""
    samples = []
    
    # 关键词 → 问题映射
    keyword_questions = {
        "质保": "质保相关的问题可以参考什么文档？",
        "索赔": "索赔流程是怎样的？",
        "故障": "故障处理的标准流程是什么？",
        "维修": "维修操作有什么规范？",
        "保养": "保养周期和项目有哪些？",
        "备件": "备件管理有什么规定？",
        "审批": "审批流程是怎样的？",
        "考核": "考核标准是什么？",
        "培训": "培训内容包括哪些？",
        "CRM": "CRM系统怎么使用？",
        "工单": "工单怎么创建和处理？",
        "客户": "客户管理有什么规范？",
    }
    
    found_keywords = []
    for keyword in keyword_questions:
        if keyword in chunk:
            found_keywords.append(keyword)
    
    # 为找到的关键词生成 Q&A（最多 2 个）
    for keyword in found_keywords[:2]:
        question = keyword_questions[keyword]
        answer = f"根据文档内容：\n\n{chunk[:1500]}"
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        })
    
    return samples


# ============================================================
# 4. 生成纯文本补全样本
# ============================================================
def generate_completion_sample(chunk, filename):
    """生成纯文本补全样本（让模型学习文档风格）"""
    # 取前 200 字作为提示，后文作为补全
    if len(chunk) < 300:
        return None
    
    prompt_text = chunk[:200]
    completion_text = chunk[200:1500]
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请继续以下文档内容：\n\n{prompt_text}"},
            {"role": "assistant", "content": completion_text}
        ]
    }


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("BYD_excerpt 数据转换脚本")
    print("=" * 60)
    
    # 1. 读取所有文件
    print("\n[1/4] 读取 BYD_excerpt 文件...")
    files = load_all_txt_files()
    
    # 2. 切分文档
    print("\n[2/4] 切分文档...")
    all_chunks = []
    for f in files:
        try:
            content = f.read_text(encoding="utf-8")
            chunks = chunk_document(content, f.name)
            for chunk in chunks:
                all_chunks.append((chunk, f.name))
        except Exception as e:
            print(f"  跳过文件 {f.name}: {e}")
    
    print(f"  → 共切分为 {len(all_chunks)} 个文档片段")
    
    # 3. 生成 Q&A 对
    print("\n[3/4] 生成 Q&A 训练数据...")
    all_samples = []
    
    for chunk, filename in all_chunks:
        # 生成 Q&A 对
        qa_samples = generate_qa_from_chunk(chunk, filename)
        all_samples.extend(qa_samples)
        
        # 生成纯文本补全样本（每 5 个 chunk 生成 1 个）
        if random.random() < 0.2:
            completion = generate_completion_sample(chunk, filename)
            if completion:
                all_samples.append(completion)
    
    print(f"  → 共生成 {len(all_samples)} 条训练样本")
    
    # 4. 打乱、划分、保存
    print("\n[4/4] 保存数据...")
    random.seed(42)
    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * TRAIN_RATIO)
    train_data = all_samples[:split_idx]
    test_data = all_samples[split_idx:]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_path = os.path.join(OUTPUT_DIR, "byd_train.jsonl")
    test_path = os.path.join(OUTPUT_DIR, "byd_test.jsonl")
    
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(test_path, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print("\n" + "=" * 60)
    print("数据转换完成！")
    print(f"  来源文件：{len(files)} 个 .txt 文件")
    print(f"  文档片段：{len(all_chunks)} 个")
    print(f"  总样本数：{len(all_samples)} 条")
    print(f"  训练集：{len(train_data)} 条 → {train_path}")
    print(f"  测试集：{len(test_data)} 条 → {test_path}")
    print("=" * 60)
    
    # 打印示例
    print("\n示例数据（第 1 条）：")
    print(json.dumps(train_data[0], ensure_ascii=False, indent=2)[:500])
    
    # 统计文档类型分布
    print("\n文档类型分布：")
    type_count = {}
    for chunk, filename in all_chunks:
        doc_type = infer_doc_type(filename, chunk)
        type_count[doc_type] = type_count.get(doc_type, 0) + 1
    for doc_type, count in sorted(type_count.items(), key=lambda x: -x[1]):
        print(f"  {doc_type}: {count} 个片段")


if __name__ == "__main__":
    main()
