"""
数据转换脚本：把 CRM 项目的 FAQ + 维修案例 + 故障码转成微调格式

输入：auto-carcrm/doc/data/ 下的 JSON 文件
输出：data/crm_train.jsonl + data/crm_test.jsonl

数据格式（和老师教程一致的 messages 格式）：
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import json
import os
import random

# ============================================================
# 配置
# ============================================================
CRM_DATA_DIR = r"D:\heimaAI\PytorchSDXX\08_掌柜智库\实战\实战\auto-carcrm\doc\data"
OUTPUT_DIR = "./data"
TRAIN_RATIO = 0.8  # 80% 训练，20% 测试

SYSTEM_PROMPT = "你是比亚迪商用车智能售后助手，精通车辆故障诊断、维修方案、质保政策和保养知识。请用专业、简洁、有条理的方式回答用户问题。"

# ============================================================
# 1. 加载 FAQ 数据
# ============================================================
def load_faq():
    """加载 FAQ 常见问题"""
    filepath = os.path.join(CRM_DATA_DIR, "FAQ常见问题.json")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for faq in data["faqs"]:
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": faq["question"]},
                {"role": "assistant", "content": faq["answer"]}
            ]
        })
    return samples


# ============================================================
# 2. 加载维修案例数据
# ============================================================
def load_repair_cases():
    """加载维修案例库，转成问答格式"""
    filepath = os.path.join(CRM_DATA_DIR, "维修案例库.json")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for case in data["cases"]:
        # 问题：描述故障症状
        question = f"我的{case['vehicle_model']}出现了以下问题：{case['fault_symptom']}，故障码是{case['fault_code']}，该怎么办？"

        # 回答：结构化维修方案
        steps = "\n".join(case["repair_steps"])
        answer = f"""根据您描述的症状和故障码{case['fault_code']}，初步判断是{case['fault_component']}问题。

**故障原因**：{case['fault_cause']}

**维修步骤**：
{steps}

**预计费用**：{case['repair_cost']}
**预计工时**：{case['repair_time']}
**质保覆盖**：{"在质保范围内，免费维修" if case["warranty_covered"] else "不在质保范围，需自费"}"""

        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        })
    return samples


# ============================================================
# 3. 加载故障码数据
# ============================================================
def load_fault_codes():
    """加载故障码大全，转成问答格式"""
    filepath = os.path.join(CRM_DATA_DIR, "故障码大全.json")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for code in data.get("fault_codes", data.get("codes", [])):
        question = f"故障码{code.get('code', '')}是什么意思？"
        answer = f"""故障码{code.get('code', '')}：{code.get('name', '')}

**含义**：{code.get('description', code.get('meaning', ''))}
**可能原因**：{code.get('cause', code.get('possible_cause', ''))}
**建议处理**：{code.get('solution', code.get('repair_suggestion', ''))}"""

        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        })
    return samples


# ============================================================
# 4. 加载三包政策数据
# ============================================================
def load_warranty():
    """加载三包政策"""
    filepath = os.path.join(CRM_DATA_DIR, "三包政策与保养手册.json")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data.get("policies", data.get("items", [])):
        question = item.get("question", item.get("title", ""))
        answer = item.get("answer", item.get("content", ""))
        if question and answer:
            samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            })
    return samples


# ============================================================
# 5. 加载车辆信息数据
# ============================================================
def load_vehicle_info():
    """加载车辆信息"""
    filepath = os.path.join(CRM_DATA_DIR, "车辆信息.json")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for vehicle in data.get("vehicles", data.get("records", [])):
        model = vehicle.get("model", vehicle.get("vehicle_model", ""))
        question = f"{model}的参数配置是什么？"

        info_parts = []
        for key, value in vehicle.items():
            if key not in ["model", "vehicle_model", "id"]:
                info_parts.append(f"- {key}：{value}")
        answer = f"{model}的主要参数：\n" + "\n".join(info_parts)

        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        })
    return samples


# ============================================================
# 6. 加载维修记录数据
# ============================================================
def load_repair_records():
    """加载维修记录"""
    filepath = os.path.join(CRM_DATA_DIR, "维修记录.json")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for record in data.get("records", data.get("repairs", [])):
        vehicle = record.get("vehicle_model", record.get("model", ""))
        fault = record.get("fault_description", record.get("symptom", ""))
        solution = record.get("repair_solution", record.get("solution", ""))

        if fault and solution:
            question = f"我的{vehicle}出现{fault}，怎么修？"
            answer = solution

            samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            })
    return samples


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 50)
    print("CRM 数据转换脚本")
    print("=" * 50)

    # 加载所有数据
    all_samples = []

    print("\n[1/6] 加载 FAQ 数据...")
    faq_samples = load_faq()
    all_samples.extend(faq_samples)
    print(f"  → FAQ: {len(faq_samples)} 条")

    print("[2/6] 加载维修案例...")
    case_samples = load_repair_cases()
    all_samples.extend(case_samples)
    print(f"  → 维修案例: {len(case_samples)} 条")

    print("[3/6] 加载故障码...")
    try:
        fault_samples = load_fault_codes()
        all_samples.extend(fault_samples)
        print(f"  → 故障码: {len(fault_samples)} 条")
    except Exception as e:
        print(f"  → 故障码加载失败: {e}")

    print("[4/6] 加载三包政策...")
    try:
        warranty_samples = load_warranty()
        all_samples.extend(warranty_samples)
        print(f"  → 三包政策: {len(warranty_samples)} 条")
    except Exception as e:
        print(f"  → 三包政策加载失败: {e}")

    print("[5/6] 加载车辆信息...")
    try:
        vehicle_samples = load_vehicle_info()
        all_samples.extend(vehicle_samples)
        print(f"  → 车辆信息: {len(vehicle_samples)} 条")
    except Exception as e:
        print(f"  → 车辆信息加载失败: {e}")

    print("[6/6] 加载维修记录...")
    try:
        record_samples = load_repair_records()
        all_samples.extend(record_samples)
        print(f"  → 维修记录: {len(record_samples)} 条")
    except Exception as e:
        print(f"  → 维修记录加载失败: {e}")

    # 打乱数据
    random.seed(42)
    random.shuffle(all_samples)

    # 划分训练集和测试集
    split_idx = int(len(all_samples) * TRAIN_RATIO)
    train_data = all_samples[:split_idx]
    test_data = all_samples[split_idx:]

    # 输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 保存训练集
    train_path = os.path.join(OUTPUT_DIR, "crm_train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 保存测试集
    test_path = os.path.join(OUTPUT_DIR, "crm_test.jsonl")
    with open(test_path, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n" + "=" * 50)
    print(f"数据转换完成！")
    print(f"  总样本数：{len(all_samples)}")
    print(f"  训练集：{len(train_data)} 条 → {train_path}")
    print(f"  测试集：{len(test_data)} 条 → {test_path}")
    print("=" * 50)

    # 打印一条示例
    print("\n示例数据：")
    print(json.dumps(train_data[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
