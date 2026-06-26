"""
推理验证脚本：测试微调后的模型效果

运行：python inference.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "./model/Qwen3-4B-merged"    # 合并后的模型
# 如果还没合并，可以用 LoRA 适配器测试：
# MODEL_PATH = "./model/Qwen3-8B"  # 基座模型（对比用）


# ============================================================
# 加载模型
# ============================================================
print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)
print("模型加载完成！\n")


# ============================================================
# 测试用例（来自你的 CRM 项目）
# ============================================================
test_questions = [
    # ====== A. 训练集内问题（考察记忆能力）======
    "发动机故障灯亮了怎么办？",                  # 训练集有
    "故障码P0562是什么意思？",                    # 训练集有
    "比亚迪T5的参数配置是什么？",                 # 训练集有

    # ====== B. 训练集外同类问题（考察泛化能力）======
    "终身质保是保哪些零部件？",                   # 质保类，训练集没这题
    "首保和定保有什么区别？",                     # 保养类，训练集没这题
    "故障码P0300是什么意思？",                    # 新故障码，训练集没这题
    "刹车异响是什么原因？",                       # 新故障场景
    "空调不制冷怎么处理？",                       # 新故障场景

    # ====== C. 边界/挑战题（考察鲁棒性）======
    "比亚迪T3的参数是什么？",                     # 训练集只有T5/T7，看是否幻觉
    "我的车打不着火了怎么办？",                   # 模糊问题，看是否追问细节
    "你们公司还招人吗？",                         # 超出业务范围，看是否拒答
]


# ============================================================
# 推理函数
# ============================================================
def chat(question, max_new_tokens=256):
    """单轮对话"""
    messages = [
        {"role": "system", "content": "你是比亚迪商用车智能售后助手，精通车辆故障诊断、维修方案、质保政策和保养知识。"},
        {"role": "user", "content": question}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    # 只取生成的部分
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


# ============================================================
# 运行测试
# ============================================================
print("=" * 50)
print("CRM 智能售后助手 · 推理测试")
print("=" * 50)

for i, question in enumerate(test_questions, 1):
    print(f"\n{'─' * 40}")
    print(f"[Q{i}] {question}")
    print(f"{'─' * 40}")
    answer = chat(question)
    print(f"[A{i}] {answer}")

print("\n" + "=" * 50)
print("测试完成！")
print("=" * 50)
