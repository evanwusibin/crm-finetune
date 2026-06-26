"""
模型合并脚本：把 LoRA 适配器合并到基座模型

为什么需要合并：LoRA 训练出来的是"适配器"（很小的权重），
需要合并到基座模型才能正常部署和推理。

运行：python merge_model.py
"""

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 配置
# ============================================================
BASE_MODEL = "./model/Qwen3-4B"            # 基座模型
PEFT_MODEL = "./finetuned/LoRA_CRM/checkpoint-200"  # LoRA 适配器（checkpoint）
MERGED_MODEL = "./model/Qwen3-4B-merged"   # 合并后输出


# ============================================================
# 合并
# ============================================================
print("=" * 50)
print("模型合并脚本")
print("=" * 50)

print(f"\n[1/4] 加载基座模型：{BASE_MODEL}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype="auto",
    device_map="auto"
)

print(f"[2/4] 加载分词器：{PEFT_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(PEFT_MODEL)

print(f"[3/4] 加载 LoRA 适配器并合并")
peft_model = PeftModel.from_pretrained(base_model, PEFT_MODEL)
merged_model = peft_model.merge_and_unload()

print(f"[4/4] 保存合并后的模型：{MERGED_MODEL}")
merged_model.save_pretrained(MERGED_MODEL)
tokenizer.save_pretrained(MERGED_MODEL)

print("\n" + "=" * 50)
print(f"合并完成！模型保存在：{MERGED_MODEL}")
print("=" * 50)
