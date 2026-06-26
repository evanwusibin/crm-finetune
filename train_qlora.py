"""
LoRA 微调脚本：用 CRM 数据微调 Qwen3-8B

参考：老师教程 06_peft_demo.py
方法：全精度 BF16 + LoRA（RTX 5070 Ti 12GB）

运行：python train_qlora.py
监控：tensorboard --logdir logs/
"""

import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import torch

# ============================================================
# 配置（按需修改）
# ============================================================
MODEL_PATH = "./model/Qwen3-4B"            # 基座模型路径（4B，12GB 卡刚好够）
DATA_DIR = "./data"                        # 数据目录
OUTPUT_DIR = "./finetuned/LoRA_CRM"        # 输出目录
LOG_DIR = "./logs/LoRA_CRM"              # 日志目录

# 训练超参数
BATCH_SIZE = 1                             # 每卡 batch size（4B 模型，12GB 显存用 1）
GRAD_ACCUM = 32                            # 梯度累积步数（有效 batch = 1*32 = 32）
MAX_STEPS = 500                            # 训练步数
LEARNING_RATE = 2e-5                       # 学习率（LoRA 用 2e-5）
MAX_LENGTH = 128                           # 最大序列长度（省显存）

# LoRA 配置
LORA_R = 4                                 # 秩（越大效果越好，显存越大）
LORA_ALPHA = 4                             # 缩放系数
LORA_DROPOUT = 0.05                        # Dropout
TARGET_MODULES = ["q_proj", "v_proj"]      # 插在哪些层


# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 50)
print("Step 1：加载数据")
print("=" * 50)

dataset = load_dataset("json", data_files={
    "train": os.path.join(DATA_DIR, "crm_train.jsonl"),
    "test": os.path.join(DATA_DIR, "crm_test.jsonl")
})
print(f"  训练集：{len(dataset['train'])} 条")
print(f"  测试集：{len(dataset['test'])} 条")


# ============================================================
# 2. 加载模型和分词器（全精度，不用量化）
# ============================================================
print("\nStep 2：加载模型（全精度 BF16）")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,            # 用 BF16 精度
    device_map="cuda:0",                   # 强制全放 GPU
    low_cpu_mem_usage=True                 # 减少 CPU 内存占用
)
print(f"  模型加载完成，显存占用：{torch.cuda.memory_allocated() / 1024**3:.1f} GB")


# ============================================================
# 3. 添加 LoRA 适配器
# ============================================================
print("\nStep 3：添加 LoRA 适配器")

# 配置 LoRA
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=TARGET_MODULES,
    task_type="CAUSAL_LM"
)

# 包装成 LoRA 模型
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()


# ============================================================
# 4. 配置训练参数
# ============================================================
print("\nStep 4：配置训练参数")

os.makedirs(LOG_DIR, exist_ok=True)
os.environ["TENSORBOARD_LOGGING_DIR"] = LOG_DIR

training_args = SFTConfig(
    # Batch 相关
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,

    # 训练步数
    max_steps=MAX_STEPS,
    num_train_epochs=1,

    # 学习率
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    # 日志和评估
    logging_strategy="steps",
    logging_steps=50,
    report_to="tensorboard",
    eval_strategy="steps",
    eval_steps=100,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # 保存
    load_best_model_at_end=True,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    output_dir=OUTPUT_DIR,

    # 精度和长度
    bf16=True,
    max_length=MAX_LENGTH,
    assistant_only_loss=True,  # 只对 assistant 部分计算 loss

    # ChatML 模板
    chat_template_path=None  # 使用模型自带的 chat template
)


# ============================================================
# 5. 构造 Trainer
# ============================================================
print("\nStep 5：构造 Trainer")

trainer = SFTTrainer(
    args=training_args,
    model=peft_model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer
)


# ============================================================
# 6. 开始训练
# ============================================================
print("\n" + "=" * 50)
print("Step 4：开始训练！")
print(f"  模型：{MODEL_PATH}")
print(f"  方法：LoRA (r={LORA_R}, alpha={LORA_ALPHA})")
print(f"  Batch：{BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"  步数：{MAX_STEPS}")
print(f"  学习率：{LEARNING_RATE}")
print("=" * 50)

trainer.train()

# 保存模型
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n" + "=" * 50)
print(f"训练完成！模型保存在：{OUTPUT_DIR}")
print("=" * 50)
