# train_qlora.py 完整流程串讲

> **日期**：2026-06-25
> **文件**：train_qlora.py
> **方法**：QLoRA（4bit 量化 + LoRA）
> **模型**：Qwen3-8B

---

## 整体流程

```
┌─────────────────────────────────────────────────────┐
│                    整体流程                          │
│                                                     │
│  Step 1 加载数据 → Step 2 量化配置 → Step 3 加载模型 │
│       ↓                                            │
│  Step 4 添加LoRA → Step 5 训练参数 → Step 6 Trainer │
│       ↓                                            │
│  Step 7 训练 → 保存模型                              │
└─────────────────────────────────────────────────────┘
```

---

## Step 1：加载数据（第 47-52 行）

```python
dataset = load_dataset("json", data_files={...})
```

📌 **知识点位置**：数据加载（老师教程 05_trl_demo.py 第 1 部分）

| 项目 | 说明 |
|---|---|
| **是什么** | 用 HuggingFace `datasets` 库加载 JSONL 文件 |
| **为什么** | 微调脚本需要 `datasets.Dataset` 格式，不是原始 JSON |
| **怎么做** | `load_dataset("json", data_files=...)` 自动解析 JSONL |
| **处于什么位置** | 整个流程的**输入端**，后面所有步骤都依赖它 |
| **好处** | 自动处理内存映射、支持大数据集、支持 map 操作 |
| **坏处** | 数据量小时开销大（但可忽略） |

---

## Step 2：配置 4bit 量化（第 60-65 行）

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

📌 **知识点位置**：QLoRA 量化（老师教程 08_QLoRA_demo.py 第 1 部分）

| 项目 | 说明 |
|---|---|
| **是什么** | 把模型权重从 FP16（16bit）压缩到 4bit |
| **为什么** | Qwen3-8B FP16 需要 ~16GB 显存，4bit 只需 ~4GB |
| **怎么做** | bitsandbytes 库的 NF4 量化算法 |
| **处于什么位置** | 模型加载前的配置，决定模型占用多少显存 |
| **好处** | 显存减少 75%，12GB 卡能跑 8B 模型 |
| **坏处** | 精度损失（但实际效果影响很小） |

**逐行解释**：

| 参数 | 值 | 含义 |
|---|---|---|
| `load_in_4bit` | True | 开启 4bit 量化 |
| `bnb_4bit_quant_type` | "nf4" | NF4 量化算法（比普通 INT4 更好） |
| `bnb_4bit_use_double_quant` | False | 不用双重量化（省一步计算） |
| `bnb_4bit_compute_dtype` | bfloat16 | 计算时用 BF16 精度 |

---

## Step 3：加载模型和分词器（第 73-79 行）

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto"
)
```

📌 **知识点位置**：模型加载（老师教程 02_sft_demo.py 第 1 部分）

| 项目 | 说明 |
|---|---|
| **是什么** | 从本地加载 Qwen3-8B 模型和分词器 |
| **为什么** | 训练前必须把模型加载到 GPU |
| **怎么做** | `AutoModelForCausalLM` 自动识别模型类型并加载 |
| **处于什么位置** | 量化配置的消费者，后续 LoRA 的载体 |
| **好处** | `device_map="auto"` 自动分配 GPU 显存 |
| **坏处** | 加载 8B 模型需要时间（约 1-2 分钟） |

**两个关键组件**：

| 组件 | 作用 |
|---|---|
| `tokenizer` | 把文字转成 token ID（输入给模型的数字） |
| `model` | 模型本体，包含所有权重参数 |

---

## Step 4：添加 LoRA 适配器（第 88-102 行）

```python
prepared_model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(r=4, lora_alpha=4, ...)
peft_model = get_peft_model(prepared_model, lora_config)
peft_model.print_trainable_parameters()
```

📌 **知识点位置**：LoRA + PEFT（老师教程 06_peft_demo.py 第 1-2 部分）

| 项目 | 说明 |
|---|---|
| **是什么** | 在量化模型上插入 LoRA 适配器，只训练少量参数 |
| **为什么** | 全参微调 8B 需要 ~32GB，LoRA 只训练 0.2% 参数 |
| **怎么做** | PEFT 库的 `get_peft_model` 自动在指定层插入 LoRA |
| **处于什么位置** | 核心步骤——决定了"训练什么"和"训练多少" |
| **好处** | 显存省、训练快、效果接近全参 |
| **坏处** | 表达能力比全参弱一点 |

**三行代码的作用**：

| 行 | 作用 |
|---|---|
| `prepare_model_for_kbit_training` | 让量化模型能参与梯度计算（冻结原始权重，启用梯度） |
| `LoraConfig` | 配置 LoRA 的参数（r、alpha、dropout、目标层） |
| `get_peft_model` | 把 LoRA 适配器"插入"到模型的 q_proj 和 v_proj 层 |

---

## Step 5：配置训练参数（第 113-150 行）

📌 **知识点位置**：SFTConfig（老师教程 01-SFTConfig参数.md）

这是整个脚本**参数最多**的部分，分 5 组：

**① Batch 相关**：
```python
per_device_train_batch_size=4,      # 每次送 4 条数据
gradient_accumulation_steps=8,       # 累积 8 次再更新
```
→ 有效 batch = 4×8 = 32

**② 学习率调度**：
```python
learning_rate=3e-4,                  # 学习率
lr_scheduler_type="cosine",          # 余弦衰减
warmup_ratio=0.1,                    # 前 10% 步预热
```
→ 开始小步走 → 中间大步走 → 最后小步走

**③ 评估和保存**：
```python
eval_strategy="steps",               # 每 N 步评估一次
eval_steps=100,                      # 每 100 步评估
save_steps=200,                      # 每 200 步保存
save_total_limit=3,                  # 最多保留 3 个检查点
```
→ 防止训练中断丢失进度

**④ 精度**：
```python
bf16=True,                           # 用 BF16 混合精度训练
```
→ 比 FP32 快一倍，精度损失可忽略

**⑤ Loss 计算**：
```python
assistant_only_loss=True,            # 只对 assistant 回答计算 loss
```
→ 不对 system/user 的输入计算 loss，只学"怎么回答"

---

## Step 6：构造 Trainer（第 158-164 行）

```python
trainer = SFTTrainer(
    args=training_args,
    model=peft_model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer
)
```

📌 **知识点位置**：SFTTrainer（老师教程 05_trl_demo.py 第 3 部分）

| 项目 | 说明 |
|---|---|
| **是什么** | TRL 库的训练器，封装了训练循环 |
| **为什么** | 手写训练循环要 287 行（02_sft_demo.py），SFTTrainer 只要 7 行 |
| **怎么做** | 传入模型、数据、参数，自动处理前向/反向/更新 |
| **处于什么位置** | 把前面所有配置"组装"在一起的容器 |
| **好处** | 代码简洁、自动处理 padding/mask/评估/保存 |
| **坏处** | 黑盒，出了问题不好调试 |

---

## Step 7：训练（第 179-183 行）

```python
trainer.train()
trainer.save_model(OUTPUT_DIR)
```

📌 **知识点位置**：训练循环（老师教程 02_sft_demo.py 第 4-6 步）

| 项目 | 说明 |
|---|---|
| **是什么** | 执行 1000 步训练，每步：前向→算loss→反向→更新参数 |
| **为什么** | 这是微调的核心——让模型从数据中学习 |
| **怎么做** | `trainer.train()` 自动执行，你只需要看 loss 曲线 |
| **处于什么位置** | 整个流程的**核心**，前面都是准备工作 |
| **好处** | 自动处理一切，有进度条和日志 |
| **坏处** | 训练时间长（约 30-60 分钟） |

---

## 整体知识图谱

```
微调知识体系
├── 数据层
│   ├── 数据格式（JSONL messages）
│   └── 数据加载（datasets 库）
├── 模型层
│   ├── 量化（BitsAndBytes 4bit NF4）
│   ├── 基座模型（Qwen3-8B）
│   └── LoRA 适配器（PEFT 库）
├── 训练层
│   ├── Trainer（SFTTrainer）
│   ├── 超参数（batch/lr/steps）
│   ├── 学习率调度（cosine + warmup）
│   └── Loss 计算（assistant_only）
└── 输出层
    ├── 模型保存（LoRA 适配器）
    └── 日志（TensorBoard）
```

---

## 对应老师教程文件

| 代码步骤 | 老师教程文件 | 对应章节 |
|---|---|---|
| Step 1 加载数据 | 05_trl_demo.py | 数据加载部分 |
| Step 2 量化配置 | 08_QLoRA_demo.py | BitsAndBytesConfig |
| Step 3 加载模型 | 02_sft_demo.py | AutoModelForCausalLM |
| Step 4 添加 LoRA | 06_peft_demo.py | LoraConfig + get_peft_model |
| Step 5 训练参数 | 01-SFTConfig参数.md | 所有参数详解 |
| Step 6 Trainer | 05_trl_demo.py | SFTTrainer 构造 |
| Step 7 训练 | 02_sft_demo.py | trainer.train() |
