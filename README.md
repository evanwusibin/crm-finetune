# CRM-Finetune：汽车售后领域大模型微调与私有化推理服务

> 基于 Qwen3-4B 的汽车售后领域 LoRA 微调项目，面向比亚迪商用车/乘用车售后知识问答、故障码解释、维修建议、质保政策和保养咨询等场景。项目包含数据构造、监督微调、训练评估、模型合并、本地推理、OpenAI 兼容 API 服务，以及与 `auto-carcrm` RAG 系统的联动接入方案。

---

## 项目定位

`crm-finetune` 解决的是 RAG 系统中的“生成模型领域适配”问题。

在汽车售后知识助手中，RAG 负责从知识库中检索事实依据，但最终回答仍依赖大模型完成组织、解释和表达。通用模型容易出现三个问题：

1. 对汽车售后术语、故障码、质保/保养表达不稳定。
2. 回答风格不符合售后客服场景，缺少结构化排查步骤。
3. 私有化部署时，希望减少对外部云模型的依赖。

本项目通过少量高质量 CRM 售后数据对 Qwen3-4B 进行 LoRA 微调，让模型更熟悉汽车售后问答格式，并通过 OpenAI 兼容接口暴露为本地模型服务，供 `auto-carcrm` 一键切换使用。

---

## 和 auto-carcrm 的关系

两个项目可以配合使用：

```text
用户问题
  ↓
auto-carcrm
  ├─ 主体识别 / 意图识别 / 实体抽取
  ├─ Milvus 向量检索 + BM25 关键词检索 + 结构化查询 + 案例召回 + HyDE + MCP Web Search
  ├─ RRF 融合 / Metadata Filter / Rerank / 置信度判断
  └─ 调用 LLM 生成最终答案
          ↓
crm-finetune 本地模型服务
  └─ /v1/chat/completions OpenAI 兼容接口
```

简言之：

- `auto-carcrm` 解决“查什么、从哪里查、怎么融合证据”。
- `crm-finetune` 解决“用更贴近汽车售后业务的模型怎么回答”。
- 两者通过 OpenAI 兼容 API 对接，RAG 项目只需要改 `.env` 中的 `OPENAI_BASE_URL` 和 `LLM_DEFAULT_MODEL`。

---

## 核心能力

| 能力 | 说明 |
|---|---|
| 数据转换 | 将 FAQ、维修案例、故障码、三包政策、车辆信息、维修记录转换为 chat messages 格式 JSONL |
| LoRA 微调 | 基于 Transformers + PEFT + TRL SFTTrainer 对 Qwen3-4B 做领域监督微调 |
| 显存友好 | 面向单卡 12GB 级别显存配置，使用 BF16 + LoRA、小 batch + 梯度累积降低门槛 |
| 训练监控 | 输出 TensorBoard 日志、训练 loss、eval loss、token accuracy 等指标 |
| 模型合并 | 支持将 LoRA adapter 合并为可独立部署的 merged 模型 |
| 推理测试 | 支持单条推理、批量 benchmark、首 token 延迟和吞吐评估 |
| API 服务 | 提供 FastAPI 版 OpenAI 兼容 `/v1/models`、`/v1/chat/completions` 接口，支持流式 SSE |
| RAG 接入 | 可作为 `auto-carcrm` 的本地 LLM 后端，替换云端模型调用 |

---

## 技术栈

| 层次 | 技术 |
|---|---|
| 基座模型 | Qwen3-4B（本地路径 `./model/Qwen3-4B`） |
| 微调框架 | Transformers、PEFT、TRL SFTTrainer |
| 训练方法 | LoRA / QLoRA 实验，当前主脚本为 BF16 + LoRA |
| 训练数据 | OpenAI chat messages JSONL 格式 |
| 推理服务 | FastAPI + Uvicorn + Transformers generate |
| API 协议 | OpenAI compatible `/v1/chat/completions` |
| 性能评估 | 自定义 benchmark、eval report、TensorBoard |
| 依赖管理 | `requirements.txt` |

---

## 目录结构

```text
crm-finetune/
├── prepare_data.py              # 从 auto-carcrm/doc/data 转换训练/测试 JSONL
├── train_qlora.py               # LoRA 微调主脚本
├── merge_model.py               # 合并 LoRA adapter 与基座模型
├── inference.py                 # 本地模型推理测试
├── benchmark_inference.py       # 推理性能 benchmark
├── benchmark_simple.py          # 简化版 benchmark
├── eval_report_gen.py           # 汇总训练/推理评估报告
├── serve_api.py                 # 简单 FastAPI 服务
├── serve_openai.py              # OpenAI 兼容 API 服务，RAG 推荐接入这个
├── check_env.py                 # 环境检查
├── test_gpu.py                  # GPU 可用性检查
├── data/                        # 小规模训练/测试 JSONL 样本
├── docs/                        # 训练过程、评估、部署和技术复盘文档
├── model/                       # 本地基座/合并模型目录，不提交 Git
├── finetuned/                   # LoRA/QLoRA 训练产物，不提交 Git
└── logs/                        # TensorBoard 与训练日志，不提交 Git
```

---

## 数据构造

数据来源于 `auto-carcrm/doc/data` 中的汽车售后知识，包括：

- FAQ 常见问题
- 维修案例库
- 故障码大全
- 三包政策与保养手册
- 车辆信息
- 维修记录

转换后的样本采用 messages 格式：

```json
{
  "messages": [
    {"role": "system", "content": "你是比亚迪商用车智能售后助手..."},
    {"role": "user", "content": "故障码P0562是什么意思？"},
    {"role": "assistant", "content": "故障码P0562表示..."}
  ]
}
```

生成数据：

```bash
python prepare_data.py
```

默认输出：

```text
data/crm_train.jsonl
data/crm_test.jsonl
```

当前样本规模较小，主要用于验证微调链路和领域表达适配，不适合宣称已覆盖完整汽车售后知识库。

---

## 训练配置

主训练脚本：`train_qlora.py`

关键配置：

```python
MODEL_PATH = "./model/Qwen3-4B"
OUTPUT_DIR = "./finetuned/LoRA_CRM"
BATCH_SIZE = 1
GRAD_ACCUM = 32
MAX_STEPS = 500
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
LORA_R = 4
LORA_ALPHA = 4
TARGET_MODULES = ["q_proj", "v_proj"]
```

训练启动：

```bash
python train_qlora.py
```

TensorBoard 监控：

```bash
tensorboard --logdir logs/
```

---

## 已记录评估结果

当前 `eval_report.json` 记录了一次 200 step 训练和推理 benchmark 结果：

| 指标 | 结果 |
|---|---:|
| final loss | 1.3349 |
| eval loss | 1.3783 |
| token accuracy | 68.50% |
| 平均首 token 延迟 TFTT | 50.01 ms |
| P95 首 token 延迟 | 54.96 ms |
| 平均吞吐 | 22.4 tok/s |
| 平均 QPS | 0.175 req/s |

质量侧结论更重要：

- 对训练覆盖过的保养、质保、常见故障类问题，回答结构和售后话术更稳定。
- 数据量只有 65 条左右，故障码覆盖不足，训练外故障码可能会答错。
- Qwen3 thinking 输出可能挤占回答 token，需要在部署时关闭 thinking 或提高 `max_new_tokens`。
- 对非业务问题需要通过 system prompt、拒答样本或 RAG 外层意图识别兜底。

因此该项目适合展示“领域微调完整工程链路”，而不是声称已经达到生产级汽车售后大模型效果。

---

## OpenAI 兼容服务

启动本地服务：

```bash
python serve_openai.py
```

默认配置：

```text
HOST = 0.0.0.0
PORT = 8100
MODEL_NAME = qwen3-crm
MODEL_PATH = ./model/Qwen3-4B-merged
```

查看模型列表：

```bash
curl http://localhost:8100/v1/models
```

聊天补全：

```bash
curl http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-crm","messages":[{"role":"user","content":"故障码P0562是什么意思？"}],"max_tokens":512}'
```

---

## 接入 auto-carcrm

在 `auto-carcrm/.env` 中将 LLM 配置切到本地服务：

```env
OPENAI_BASE_URL=http://localhost:8100/v1
OPENAI_API_KEY=not-needed
LLM_DEFAULT_MODEL=qwen3-crm
```

这样 RAG 系统中的意图识别、主体识别、HyDE、实体抽取、答案生成等 LLM 调用点可以复用同一个本地模型服务。

---

## 运行顺序建议

```bash
# 1. 准备数据
python prepare_data.py

# 2. 微调
python train_qlora.py

# 3. 合并模型
python merge_model.py

# 4. 本地推理检查
python inference.py

# 5. 启动 OpenAI 兼容服务
python serve_openai.py

# 6. 在 auto-carcrm 中切换 .env 并重启 RAG 后端
```

---

## 项目边界

- 不提交基座模型、合并模型、LoRA 权重和训练日志到 GitHub。
- 当前数据集规模偏小，适合教学、验证和面试讲解，不代表完整生产效果。
- 当前部署脚本是单机推理服务；高并发生产建议替换为 vLLM、TGI 或企业模型网关。
- 微调和 RAG 是互补关系：微调提升表达和领域习惯，RAG 负责事实召回和可追溯引用。

---

## 推荐仓库描述

> Qwen3-4B LoRA fine-tuning project for automotive CRM after-sales QA, with data preparation, evaluation, model merge, OpenAI-compatible FastAPI serving, and integration with a LangGraph + Milvus RAG system.
