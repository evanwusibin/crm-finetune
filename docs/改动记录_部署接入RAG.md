# 改动记录：微调模型接入 RAG 项目

> 日期：2026-06-26

---

## 改动清单

### 1. 新建文件

| 文件 | 说明 |
|------|------|
| `crm-finetune/serve_openai.py` | OpenAI 兼容 API 服务器，加载合并模型，暴露 `/v1/chat/completions` |
| `auto-carcrm/.env.backup` | RAG 项目原始 .env 备份 |

### 2. 修改文件

| 文件 | 改动 |
|------|------|
| `auto-carcrm/.env` | LLM 配置 3 行（见下方） |

### 3. 安装依赖

| 包 | 说明 |
|------|------|
| `fastapi` | API 框架 |
| `uvicorn` | ASGI 服务器 |
| `sse-starlette` | 流式 SSE 支持 |

---

## .env 具体改动

```diff
- LLM_DEFAULT_MODEL=mimo-v2.5-pro
+ LLM_DEFAULT_MODEL=qwen3-crm

- OPENAI_API_KEY=tp-c6uajm09kmtfjujxjnnkul0lhieb6t0j252put1vlbd4znbg
+ OPENAI_API_KEY=not-needed

- OPENAI_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
+ OPENAI_BASE_URL=http://localhost:8100/v1

- VL_ENABLED=true
+ VL_ENABLED=false
```

**VL_ENABLED 改为 false**：微调模型不支持视觉，关闭多模态避免报错。

---

## 启动步骤

### 终端 1：启动模型 API 服务

```bash
cd D:\heimaAI\PytorchSDXX\10_微调\crm-finetune
.venv\python.exe serve_openai.py
```

等待看到：
```
模型加载完成！
启动 API 服务：http://0.0.0.0:8100
```

### 终端 2：启动 RAG 项目

```bash
cd D:\heimaAI\PytorchSDXX\08_掌柜智库\实战\实战\auto-carcrm
python app/main.py
```

### 验证

```bash
# 直接测试模型 API
curl http://localhost:8100/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen3-crm\",\"messages\":[{\"role\":\"system\",\"content\":\"你是比亚迪商用车智能售后助手\"},{\"role\":\"user\",\"content\":\"故障码P0562是什么意思？\"}]}"

# 通过 RAG 项目测试（浏览器打开前端搜索页面）
```

---

## 回滚方法

如果要切回小米 MiMo API：

```bash
# 恢复原始 .env
cp auto-carcrm/.env.backup auto-carcrm/.env
```

或者手动改回：
```env
LLM_DEFAULT_MODEL=mimo-v2.5-pro
OPENAI_API_KEY=tp-c6uajm09kmtfjujxjnnkul0lhieb6t0j252put1vlbd4znbg
OPENAI_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
VL_ENABLED=true
```

---

## RAG 项目受影响的 6 个 LLM 调用点

全部自动切换，无需改代码：

1. `app/rag/query/intent_recognition_service.py` — 意图识别
2. `app/rag/query/entity_extraction_service.py` — 实体提取
3. `app/rag/query/hyde_search_sevice.py` — HyDE 查询扩展
4. `app/rag/query/rerank_service.py` — 文本压缩
5. `app/rag/query/answer_service.py` — 回答生成（支持流式）
6. `app/resources/prompts/image_summary.prompt` — 图片摘要（已关闭 VL）
