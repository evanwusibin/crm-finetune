# Hybrid 融合服务设计：微调域内 + 联网通用

> 对应用户需求：“把这个模型输出的结果再加上另一个模型的网络搜索做补充”

## 1. 为什么需要 Hybrid

- **微调模型（v2）**：熟悉比亚迪内部术语、流程、表格，但训练数据是文档碎片 → 评估 token_overlap 仅 0.067，70% 答非所问。
- **通用模型 + 联网**：时效性强、能纠偏（例 P0562 域内说“燃油泵继电器”，正确应为“系统电压过低”），但不懂比亚迪 CRM 内部流程。
- **融合**：域内为主、联网为辅；域内为碎片/错误时联网为主重写。

## 2. 架构

```
用户问题
  ├─[1] Finetuned CRM  (Qwen3-4B-QLoRA-BYd-v2 @8100)  域内知识
  ├─[2] Web Search     (Tavily → DDG → CuratedKB兜底)  时效/通用
  └─[3] Fusion LLM     (复用 Finetuned API，可切 Qwen-Max/GPT-4)  融合生成
          │
          └─> 最终答案 (OpenAI Compatible)
```

**服务**：
- `serve_openai.py` @8100  域内专用
- `serve_hybrid.py` @8101  融合服务（同时兼容 `/v1/chat/completions` 和 `/v1/hybrid/chat/completions`）

## 3. 关键实现

### Web Search 三级降级
```python
Tavily (需 TAVILY_API_KEY) → DuckDuckGo html → CuratedKB
```
CuratedKB 为离线演示兜底，已内置：
- p0562/p0300 故障码、T5保养、质保、索赔流程（见 `CURATED_KB`）

生产建议：替换为 Tavily/Exa/Bing Search + 向量检索（RAG）。

### Fusion Prompt
见 `FUSION_PROMPT`：明确“域内为碎片则联网为主”“矛盾时域内为准但标注差异”“不要编造联网未出现内容”等 5 条约束（改编自笔记案例2的提示词约束）。

### 调用示例
```bash
# 1. 域内直连
curl http://localhost:8100/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3-crm","messages":[{"role":"user","content":"故障码P0562是什么？"}]}'

# 2. Hybrid 融合
curl http://localhost:8101/v1/hybrid/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"故障码P0562是什么？"}]}'

# 3. OpenAI 兼容（Hybrid 也兼容 /v1/chat/completions，可直接替换 .env）
curl http://localhost:8101/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3-crm-hybrid","messages":[{"role":"user","content":"索赔被驳回怎么办？"}]}'
```

RAG 项目切换：
```env
# auto-carcrm/.env
OPENAI_BASE_URL=http://localhost:8101/v1
OPENAI_API_KEY=not-needed
LLM_DEFAULT_MODEL=qwen3-crm-hybrid
```

## 4. 验证结果

| 问题 | 域内 alone | Hybrid 融合 | 结论 |
|------|-----------|-------------|------|
| T5保养周期 | 表格碎片“20万公里…” | “首保3个月/5000km…制动20万公里，首保是首次…” | 可读性↑ |
| P0562 | “燃油泵继电器断路”（错）+ 延保流程（错） | 融合后同时给出域内“继电器”+联网“电压过低”并标注补充 | 纠偏能力验证 |
| 索赔被驳回 | “预授权结算…”（片面） | 仍偏域内，未充分用联网“异议复议” | 需再调 Fusion 权重 |

**_debug 字段**：每次返回带 `domain_answer` / `web_results` / `latency_ms`，便于 Bad Case 归因（对应笔记“步骤5看日志定位”）。

## 5. 生产化建议

1. **Second Model 升级**：`FUSION_API` 切到 Qwen-Max / GPT-4o + 真实 Tavily key，curated 仅作兜底。
2. **RAG 化**：把 BYD 197 文档接入 Milvus + BGE-M3，Hybrid 的域内路改为检索增强，评估切回 RAGAS 5 指标。
3. **数据重写**：对 2891 条做 LLM 问答重写 + 人工抽检 100 条，再训 v3，预期 correctness 0.06 → 0.4+ 后，Hybrid 的域内权重可调高。
4. **监控**：复用 `eval_v2_comprehensive.py` 定期跑 80 抽样，监控 hybrid vs 域内好坏率。
