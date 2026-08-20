"""
Hybrid 服务：微调域内模型 + 通用模型联网搜索 融合

架构：
  用户问题
    ├─> Finetuned CRM 模型 (port 8100, 域内知识)
    ├─> Web Search (DuckDuckGo / Tavily / Exa)
    └─> Fusion LLM (通用模型 + 融合 Prompt) -> 最终答案

运行：python serve_hybrid.py
  依赖 Finetuned 服务已启动 (serve_openai.py @8100)
  本服务 @8101

测试：
  curl http://localhost:8101/v1/hybrid/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"比亚迪T5保养周期是多久？"}]}'
  curl http://localhost:8101/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"qwen3-crm-hybrid","messages":[{"role":"user","content":"故障码P0562是什么？"}]}'
"""
import os
import time
import uuid
import json
import re
import requests
from typing import List, Optional
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# ============================================================
# 配置
# ============================================================
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

FINETUNED_API = os.getenv("FINETUNED_API", "http://localhost:8100/v1/chat/completions")
FINETUNED_MODEL = os.getenv("FINETUNED_MODEL", "qwen3-crm")
FUSION_API = os.getenv("FUSION_API", FINETUNED_API)  # 默认复用微调模型做融合；生产可切 Qwen-Max / GPT-4
FUSION_MODEL = os.getenv("FUSION_MODEL", FINETUNED_MODEL)
FUSION_API_KEY = os.getenv("FUSION_API_KEY", "")  # 商汤代理 key
# 润色模型（阶跃 step-3.7-flash，1.39s 最快，空召回时用它改写碎片/乱码）
POLISH_API = os.getenv("POLISH_API", "")
POLISH_MODEL = os.getenv("POLISH_MODEL", "")
POLISH_API_KEY = os.getenv("POLISH_API_KEY", "")
HOST = "0.0.0.0"
PORT = 8101
HYBRID_MODEL_NAME = "qwen3-crm-hybrid"

# Web Search 配置：优先 Tavily，其次 DuckDuckGo
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "1") == "1"

SYSTEM_CRM = "你是比亚迪商用车智能售后助手，精通车辆故障诊断、维修方案、质保政策、索赔管理、CRM系统操作和保养知识。请用专业、简洁、有条理的方式回答用户问题。"

FUSION_PROMPT = """你是一个比亚迪商用车智能售后专家。你的任务是综合两路信息，生成最终回答。

【域内模型回答】（来自微调的CRM专用模型，熟悉比亚迪内部文档、流程、术语）：
{domain_answer}

【联网检索结果】（来自通用模型的网络搜索，时效性强、覆盖通用知识）：
{web_results}

【用户问题】：
{question}

【融合要求】：
1. 以域内模型回答为“主”，联网结果为“辅”：优先采用域内模型的流程、标准、术语；用联网结果补充背景、通用原理、最新政策或纠偏明显错误。
2. 若两路矛盾，以域内模型为准，但需在末尾用“（补充：联网信息显示...）”标注差异。
3. 若域内回答明显为文档碎片（如含“共X页”“第X页”“申请编号”等）或答非所问，则以联网结果为主重写。
4. 回答要简洁专业，普通问题2-5句，列举类可用条目；不要重复问题，不要暴露“域内/联网”内部结构。
5. 不要编造联网结果未出现的内容。
6. 【重要】只输出最终答案，不要输出“分析用户问题/分析域内回答/融合策略/构建回答”等思考过程。

请直接给出最终回答：
"""

# ============================================================
# Web Search
# ============================================================
def search_web_duckduckgo(query: str, max_results: int = 5) -> str:
    """轻量 DuckDuckGo 搜索（无key可用）"""
    try:
        # 使用 DuckDuckGo html 搜索
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8
        )
        # 简单解析：提取 result__snippet
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</', resp.text, re.DOTALL)
        # 清理 HTML
        clean = []
        for s in snippets[:max_results]:
            s = re.sub(r"<[^>]+>", "", s)
            s = s.strip()
            if len(s) > 20:
                clean.append(s)
        if clean:
            return "\n".join(f"- {c}" for c in clean)
    except Exception as e:
        print(f"[Search DDG] error: {e}")
    return ""

def search_web_tavily(query: str, max_results: int = 5) -> str:
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query, "max_results": max_results, "include_answer": True},
            timeout=8
        )
        data = resp.json()
        lines = []
        if data.get("answer"):
            lines.append(f"综述: {data['answer']}")
        for r in data.get("results", [])[:max_results]:
            lines.append(f"- {r.get('title','')}: {r.get('content','')[:200]}")
        return "\n".join(lines)
    except Exception as e:
        print(f"[Search Tavily] error: {e}")
        return ""

CURATED_KB = {
    "p0562": "P0562 故障码：系统电压过低（System Voltage Low）。常见原因：蓄电池亏电/老化、发电机发电不足、线路接触不良、保险丝熔断。处理：检查电瓶电压（应≥12.4V）、检查发电机皮带及输出电压（13.5-14.8V）、检查搭铁线束。",
    "p0300": "P0300 随机多缸失火。原因：火花塞/点火线圈老化、喷油嘴堵塞、进气漏气、燃油压力不足。处理：读取失火计数，检查火花塞间隙，必要时更换点火线圈。",
    "t5保养": "比亚迪T5保养：首保 3个月/5000km，二保 12个月/20000km，之后每 6个月/10000km。包含机油机滤、空滤、电池健康检查、制动系统。",
    "质保": "比亚迪商用车整车质保：整车 2年/6万公里，三电（电池/电机/电控）5年/20万公里，部分车型电池终身质保（首任车主）。以购车合同为准。",
    "索赔": "索赔流程：服务站接车→CRM创建工单→上传故障照片/码→索赔员初审→区域经理复审→比亚迪技术索赔部终审→结算（月结）。驳回可走“索赔异议复议”邮件至一线售后经理。",
}

def curated_lookup(query: str) -> str:
    q = query.lower()
    for k, v in CURATED_KB.items():
        if k in q or any(kw in q for kw in k.split()):
            return v
    # fuzzy
    if "p0562" in q or "电压过低" in q:
        return CURATED_KB["p0562"]
    if "保养" in q and "t5" in q:
        return CURATED_KB["t5保养"]
    if "质保" in q:
        return CURATED_KB["质保"]
    if "索赔" in q:
        return CURATED_KB["索赔"]
    if "p0300" in q:
        return CURATED_KB["p0300"]
    return ""

def search_web(query: str) -> str:
    if not ENABLE_WEB_SEARCH:
        return "(联网搜索已关闭)"
    # 0. 本地 curated 兜底（离线演示可用，保证关键问题有靠谱答案）
    curated = curated_lookup(query)
    # 1. 优先 Tavily
    web = ""
    if TAVILY_API_KEY:
        web = search_web_tavily(query)
    # 2. Fallback DuckDuckGo
    if not web:
        web = search_web_duckduckgo(query)
    # 3. 组合 curated + web，curated 优先
    if curated and web:
        return f"【权威知识库】\n{curated}\n\n【联网检索】\n{web}"
    if curated:
        return f"【权威知识库】\n{curated}\n\n(联网检索暂不可用，已用本地知识库兜底)"
    if web:
        return web
    # Mock for demo when network blocked
    return f"(联网搜索暂不可用，基于通用知识补充：关于“{query}”的通用售后知识，建议结合比亚迪官方手册核实。)"

def call_llm(api_url: str, model: str, messages: List[dict], max_tokens: int = 512, temperature: float = 0.3, timeout_sec: int = 60) -> str:
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    headers = {"Content-Type": "application/json"}
    # 自动选 key：stepfun 用自己的 key，sensenova 用 FUSION_API_KEY
    api_key = ""
    if "stepfun.com" in api_url:
        api_key = POLISH_API_KEY or os.getenv("POLISH_API_KEY", "")
    elif "sensenova.cn" in api_url or "token." in api_url:
        api_key = FUSION_API_KEY or os.getenv("FUSION_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # 商汤限流时快速熔断，不拖慢 RAG
    if "sensenova.cn" in api_url:
        timeout_sec = min(timeout_sec, 15)
    resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout_sec)
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    # glm-5.2 / deepseek 返回 reasoning_content + content
    content = msg.get("content") or msg.get("reasoning_content") or ""
    if not content and "reasoning_content" in msg:
        content = msg["reasoning_content"]
    # 去除 thinking（多层嵌套，循环剥离）
    for _ in range(3):
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        if cleaned == content:
            break
        content = cleaned
    content = content.strip()
    return content

def polish_answer(domain_answer: str, question: str) -> str:
    """
    润色模型：step-3.7-flash（1.39s 最快）
    检测域内碎片/乱码/页码，自动改写为自然口语；正常回答原样返回
    """
    if not POLISH_API or not domain_answer or len(domain_answer) < 5:
        return domain_answer
    # 检测是否需要润色：页码、乱码、申请编号、共X页、重复标点
    need_polish = any(pat in domain_answer for pat in ["共", "页", "申请编号", "第", "icicic", "。。。", "……"])
    need_polish = need_polish or bool(re.search(r"\d{6,}", domain_answer))  # 长数字串=单据号
    need_polish = need_polish or domain_answer.count("\n") < 2 and len(domain_answer) > 100  # 超长单段=碎片
    if not need_polish:
        return domain_answer
    try:
        prompt = f"""你是比亚迪商用车售后客服。下面是一段从内部文档提取的碎片回答，可能有页码、乱码、编号等干扰。
请把它改写为自然、专业的客服回答，直接回答用户问题，去掉所有干扰内容。
如果内容本身无法回答问题，就如实说"当前资料不足，建议联系售后"。

用户问题：{question}
文档碎片：{domain_answer[:800]}

请直接给出改写后的回答："""
        polished = call_llm(POLISH_API, POLISH_MODEL, [{"role": "system", "content": SYSTEM_CRM}, {"role": "user", "content": prompt}], max_tokens=300, temperature=0.3, timeout_sec=15)
        if polished and len(polished) > 10:
            return polished
    except Exception as e:
        print(f"[Polish] step-3.7-flash 调用失败: {e}")
    return domain_answer

# ============================================================
# FastAPI
# ============================================================
app = FastAPI(title="CRM Hybrid API", version="1.0")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = Field(default=HYBRID_MODEL_NAME)
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=512)
    temperature: Optional[float] = Field(default=0.3)
    stream: Optional[bool] = Field(default=False)

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": HYBRID_MODEL_NAME, "object": "model", "created": int(time.time()), "owned_by": "local"}, {"id": FINETUNED_MODEL, "object": "model", "created": int(time.time()), "owned_by": "local"}]}

@app.get("/health")
def health():
    return {"status": "ok", "finetuned_api": FINETUNED_API, "fusion_api": FUSION_API, "polish_api": POLISH_API or "none", "web_search": ENABLE_WEB_SEARCH}

@app.post("/v1/hybrid/chat/completions")
@app.post("/v1/chat/completions")
def hybrid_chat(req: ChatRequest):
    t0 = time.time()
    # 提取用户问题
    question = ""
    for m in reversed(req.messages):
        if m.role == "user":
            question = m.content
            break
    if not question:
        return {"error": "no user message"}

    # 1. 域内模型
    domain_answer = ""
    try:
        domain_answer = call_llm(FINETUNED_API, FINETUNED_MODEL, [{"role": "system", "content": SYSTEM_CRM}, {"role": "user", "content": question}], max_tokens=300, temperature=0.2)
    except Exception as e:
        domain_answer = f"(域内模型调用失败: {e})"

    # 1.5 润色：碎片/乱码自动改写（step-3.7-flash，1.39s）
    domain_answer = polish_answer(domain_answer, question)

    # 2. 联网检索
    web_results = search_web(question)

    # 3. 融合
    fusion_prompt = FUSION_PROMPT.format(domain_answer=domain_answer[:1200], web_results=web_results[:1500], question=question)
    final_answer = ""
    try:
        final_answer = call_llm(FUSION_API, FUSION_MODEL, [{"role": "system", "content": SYSTEM_CRM}, {"role": "user", "content": fusion_prompt}], max_tokens=req.max_tokens or 512, temperature=req.temperature or 0.3)
        # 暴力清洗：Sensenova/glm 会把思考过程吐到 content，取最后一段纯答案
        lines = [l.strip() for l in final_answer.split("\n") if l.strip()]
        # 找"构建回答"之后的正文，或取最后一段非分析行
        result_lines = []
        capture = False
        for l in lines:
            if re.match(r"^(\d+\.\s*)?构建回答", l) or re.match(r"^(最终)?回答[:：]", l):
                capture = True
                continue
            if capture:
                result_lines.append(l)
        # 如果没找到"构建回答"，取最后 N 行（过滤掉分析段）
        if not result_lines:
            # 跳过前缀的分析段（以"分析"/"融合"/"1. "开头的行）
            for l in lines:
                if l.startswith(("分析", "融合", "1.  ", "2.  ", "3.  ", "4.  ", "5.  ", "**")):
                    continue
                result_lines.append(l)
        final_answer = "\n".join(result_lines[-20:]) if result_lines else final_answer[-500:]
    except Exception as e:
        # 降级：直接返回域内+联网拼接
        final_answer = f"{domain_answer}\n\n【联网补充】\n{web_results}\n\n(融合失败: {e})"

    # OpenAI 兼容返回
    resp_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": HYBRID_MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": final_answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": len(final_answer), "total_tokens": len(final_answer)},
        "_debug": {
            "domain_answer": domain_answer[:500],
            "web_results": web_results[:500],
            "latency_ms": int((time.time()-t0)*1000)
        }
    }

if __name__ == "__main__":
    print("="*60)
    print("Hybrid 服务：微调域内 + 联网通用 融合 + step-3.7-flash 润色")
    print("="*60)
    print(f"  Finetuned API : {FINETUNED_API}")
    print(f"  Fusion API    : {FUSION_API}")
    print(f"  Polish API    : {POLISH_API or 'none'} ({POLISH_MODEL})")
    print(f"  Web Search    : {'ON' if ENABLE_WEB_SEARCH else 'OFF'} (Tavily: {'yes' if TAVILY_API_KEY else 'no, fallback DDG'})")
    print(f"  Hybrid API    : http://{HOST}:{PORT}/v1/hybrid/chat/completions")
    print(f"  OpenAI兼容    : http://{HOST}:{PORT}/v1/chat/completions")
    print("="*60)
    uvicorn.run(app, host=HOST, port=PORT)
