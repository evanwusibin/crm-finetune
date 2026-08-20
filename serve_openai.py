"""
OpenAI 兼容 API 服务器

用途：部署微调模型，暴露 /v1/chat/completions 接口
      RAG 项目改 .env 即可无缝切换

运行：python serve_openai.py
测试：curl http://localhost:8100/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"qwen3-crm","messages":[{"role":"user","content":"故障码P0562是什么意思？"}]}'
"""

import json
import time
import uuid
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from typing import Optional
import uvicorn

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "./model/Qwen3-4B-QLoRA-BYD-v2-merged"  # v2 清洗数据微调模型
HOST = "0.0.0.0"
PORT = 8100
MODEL_NAME = "qwen3-crm"

# ============================================================
# 加载模型
# ============================================================
print("=" * 50, flush=True)
print("OpenAI 兼容 API 服务器", flush=True)
print("=" * 50, flush=True)
print(f"\n模型：{MODEL_PATH}", flush=True)
print(f"服务地址：http://{HOST}:{PORT}", flush=True)
print(f"模型名称：{MODEL_NAME}", flush=True)

print("\n加载模型...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)
print("模型加载完成！\n", flush=True)

# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="CRM 微调模型 API",
    description="OpenAI 兼容的本地模型服务",
    version="1.0.0"
)


# ============================================================
# 请求/响应模型（OpenAI 格式）
# ============================================================
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


# ============================================================
# 核心推理函数
# ============================================================
import re

def generate_response(messages: list[dict], max_tokens: int, temperature: float, top_p: float):
    """非流式生成"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
        )

    generated_ids = outputs[0][input_len:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    # 去除 Qwen3 thinking 标签（多层嵌套）
    for _ in range(3):
        cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
        if cleaned == response_text:
            break
        response_text = cleaned
    response_text = response_text.strip()
    output_tokens = len(generated_ids)

    return response_text, input_len, output_tokens


def generate_stream(messages: list[dict], max_tokens: int, temperature: float, top_p: float):
    """流式生成（SSE）"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else 1.0,
        "top_p": top_p,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    total_tokens = 0

    for new_text in streamer:
        if not new_text:
            continue
        total_tokens += 1
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "delta": {"content": new_text},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    # 发送结束标记
    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

    thread.join()


# ============================================================
# API 端点
# ============================================================
@app.get("/v1/models")
def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local"
        }]
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """OpenAI 兼容的聊天补全接口"""
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    # 流式返回
    if req.stream:
        return StreamingResponse(
            generate_stream(messages, req.max_tokens, req.temperature, req.top_p),
            media_type="text/event-stream"
        )

    # 非流式返回
    response_text, prompt_tokens, completion_tokens = generate_response(
        messages, req.max_tokens, req.temperature, req.top_p
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=MODEL_NAME,
        choices=[Choice(
            index=0,
            message=ChoiceMessage(content=response_text),
            finish_reason="stop"
        )],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )


@app.get("/")
def root():
    return {
        "message": "CRM 微调模型 OpenAI 兼容 API",
        "model": MODEL_NAME,
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models"
        }
    }


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    print(f"启动 API 服务：http://{HOST}:{PORT}", flush=True)
    print(f"OpenAI 兼容端点：http://{HOST}:{PORT}/v1/chat/completions", flush=True)
    uvicorn.run(app, host=HOST, port=PORT)
