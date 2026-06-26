"""
API 部署脚本：把微调后的模型部署成 FastAPI 服务

运行：python serve_api.py
访问：http://localhost:8000/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "./model/Qwen3-4B-merged"
SYSTEM_PROMPT = "你是比亚迪商用车智能售后助手，精通车辆故障诊断、维修方案、质保政策和保养知识。请用专业、简洁、有条理的方式回答用户问题。"

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
print("模型加载完成！")

# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="CRM 智能售后助手 API",
    description="基于 Qwen3-8B 微调的汽车售后问答服务",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    question: str
    max_tokens: int = 256
    temperature: float = 0.7


class ChatResponse(BaseModel):
    answer: str
    model: str


@app.get("/")
def root():
    return {"message": "CRM 智能售后助手 API", "status": "running"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """单轮问答接口"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": req.question}
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
            max_new_tokens=req.max_tokens,
            do_sample=True,
            temperature=req.temperature,
            top_p=0.9
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return ChatResponse(answer=answer, model=MODEL_PATH)


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    print("\n启动 API 服务...")
    print("访问 http://localhost:8000/docs 查看文档")
    uvicorn.run(app, host="0.0.0.0", port=8000)
