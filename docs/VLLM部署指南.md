# vLLM 部署指南（Qwen3-4B-QLoRA-BYD-v2）

> 你本地 `Windows + RTX 5070 Ti` 用 `transformers` 直跑最稳，**上 Linux 服务器再切 `vLLM`**，接口与 `serve_openai.py @8100` 完全兼容，`auto-carcrm` 改一行 `.env` 就能切。

---

## 一、一句话定位

- **transformers**：学原理、本地验证、单并发，开箱即用
- **vLLM**：`PagedAttention` + 连续批处理，高并发低延迟，生产用

---

## 二、环境要求（坑最多的地方）

| 项 | 要求 | 踩坑记录 |
|---|---|---|
| OS | **Linux**（Ubuntu 22.04 推荐） | Windows 原生不支持 vLLM，本地别装；用 **WSL2 Ubuntu** 或 **AutoDL/矩池云 4090/A10** |
| Python | 3.10-3.12 | 3.13 还不稳 |
| CUDA | 12.1+ / 12.4 推荐，驱动 `>=535` | `nvidia-smi` 看驱动，`nvcc --version` 看 CUDA，**驱动和 CUDA 版本要匹配**，不匹配直接 `CUDA error` |
| 显存 | Qwen3-4B FP16 `~10GB`，`gpu_memory_utilization 0.9` 时 12GB 刚好 | 4B 单卡 4090/A10 可跑，70B 要多卡/A100 |
| 模型 | 已合并的 `model/Qwen3-4B-QLoRA-BYD-v2-merged`（同 `serve_openai.py` 的 `MODEL_PATH`） | 不要指到 `finetuned/adapter`，vLLM 要 **merged** 权重 |

---

## 三、安装（Linux 服务器上）

```bash
# 1. 建 env
conda create -n vllm python=3.11 -y && conda activate vllm

# 2. 装 vLLM（CUDA 12.1 轮子，AutoDL 预装 CUDA 12.4 也能用）
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121

# 验证
python -c "import vllm; print(vllm.__version__)"
nvidia-smi
```

**坑：**
- `pip install vllm` 在 Windows 下会 `No matching distribution` → 别在 Windows 装
- AutoDL 已有 `CUDA`，直接 `pip install vllm` 即可，不用重装 CUDA
- 3286 端口被占就换端口

---

## 四、启动（OpenAI 兼容，最常用）

```bash
# 模型已在服务器 /opt/models/Qwen3-4B-QLoRA-BYD-v2-merged
vllm serve /opt/models/Qwen3-4B-QLoRA-BYD-v2-merged \
  --served-model-name qwen3-crm \
  --host 0.0.0.0 --port 8100 \
  --dtype auto \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096
```

**参数说明：**
- `--served-model-name qwen3-crm`：必须和 `auto-carcrm/.env` 的 `LLM_DEFAULT_MODEL` 一致
- `--max-model-len 4096`：你微调 `MAX_LENGTH=512`，但 RAG 的 `prompt_builder` 会拼到 4000+，设小了会 `Input too long`
- `--gpu-memory-utilization 0.9`：12GB 卡设 0.9，24GB 设 0.95

**验证：**
```bash
curl http://localhost:8100/v1/models
curl http://localhost:8100/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3-crm","messages":[{"role":"user","content":"故障码P0562是什么？"}],"max_tokens":200}'
# 期望：无 <think> 泄露，200字内干净回答
```

**Docker 版（生产推荐）：**
```dockerfile
# Dockerfile.vllm
FROM vllm/vllm-openai:latest
COPY model/Qwen3-4B-QLoRA-BYD-v2-merged /model
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", "/model", "--served-model-name", "qwen3-crm", "--host", "0.0.0.0", "--port", "8100"]
```
```bash
docker build -f Dockerfile.vllm -t crm-vllm:latest .
docker run -d --gpus all -p 8100:8100 crm-vllm:latest
```

---

## 五、RAG 一键切换

```env
# auto-carcrm/.env  本地 transformers
OPENAI_BASE_URL=http://localhost:8100/v1
LLM_DEFAULT_MODEL=qwen3-crm

# 切到 vLLM（同一接口，不用改代码）
OPENAI_BASE_URL=http://<vllm-server>:8100/v1
LLM_DEFAULT_MODEL=qwen3-crm
OPENAI_API_KEY=not-needed
```

`Hybrid 8101` 也能切：`FUSION_API=http://<vllm-server>:8100/v1/chat/completions`，`FUSION_MODEL=qwen3-crm`，就是把商汤的 `glm-5.2` 换成本地 vLLM 融合。

---

## 六、踩坑清单（直接抄到面试）

1. **Windows 装 vLLM 失败**：`No matching distribution` → 用 WSL2 或 AutoDL Linux
2. **CUDA 不匹配**：`CUDA error: no kernel image` → `nvidia-smi` 驱动 535+，`pip` 用 `cu121` 轮子
3. **显存 OOM**：`CUDA out of memory` → 降 `gpu_memory_utilization 0.9→0.85` 或降 `max-model-len 4096→2048`
4. **模型路径错**：`Model not found` → 指到 **merged** 目录，不是 `finetuned/QLoRA` adapter
5. **served-model-name 对不上**：RAG 报 `model not found` → `vllm serve --served-model-name` 必须等于 `LLM_DEFAULT_MODEL`
6. **max-model-len 太小**：`Input too long` → RAG 的 prompt 要 4000，设 4096
7. **thinking 泄露**：Qwen3 默认带 `<think>` → `serve_openai.py` 已加多层 `re.sub` 清洗，vLLM 侧加 `--enable-auto-tool-choice` 无关，靠后处理
8. **端口冲突**：`8100` 被 `contract-review` 占 → `docker ps` 查，换 `8102` 或 `docker-compose`改宿主端口（同 Milvus 9000→19000 的修法）
9. **权重没传**：`model/` 被 `.gitignore` → 用 `rsync -avz -P model/Qwen3-4B-QLoRA-BYD-v2-merged/ user@server:/opt/models/...` 或 `ModelScope` 拉

---

## 七、本地 vs 线上对照

| 环境 | 命令 | 用途 |
|------|------|------|
| 本地 Windows | `python serve_openai.py` | 调试，单并发 |
| 本地 Windows | `python serve_hybrid.py` | 演示融合+润色（已加 step-3.7-flash 1.39s） |
| 线上 Linux | `vllm serve ... --port 8100` | 生产，高并发 |

> 面试话术：*“本地用 transformers 验证链路，线上用 vLLM 的 PagedAttention 做高并发，接口都是 OpenAI 兼容，RAG 改一行配置就私有化了。”*
