# 小插曲：QLoRA 训练异常慢（每步 5 分钟）

> **日期**：2026-06-25
> **现象**：QLoRA 训练每步 317 秒，1000 步需要 88 小时
> **根因**：bitsandbytes 4bit 量化不支持 RTX 5070 Ti（sm_120），实际用 CPU 跑
> **解决**：去掉量化，改用全精度 BF16 + LoRA

---

## 一、发生了什么

| 项目 | 预期 | 实际 |
|---|---|---|
| 每步时间 | 5-15 秒 | 317 秒 |
| 1000 步总时间 | 1.5-4 小时 | 88 小时 |
| GPU 利用率 | >80% | 接近 0% |
| 显存占用 | ~6GB | 很少 |

---

## 二、为什么会出现这种情况

### 根因链条

```
RTX 5070 Ti（sm_120，Blackwell 架构）
    ↓
PyTorch 2.12.1 支持 sm_50~sm_90，不支持 sm_120
    ↓
bitsandbytes 的 CUDA kernel 无法在 sm_120 上运行
    ↓
4bit 量化的计算 fallback 到 CPU
    ↓
CPU 算 8B 模型 → 每步 5 分钟
```

### 关键警告（被忽略了）

```
UserWarning: NVIDIA GeForce RTX 5070 Ti Laptop GPU with CUDA capability sm_120
is not compatible with the current PyTorch installation.
```

这个警告不是"可能有问题"，而是**一定有问题**——量化计算会 fallback 到 CPU。

### 为什么之前 GPU 测试通过了？

之前的测试：
```python
x = torch.randn(1000, 1000).cuda()
y = torch.matmul(x, x)  # OK
```

这个测试只验证了**基本矩阵运算**能用 GPU，没有验证 **bitsandbytes 的量化 kernel** 能用 GPU。

---

## 三、怎么解决的

| 改动 | 原来（QLoRA） | 现在（LoRA） |
|---|---|---|
| 量化 | 4bit NF4 | 不量化，全精度 BF16 |
| 显存占用 | ~6GB | ~16GB |
| 学习率 | 3e-4 | 2e-5 |
| 每步时间 | 317 秒 | 预期 5-15 秒 |

**为什么可以不量化**：
- Qwen3-8B BF16 约 16GB，12GB 卡可能装不下
- 但 LoRA 冻结了大部分参数，实际显存需求约 10-12GB
- 通过减小 batch_size（2）和序列长度（256）可以塞进去

---

## 四、后续怎么杜绝

### 4.1 训练前必做检查清单

```markdown
## 微调前环境检查清单

- [ ] GPU 架构兼容性：nvidia-smi 看 Compute Capability，确认 PyTorch 支持
- [ ] 量化库兼容性：bitsandbytes 是否支持当前 GPU 架构
- [ ] 显存预估：模型大小 × 精度系数 ≤ 可用显存
- [ ] 小规模测试：先跑 5 步，确认速度正常
- [ ] GPU 利用率：nvidia-smi 确认 GPU 在工作，不是 CPU fallback
```

### 4.2 快速验证脚本

```python
# test_gpu_training.py - 训练前跑这个
import torch
import time

# 1. 基本 CUDA 测试
x = torch.randn(1000, 1000).cuda()
y = torch.matmul(x, x)
print("基本 CUDA 测试: OK")

# 2. bitsandbytes 量化测试
try:
    import bitsandbytes as bnb
    linear = bnb.nn.Linear4bit(1000, 1000).cuda()
    x = torch.randn(1, 1000).cuda()
    y = linear(x)
    print("bitsandbytes 4bit 测试: OK")
except Exception as e:
    print(f"bitsandbytes 4bit 测试: FAILED - {e}")
    print("建议：使用全精度 LoRA，不用 QLoRA")

# 3. 模型加载速度测试
from transformers import AutoModelForCausalLM
start = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "./model/Qwen3-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print(f"模型加载时间: {time.time()-start:.1f}s")
print(f"显存占用: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
```

### 4.3 训练时监控

```bash
# 另开一个终端，监控 GPU 使用
nvidia-smi -l 1

# 正常情况：GPU 利用率 >80%，显存占用 >8GB
# 异常情况：GPU 利用率 <10%，显存占用很少 → 可能在用 CPU
```

---

## 五、企业中应该怎么做

### 5.1 GPU 选型阶段

| 步骤 | 做什么 | 工具 |
|---|---|---|
| 1 | 确认模型大小和精度需求 | 显存计算器 |
| 2 | 确认 GPU 架构兼容性 | PyTorch 官方文档 |
| 3 | 确认量化库支持 | bitsandbytes GitHub |
| 4 | 做 benchmark | 小规模测试 |

### 5.2 企业常见 GPU 兼容性表

| GPU | 架构 | Compute Capability | PyTorch 支持 | bitsandbytes 支持 |
|---|---|---|---|---|
| V100 | Volta | sm_70 | ✅ | ✅ |
| A100 | Ampere | sm_80 | ✅ | ✅ |
| A10 | Ampere | sm_86 | ✅ | ✅ |
| RTX 3090 | Ampere | sm_86 | ✅ | ✅ |
| RTX 4090 | Ada | sm_89 | ✅ | ✅ |
| H100 | Hopper | sm_90 | ✅ | ✅ |
| RTX 5070 Ti | Blackwell | sm_120 | ⚠️ 部分 | ❌ 不支持 |
| RTX 5090 | Blackwell | sm_120 | ⚠️ 部分 | ❌ 不支持 |

### 5.3 企业标准流程

```markdown
## 微调项目标准流程

### Phase 0：环境验证（30 分钟）
1. 确认 GPU 型号和架构
2. 运行 GPU 兼容性测试脚本
3. 确认 PyTorch + bitsandbytes 版本
4. 记录测试结果

### Phase 1：小规模验证（1 小时）
1. 用 10 条数据跑 5 步
2. 确认每步时间 < 30 秒
3. 确认 GPU 利用率 > 80%
4. 确认显存占用合理

### Phase 2：正式训练
1. 全量数据训练
2. 监控 GPU 利用率和显存
3. 监控 loss 曲线
4. 定期保存检查点
```

---

## 六、经验总结

| 教训 | 做法 |
|---|---|
| 警告不能忽略 | 看到不兼容警告必须验证，不能假设"应该没问题" |
| 先小规模测试 | 1000 步之前先跑 5 步，确认速度正常 |
| 监控 GPU 利用率 | 训练时开 nvidia-smi，确认 GPU 在工作 |
| 量化不是万能的 | 新架构 GPU 可能不支持量化，要有 fallback 方案 |
| 知识要落地 | 知道"QLoRA 省显存"不够，还要知道"哪些 GPU 不支持" |
