import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

# 测试 GPU 计算
try:
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print("GPU 计算测试:", "OK" if y.shape == (1000, 1000) else "FAILED")
    print("显存占用:", f"{torch.cuda.memory_allocated()/1024**3:.2f} GB")
except Exception as e:
    print("GPU 计算失败:", e)
