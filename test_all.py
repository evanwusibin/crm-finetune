import torch
import transformers
import peft
import trl
import bitsandbytes
import datasets
print("所有依赖 OK")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
