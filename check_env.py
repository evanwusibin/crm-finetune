import sys
print("Python:", sys.version)
print()

import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("显存总量:", f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
print("显存占用:", f"{torch.cuda.memory_allocated()/1024**3:.2f} GB")
print()

import transformers
print("transformers:", transformers.__version__)

import peft
print("peft:", peft.__version__)

import trl
print("trl:", trl.__version__)

import bitsandbytes
print("bitsandbytes:", bitsandbytes.__version__)

import datasets
print("datasets:", datasets.__version__)

import accelerate
print("accelerate:", accelerate.__version__)

print()
print("所有依赖 OK")
