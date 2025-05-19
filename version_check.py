# check_env.py
import sys, subprocess, numpy as np, torch
try:
    import torchvision
    tv = torchvision.__version__
except ImportError:
    tv = 'not installed'
try:
    import voxelmorph as vxm
    vm = getattr(vxm, '__version__', '(no __version__ attr)')
except ImportError:
    vm = 'not installed'

print(f"Python       : {sys.version.split()[0]}")
print(f"NumPy        : {np.__version__}")
print(f"PyTorch      : {torch.__version__}")
print(f"  ‣ built with CUDA: {torch.version.cuda}")
print(f"  ‣ cuDNN version : {torch.backends.cudnn.version()}")
print(f"TorchVision  : {tv}")
print(f"VoxelMorph   : {vm}")
print(f"GPU count    : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"  ‣ CUDA capability: {torch.cuda.get_device_capability(i)}")
print(f"  ‣ Memory Allocated: {torch.cuda.memory_allocated(i)}")