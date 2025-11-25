import torch
"""
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 - cmd to install torch with CUDA
"""
print(torch.version.cuda)       # version of  CUDA, that was build with  PyTorch
print(torch.backends.cudnn.version())  # cuDNN
print(torch.cuda.is_available()) # True/False
print(torch.cuda.device_count()) # amount of exists GPU
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")