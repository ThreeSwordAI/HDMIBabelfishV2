import torch

print("="*50)
print("JETSON ORIN NX – CUDA AVAILABILITY TEST")
print("="*50)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print("✅ CUDA is working correctly!")
else:
    print("❌ CUDA is NOT available. PyTorch is using CPU.")
