import torch
import torchvision
import torchaudio

print("="*50)
print("JETSON ORIN NX CUDA TEST")
print("="*50)

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("Torchaudio version:", torchaudio.__version__)

# CUDA Tests
print("\nCUDA Information:")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Memory info
    print(f"\nGPU Memory:")
    print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
    
    # Test computation
    print("\nTesting CUDA computation...")
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"Matrix multiplication successful!")
    print(f"Result shape: {z.shape}, Device: {z.device}")
    
    # Performance test
    import time
    start = time.time()
    for _ in range(100):
        _ = torch.matmul(x, y)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"100 matrix multiplications took: {elapsed:.3f} seconds")
else:
    print("CUDA is NOT available!")

print("\n✅ Setup complete! Jetson is in MAXN mode with CUDA enabled.")