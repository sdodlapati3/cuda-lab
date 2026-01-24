#!/usr/bin/env python
"""Quick GPU profiling script using PyTorch Profiler."""

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

def main():
    print("=" * 80)
    print("CUDA KERNEL PROFILING ON H100")
    print("=" * 80)
    print()
    
    # Check GPU
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print()
    
    # Create model
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024)
    ).cuda()
    
    x = torch.randn(256, 1024, device="cuda")
    
    # Warmup
    for _ in range(5):
        _ = model(x)
    torch.cuda.synchronize()
    
    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function("model_forward"):
            for _ in range(10):
                y = model(x)
        torch.cuda.synchronize()
    
    # Print results
    print("CUDA KERNEL TIMES (sorted by GPU time)")
    print("-" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    print()
    
    print("MEMORY ALLOCATION")
    print("-" * 80)
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

if __name__ == "__main__":
    main()
