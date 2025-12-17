#!/usr/bin/env python3
"""Verify CUDA setup is working on T4 GPU"""

import numpy as np
from numba import cuda
import math

print("=" * 50)
print("CUDA SETUP VERIFICATION")
print("=" * 50)

# Check CUDA
if not cuda.is_available():
    print("❌ CUDA is NOT available!")
    print("   Make sure you're on a GPU node:")
    print("   srun --partition=t4flex --gres=gpu:1 --pty bash")
    exit(1)

print("✅ CUDA is available")

# Get device info
device = cuda.get_current_device()
print(f"✅ GPU: {device.name.decode()}")
cc = device.compute_capability
print(f"   Compute Capability: {cc[0]}.{cc[1]}")
print(f"   SMs: {device.MULTIPROCESSOR_COUNT}")
print(f"   Max threads/block: {device.MAX_THREADS_PER_BLOCK}")

# Test a simple kernel
@cuda.jit
def test_kernel(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] = idx * 2

# Run test
n = 1000
arr = cuda.device_array(n, dtype=np.float32)
threads = 256
blocks = math.ceil(n / threads)

test_kernel[blocks, threads](arr)
result = arr.copy_to_host()

if result[10] == 20.0 and result[100] == 200.0:
    print("✅ Kernel execution successful")
else:
    print("❌ Kernel execution failed")
    exit(1)

# Memory test
ctx = cuda.current_context()
free, total = ctx.get_memory_info()
print(f"✅ GPU Memory: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

# Performance test
import time

@cuda.jit
def vector_add(a, b, c):
    idx = cuda.grid(1)
    if idx < c.size:
        c[idx] = a[idx] + b[idx]

n = 10_000_000
a = cuda.to_device(np.random.randn(n).astype(np.float32))
b = cuda.to_device(np.random.randn(n).astype(np.float32))
c = cuda.device_array(n, dtype=np.float32)

# Warmup
vector_add[math.ceil(n/256), 256](a, b, c)
cuda.synchronize()

# Benchmark
start = time.perf_counter()
for _ in range(100):
    vector_add[math.ceil(n/256), 256](a, b, c)
cuda.synchronize()
elapsed = (time.perf_counter() - start) / 100

bandwidth = (3 * n * 4) / elapsed / 1e9  # Read a, b, write c
print(f"✅ Vector add bandwidth: {bandwidth:.1f} GB/s")

print("=" * 50)
print("🚀 T4 GPU ready for CUDA learning!")
print("=" * 50)
print()
print("Next steps:")
print("  cd ~/cuda-lab/learning-path/week-01")
print("  jupyter notebook day-1-gpu-basics.ipynb")
