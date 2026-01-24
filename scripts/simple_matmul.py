#!/usr/bin/env python
"""Simple matmul for Nsight Compute profiling."""
import torch

# Create matrices
A = torch.randn(2048, 2048, device='cuda')
B = torch.randn(2048, 2048, device='cuda')
torch.cuda.synchronize()

# Profile this operation
C = torch.mm(A, B)
torch.cuda.synchronize()

print(f"Matrix multiply: {A.shape} x {B.shape} = {C.shape}")
print("Done")
