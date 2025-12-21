# Starter 04: PyTorch CUDA Extension

**The Integration Pattern** - Make your CUDA kernels usable in real ML workflows.

## Why This Matters

Writing fast CUDA is only half the job. You need to:
1. **Integrate with autograd** (forward + backward)
2. **Handle types** (FP32, FP16, BF16)
3. **Validate gradients** (numerical checks)
4. **Benchmark properly** (vs PyTorch baseline)

This starter shows the complete pattern for a **fused LayerNorm**.

## Structure

```
pytorch-extension/
├── csrc/
│   ├── layernorm_cuda.cu      # CUDA kernels
│   └── layernorm.cpp          # PyTorch bindings
├── fused_layernorm/
│   ├── __init__.py
│   └── functional.py          # Autograd function
├── setup.py                    # Build script
├── test_correctness.py         # Gradient checks
└── test_benchmark.py           # Performance comparison
```

## Quick Start

```bash
# Build and install
pip install -e .

# Run tests
python test_correctness.py
python test_benchmark.py
```

## Usage

```python
import torch
from fused_layernorm import fused_layer_norm

# Forward pass
x = torch.randn(32, 1024, device='cuda', requires_grad=True)
weight = torch.ones(1024, device='cuda', requires_grad=True)
bias = torch.zeros(1024, device='cuda', requires_grad=True)

output = fused_layer_norm(x, weight, bias, eps=1e-5)

# Backward pass works automatically!
loss = output.sum()
loss.backward()

print(x.grad.shape)  # Gradients flow correctly
```

## What This Teaches

| File | Concept |
|------|---------|
| `layernorm_cuda.cu` | CUDA kernel for LN forward/backward |
| `layernorm.cpp` | `pybind11` bindings, input checks |
| `functional.py` | `torch.autograd.Function` with `save_for_backward` |
| `test_correctness.py` | `torch.autograd.gradcheck` |
| `test_benchmark.py` | CUDA events for timing |

## Key Patterns

### 1. Input Checking (C++)
```cpp
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
```

### 2. Autograd Integration (Python)
```python
class FusedLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, eps):
        # Save for backward
        ctx.save_for_backward(input, weight, bias)
        ctx.eps = eps
        return fused_layernorm_cuda.forward(input, weight, bias, eps)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        return fused_layernorm_cuda.backward(grad_output, input, weight, ctx.eps)
```

### 3. Gradient Checking
```python
def test_gradients():
    x = torch.randn(8, 32, device='cuda', dtype=torch.float64, requires_grad=True)
    w = torch.randn(32, device='cuda', dtype=torch.float64, requires_grad=True)
    b = torch.randn(32, device='cuda', dtype=torch.float64, requires_grad=True)
    
    assert torch.autograd.gradcheck(
        lambda x, w, b: fused_layer_norm(x, w, b, 1e-5),
        (x, w, b),
        eps=1e-6,
        atol=1e-4
    )
```

## Exercises

After understanding this:

1. **Add FP16 support**
   - Use `AT_DISPATCH_FLOATING_TYPES_AND_HALF`
   - Handle accumulation in FP32

2. **Implement RMSNorm**
   - Simpler than LayerNorm (no mean subtraction)
   - Popular in LLaMA, Mistral

3. **Fuse with dropout**
   - Add dropout after normalization
   - Single kernel read/write

4. **Add gradient accumulation**
   - For `grad_weight` and `grad_bias` across batch

5. **Profile and optimize**
   - Use Nsight Compute to find bottlenecks
   - Compare with Apex FusedLayerNorm
