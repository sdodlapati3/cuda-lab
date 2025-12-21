# PyTorch CUDA Extension Template

Template for building production-quality PyTorch CUDA extensions with forward and backward passes.

## Structure

```
pytorch_extension/
├── csrc/
│   ├── cuda/
│   │   ├── fused_op_kernel.cu    # CUDA kernels
│   │   └── fused_op_kernel.cuh   # CUDA headers
│   ├── fused_op.cpp              # C++ binding
│   └── fused_op.h
├── fused_op/
│   ├── __init__.py               # Python interface
│   └── functional.py             # Functional API
├── setup.py                       # Build script
├── test_correctness.py           # Gradient checks
├── test_benchmark.py             # Performance tests
└── README.md
```

## Quick Start

```bash
# Build extension
pip install -e .

# Run tests
python test_correctness.py
python test_benchmark.py
```

## Usage

```python
import torch
from fused_op import fused_layer_norm

# Forward pass (uses custom CUDA kernel)
output = fused_layer_norm(input, weight, bias, eps=1e-5)

# Backward pass (automatically uses custom CUDA backward kernel)
loss = output.sum()
loss.backward()
```

## Files Explained

### 1. `csrc/cuda/fused_op_kernel.cu` - The CUDA Kernels

```cuda
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Forward kernel
template <typename T>
__global__ void fused_op_forward_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int N, int C
) {
    // Your optimized CUDA kernel here
}

// Backward kernel
template <typename T>
__global__ void fused_op_backward_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ grad_input,
    T* __restrict__ grad_weight,
    T* __restrict__ grad_bias,
    int N, int C
) {
    // Your backward CUDA kernel here
}
```

### 2. `csrc/fused_op.cpp` - C++ Binding

```cpp
#include <torch/extension.h>
#include "cuda/fused_op_kernel.cuh"

// Forward declaration
torch::Tensor fused_op_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
fused_op_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight
);

// Input checking
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_op_forward_cuda, "Fused Op Forward (CUDA)");
    m.def("backward", &fused_op_backward_cuda, "Fused Op Backward (CUDA)");
}
```

### 3. `fused_op/functional.py` - Autograd Function

```python
import torch
from torch.autograd import Function
import fused_op_cuda  # The compiled extension

class FusedOpFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight)
        output = fused_op_cuda.forward(input, weight, bias)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_op_cuda.backward(
            grad_output.contiguous(), input, weight
        )
        return grad_input, grad_weight, grad_bias

def fused_op(input, weight, bias):
    return FusedOpFunction.apply(input, weight, bias)
```

### 4. `setup.py` - Build Script

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fused_op_cuda',
    ext_modules=[
        CUDAExtension(
            name='fused_op_cuda',
            sources=[
                'csrc/fused_op.cpp',
                'csrc/cuda/fused_op_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-lineinfo',
                    '--use_fast_math',
                    '-gencode=arch=compute_70,code=sm_70',  # V100
                    '-gencode=arch=compute_80,code=sm_80',  # A100
                    '-gencode=arch=compute_89,code=sm_89',  # RTX 4090
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

### 5. `test_correctness.py` - Gradient Checking

```python
import torch
from torch.autograd import gradcheck
from fused_op import fused_op

def test_forward_correctness():
    """Compare with PyTorch reference implementation."""
    torch.manual_seed(42)
    
    input = torch.randn(32, 128, device='cuda', dtype=torch.float64)
    weight = torch.randn(128, device='cuda', dtype=torch.float64)
    bias = torch.randn(128, device='cuda', dtype=torch.float64)
    
    # Reference implementation
    ref_output = torch.layer_norm(input, [128], weight, bias)
    
    # Custom implementation
    custom_output = fused_op(input, weight, bias)
    
    torch.testing.assert_close(custom_output, ref_output, atol=1e-5, rtol=1e-5)
    print("✓ Forward pass matches reference")

def test_backward_correctness():
    """Numerical gradient check."""
    torch.manual_seed(42)
    
    input = torch.randn(8, 32, device='cuda', dtype=torch.float64, requires_grad=True)
    weight = torch.randn(32, device='cuda', dtype=torch.float64, requires_grad=True)
    bias = torch.randn(32, device='cuda', dtype=torch.float64, requires_grad=True)
    
    assert gradcheck(fused_op, (input, weight, bias), eps=1e-6, atol=1e-4)
    print("✓ Gradient check passed")

if __name__ == '__main__':
    test_forward_correctness()
    test_backward_correctness()
    print("\n✅ All tests passed!")
```

### 6. `test_benchmark.py` - Performance Testing

```python
import torch
import time
from fused_op import fused_op

def benchmark(fn, warmup=10, iterations=100):
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations

def main():
    sizes = [(32, 512), (64, 1024), (128, 2048), (256, 4096)]
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                   FUSED OP BENCHMARK                         ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║ Shape          │ PyTorch (μs) │ Custom (μs) │ Speedup        ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    
    for batch, hidden in sizes:
        input = torch.randn(batch, hidden, device='cuda')
        weight = torch.randn(hidden, device='cuda')
        bias = torch.randn(hidden, device='cuda')
        
        pytorch_time = benchmark(lambda: torch.layer_norm(input, [hidden], weight, bias))
        custom_time = benchmark(lambda: fused_op(input, weight, bias))
        speedup = pytorch_time / custom_time
        
        print(f"║ ({batch:3d}, {hidden:4d})    │ {pytorch_time*1000:11.1f} │ {custom_time*1000:11.1f} │ {speedup:6.2f}×         ║")
    
    print("╚══════════════════════════════════════════════════════════════╝")

if __name__ == '__main__':
    main()
```

## Checklist for Production Extensions

- [ ] Input validation (CUDA, contiguous, dtype)
- [ ] Support for multiple dtypes (FP32, FP16, BF16)
- [ ] Gradient checking with `torch.autograd.gradcheck`
- [ ] Benchmark against PyTorch baseline
- [ ] Memory leak testing
- [ ] Multi-GPU testing (if applicable)
- [ ] Edge cases (empty tensors, single element, etc.)
- [ ] Documentation and usage examples
- [ ] CI/CD integration
