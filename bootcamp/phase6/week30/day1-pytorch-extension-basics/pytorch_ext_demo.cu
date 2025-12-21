/**
 * Week 30, Day 1: PyTorch Extension Basics
 * Understanding the extension structure.
 */
#include <cstdio>

/*
 * PyTorch CUDA Extension Structure:
 *
 * 1. setup.py - Build configuration
 *    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
 *    setup(
 *        ext_modules=[CUDAExtension('my_cuda_op', ['my_op.cpp', 'my_op_kernel.cu'])],
 *        cmdclass={'build_ext': BuildExtension}
 *    )
 *
 * 2. my_op.cpp - C++ bindings
 *    #include <torch/extension.h>
 *    torch::Tensor my_op_cuda(torch::Tensor input);
 *    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
 *        m.def("forward", &my_op_cuda, "My Op (CUDA)");
 *    }
 *
 * 3. my_op_kernel.cu - CUDA implementation
 *    __global__ void my_kernel(...) { ... }
 *    torch::Tensor my_op_cuda(torch::Tensor input) {
 *        // Launch kernel
 *        return output;
 *    }
 */

int main() {
    printf("Week 30 Day 1: PyTorch Extension Basics\n\n");
    
    printf("Extension File Structure:\n");
    printf("  my_extension/\n");
    printf("  ├── setup.py          # Build script\n");
    printf("  ├── my_op.cpp         # C++ interface\n");
    printf("  ├── my_op_kernel.cu   # CUDA kernels\n");
    printf("  └── __init__.py       # Python package\n\n");
    
    printf("Key PyTorch Types:\n");
    printf("  torch::Tensor        - Multi-dimensional array\n");
    printf("  tensor.data_ptr<T>() - Get raw pointer\n");
    printf("  tensor.sizes()       - Get dimensions\n");
    printf("  tensor.device()      - Get device (CPU/CUDA)\n\n");
    
    printf("Build Methods:\n");
    printf("  1. python setup.py install  (ahead-of-time)\n");
    printf("  2. torch.utils.cpp_extension.load()  (JIT)\n\n");
    
    printf("JIT Compilation Example:\n");
    printf("  from torch.utils.cpp_extension import load\n");
    printf("  my_op = load('my_op', ['my_op.cpp', 'my_op.cu'])\n");
    printf("  output = my_op.forward(input)\n");
    
    return 0;
}
