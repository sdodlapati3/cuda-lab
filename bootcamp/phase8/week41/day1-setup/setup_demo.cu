/**
 * Week 41, Day 1: PyTorch Extension Setup
 */
#include <cstdio>

int main() {
    printf("Week 41 Day 1: PyTorch CUDA Extension Setup\n\n");
    
    printf("Two Ways to Build PyTorch Extensions:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ 1. JIT Compilation (torch.utils.cpp_extension.load)               ║\n");
    printf("║    • Quick prototyping, no setup.py needed                        ║\n");
    printf("║    • Compiles on first import                                     ║\n");
    printf("║                                                                   ║\n");
    printf("║ 2. Ahead-of-Time (setup.py with CUDAExtension)                    ║\n");
    printf("║    • Production deployment                                        ║\n");
    printf("║    • pip installable package                                      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Minimal setup.py:\n");
    printf("```python\n");
    printf("from setuptools import setup\n");
    printf("from torch.utils.cpp_extension import BuildExtension, CUDAExtension\n");
    printf("\n");
    printf("setup(\n");
    printf("    name='my_cuda_ext',\n");
    printf("    ext_modules=[\n");
    printf("        CUDAExtension(\n");
    printf("            name='my_cuda_ext',\n");
    printf("            sources=['my_kernel.cu', 'binding.cpp'],\n");
    printf("            extra_compile_args={'cxx': ['-O3'],\n");
    printf("                               'nvcc': ['-O3', '--use_fast_math']}\n");
    printf("        )\n");
    printf("    ],\n");
    printf("    cmdclass={'build_ext': BuildExtension}\n");
    printf(")\n");
    printf("```\n\n");
    
    printf("JIT Example:\n");
    printf("```python\n");
    printf("from torch.utils.cpp_extension import load\n");
    printf("\n");
    printf("my_ext = load(\n");
    printf("    name='my_ext',\n");
    printf("    sources=['kernel.cu'],\n");
    printf("    verbose=True\n");
    printf(")\n");
    printf("```\n\n");
    
    printf("Directory Structure:\n");
    printf("my_cuda_extension/\n");
    printf("├── setup.py\n");
    printf("├── my_ext/\n");
    printf("│   ├── __init__.py\n");
    printf("│   ├── csrc/\n");
    printf("│   │   ├── binding.cpp    # pybind11 bindings\n");
    printf("│   │   └── kernel.cu      # CUDA kernels\n");
    printf("│   └── functional.py      # Python wrapper\n");
    printf("└── tests/\n");
    printf("    └── test_kernel.py\n");
    
    return 0;
}
