/**
 * Week 41, Day 6: Week Summary
 */
#include <cstdio>

int main() {
    printf("Week 41 Summary: PyTorch Extensions Basics\n");
    printf("══════════════════════════════════════════════════════════════════════\n\n");
    
    printf("Day 1: Extension Setup\n");
    printf("  • JIT vs AOT compilation\n");
    printf("  • setup.py with CUDAExtension\n");
    printf("  • Project structure\n\n");
    
    printf("Day 2: Tensor Accessors\n");
    printf("  • PackedTensorAccessor for bounds checking\n");
    printf("  • data_ptr<T>() for raw access\n");
    printf("  • AT_DISPATCH_FLOATING_TYPES macro\n\n");
    
    printf("Day 3: pybind11 Bindings\n");
    printf("  • PYBIND11_MODULE for basic binding\n");
    printf("  • TORCH_LIBRARY for modern registration\n\n");
    
    printf("Day 4: Error Handling\n");
    printf("  • TORCH_CHECK for input validation\n");
    printf("  • AT_CUDA_CHECK for CUDA errors\n\n");
    
    printf("Day 5: Testing\n");
    printf("  • torch.testing.assert_close()\n");
    printf("  • Parametrized tests for shapes/dtypes\n\n");
    
    printf("Key Takeaways:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ • Always validate inputs (device, dtype, contiguity)              ║\n");
    printf("║ • Use raw pointers for performance, accessors for debugging       ║\n");
    printf("║ • Test against PyTorch reference implementations                  ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
