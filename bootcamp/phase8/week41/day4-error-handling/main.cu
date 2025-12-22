/**
 * Week 41, Day 4: Error Handling
 */
#include <cstdio>

int main() {
    printf("Week 41 Day 4: Error Handling in Extensions\n\n");
    
    printf("PyTorch Error Checking Macros:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ TORCH_CHECK(cond, msg)  - Throws c10::Error if false              ║\n");
    printf("║ TORCH_INTERNAL_ASSERT   - Debug-only assertion                    ║\n");
    printf("║ AT_CUDA_CHECK           - Checks CUDA API return codes            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Input Validation:\n");
    printf("```cpp\n");
    printf("torch::Tensor my_kernel(torch::Tensor input) {\n");
    printf("    // Device check\n");
    printf("    TORCH_CHECK(input.is_cuda(), \"Input must be a CUDA tensor\");\n");
    printf("    \n");
    printf("    // Contiguity check\n");
    printf("    TORCH_CHECK(input.is_contiguous(), \"Input must be contiguous\");\n");
    printf("    \n");
    printf("    // Dtype check\n");
    printf("    TORCH_CHECK(input.scalar_type() == torch::kFloat32,\n");
    printf("               \"Expected float32, got \", input.scalar_type());\n");
    printf("    \n");
    printf("    // Dimension check\n");
    printf("    TORCH_CHECK(input.dim() == 2, \"Expected 2D tensor\");\n");
    printf("    \n");
    printf("    // ... kernel launch\n");
    printf("}\n");
    printf("```\n\n");
    
    printf("CUDA Error Checking:\n");
    printf("```cpp\n");
    printf("#define CUDA_CHECK(call) do { \\\n");
    printf("    cudaError_t err = call; \\\n");
    printf("    TORCH_CHECK(err == cudaSuccess, \\\n");
    printf("               \"CUDA error: \", cudaGetErrorString(err)); \\\n");
    printf("} while(0)\n");
    printf("\n");
    printf("// Or use PyTorch's macro\n");
    printf("AT_CUDA_CHECK(cudaGetLastError());\n");
    printf("```\n");
    
    return 0;
}
