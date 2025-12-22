/**
 * Week 41, Day 2: Tensor Accessors
 */
#include <cstdio>

int main() {
    printf("Week 41 Day 2: PyTorch Tensor Accessors\n\n");
    
    printf("Accessing Tensor Data in CUDA:\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ Method 1: PackedTensorAccessor (bounds-checked, slower)           ║\n");
    printf("║ Method 2: data_ptr<T>() (raw pointer, faster)                     ║\n");
    printf("║                                                                   ║\n");
    printf("║ Always check: tensor.is_cuda(), tensor.is_contiguous()            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("PackedTensorAccessor Example:\n");
    printf("```cpp\n");
    printf("// In binding.cpp\n");
    printf("torch::Tensor my_kernel(torch::Tensor input) {\n");
    printf("    TORCH_CHECK(input.is_cuda(), \"Input must be CUDA tensor\");\n");
    printf("    TORCH_CHECK(input.is_contiguous(), \"Input must be contiguous\");\n");
    printf("    \n");
    printf("    auto output = torch::empty_like(input);\n");
    printf("    \n");
    printf("    // Create accessors (32 = index type, 2 = dimensions)\n");
    printf("    auto input_a = input.packed_accessor32<float, 2>();\n");
    printf("    auto output_a = output.packed_accessor32<float, 2>();\n");
    printf("    \n");
    printf("    my_cuda_kernel<<<grid, block>>>(input_a, output_a);\n");
    printf("    return output;\n");
    printf("}\n");
    printf("```\n\n");
    
    printf("Raw Pointer Access (Faster):\n");
    printf("```cpp\n");
    printf("torch::Tensor my_kernel(torch::Tensor input) {\n");
    printf("    float* input_ptr = input.data_ptr<float>();\n");
    printf("    float* output_ptr = output.data_ptr<float>();\n");
    printf("    \n");
    printf("    int n = input.size(0);\n");
    printf("    int d = input.size(1);\n");
    printf("    \n");
    printf("    my_cuda_kernel<<<grid, block>>>(input_ptr, output_ptr, n, d);\n");
    printf("    return output;\n");
    printf("}\n");
    printf("```\n\n");
    
    printf("Data Type Dispatch:\n");
    printf("```cpp\n");
    printf("AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), \"my_kernel\", [&] {\n");
    printf("    scalar_t* ptr = input.data_ptr<scalar_t>();\n");
    printf("    // kernel launch with scalar_t\n");
    printf("});\n");
    printf("```\n");
    
    return 0;
}
