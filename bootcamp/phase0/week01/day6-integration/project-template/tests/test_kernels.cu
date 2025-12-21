/**
 * Kernel correctness tests
 */

#include "cuda_utils.cuh"
#include <cmath>
#include <cstdlib>

// Kernel to test
__global__ void test_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f + 1.0f;
    }
}

// CPU reference
void cpu_reference(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = input[i] * 2.0f + 1.0f;
    }
}

bool test_basic_correctness() {
    printf("Test: Basic correctness... ");
    
    const int N = 10000;
    float* h_input = new float[N];
    float* h_output_gpu = new float[N];
    float* h_output_cpu = new float[N];
    
    // Random input
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
    }
    
    // GPU compute
    DeviceBuffer<float> d_input(N);
    DeviceBuffer<float> d_output(N);
    d_input.copy_from_host(h_input);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    test_kernel<<<blocks, threads>>>(d_input.data(), d_output.data(), N);
    CUDA_KERNEL_CHECK();
    
    d_output.copy_to_host(h_output_gpu);
    
    // CPU compute
    cpu_reference(h_input, h_output_cpu, N);
    
    // Compare
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabs(h_output_gpu[i] - h_output_cpu[i]);
        max_diff = fmax(max_diff, diff);
    }
    
    bool passed = max_diff < 1e-5f;
    printf("%s (max diff: %e)\n", passed ? "PASSED" : "FAILED", max_diff);
    
    delete[] h_input;
    delete[] h_output_gpu;
    delete[] h_output_cpu;
    
    return passed;
}

bool test_edge_cases() {
    printf("Test: Edge cases... ");
    
    // Test with N=0
    {
        DeviceBuffer<float> d_data(1);  // Minimum allocation
        test_kernel<<<1, 1>>>(d_data.data(), d_data.data(), 0);
        CUDA_KERNEL_CHECK();
    }
    
    // Test with N=1
    {
        float val = 3.0f;
        DeviceBuffer<float> d_input(1);
        DeviceBuffer<float> d_output(1);
        d_input.copy_from_host(&val);
        
        test_kernel<<<1, 1>>>(d_input.data(), d_output.data(), 1);
        CUDA_KERNEL_CHECK();
        
        float result;
        d_output.copy_to_host(&result);
        
        if (fabs(result - 7.0f) > 1e-5f) {
            printf("FAILED (expected 7.0, got %f)\n", result);
            return false;
        }
    }
    
    printf("PASSED\n");
    return true;
}

int main() {
    printf("\n=== Kernel Tests ===\n\n");
    
    int passed = 0;
    int total = 0;
    
    total++; if (test_basic_correctness()) passed++;
    total++; if (test_edge_cases()) passed++;
    
    printf("\n=== Results: %d/%d passed ===\n\n", passed, total);
    
    return (passed == total) ? 0 : 1;
}
