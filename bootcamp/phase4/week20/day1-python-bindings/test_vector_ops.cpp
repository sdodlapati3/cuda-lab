/**
 * Test the vector operations library
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>

extern "C" {
    void cuda_vector_add(const float* a, const float* b, float* c, int n);
    void cuda_vector_mul(const float* a, const float* b, float* c, int n);
    float cuda_sum(const float* input, int n);
    int cuda_get_device_count();
    const char* cuda_get_device_name(int device);
}

int main() {
    printf("=== Testing CUDA Vector Operations ===\n\n");
    
    // Device info
    int deviceCount = cuda_get_device_count();
    printf("CUDA Devices: %d\n", deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        printf("  Device %d: %s\n", i, cuda_get_device_name(i));
    }
    printf("\n");
    
    // Test data
    const int N = 1000000;
    float* a = (float*)malloc(N * sizeof(float));
    float* b = (float*)malloc(N * sizeof(float));
    float* c = (float*)malloc(N * sizeof(float));
    
    // Initialize
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    
    // Test vector add
    printf("Testing vector_add with %d elements...\n", N);
    cuda_vector_add(a, b, c, N);
    
    // Verify
    bool pass = true;
    for (int i = 0; i < N && pass; i++) {
        if (fabs(c[i] - 3.0f) > 1e-5f) pass = false;
    }
    printf("  Result: %s\n", pass ? "PASS" : "FAIL");
    
    // Test vector mul
    printf("Testing vector_mul with %d elements...\n", N);
    cuda_vector_mul(a, b, c, N);
    
    pass = true;
    for (int i = 0; i < N && pass; i++) {
        if (fabs(c[i] - 2.0f) > 1e-5f) pass = false;
    }
    printf("  Result: %s\n", pass ? "PASS" : "FAIL");
    
    // Test sum
    printf("Testing array_sum with %d elements...\n", N);
    float sum = cuda_sum(a, N);
    printf("  Sum: %.0f (expected: %d)\n", sum, N);
    printf("  Result: %s\n", fabs(sum - N) < 1.0f ? "PASS" : "FAIL");
    
    // Cleanup
    free(a);
    free(b);
    free(c);
    
    printf("\nAll tests complete!\n");
    return 0;
}
