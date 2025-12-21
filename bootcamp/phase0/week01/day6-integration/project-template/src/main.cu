/**
 * Main entry point - Template
 * 
 * Replace this with your actual application.
 */

#include "cuda_utils.cuh"
#include <cstdio>

// Example kernel
__global__ void example_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    // Print device info
    auto info = get_gpu_info();
    info.print();
    
    // Example: allocate, compute, verify
    const int N = 1024;
    
    // Host data
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = (float)i;
    
    // Device data (RAII)
    DeviceBuffer<float> d_data(N);
    d_data.copy_from_host(h_data);
    
    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    CudaTimer timer;
    timer.start();
    example_kernel<<<blocks, threads>>>(d_data.data(), N);
    timer.stop();
    CUDA_KERNEL_CHECK();
    
    printf("Kernel time: %.3f ms\n", timer.elapsed_ms());
    
    // Copy back and verify
    d_data.copy_to_host(h_data);
    
    bool correct = true;
    for (int i = 0; i < N && correct; i++) {
        if (fabs(h_data[i] - 2.0f * i) > 1e-5f) {
            printf("Mismatch at %d: expected %.1f, got %.1f\n", 
                   i, 2.0f * i, h_data[i]);
            correct = false;
        }
    }
    
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    
    delete[] h_data;
    return correct ? 0 : 1;
}
