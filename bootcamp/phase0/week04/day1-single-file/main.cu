/**
 * Single-File CUDA Project Template
 * 
 * A minimal but complete template for CUDA development.
 * Includes error checking, timing, and device info.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ============================================================================
// Error Checking
// ============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUDA_CHECK_KERNEL() do { \
    CUDA_CHECK(cudaGetLastError()); \
    CUDA_CHECK(cudaDeviceSynchronize()); \
} while(0)

// ============================================================================
// Timer
// ============================================================================
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
private:
    cudaEvent_t start_, stop_;
};

// ============================================================================
// Your Kernel Here
// ============================================================================
__global__ void my_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

// ============================================================================
// Device Info
// ============================================================================
void print_device_info() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Memory: %.1f GB, SMs: %d\n", 
           prop.totalGlobalMem / 1e9, prop.multiProcessorCount);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    printf("CUDA Single-File Template\n");
    printf("=========================\n\n");
    
    print_device_info();
    
    // Configuration
    const int N = 1 << 20;  // 1M elements
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    
    printf("\nProblem size: %d elements\n", N);
    
    // Allocate
    float* h_data = (float*)malloc(N * sizeof(float));
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Run kernel
    CudaTimer timer;
    timer.start();
    my_kernel<<<numBlocks, blockSize>>>(d_data, N);
    CUDA_CHECK_KERNEL();
    float time_ms = timer.stop();
    
    printf("Kernel time: %.3f ms\n", time_ms);
    
    // Verify (optional)
    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < N && correct; i++) {
        float expected = (float)i * 2.0f + 1.0f;
        if (fabs(h_data[i] - expected) > 1e-5f) {
            printf("Mismatch at %d: got %f, expected %f\n", i, h_data[i], expected);
            correct = false;
        }
    }
    printf("Result: %s\n", correct ? "PASS" : "FAIL");
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);
    
    return correct ? 0 : 1;
}
