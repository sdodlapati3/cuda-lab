/**
 * Day 5: Timeline Demo for Nsight Systems
 * 
 * Creates recognizable patterns in timeline view.
 */

#include <cstdio>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Kernel that does real work (visible in timeline)
__global__ void compute_kernel(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        // Do enough work to be visible in timeline
        for (int i = 0; i < iterations; i++) {
            val = sinf(val) * cosf(val) + 0.1f;
        }
        data[idx] = val;
    }
}

// Short kernel (to show overhead)
__global__ void tiny_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

void demo_serial_pattern() {
    printf("\n=== Serial Pattern ===\n");
    printf("CPU busy -> H2D -> Kernel -> D2H -> CPU busy\n");
    
    const int N = 1 << 22;  // 4M elements
    size_t size = N * sizeof(float);
    
    float* h_data = (float*)malloc(size);
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    
    // Serial pattern: each step waits for previous
    for (int iter = 0; iter < 5; iter++) {
        // CPU work (visible as gap)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
        compute_kernel<<<(N + 255) / 256, 256>>>(d_data, N, 100);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    }
    
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);
}

void demo_overlapped_pattern() {
    printf("\n=== Overlapped Pattern (Streams) ===\n");
    printf("Multiple streams allow concurrent execution\n");
    
    const int N = 1 << 20;  // 1M elements per chunk
    const int NUM_STREAMS = 4;
    size_t size = N * sizeof(float);
    
    cudaStream_t streams[NUM_STREAMS];
    float* h_data[NUM_STREAMS];
    float* d_data[NUM_STREAMS];
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaMallocHost(&h_data[i], size));  // Pinned memory
        CUDA_CHECK(cudaMalloc(&d_data[i], size));
        for (int j = 0; j < N; j++) h_data[i][j] = 1.0f;
    }
    
    // Overlap pattern: issue all work to streams
    for (int iter = 0; iter < 3; iter++) {
        for (int s = 0; s < NUM_STREAMS; s++) {
            CUDA_CHECK(cudaMemcpyAsync(d_data[s], h_data[s], size, 
                       cudaMemcpyHostToDevice, streams[s]));
            compute_kernel<<<(N + 255) / 256, 256, 0, streams[s]>>>(
                d_data[s], N, 50);
            CUDA_CHECK(cudaMemcpyAsync(h_data[s], d_data[s], size,
                       cudaMemcpyDeviceToHost, streams[s]));
        }
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaFreeHost(h_data[i]));
        CUDA_CHECK(cudaFree(d_data[i]));
    }
}

void demo_many_small_kernels() {
    printf("\n=== Many Small Kernels ===\n");
    printf("Launch overhead becomes visible\n");
    
    const int N = 1 << 16;  // Small
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(float)));
    
    // Launch many tiny kernels
    for (int i = 0; i < 100; i++) {
        tiny_kernel<<<(N + 255) / 256, 256>>>(d_data, N);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_data));
}

void demo_memory_transfer_bound() {
    printf("\n=== Memory Transfer Bound ===\n");
    printf("Transfers dominate; kernel is tiny\n");
    
    const int N = 1 << 24;  // 16M elements (64MB)
    size_t size = N * sizeof(float);
    
    float* h_data = (float*)malloc(size);
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    
    for (int iter = 0; iter < 3; iter++) {
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
        tiny_kernel<<<(N + 255) / 256, 256>>>(d_data, N);  // Very short
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    }
    
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);
}

int main() {
    printf("Timeline Demo for Nsight Systems\n");
    printf("================================\n");
    printf("Profile with: nsys profile -o report ./build/timeline_demo\n");
    
    demo_serial_pattern();
    demo_overlapped_pattern();
    demo_many_small_kernels();
    demo_memory_transfer_bound();
    
    printf("\nDone! View report.nsys-rep in Nsight Systems GUI\n");
    
    return 0;
}
