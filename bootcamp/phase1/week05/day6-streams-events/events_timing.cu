/**
 * events_timing.cu - Accurate GPU timing with CUDA events
 * 
 * Learning objectives:
 * - Use events for precise kernel timing
 * - Understand event synchronization
 * - Compare CPU vs GPU timing methods
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

// Kernel to benchmark
__global__ void saxpy(float* y, const float* x, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

// Event-based timer class
class GpuTimer {
public:
    cudaEvent_t start, stop;
    
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void Start(cudaStream_t stream = 0) {
        cudaEventRecord(start, stream);
    }
    
    void Stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop, stream);
    }
    
    float ElapsedMs() {
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

int main() {
    printf("=== CUDA Events Timing Demo ===\n\n");
    
    const int N = 1 << 24;  // 16M elements
    const int TRIALS = 100;
    
    float* d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    
    // Initialize
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    cudaMemcpy(d_x, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    
    // Demo 1: Compare timing methods
    printf("=== Demo 1: CPU vs GPU Timing ===\n\n");
    
    // Method 1: CPU timing (WRONG - doesn't account for async)
    auto cpu_start = std::chrono::high_resolution_clock::now();
    saxpy<<<num_blocks, block_size>>>(d_y, d_x, 2.0f, N);
    // Note: kernel launches are async! CPU time is wrong without sync
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_time_wrong = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    
    // Method 2: CPU timing with sync (CORRECT but overhead)
    cudaDeviceSynchronize();  // Ensure previous kernel done
    cpu_start = std::chrono::high_resolution_clock::now();
    saxpy<<<num_blocks, block_size>>>(d_y, d_x, 2.0f, N);
    cudaDeviceSynchronize();
    cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_time_sync = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    
    // Method 3: CUDA events (BEST)
    GpuTimer timer;
    cudaDeviceSynchronize();
    timer.Start();
    saxpy<<<num_blocks, block_size>>>(d_y, d_x, 2.0f, N);
    timer.Stop();
    float gpu_time = timer.ElapsedMs();
    
    printf("CPU timing (no sync):      %.4f ms (WRONG!)\n", cpu_time_wrong);
    printf("CPU timing (with sync):    %.4f ms\n", cpu_time_sync);
    printf("CUDA events:               %.4f ms (BEST)\n\n", gpu_time);
    
    // Demo 2: Accurate benchmarking
    printf("=== Demo 2: Proper Benchmarking ===\n\n");
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        saxpy<<<num_blocks, block_size>>>(d_y, d_x, 2.0f, N);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    float total_time = 0.0f;
    float min_time = 1e9f;
    float max_time = 0.0f;
    
    for (int t = 0; t < TRIALS; t++) {
        timer.Start();
        saxpy<<<num_blocks, block_size>>>(d_y, d_x, 2.0f, N);
        timer.Stop();
        float ms = timer.ElapsedMs();
        
        total_time += ms;
        min_time = fminf(min_time, ms);
        max_time = fmaxf(max_time, ms);
    }
    
    float avg_time = total_time / TRIALS;
    float bandwidth = 3.0f * N * sizeof(float) / avg_time / 1e6;  // GB/s
    
    printf("SAXPY Benchmark (%d trials):\n", TRIALS);
    printf("  Array size:     %d elements (%.1f MB)\n", N, N * sizeof(float) / 1e6f);
    printf("  Min time:       %.4f ms\n", min_time);
    printf("  Max time:       %.4f ms\n", max_time);
    printf("  Avg time:       %.4f ms\n", avg_time);
    printf("  Bandwidth:      %.1f GB/s\n\n", bandwidth);
    
    // Demo 3: Event inter-stream synchronization
    printf("=== Demo 3: Event-Based Stream Sync ===\n\n");
    
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    cudaEvent_t event;
    cudaEventCreate(&event);
    
    // Stream 1 does work, records event
    printf("Stream 1: Starting work\n");
    saxpy<<<num_blocks, block_size, 0, stream1>>>(d_y, d_x, 2.0f, N);
    cudaEventRecord(event, stream1);  // Record when stream1 finishes
    
    // Stream 2 waits for stream 1's event before proceeding
    cudaStreamWaitEvent(stream2, event);  // Stream 2 waits here
    printf("Stream 2: Waiting for Stream 1's event\n");
    saxpy<<<num_blocks, block_size, 0, stream2>>>(d_y, d_x, 2.0f, N);
    
    cudaDeviceSynchronize();
    printf("Both streams complete\n\n");
    
    cudaEventDestroy(event);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
    // Cleanup
    delete[] h_data;
    cudaFree(d_x);
    cudaFree(d_y);
    
    printf("=== Timing Best Practices ===\n");
    printf("1. Always use CUDA events for GPU timing\n");
    printf("2. Do warmup runs before benchmarking\n");
    printf("3. Run multiple trials and report min/avg\n");
    printf("4. Report bandwidth for memory-bound kernels\n");
    printf("5. Use events for cross-stream synchronization\n");
    
    return 0;
}
