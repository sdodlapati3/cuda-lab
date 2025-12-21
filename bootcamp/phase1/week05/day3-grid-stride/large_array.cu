/**
 * large_array.cu - Grid-stride for large arrays with benchmarking
 * 
 * Learning objectives:
 * - Process arrays larger than GPU can hold threads
 * - Benchmark different grid sizes
 * - Find optimal launch configuration
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

// Simple SAXPY kernel with grid-stride
__global__ void saxpy_grid_stride(float* y, const float* x, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

// Helper for timing
class GpuTimer {
    cudaEvent_t start, stop;
public:
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void Start() { cudaEventRecord(start, 0); }
    void Stop() { cudaEventRecord(stop, 0); cudaEventSynchronize(stop); }
    float ElapsedMs() {
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

int main() {
    printf("=== Large Array Grid-Stride Benchmark ===\n\n");
    
    // Large array: 100 million elements
    const int N = 100'000'000;
    const size_t bytes = N * sizeof(float);
    
    printf("Array size: %d elements (%.1f MB)\n", N, bytes / 1e6);
    
    // Allocate
    float* h_x = new float[N];
    float* h_y = new float[N];
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }
    
    float* d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("SMs: %d, Max threads/SM: %d\n", 
           prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor);
    printf("\n");
    
    // Benchmark different configurations
    printf("=== Benchmark: Varying Number of Blocks ===\n");
    printf("Block size fixed at 256\n\n");
    
    const int BLOCK_SIZE = 256;
    const float a = 2.0f;
    
    // Test different grid sizes
    int test_blocks[] = {1, 10, 100, 1000, 10000, 
                         prop.multiProcessorCount,
                         prop.multiProcessorCount * 2,
                         prop.multiProcessorCount * 4,
                         (N + BLOCK_SIZE - 1) / BLOCK_SIZE};  // Max needed
    
    printf("%10s %12s %12s %12s\n", "Blocks", "Time (ms)", "GB/s", "Note");
    printf("------------------------------------------------------\n");
    
    GpuTimer timer;
    
    for (int blocks : test_blocks) {
        if (blocks <= 0) continue;
        
        // Reset y
        cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);
        
        // Warm up
        saxpy_grid_stride<<<blocks, BLOCK_SIZE>>>(d_y, d_x, a, N);
        cudaDeviceSynchronize();
        
        // Reset and time
        cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);
        
        timer.Start();
        saxpy_grid_stride<<<blocks, BLOCK_SIZE>>>(d_y, d_x, a, N);
        timer.Stop();
        
        float ms = timer.ElapsedMs();
        float gb_s = 3.0f * bytes / ms / 1e6;  // 2 reads + 1 write
        
        const char* note = "";
        if (blocks == prop.multiProcessorCount) note = "<-- SMs";
        if (blocks == prop.multiProcessorCount * 2) note = "<-- 2×SMs";
        if (blocks == (N + BLOCK_SIZE - 1) / BLOCK_SIZE) note = "<-- 1 thread/elem";
        
        printf("%10d %12.3f %12.1f %s\n", blocks, ms, gb_s, note);
    }
    
    // Verify correctness
    printf("\n=== Verification ===\n");
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);
    saxpy_grid_stride<<<prop.multiProcessorCount * 2, BLOCK_SIZE>>>(d_y, d_x, a, N);
    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
    
    // Check first and last elements
    float expected = 2.0f * 1.0f + 2.0f;  // a*x + y
    bool pass = true;
    for (int i = 0; i < 10; i++) {
        if (h_y[i] != expected) {
            printf("FAIL at %d: got %f, expected %f\n", i, h_y[i], expected);
            pass = false;
            break;
        }
    }
    if (pass) printf("Result: CORRECT (y[i] = %.1f)\n", expected);
    
    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    delete[] h_x;
    delete[] h_y;
    
    printf("\n=== Key Insight ===\n");
    printf("Optimal grid size is often 2-4× the number of SMs.\n");
    printf("Too few blocks: Underutilize GPU\n");
    printf("Too many blocks: Diminishing returns, more overhead\n");
    printf("Grid-stride lets you use optimal config regardless of N.\n");
    
    return 0;
}
