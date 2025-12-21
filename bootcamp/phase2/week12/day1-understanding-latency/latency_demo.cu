/**
 * latency_demo.cu - Demonstrate and measure latencies
 * 
 * Learning objectives:
 * - See how latency affects performance
 * - Understand latency hiding
 */

#include <cuda_runtime.h>
#include <cstdio>

// Measure register "latency" (should be ~0)
__global__ void register_chain(float* out, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 1.0f;
    
    // Serial dependency chain - exposes instruction latency
    for (int i = 0; i < iters; i++) {
        val = val * 1.0001f + 0.0001f;
    }
    
    if (idx < n) out[idx] = val;
}

// Measure shared memory latency
__global__ void shared_mem_chain(float* out, int n, int iters) {
    __shared__ float smem[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    smem[tid] = 1.0f;
    __syncthreads();
    
    // Pointer chasing in shared memory
    float val = smem[tid];
    for (int i = 0; i < iters; i++) {
        val = val + smem[(int)val % 256];
    }
    
    if (idx < n) out[idx] = val;
}

// Measure global memory latency (strided to avoid cache)
__global__ void global_mem_chain(const float* in, float* out, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Pointer chasing in global memory - exposed latency
    float val = 0;
    int ptr = idx;
    for (int i = 0; i < 100; i++) {
        val += in[ptr % n];
        ptr = (ptr + stride) % n;  // Strided access defeats cache
    }
    
    if (idx < n) out[idx] = val;
}

// Latency hidden with enough parallelism
__global__ void global_mem_parallel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Independent accesses - can overlap
        out[idx] = in[idx] * 2.0f;
    }
}

int main() {
    printf("=== Latency Demonstration ===\n\n");
    
    const int N = 1 << 20;  // 1M
    const int TRIALS = 50;
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = (float)(i % 256);
    cudaMemcpy(d_in, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    printf("=== Instruction Latency (Register Chain) ===\n\n");
    
    // Measure instruction throughput with dependency chain
    for (int iters : {100, 1000, 10000}) {
        cudaEventRecord(start);
        for (int t = 0; t < TRIALS; t++) {
            register_chain<<<blocks, threads>>>(d_out, N, iters);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        
        float time_per = ms / TRIALS;
        float ops = (float)N * iters * 2;  // mul + add
        float gflops = ops / (time_per / 1000) / 1e9;
        
        printf("Iterations=%5d: %.3f ms, %.1f GFLOPS (serial deps limit throughput)\n",
               iters, time_per, gflops);
    }
    printf("\n");
    
    printf("=== Memory Latency (Pointer Chasing) ===\n\n");
    
    // Global memory with varying stride (defeats cache)
    for (int stride : {1, 64, 4096}) {
        cudaEventRecord(start);
        for (int t = 0; t < TRIALS; t++) {
            global_mem_chain<<<blocks, threads>>>(d_in, d_out, N, stride);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        
        printf("Stride=%4d: %.3f ms (larger stride = more cache misses = higher latency)\n",
               stride, ms / TRIALS);
    }
    printf("\n");
    
    printf("=== Latency Hidden (Parallel Access) ===\n\n");
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        global_mem_parallel<<<blocks, threads>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    float time_per = ms / TRIALS;
    float bytes = N * 8.0f;  // read + write
    float bw = bytes / (time_per / 1000) / 1e9;
    
    printf("Parallel access: %.3f ms, %.1f GB/s\n", time_per, bw);
    printf("(Latency hidden by many independent threads)\n\n");
    
    printf("=== Key Insights ===\n\n");
    printf("1. Memory latency: 400-800 cycles (global), ~30 cycles (shared)\n");
    printf("2. Instruction latency: ~4 cycles (FMA)\n");
    printf("3. Serial dependencies expose latency\n");
    printf("4. Parallelism hides latency:\n");
    printf("   - Many threads → scheduler switches during waits\n");
    printf("   - Independent operations → can issue in parallel\n");
    printf("5. Goal: Keep enough work in flight to hide all latency\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_data;
    
    return 0;
}
