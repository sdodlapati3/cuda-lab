/**
 * optimization_cycle.cu - Demonstrate iterative optimization
 * 
 * Shows multiple versions of same kernel, each with one improvement
 */

#include <cuda_runtime.h>
#include <cstdio>

// ========== Version 0: Naive implementation ==========
__global__ void transpose_v0(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Naive: uncoalesced writes
        out[x * height + y] = in[y * width + x];
    }
}

// ========== Version 1: Coalesced reads ==========
__global__ void transpose_v1(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Same as v0, but let's think about it differently
        // Reads are coalesced (row-major in)
        // Writes are strided (column-major out) - still bad
        out[x * height + y] = in[y * width + x];
    }
}

// ========== Version 2: Shared memory tiling ==========
#define TILE 32
__global__ void transpose_v2(const float* in, float* out, int width, int height) {
    __shared__ float tile[TILE][TILE];
    
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    
    // Coalesced read into shared memory
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads();
    
    // Calculate transposed output position
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    
    // Coalesced write from shared memory
    if (x < height && y < width) {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// ========== Version 3: Avoid bank conflicts ==========
#define TILE_PAD 33  // 32 + 1 padding
__global__ void transpose_v3(const float* in, float* out, int width, int height) {
    __shared__ float tile[TILE][TILE_PAD];  // Padding avoids bank conflicts
    
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads();
    
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    
    if (x < height && y < width) {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

struct BenchResult {
    float time_ms;
    float bandwidth_gbps;
    float efficiency;
};

template<typename Func>
BenchResult benchmark_kernel(const char* name, Func kernel_launch, 
                             int width, int height, int trials) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 3; i++) kernel_launch();
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < trials; i++) kernel_launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    BenchResult result;
    result.time_ms = ms / trials;
    
    // Bandwidth: read + write
    size_t bytes = 2 * width * height * sizeof(float);
    result.bandwidth_gbps = bytes / (result.time_ms / 1000) / 1e9;
    
    // Rough efficiency (assuming ~1500 GB/s peak)
    result.efficiency = result.bandwidth_gbps / 1500.0f * 100.0f;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}

int main() {
    printf("=== Optimization Cycle Demo: Matrix Transpose ===\n\n");
    
    const int WIDTH = 4096;
    const int HEIGHT = 4096;
    const int TRIALS = 100;
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc(&d_out, WIDTH * HEIGHT * sizeof(float));
    
    // Initialize
    float* h_data = new float[WIDTH * HEIGHT];
    for (int i = 0; i < WIDTH * HEIGHT; i++) h_data[i] = (float)i;
    cudaMemcpy(d_in, h_data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threads(32, 32);
    dim3 blocks((WIDTH + 31) / 32, (HEIGHT + 31) / 32);
    
    printf("Matrix: %d x %d (%zu MB)\n\n", WIDTH, HEIGHT, 
           WIDTH * HEIGHT * sizeof(float) / (1024 * 1024));
    
    printf("%-20s | %10s | %10s | %10s | %s\n",
           "Version", "Time(ms)", "BW(GB/s)", "Efficiency", "Change");
    printf("-----------------------------------------------------------------------\n");
    
    // Version 0
    auto v0 = [&]() { transpose_v0<<<blocks, threads>>>(d_in, d_out, WIDTH, HEIGHT); };
    BenchResult r0 = benchmark_kernel("v0_naive", v0, WIDTH, HEIGHT, TRIALS);
    printf("%-20s | %10.3f | %10.1f | %9.1f%% | (baseline)\n",
           "v0_naive", r0.time_ms, r0.bandwidth_gbps, r0.efficiency);
    
    // Version 1
    auto v1 = [&]() { transpose_v1<<<blocks, threads>>>(d_in, d_out, WIDTH, HEIGHT); };
    BenchResult r1 = benchmark_kernel("v1_coalesced_read", v1, WIDTH, HEIGHT, TRIALS);
    printf("%-20s | %10.3f | %10.1f | %9.1f%% | Same as v0 actually\n",
           "v1_coalesced_read", r1.time_ms, r1.bandwidth_gbps, r1.efficiency);
    
    // Version 2
    auto v2 = [&]() { transpose_v2<<<blocks, threads>>>(d_in, d_out, WIDTH, HEIGHT); };
    BenchResult r2 = benchmark_kernel("v2_shared_memory", v2, WIDTH, HEIGHT, TRIALS);
    printf("%-20s | %10.3f | %10.1f | %9.1f%% | +Shared mem tiling\n",
           "v2_shared_memory", r2.time_ms, r2.bandwidth_gbps, r2.efficiency);
    
    // Version 3
    auto v3 = [&]() { transpose_v3<<<blocks, threads>>>(d_in, d_out, WIDTH, HEIGHT); };
    BenchResult r3 = benchmark_kernel("v3_no_bank_conflict", v3, WIDTH, HEIGHT, TRIALS);
    printf("%-20s | %10.3f | %10.1f | %9.1f%% | +Padding for banks\n",
           "v3_no_bank_conflict", r3.time_ms, r3.bandwidth_gbps, r3.efficiency);
    
    printf("\n=== Optimization Summary ===\n\n");
    printf("v0 → v2: %.1fx speedup (shared memory tiling)\n", r0.time_ms / r2.time_ms);
    printf("v2 → v3: %.1fx speedup (bank conflict avoidance)\n", r2.time_ms / r3.time_ms);
    printf("v0 → v3: %.1fx total speedup\n", r0.time_ms / r3.time_ms);
    printf("\n");
    
    printf("=== The Optimization Cycle Applied ===\n\n");
    printf("Iteration 1:\n");
    printf("  Profile v0 → Bottleneck: uncoalesced writes\n");
    printf("  Hypothesis: Shared memory can enable coalesced writes\n");
    printf("  Result: v2 is %.1fx faster\n\n", r0.time_ms / r2.time_ms);
    
    printf("Iteration 2:\n");
    printf("  Profile v2 → Bottleneck: shared memory bank conflicts\n");
    printf("  Hypothesis: Padding shared memory avoids conflicts\n");
    printf("  Result: v3 is %.1fx faster\n\n", r2.time_ms / r3.time_ms);
    
    printf("Iteration 3:\n");
    printf("  Profile v3 → Efficiency: %.0f%% of peak\n", r3.efficiency);
    printf("  Decision: Close to memory ceiling, optimization complete!\n");
    printf("\n");
    
    printf("=== Key Takeaways ===\n\n");
    printf("1. Profile first - don't guess the bottleneck\n");
    printf("2. One change at a time - isolate improvements\n");
    printf("3. Measure after each change - verify hypothesis\n");
    printf("4. Compare to theoretical peak - know when to stop\n");
    printf("5. Document the journey - helps future optimization\n");
    
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_data;
    
    return 0;
}
