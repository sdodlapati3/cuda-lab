/**
 * occupancy_tradeoffs.cu - When occupancy matters (and when it doesn't)
 * 
 * Learning objectives:
 * - See that high occupancy doesn't always mean better performance
 * - Understand the occupancy-performance relationship
 * - Learn to make informed trade-offs
 */

#include <cuda_runtime.h>
#include <cstdio>

// ========== Case 1: Memory-bound - occupancy HELPS ==========

// Low occupancy version
__global__ __launch_bounds__(64, 4)
void membound_low_occ(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * 2.0f;
}

// High occupancy version
__global__ __launch_bounds__(256, 8)
void membound_high_occ(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * 2.0f;
}

// ========== Case 2: Compute-bound - occupancy matters LESS ==========

// Low occupancy but more registers per thread
__global__ __launch_bounds__(128, 2)
void compute_low_occ(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        // Heavy compute with many registers
        float r[16];
        for (int i = 0; i < 16; i++) r[i] = val * (float)(i + 1);
        for (int j = 0; j < 50; j++) {
            for (int i = 0; i < 16; i++) r[i] = r[i] * 1.0001f + r[(i+1)%16] * 0.0001f;
        }
        float sum = 0;
        for (int i = 0; i < 16; i++) sum += r[i];
        out[idx] = sum;
    }
}

// Higher occupancy but more register pressure
__global__ __launch_bounds__(256, 4)
void compute_high_occ(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        // Same compute, but compiler may spill registers
        float r[16];
        for (int i = 0; i < 16; i++) r[i] = val * (float)(i + 1);
        for (int j = 0; j < 50; j++) {
            for (int i = 0; i < 16; i++) r[i] = r[i] * 1.0001f + r[(i+1)%16] * 0.0001f;
        }
        float sum = 0;
        for (int i = 0; i < 16; i++) sum += r[i];
        out[idx] = sum;
    }
}

// ========== Case 3: Cache-friendly - LOW occupancy can WIN ==========

// Tiled matrix multiply - uses lots of smem, low occupancy
// But the smem reuse makes it fast!
#define TILE_32 32
__global__ void matmul_tile32(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_32][TILE_32];
    __shared__ float Bs[TILE_32][TILE_32];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_32 + ty;
    int col = blockIdx.x * TILE_32 + tx;
    
    float sum = 0;
    for (int t = 0; t < N / TILE_32; t++) {
        As[ty][tx] = A[row * N + t * TILE_32 + tx];
        Bs[ty][tx] = B[(t * TILE_32 + ty) * N + col];
        __syncthreads();
        
        for (int k = 0; k < TILE_32; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}

// Smaller tile - higher occupancy but worse cache reuse
#define TILE_16 16
__global__ void matmul_tile16(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_16][TILE_16];
    __shared__ float Bs[TILE_16][TILE_16];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_16 + ty;
    int col = blockIdx.x * TILE_16 + tx;
    
    float sum = 0;
    for (int t = 0; t < N / TILE_16; t++) {
        As[ty][tx] = A[row * N + t * TILE_16 + tx];
        Bs[ty][tx] = B[(t * TILE_16 + ty) * N + col];
        __syncthreads();
        
        for (int k = 0; k < TILE_16; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}

template<typename Func>
float benchmark(const char* name, Func kernel_launch, int trials) {
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
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / trials;
}

int main() {
    printf("=== When Does Occupancy Matter? ===\n\n");
    
    const int N = 1 << 24;  // 16M
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    cudaMemcpy(d_in, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Case 1: Memory-bound
    printf("=== Case 1: Memory-Bound Kernel ===\n");
    printf("(Higher occupancy should help hide memory latency)\n\n");
    
    auto mem_low = [&]() { membound_low_occ<<<N/64, 64>>>(d_in, d_out, N); };
    auto mem_high = [&]() { membound_high_occ<<<N/256, 256>>>(d_in, d_out, N); };
    
    float t1 = benchmark("membound_low_occ", mem_low, 100);
    float t2 = benchmark("membound_high_occ", mem_high, 100);
    
    // Get occupancies
    int b1, b2;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b1, membound_low_occ, 64, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b2, membound_high_occ, 256, 0);
    
    printf("Low occupancy  (%.0f%%): %.4f ms\n", 100.0f * b1 * 2 / 64, t1);
    printf("High occupancy (%.0f%%): %.4f ms\n", 100.0f * b2 * 8 / 64, t2);
    printf("Winner: %s (%.1fx)\n\n", t2 < t1 ? "HIGH occupancy" : "LOW occupancy", 
           t2 < t1 ? t1/t2 : t2/t1);
    
    // Case 2: Compute-bound
    printf("=== Case 2: Compute-Bound Kernel ===\n");
    printf("(Occupancy matters less when compute is the bottleneck)\n\n");
    
    auto comp_low = [&]() { compute_low_occ<<<N/128, 128>>>(d_in, d_out, N); };
    auto comp_high = [&]() { compute_high_occ<<<N/256, 256>>>(d_in, d_out, N); };
    
    t1 = benchmark("compute_low_occ", comp_low, 20);
    t2 = benchmark("compute_high_occ", comp_high, 20);
    
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b1, compute_low_occ, 128, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b2, compute_high_occ, 256, 0);
    
    printf("Low occupancy  (%.0f%%): %.4f ms\n", 100.0f * b1 * 4 / 64, t1);
    printf("High occupancy (%.0f%%): %.4f ms\n", 100.0f * b2 * 8 / 64, t2);
    printf("Result: Difference is %.1f%% (occupancy matters less)\n\n",
           100.0f * abs(t1 - t2) / (0.5f * (t1 + t2)));
    
    // Case 3: Cache-friendly
    printf("=== Case 3: Tiled MatMul (Cache-Friendly) ===\n");
    printf("(Lower occupancy with better cache reuse can WIN)\n\n");
    
    const int MAT_N = 1024;
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, MAT_N * MAT_N * sizeof(float));
    cudaMalloc(&d_B, MAT_N * MAT_N * sizeof(float));
    cudaMalloc(&d_C, MAT_N * MAT_N * sizeof(float));
    
    dim3 block32(32, 32);
    dim3 grid32(MAT_N / 32, MAT_N / 32);
    dim3 block16(16, 16);
    dim3 grid16(MAT_N / 16, MAT_N / 16);
    
    auto mm32 = [&]() { matmul_tile32<<<grid32, block32>>>(d_A, d_B, d_C, MAT_N); };
    auto mm16 = [&]() { matmul_tile16<<<grid16, block16>>>(d_A, d_B, d_C, MAT_N); };
    
    t1 = benchmark("matmul_tile32", mm32, 50);
    t2 = benchmark("matmul_tile16", mm16, 50);
    
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b1, matmul_tile32, 1024, 32*32*4*2);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b2, matmul_tile16, 256, 16*16*4*2);
    
    printf("Tile 32x32 (%.0f%% occ): %.4f ms\n", 100.0f * b1 * 32 / 64, t1);
    printf("Tile 16x16 (%.0f%% occ): %.4f ms\n", 100.0f * b2 * 8 / 64, t2);
    printf("Winner: %s\n\n", t1 < t2 ? "LARGER tile (lower occ)" : "SMALLER tile (higher occ)");
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("=== Key Takeaways ===\n\n");
    printf("1. Memory-bound: Higher occupancy helps hide latency\n");
    printf("2. Compute-bound: Occupancy matters less\n");
    printf("3. Cache-friendly: Lower occupancy with better reuse can win\n");
    printf("4. ALWAYS BENCHMARK - don't assume higher is better!\n");
    printf("5. Target: 50%% occupancy is often enough\n");
    printf("6. Watch for: register spilling when forcing high occupancy\n");
    
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_data;
    
    return 0;
}
