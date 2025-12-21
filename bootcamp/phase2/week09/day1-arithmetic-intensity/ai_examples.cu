/**
 * ai_examples.cu - Calculate AI for various kernel patterns
 * 
 * Hands-on practice calculating arithmetic intensity
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Pattern 1: Pointwise operations (vary compute per element)
template<int OPS_PER_ELEMENT>
__global__ void pointwise_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        // Simulate more compute
        #pragma unroll
        for (int i = 0; i < OPS_PER_ELEMENT; i++) {
            val = val * 1.01f + 0.01f;  // 2 FLOPs
        }
        out[idx] = val;
    }
}

// Pattern 2: Stencil (data reuse across threads)
__global__ void stencil_3pt(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < n - 1) {
        // 3 loads but only 1 unique per thread (neighbors shared)
        // Effective: 4 bytes load, 4 bytes store = 8 bytes
        // 2 adds, 1 multiply = 3 FLOPs
        // AI ≈ 3/8 = 0.375 (with perfect caching)
        out[idx] = 0.25f * in[idx-1] + 0.5f * in[idx] + 0.25f * in[idx+1];
    }
}

// Pattern 3: Reduction (low AI due to read-only)
__global__ void reduction_step(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// Pattern 4: Matrix multiply tile (high AI with reuse)
#define TILE_SIZE 16

__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < N / TILE_SIZE; t++) {
        // Load tiles
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();
        
        // Compute: TILE_SIZE FMAs = 2*TILE_SIZE FLOPs
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}

void benchmark_ai_patterns() {
    printf("=== Arithmetic Intensity Patterns ===\n\n");
    
    const int N = 1 << 22;  // 4M elements
    const int TRIALS = 50;
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemset(d_in, 1, N * sizeof(float));
    
    int blocks = (N + 255) / 256;
    int threads = 256;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    printf("%-30s %10s %10s %10s %10s\n", 
           "Pattern", "AI", "GB/s", "GFLOPS", "Bound");
    printf("-----------------------------------------------------------------------\n");
    
    // Pattern 1a: 1 op per element
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        pointwise_kernel<1><<<blocks, threads>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    float bytes = N * 8.0f;  // 1 read + 1 write
    float flops = N * 2.0f;  // 1 multiply + 1 add
    float ai = flops / bytes;
    float bw = (bytes * TRIALS) / (ms / 1000) / 1e9;
    float gflops = (flops * TRIALS) / (ms / 1000) / 1e9;
    printf("%-30s %10.3f %10.1f %10.1f %10s\n", 
           "Pointwise (1 op)", ai, bw, gflops, "memory");
    
    // Pattern 1b: 10 ops per element
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        pointwise_kernel<10><<<blocks, threads>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    flops = N * 20.0f;  // 10 * 2 FLOPs
    ai = flops / bytes;
    bw = (bytes * TRIALS) / (ms / 1000) / 1e9;
    gflops = (flops * TRIALS) / (ms / 1000) / 1e9;
    printf("%-30s %10.3f %10.1f %10.1f %10s\n", 
           "Pointwise (10 ops)", ai, bw, gflops, "memory");
    
    // Pattern 1c: 100 ops per element
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        pointwise_kernel<100><<<blocks, threads>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    flops = N * 200.0f;  // 100 * 2 FLOPs
    ai = flops / bytes;
    bw = (bytes * TRIALS) / (ms / 1000) / 1e9;
    gflops = (flops * TRIALS) / (ms / 1000) / 1e9;
    printf("%-30s %10.3f %10.1f %10.1f %10s\n", 
           "Pointwise (100 ops)", ai, bw, gflops, "transitioning");
    
    // Pattern 2: Stencil
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        stencil_3pt<<<blocks, threads>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    bytes = N * 8.0f;  // Approximate with caching
    flops = N * 5.0f;  // 3 multiplies + 2 adds
    ai = flops / bytes;
    bw = (bytes * TRIALS) / (ms / 1000) / 1e9;
    gflops = (flops * TRIALS) / (ms / 1000) / 1e9;
    printf("%-30s %10.3f %10.1f %10.1f %10s\n", 
           "3-point Stencil", ai, bw, gflops, "memory");
    
    // Pattern 3: Reduction
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        reduction_step<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    bytes = N * 4.0f;  // Just reads
    flops = N * 1.0f;  // 1 add per element
    ai = flops / bytes;
    bw = (bytes * TRIALS) / (ms / 1000) / 1e9;
    gflops = (flops * TRIALS) / (ms / 1000) / 1e9;
    printf("%-30s %10.3f %10.1f %10.1f %10s\n", 
           "Reduction", ai, bw, gflops, "memory");
    
    // Pattern 4: Matrix multiply (smaller for timing)
    const int M = 1024;
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * M * sizeof(float));
    cudaMalloc(&d_B, M * M * sizeof(float));
    cudaMalloc(&d_C, M * M * sizeof(float));
    
    dim3 gridDim(M / TILE_SIZE, M / TILE_SIZE);
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    // For tiled matmul: 2*M^3 FLOPs, ~3*M^2*4 bytes (with tile reuse)
    flops = 2.0f * M * M * M;
    bytes = 3.0f * M * M * 4;  // Simplified (ignores tile reuse complexity)
    ai = flops / bytes;
    bw = (bytes * TRIALS) / (ms / 1000) / 1e9;
    gflops = (flops * TRIALS) / (ms / 1000) / 1e9;
    printf("%-30s %10.1f %10.1f %10.1f %10s\n", 
           "Tiled MatMul 1024x1024", ai, bw, gflops, "compute");
    
    printf("\n");
    printf("Observation: As AI increases, kernel transitions from\n");
    printf("             memory-bound to compute-bound.\n");
    printf("             Most real kernels have low AI and are memory-bound.\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    benchmark_ai_patterns();
    return 0;
}
