/**
 * analyze_kernels.cu - Roofline analysis of common kernels
 * 
 * Learning objectives:
 * - Calculate theoretical AI for real kernels
 * - Measure achieved performance
 * - Position on roofline and interpret
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

// Vector Add: AI = 2 FLOPS / 12 bytes = 0.17
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // 1 FLOP, 12 bytes
    }
}

// SAXPY: AI = 2 FLOPS / 12 bytes = 0.17
__global__ void saxpy(float a, const float* x, const float* y, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a * x[idx] + y[idx];  // 2 FLOPs (1 FMA), 12 bytes
    }
}

// Reduction: AI depends on N and implementation
// First pass: read N values, write 1 per block
// AI = N FLOPS / N*4 bytes = 0.25 per pass
template<int BLOCK_SIZE>
__global__ void reduce_sum(const float* in, float* out, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // Load with first reduction
    float sum = 0;
    if (idx < n) sum = in[idx];
    if (idx + blockDim.x < n) sum += in[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduce in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// Stencil (1D 3-point): AI = 2 FLOPS / 12 bytes = 0.17 (naive)
// With caching: AI = 2 FLOPS / 4 bytes = 0.5
__global__ void stencil_1d(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < n - 1) {
        out[idx] = 0.25f * in[idx-1] + 0.5f * in[idx] + 0.25f * in[idx+1];
    }
}

// Matrix multiply (naive): AI = 2*N^3 / 3*N^2*4 = N/6
// For N=1024: AI ≈ 170 (very compute-bound)
// For small tiles, AI is lower
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];  // 2 FLOPs
        }
        C[row * N + col] = sum;
    }
}

// Transpose: AI = 0 (just moving data), bandwidth matters
__global__ void transpose(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        out[x * height + y] = in[y * width + x];
    }
}

struct KernelProfile {
    const char* name;
    float theoretical_ai;
    float measured_gflops;
    float measured_bw_gbps;
    float measured_ai;
    const char* bound_type;
};

int main() {
    printf("=== Kernel Roofline Analysis ===\n\n");
    
    // Get device info for ceilings
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    
    float gpu_clock_ghz = prop.clockRate / 1e6;
    int cuda_cores_per_sm = (prop.major == 8) ? ((prop.minor == 0) ? 64 : 128) : 64;
    float peak_gflops = 2.0f * gpu_clock_ghz * prop.multiProcessorCount * cuda_cores_per_sm * 1000;
    
    printf("Peak Compute: ~%.0f GFLOPS\n", peak_gflops);
    
    // Measure peak BW
    const int BW_SIZE = 1 << 26;  // 64M elements
    float *d_bw_in, *d_bw_out;
    cudaMalloc(&d_bw_in, BW_SIZE * sizeof(float));
    cudaMalloc(&d_bw_out, BW_SIZE * sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        cudaMemcpy(d_bw_out, d_bw_in, BW_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float peak_bw = 2.0f * BW_SIZE * sizeof(float) * 10 / (ms / 1000) / 1e9;
    printf("Peak Bandwidth: ~%.0f GB/s\n", peak_bw);
    
    float ridge_ai = peak_gflops / peak_bw;
    printf("Ridge Point: %.1f FLOPS/byte\n\n", ridge_ai);
    
    cudaFree(d_bw_in);
    cudaFree(d_bw_out);
    
    const int TRIALS = 20;
    KernelProfile profiles[6];
    
    // ========== Vector Add ==========
    const int VEC_N = 1 << 24;
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, VEC_N * sizeof(float));
    cudaMalloc(&d_b, VEC_N * sizeof(float));
    cudaMalloc(&d_c, VEC_N * sizeof(float));
    
    int threads = 256;
    int blocks = (VEC_N + threads - 1) / threads;
    
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, VEC_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    float flops = (float)VEC_N * 1.0f;
    float bytes = (float)VEC_N * 12.0f;  // 2 read + 1 write
    profiles[0] = {
        "vector_add",
        1.0f / 12.0f,  // Theoretical AI
        flops * TRIALS / (ms / 1000) / 1e9,  // GFLOPS
        bytes * TRIALS / (ms / 1000) / 1e9,  // GB/s
        0, "memory"
    };
    profiles[0].measured_ai = profiles[0].measured_gflops * 1e9 / (profiles[0].measured_bw_gbps * 1e9);
    
    // ========== SAXPY ==========
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        saxpy<<<blocks, threads>>>(2.0f, d_a, d_b, d_c, VEC_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    flops = (float)VEC_N * 2.0f;  // 1 FMA = 2 FLOPs
    bytes = (float)VEC_N * 12.0f;
    profiles[1] = {
        "saxpy",
        2.0f / 12.0f,
        flops * TRIALS / (ms / 1000) / 1e9,
        bytes * TRIALS / (ms / 1000) / 1e9,
        0, "memory"
    };
    profiles[1].measured_ai = profiles[1].measured_gflops * 1e9 / (profiles[1].measured_bw_gbps * 1e9);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // ========== Reduction ==========
    const int RED_N = 1 << 24;
    float *d_in, *d_out;
    cudaMalloc(&d_in, RED_N * sizeof(float));
    cudaMalloc(&d_out, (RED_N / 512) * sizeof(float));
    
    float* h_in = new float[RED_N];
    for (int i = 0; i < RED_N; i++) h_in[i] = 1.0f;
    cudaMemcpy(d_in, h_in, RED_N * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        reduce_sum<256><<<RED_N / 512, 256>>>(d_in, d_out, RED_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    // Reduction: N-1 adds for N elements (first pass)
    flops = (float)RED_N * 1.0f;
    bytes = (float)RED_N * 4.0f + (RED_N / 512) * 4.0f;  // Read all + write partial
    profiles[2] = {
        "reduction",
        flops / bytes,
        flops * TRIALS / (ms / 1000) / 1e9,
        bytes * TRIALS / (ms / 1000) / 1e9,
        0, "memory"
    };
    profiles[2].measured_ai = profiles[2].measured_gflops * 1e9 / (profiles[2].measured_bw_gbps * 1e9);
    
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    
    // ========== Stencil ==========
    const int STEN_N = 1 << 24;
    cudaMalloc(&d_in, STEN_N * sizeof(float));
    cudaMalloc(&d_out, STEN_N * sizeof(float));
    
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        stencil_1d<<<(STEN_N + threads - 1) / threads, threads>>>(d_in, d_out, STEN_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    flops = (float)STEN_N * 4.0f;  // 3 muls + adds
    bytes = (float)STEN_N * 8.0f;  // ~1 read + 1 write (cache helps)
    profiles[3] = {
        "stencil_1d",
        4.0f / 8.0f,  // With caching
        flops * TRIALS / (ms / 1000) / 1e9,
        bytes * TRIALS / (ms / 1000) / 1e9,
        0, "memory"
    };
    profiles[3].measured_ai = profiles[3].measured_gflops * 1e9 / (profiles[3].measured_bw_gbps * 1e9);
    
    cudaFree(d_in);
    cudaFree(d_out);
    
    // ========== Matrix Multiply ==========
    const int MAT_N = 1024;
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, MAT_N * MAT_N * sizeof(float));
    cudaMalloc(&d_B, MAT_N * MAT_N * sizeof(float));
    cudaMalloc(&d_C, MAT_N * MAT_N * sizeof(float));
    
    dim3 mm_threads(16, 16);
    dim3 mm_blocks((MAT_N + 15) / 16, (MAT_N + 15) / 16);
    
    cudaEventRecord(start);
    for (int i = 0; i < 5; i++) {  // Fewer trials (slow)
        matmul_naive<<<mm_blocks, mm_threads>>>(d_A, d_B, d_C, MAT_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    flops = 2.0f * MAT_N * MAT_N * MAT_N;  // 2*N^3
    bytes = 3.0f * MAT_N * MAT_N * sizeof(float);  // Naive: 3*N^2 per element access
    // But naive reads each element N times, so actual:
    float actual_bytes = (2.0f * MAT_N * MAT_N * MAT_N + MAT_N * MAT_N) * sizeof(float);
    profiles[4] = {
        "matmul_naive",
        (float)MAT_N / 6.0f,  // Theoretical optimal
        flops * 5 / (ms / 1000) / 1e9,
        actual_bytes * 5 / (ms / 1000) / 1e9,
        0, "memory"  // Naive is actually memory-bound
    };
    profiles[4].measured_ai = profiles[4].measured_gflops * 1e9 / (profiles[4].measured_bw_gbps * 1e9);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // ========== Transpose ==========
    const int TRANS_W = 4096, TRANS_H = 4096;
    cudaMalloc(&d_in, TRANS_W * TRANS_H * sizeof(float));
    cudaMalloc(&d_out, TRANS_W * TRANS_H * sizeof(float));
    
    dim3 trans_threads(16, 16);
    dim3 trans_blocks((TRANS_W + 15) / 16, (TRANS_H + 15) / 16);
    
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        transpose<<<trans_blocks, trans_threads>>>(d_in, d_out, TRANS_W, TRANS_H);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    flops = 0;  // No compute
    bytes = 2.0f * TRANS_W * TRANS_H * sizeof(float);
    profiles[5] = {
        "transpose",
        0.0f,  // Pure memory
        0.0f,
        bytes * TRIALS / (ms / 1000) / 1e9,
        0, "memory"
    };
    
    cudaFree(d_in);
    cudaFree(d_out);
    
    // Print analysis
    printf("=== Kernel Analysis ===\n\n");
    printf("%-15s %10s %10s %10s %12s %s\n", 
           "Kernel", "Theo AI", "Meas AI", "GFLOPS", "GB/s", "Bound");
    printf("-------------------------------------------------------------------------\n");
    
    for (int i = 0; i < 6; i++) {
        printf("%-15s %10.2f %10.2f %10.1f %12.1f %s\n",
               profiles[i].name,
               profiles[i].theoretical_ai,
               profiles[i].measured_ai,
               profiles[i].measured_gflops,
               profiles[i].measured_bw_gbps,
               profiles[i].bound_type);
    }
    
    printf("\n=== Interpretation ===\n\n");
    printf("1. vector_add, saxpy, reduction, stencil: Low AI → memory-bound\n");
    printf("   - Optimization: improve memory access patterns, use shared memory\n");
    printf("\n");
    printf("2. matmul_naive: High theoretical AI but measured is lower\n");
    printf("   - Problem: redundant memory access (no tiling)\n");
    printf("   - Optimization: use shared memory tiling (Week 6)\n");
    printf("\n");
    printf("3. transpose: AI = 0 (no FLOPs)\n");
    printf("   - Pure bandwidth test\n");
    printf("   - Optimization: coalesced access patterns\n");
    printf("\n");
    printf("Ridge point AI = %.1f FLOPS/byte\n", ridge_ai);
    printf("Kernels below this are memory-bound.\n");
    printf("Kernels above this SHOULD be compute-bound (if efficient).\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
