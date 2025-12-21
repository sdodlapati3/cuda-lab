/**
 * dot_product.cu - Reduction applied: dot product
 * 
 * Learning objectives:
 * - Apply reduction to real problem
 * - Compare with cuBLAS
 * - Fused vs unfused approach
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 256
#define FULL_MASK 0xFFFFFFFF

// Warp-level reduction helper
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// Unfused: multiply then reduce separately
__global__ void multiply_kernel(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void reduce_sum(const float* input, float* output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += input[i];
    }
    
    sum = warp_reduce_sum(sum);
    
    __shared__ float warp_sums[8];
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        sum = (tid < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (tid == 0) {
            atomicAdd(output, sum);
        }
    }
}

// Fused: multiply and reduce in one kernel
__global__ void fused_dot_product(const float* a, const float* b, float* result, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    // Grid-stride loop with fused multiply-add
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += a[i] * b[i];
    }
    
    // Warp reduction
    sum = warp_reduce_sum(sum);
    
    // Block reduction
    __shared__ float warp_sums[8];
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        sum = (tid < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (tid == 0) {
            atomicAdd(result, sum);
        }
    }
}

double dot_cpu(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)a[i] * (double)b[i];
    }
    return sum;
}

int main() {
    printf("=== Dot Product: Reduction in Action ===\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    
    float peak_bw = prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) * 2 / 1e9;
    printf("Peak bandwidth: %.0f GB/s\n\n", peak_bw);
    
    const int N = 1 << 24;
    const int TRIALS = 100;
    size_t bytes = N * sizeof(float);
    
    printf("Vector size: %d elements (%.1f MB each)\n\n", N, bytes / 1e6);
    
    // Allocate and initialize
    float* h_a = new float[N];
    float* h_b = new float[N];
    
    for (int i = 0; i < N; i++) {
        h_a[i] = (rand() / (float)RAND_MAX) * 2 - 1;
        h_b[i] = (rand() / (float)RAND_MAX) * 2 - 1;
    }
    
    double cpu_result = dot_cpu(h_a, h_b, N);
    printf("CPU result: %.6f\n\n", cpu_result);
    
    float *d_a, *d_b, *d_temp, *d_result;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_temp, bytes);
    cudaMalloc(&d_result, sizeof(float));
    
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    int blocks = 256;
    
    printf("%-20s %-12s %-15s %-12s %-12s\n",
           "Method", "Time(ms)", "Bandwidth(GB/s)", "Efficiency", "Error");
    printf("------------------------------------------------------------------------\n");
    
    // Method 1: Unfused (multiply + reduce)
    float gpu_result;
    
    cudaMemset(d_result, 0, sizeof(float));
    multiply_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_temp, d_a, d_b, N);
    reduce_sum<<<blocks, BLOCK_SIZE>>>(d_temp, d_result, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        cudaMemset(d_result, 0, sizeof(float));
        multiply_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_temp, d_a, d_b, N);
        reduce_sum<<<blocks, BLOCK_SIZE>>>(d_temp, d_result, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    ms /= TRIALS;
    
    cudaMemcpy(&gpu_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Bandwidth: read A+B, write temp, read temp = 4N floats = 16N bytes
    float bw = 4.0f * bytes / ms / 1e6;
    float error = fabsf(gpu_result - cpu_result) / fabsf(cpu_result);
    printf("%-20s %-12.3f %-15.1f %-12.1f%% %-12.2e\n",
           "Unfused (2 kernels)", ms, bw, 100.0f * bw / peak_bw, error);
    
    // Method 2: Fused
    cudaMemset(d_result, 0, sizeof(float));
    fused_dot_product<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_result, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        cudaMemset(d_result, 0, sizeof(float));
        fused_dot_product<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_result, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    ms /= TRIALS;
    
    cudaMemcpy(&gpu_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Bandwidth: read A+B only = 2N floats = 8N bytes
    bw = 2.0f * bytes / ms / 1e6;
    error = fabsf(gpu_result - cpu_result) / fabsf(cpu_result);
    printf("%-20s %-12.3f %-15.1f %-12.1f%% %-12.2e\n",
           "Fused (1 kernel)", ms, bw, 100.0f * bw / peak_bw, error);
    
    // Method 3: cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cublasSdot(handle, N, d_a, 1, d_b, 1, &gpu_result);  // Warmup
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        cublasSdot(handle, N, d_a, 1, d_b, 1, &gpu_result);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    ms /= TRIALS;
    
    bw = 2.0f * bytes / ms / 1e6;
    error = fabsf(gpu_result - cpu_result) / fabsf(cpu_result);
    printf("%-20s %-12.3f %-15.1f %-12.1f%% %-12.2e\n",
           "cuBLAS", ms, bw, 100.0f * bw / peak_bw, error);
    
    cublasDestroy(handle);
    
    printf("\n=== Analysis ===\n");
    printf("Dot product = multiply + reduce\n");
    printf("\nUnfused approach:\n");
    printf("  - Read A, B → Write temp (3N)\n");
    printf("  - Read temp (1N)\n");
    printf("  - Total: 4N memory operations\n");
    printf("\nFused approach:\n");
    printf("  - Read A, B (2N)\n");
    printf("  - Multiply in registers, reduce to shared\n");
    printf("  - Total: 2N memory operations (50%% reduction)\n");
    printf("\nFusion halves memory traffic!\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_temp);
    cudaFree(d_result);
    delete[] h_a;
    delete[] h_b;
    
    return 0;
}
