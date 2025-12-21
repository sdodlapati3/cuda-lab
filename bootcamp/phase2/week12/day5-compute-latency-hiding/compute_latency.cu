/**
 * compute_latency.cu - Hiding compute latency
 * 
 * Learning objectives:
 * - Break dependency chains
 * - Use multiple accumulators
 * - Loop unrolling effects
 */

#include <cuda_runtime.h>
#include <cstdio>

// Long dependency chain
__global__ void chain_reduction(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum = sum + in[i];  // Each depends on previous sum
    }
    
    out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

// Multiple accumulators - breaks chain
__global__ void multi_accum_reduction(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // 4 independent accumulators
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
    int stride4 = stride * 4;
    for (int i = idx; i < n; i += stride4) {
        if (i < n) sum0 += in[i];
        if (i + stride < n) sum1 += in[i + stride];
        if (i + 2*stride < n) sum2 += in[i + 2*stride];
        if (i + 3*stride < n) sum3 += in[i + 3*stride];
    }
    
    out[blockIdx.x * blockDim.x + threadIdx.x] = sum0 + sum1 + sum2 + sum3;
}

// Dependency chain in compute
__global__ void chained_compute(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float x = data[idx];
    
    // Long chain: each op depends on previous
    for (int i = 0; i < 20; i++) {
        x = x * 1.01f;
        x = x + 0.01f;
        x = x * 0.99f;
        x = x - 0.01f;
    }
    
    data[idx] = x;
}

// Independent computations
__global__ void independent_compute(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float x = data[idx];
    
    // Independent ops that can execute in parallel
    for (int i = 0; i < 20; i++) {
        float a = x * 1.01f;  // Independent
        float b = x + 0.01f;  // Independent  
        float c = x * 0.99f;  // Independent
        float d = x - 0.01f;  // Independent
        x = a + b + c + d;
    }
    
    data[idx] = x;
}

// Matrix chain - bad (column access)
__global__ void chain_matmul_bad(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum = sum + A[row * n + k] * B[k * n + col];  // Accumulator dependency
        }
        C[row * n + col] = sum;
    }
}

// Matrix with unrolled inner loop
__global__ void chain_matmul_unrolled(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        
        int k;
        for (k = 0; k + 3 < n; k += 4) {
            sum0 += A[row * n + k] * B[k * n + col];
            sum1 += A[row * n + k+1] * B[(k+1) * n + col];
            sum2 += A[row * n + k+2] * B[(k+2) * n + col];
            sum3 += A[row * n + k+3] * B[(k+3) * n + col];
        }
        
        // Remainder
        for (; k < n; k++) {
            sum0 += A[row * n + k] * B[k * n + col];
        }
        
        C[row * n + col] = sum0 + sum1 + sum2 + sum3;
    }
}

int main() {
    printf("=== Compute Latency Hiding Demo ===\n\n");
    
    // Test 1: Reduction with dependency chains
    {
        const int N = 1 << 24;
        float *d_in, *d_out;
        cudaMalloc(&d_in, N * sizeof(float));
        cudaMalloc(&d_out, N * sizeof(float));
        
        float* h_data = new float[N];
        for (int i = 0; i < N; i++) h_data[i] = 1.0f;
        cudaMemcpy(d_in, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        printf("=== Reduction: Dependency Chain vs Multiple Accumulators ===\n\n");
        printf("%-25s | %10s | %10s\n", "Method", "Time (ms)", "Speedup");
        printf("--------------------------------------------------\n");
        
        int blocks = 256;
        int threads = 256;
        
        // Chained
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            chain_reduction<<<blocks, threads>>>(d_in, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float t1;
        cudaEventElapsedTime(&t1, start, stop);
        printf("%-25s | %10.4f | %9.2fx\n", "Single accumulator", t1/100, 1.0);
        
        // Multi-accumulator
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            multi_accum_reduction<<<blocks, threads>>>(d_in, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float t2;
        cudaEventElapsedTime(&t2, start, stop);
        printf("%-25s | %10.4f | %9.2fx\n", "4 accumulators", t2/100, t1/t2);
        
        cudaFree(d_in);
        cudaFree(d_out);
        delete[] h_data;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // Test 2: Compute chains
    {
        const int N = 1 << 20;
        float *d_data;
        cudaMalloc(&d_data, N * sizeof(float));
        
        float* h_data = new float[N];
        for (int i = 0; i < N; i++) h_data[i] = 1.0f;
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        printf("\n=== Compute: Chained vs Independent Ops ===\n\n");
        printf("%-25s | %10s | %10s\n", "Method", "Time (ms)", "Speedup");
        printf("--------------------------------------------------\n");
        
        int blocks = 256;
        int threads = 256;
        
        // Chained
        cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            chained_compute<<<blocks, threads>>>(d_data, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float t1;
        cudaEventElapsedTime(&t1, start, stop);
        printf("%-25s | %10.4f | %9.2fx\n", "Chained operations", t1/100, 1.0);
        
        // Independent
        cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            independent_compute<<<blocks, threads>>>(d_data, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float t2;
        cudaEventElapsedTime(&t2, start, stop);
        printf("%-25s | %10.4f | %9.2fx\n", "Independent operations", t2/100, t1/t2);
        
        cudaFree(d_data);
        delete[] h_data;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // Test 3: Matrix multiply with unrolling
    {
        const int N = 512;
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, N * N * sizeof(float));
        cudaMalloc(&d_B, N * N * sizeof(float));
        cudaMalloc(&d_C, N * N * sizeof(float));
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        printf("\n=== MatMul: Loop Unrolling ===\n\n");
        printf("%-25s | %10s | %10s\n", "Method", "Time (ms)", "Speedup");
        printf("--------------------------------------------------\n");
        
        dim3 threads(16, 16);
        dim3 blocks((N + 15) / 16, (N + 15) / 16);
        
        // Basic
        cudaEventRecord(start);
        for (int i = 0; i < 50; i++) {
            chain_matmul_bad<<<blocks, threads>>>(d_A, d_B, d_C, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float t1;
        cudaEventElapsedTime(&t1, start, stop);
        printf("%-25s | %10.4f | %9.2fx\n", "Basic inner loop", t1/50, 1.0);
        
        // Unrolled
        cudaEventRecord(start);
        for (int i = 0; i < 50; i++) {
            chain_matmul_unrolled<<<blocks, threads>>>(d_A, d_B, d_C, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float t2;
        cudaEventElapsedTime(&t2, start, stop);
        printf("%-25s | %10.4f | %9.2fx\n", "Unrolled x4", t2/50, t1/t2);
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    printf("\n=== Key Takeaways ===\n\n");
    printf("1. Single accumulator = long dependency chain\n");
    printf("2. Multiple accumulators = independent ops = parallelism\n");
    printf("3. Loop unrolling exposes more ILP\n");
    printf("4. Think about data dependencies in YOUR code\n");
    printf("5. Profile with: ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct\n");
    
    return 0;
}
