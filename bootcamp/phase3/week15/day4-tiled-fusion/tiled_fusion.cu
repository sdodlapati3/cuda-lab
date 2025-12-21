/**
 * tiled_fusion.cu - Fuse operations in shared memory tiles
 * 
 * Learning objectives:
 * - Tiled matrix multiply + bias
 * - Tiled matmul + ReLU
 * - Row-wise softmax in tiles
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>

#define TILE_SIZE 32

// ============================================================================
// Unfused: MatMul then Add Bias
// ============================================================================

__global__ void matmul_kernel(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void add_bias_kernel(float* C, const float* bias, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        C[row * N + col] += bias[col];
    }
}

__global__ void relu_matrix_kernel(float* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        int idx = row * N + col;
        C[idx] = fmaxf(0.0f, C[idx]);
    }
}

// ============================================================================
// Fused: MatMul + Bias + ReLU in one kernel
// ============================================================================

__global__ void matmul_bias_relu_kernel(const float* A, const float* B, 
                                         const float* bias, float* C,
                                         int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        // Fused: add bias and ReLU in same write
        float val = sum + bias[col];
        C[row * N + col] = fmaxf(0.0f, val);
    }
}

// ============================================================================
// Fused Row-wise Softmax
// ============================================================================

// Each block handles one row
__global__ void fused_row_softmax(const float* input, float* output, int N) {
    extern __shared__ float sdata[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_in = input + row * N;
    float* row_out = output + row * N;
    
    // Step 1: Find max (reduction)
    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, row_in[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();
    
    // Step 2: Compute exp and sum (fused)
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float exp_val = expf(row_in[i] - row_max);
        row_out[i] = exp_val;  // Store temporarily
        local_sum += exp_val;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    
    // Step 3: Normalize
    for (int i = tid; i < N; i += blockDim.x) {
        row_out[i] /= row_sum;
    }
}

int main() {
    printf("=== Tiled Fusion Demo ===\n\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========================================================================
    // Test 1: MatMul + Bias + ReLU
    // ========================================================================
    {
        printf("1. MatMul + Bias + ReLU\n");
        printf("─────────────────────────────────────────\n");
        
        const int M = 1024, N = 1024, K = 1024;
        
        float *d_A, *d_B, *d_C, *d_C_fused, *d_bias;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));
        cudaMalloc(&d_C_fused, M * N * sizeof(float));
        cudaMalloc(&d_bias, N * sizeof(float));
        
        // Initialize
        float* h_data = new float[M * K];
        for (int i = 0; i < M * K; i++) h_data[i] = 0.01f;
        cudaMemcpy(d_A, h_data, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_data, K * N * sizeof(float), cudaMemcpyHostToDevice);
        for (int i = 0; i < N; i++) h_data[i] = -0.5f;  // Negative bias to test ReLU
        cudaMemcpy(d_bias, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        
        // Unfused: 3 kernels
        cudaEventRecord(start);
        for (int i = 0; i < 10; i++) {
            matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            add_bias_kernel<<<grid, block>>>(d_C, d_bias, M, N);
            relu_matrix_kernel<<<grid, block>>>(d_C, M, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float unfused_ms;
        cudaEventElapsedTime(&unfused_ms, start, stop);
        unfused_ms /= 10;
        
        // Fused: 1 kernel
        cudaEventRecord(start);
        for (int i = 0; i < 10; i++) {
            matmul_bias_relu_kernel<<<grid, block>>>(d_A, d_B, d_bias, d_C_fused, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float fused_ms;
        cudaEventElapsedTime(&fused_ms, start, stop);
        fused_ms /= 10;
        
        // Verify
        float* h_unfused = new float[100];
        float* h_fused = new float[100];
        cudaMemcpy(h_unfused, d_C, 100 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fused, d_C_fused, 100 * sizeof(float), cudaMemcpyDeviceToHost);
        
        bool correct = true;
        for (int i = 0; i < 100; i++) {
            if (fabsf(h_unfused[i] - h_fused[i]) > 1e-3) correct = false;
        }
        
        printf("   Matrix size: %dx%d @ %dx%d\n", M, N, K, N);
        printf("   Unfused (3 kernels): %.2f ms\n", unfused_ms);
        printf("   Fused   (1 kernel):  %.2f ms\n", fused_ms);
        printf("   Speedup: %.2fx\n", unfused_ms / fused_ms);
        printf("   Verification: %s\n\n", correct ? "PASSED" : "FAILED");
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_C_fused);
        cudaFree(d_bias);
        delete[] h_data;
        delete[] h_unfused;
        delete[] h_fused;
    }
    
    // ========================================================================
    // Test 2: Fused Row-wise Softmax
    // ========================================================================
    {
        printf("2. Fused Row-wise Softmax\n");
        printf("─────────────────────────────────────────\n");
        
        const int ROWS = 1024;
        const int COLS = 1024;
        const int BLOCK = 256;
        
        float *d_in, *d_out;
        cudaMalloc(&d_in, ROWS * COLS * sizeof(float));
        cudaMalloc(&d_out, ROWS * COLS * sizeof(float));
        
        // Initialize
        float* h_in = new float[ROWS * COLS];
        for (int i = 0; i < ROWS * COLS; i++) {
            h_in[i] = 0.01f * (i % 100);
        }
        cudaMemcpy(d_in, h_in, ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);
        
        // Warmup
        fused_row_softmax<<<ROWS, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, COLS);
        cudaDeviceSynchronize();
        
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            fused_row_softmax<<<ROWS, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, COLS);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= 100;
        
        // Verify one row sums to 1
        float* h_out = new float[COLS];
        cudaMemcpy(h_out, d_out, COLS * sizeof(float), cudaMemcpyDeviceToHost);
        float sum = 0;
        for (int i = 0; i < COLS; i++) sum += h_out[i];
        
        printf("   Matrix: %d rows x %d cols\n", ROWS, COLS);
        printf("   Fused softmax: %.3f ms\n", ms);
        printf("   Row sum check: %.6f (should be 1.0)\n\n", sum);
        
        cudaFree(d_in);
        cudaFree(d_out);
        delete[] h_in;
        delete[] h_out;
    }
    
    printf("=== Key Points ===\n\n");
    printf("1. Tiled fusion: load once, compute multiple operations\n");
    printf("2. MatMul + bias + activation is a common pattern\n");
    printf("3. Row-wise softmax: fuse max, exp, sum in one kernel\n");
    printf("4. Shared memory enables data reuse across operations\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
