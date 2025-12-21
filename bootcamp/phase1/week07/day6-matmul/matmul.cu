/**
 * matmul.cu - Matrix multiplication optimization
 * 
 * Learning objectives:
 * - Tiled matrix multiply with shared memory
 * - Understanding compute vs memory bound
 * - Comparison with cuBLAS
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#define TILE_SIZE 16

// Version 1: Naive - each thread computes one element
__global__ void matmul_naive(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Version 2: Tiled with shared memory
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A into shared memory
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile of B into shared memory
        if (t * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Version 3: Tiled with bank conflict avoidance
__global__ void matmul_tiled_padded(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    // Pad shared memory to avoid bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 padding
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (t * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Version 4: Register tiling (each thread computes 2x2 block)
#define TILE_M 32
#define TILE_N 32
#define TILE_K 16

__global__ void matmul_register_tiled(const float* A, const float* B, float* C,
                                       int M, int N, int K) {
    // Each thread computes 2x2 output elements
    __shared__ float As[TILE_K][TILE_M];
    __shared__ float Bs[TILE_K][TILE_N];
    
    int tx = threadIdx.x;  // 0-15
    int ty = threadIdx.y;  // 0-15
    
    int row = blockIdx.y * TILE_M + ty * 2;
    int col = blockIdx.x * TILE_N + tx * 2;
    
    float sum00 = 0, sum01 = 0, sum10 = 0, sum11 = 0;
    
    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
        // Collaborative loading - each thread loads 2 elements
        int aRow = blockIdx.y * TILE_M + ty * 2;
        int aCol = t * TILE_K + tx;
        
        if (aRow < M && aCol < K) As[tx][ty * 2] = A[aRow * K + aCol];
        else As[tx][ty * 2] = 0;
        
        if (aRow + 1 < M && aCol < K) As[tx][ty * 2 + 1] = A[(aRow + 1) * K + aCol];
        else As[tx][ty * 2 + 1] = 0;
        
        int bRow = t * TILE_K + ty;
        int bCol = blockIdx.x * TILE_N + tx * 2;
        
        if (bRow < K && bCol < N) Bs[ty][tx * 2] = B[bRow * N + bCol];
        else Bs[ty][tx * 2] = 0;
        
        if (bRow < K && bCol + 1 < N) Bs[ty][tx * 2 + 1] = B[bRow * N + bCol + 1];
        else Bs[ty][tx * 2 + 1] = 0;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float a0 = As[k][ty * 2];
            float a1 = As[k][ty * 2 + 1];
            float b0 = Bs[k][tx * 2];
            float b1 = Bs[k][tx * 2 + 1];
            
            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) C[row * N + col] = sum00;
    if (row < M && col + 1 < N) C[row * N + col + 1] = sum01;
    if (row + 1 < M && col < N) C[(row + 1) * N + col] = sum10;
    if (row + 1 < M && col + 1 < N) C[(row + 1) * N + col + 1] = sum11;
}

void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

float max_error(const float* a, const float* b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]) / (fabsf(b[i]) + 1e-6f);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

int main() {
    printf("=== Matrix Multiplication Optimization ===\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    
    // Peak TFLOPS (approximate for A100)
    float peak_tflops = 19.5f;  // FP32 tensor cores
    printf("Approximate peak FP32: %.1f TFLOPS\n\n", peak_tflops);
    
    // Matrix dimensions
    const int M = 2048, N = 2048, K = 2048;
    const int TRIALS = 10;
    
    printf("Matrix dimensions: A[%d×%d] × B[%d×%d] = C[%d×%d]\n", M, K, K, N, M, N);
    printf("FLOPs per multiply: %lld (2MNK)\n\n", 2LL * M * N * K);
    
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    
    // Allocate host memory
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];
    float* h_C_ref = new float[M * N];
    
    // Initialize
    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (rand() / (float)RAND_MAX) * 2 - 1;
    for (int i = 0; i < K * N; i++) h_B[i] = (rand() / (float)RAND_MAX) * 2 - 1;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    printf("%-25s %-12s %-12s %-12s\n", "Version", "Time(ms)", "GFLOPS", "% Peak");
    printf("----------------------------------------------------------------\n");
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    auto benchmark = [&](const char* name, auto kernel_fn, dim3 grid, dim3 block) {
        // Warmup
        kernel_fn<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        
        cudaEventRecord(start);
        for (int t = 0; t < TRIALS; t++) {
            kernel_fn<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        ms /= TRIALS;
        
        float gflops = 2.0f * M * N * K / ms / 1e6;
        float pct_peak = 100.0f * gflops / (peak_tflops * 1000);
        
        printf("%-25s %-12.2f %-12.1f %-12.1f%%\n", name, ms, gflops, pct_peak);
        
        // Verify first run
        cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);
        return h_C;
    };
    
    // V1: Naive
    benchmark("V1: Naive", matmul_naive, grid, block);
    
    // V2: Tiled
    benchmark("V2: Tiled (16x16)", matmul_tiled, grid, block);
    
    // V3: Tiled + padded
    benchmark("V3: Tiled + padded", matmul_tiled_padded, grid, block);
    
    // V4: Register tiling
    dim3 block4(16, 16);
    dim3 grid4((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    float* result_v4 = benchmark("V4: Register tiling", matmul_register_tiled, grid4, block4);
    
    // cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    ms /= TRIALS;
    
    float gflops = 2.0f * M * N * K / ms / 1e6;
    float pct_peak = 100.0f * gflops / (peak_tflops * 1000);
    printf("%-25s %-12.2f %-12.1f %-12.1f%%\n", "cuBLAS (reference)", ms, gflops, pct_peak);
    
    // Verify against cuBLAS
    cudaMemcpy(h_C_ref, d_C, bytes_C, cudaMemcpyDeviceToHost);
    float err = max_error(result_v4, h_C_ref, M * N);
    printf("\nV4 vs cuBLAS max relative error: %.2e\n", err);
    
    cublasDestroy(handle);
    
    printf("\n=== Key Insights ===\n");
    printf("1. Naive: Each element loads K elements from A and B (2K loads)\n");
    printf("   - Memory bound, very low arithmetic intensity\n");
    printf("\n2. Tiled: Load tiles into shared memory, reuse TILE_SIZE times\n");
    printf("   - 16x data reuse → 16x less memory traffic\n");
    printf("   - Transitions from memory-bound to compute-bound\n");
    printf("\n3. Register tiling: Each thread computes multiple outputs\n");
    printf("   - More work per thread = better instruction parallelism\n");
    printf("   - Amortize shared memory loads across more FLOPs\n");
    printf("\n4. cuBLAS uses advanced techniques:\n");
    printf("   - Larger tiles (128x128 or bigger)\n");
    printf("   - Prefetching and double-buffering\n");
    printf("   - Tensor cores (on supported GPUs)\n");
    printf("   - Assembly-level optimization\n");
    printf("\nGap from cuBLAS shows room for expert-level optimization!\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    
    return 0;
}
