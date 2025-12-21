/**
 * reduction_fusion.cu - Fuse transforms with reductions
 * 
 * Learning objectives:
 * - Fused sum of squares
 * - Fused dot product
 * - Fused softmax
 */

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cmath>
#include <cfloat>

// ============================================================================
// Unfused: Square then Reduce
// ============================================================================

__global__ void square_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = x[idx] * x[idx];
}

// ============================================================================
// Fused: Transform-Reduce (Sum of Squares)
// ============================================================================

template<int BLOCK_SIZE>
__global__ void fused_sum_squares(const float* x, float* partial, int n) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Transform: load and square in one step
    float val = (idx < n) ? x[idx] : 0.0f;
    float squared = val * val;  // Transform in register
    
    // Reduce squared values
    float block_sum = BlockReduce(temp_storage).Sum(squared);
    
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = block_sum;
    }
}

// ============================================================================
// Unfused: Multiply then Reduce (Dot Product)
// ============================================================================

__global__ void multiply_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * b[idx];
}

// ============================================================================
// Fused: Dot Product
// ============================================================================

template<int BLOCK_SIZE>
__global__ void fused_dot_product(const float* a, const float* b, 
                                   float* partial, int n) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Transform: multiply in register
    float prod = (idx < n) ? a[idx] * b[idx] : 0.0f;
    
    // Reduce products
    float block_sum = BlockReduce(temp_storage).Sum(prod);
    
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = block_sum;
    }
}

// ============================================================================
// Softmax: Unfused (4 kernels)
// ============================================================================

__global__ void find_max_kernel(const float* x, float* max_vals, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? x[idx] : -FLT_MAX;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    
    if (tid == 0) atomicMax((int*)max_vals, __float_as_int(sdata[0]));
}

__global__ void exp_subtract_kernel(const float* x, float max_val, 
                                     float* exp_x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) exp_x[idx] = expf(x[idx] - max_val);
}

__global__ void sum_kernel(const float* x, float* sum, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? x[idx] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(sum, sdata[0]);
}

__global__ void divide_kernel(const float* x, float sum, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = x[idx] / sum;
}

// ============================================================================
// Softmax: Fused (2-pass)
// ============================================================================

// Pass 1: Find max
template<int BLOCK_SIZE>
__global__ void softmax_pass1(const float* x, float* block_max, int n) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? x[idx] : -FLT_MAX;
    
    float max_val = BlockReduce(temp_storage).Reduce(val, cub::Max());
    
    if (threadIdx.x == 0) block_max[blockIdx.x] = max_val;
}

// Pass 2: exp, sum, normalize (fused)
template<int BLOCK_SIZE>
__global__ void softmax_pass2(const float* x, float global_max, 
                               float* y, float* block_sums, int n) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float shared_sum;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute exp(x - max)
    float exp_val = (idx < n) ? expf(x[idx] - global_max) : 0.0f;
    
    // Sum exp values in block
    float block_sum = BlockReduce(temp_storage).Sum(exp_val);
    
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = block_sum;
    }
    
    // Store exp for later normalization
    if (idx < n) y[idx] = exp_val;
}

__global__ void normalize_kernel(float* y, float sum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] /= sum;
}

int main() {
    printf("=== Reduction Fusion Demo ===\n\n");
    
    const int N = 1 << 20;  // 1M elements
    const int BLOCK_SIZE = 256;
    const int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========================================================================
    // Test 1: Sum of Squares
    // ========================================================================
    {
        printf("1. Sum of Squares (Transform + Reduce)\n");
        printf("─────────────────────────────────────────\n");
        
        float *d_x, *d_temp, *d_partial;
        cudaMalloc(&d_x, N * sizeof(float));
        cudaMalloc(&d_temp, N * sizeof(float));
        cudaMalloc(&d_partial, num_blocks * sizeof(float));
        
        // Initialize with 1s
        float* h_x = new float[N];
        for (int i = 0; i < N; i++) h_x[i] = 1.0f;
        cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
        
        // CUB device reduce setup
        void* d_cub_temp = nullptr;
        size_t temp_bytes = 0;
        float* d_result;
        cudaMalloc(&d_result, sizeof(float));
        cub::DeviceReduce::Sum(d_cub_temp, temp_bytes, d_temp, d_result, N);
        cudaMalloc(&d_cub_temp, temp_bytes);
        
        // Unfused: square then CUB reduce
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            square_kernel<<<num_blocks, BLOCK_SIZE>>>(d_x, d_temp, N);
            cub::DeviceReduce::Sum(d_cub_temp, temp_bytes, d_temp, d_result, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float unfused_ms;
        cudaEventElapsedTime(&unfused_ms, start, stop);
        unfused_ms /= 100;
        
        // Fused
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            fused_sum_squares<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(
                d_x, d_partial, N);
            // Final reduction of partial sums
            cub::DeviceReduce::Sum(d_cub_temp, temp_bytes, d_partial, d_result, num_blocks);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float fused_ms;
        cudaEventElapsedTime(&fused_ms, start, stop);
        fused_ms /= 100;
        
        float result;
        cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        
        printf("   Unfused: %.3f ms (square kernel + reduce)\n", unfused_ms);
        printf("   Fused:   %.3f ms (combined)\n", fused_ms);
        printf("   Speedup: %.2fx\n", unfused_ms / fused_ms);
        printf("   Result:  %.0f (expected %d)\n\n", result, N);
        
        cudaFree(d_x);
        cudaFree(d_temp);
        cudaFree(d_partial);
        cudaFree(d_cub_temp);
        cudaFree(d_result);
        delete[] h_x;
    }
    
    // ========================================================================
    // Test 2: Dot Product
    // ========================================================================
    {
        printf("2. Dot Product (Transform + Reduce)\n");
        printf("─────────────────────────────────────────\n");
        
        float *d_a, *d_b, *d_temp, *d_partial, *d_result;
        cudaMalloc(&d_a, N * sizeof(float));
        cudaMalloc(&d_b, N * sizeof(float));
        cudaMalloc(&d_temp, N * sizeof(float));
        cudaMalloc(&d_partial, num_blocks * sizeof(float));
        cudaMalloc(&d_result, sizeof(float));
        
        // Initialize: a = 2, b = 3
        float* h_data = new float[N];
        for (int i = 0; i < N; i++) h_data[i] = 2.0f;
        cudaMemcpy(d_a, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
        for (int i = 0; i < N; i++) h_data[i] = 3.0f;
        cudaMemcpy(d_b, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
        
        void* d_cub_temp = nullptr;
        size_t temp_bytes = 0;
        cub::DeviceReduce::Sum(d_cub_temp, temp_bytes, d_temp, d_result, N);
        cudaMalloc(&d_cub_temp, temp_bytes);
        
        // Unfused
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            multiply_kernel<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_temp, N);
            cub::DeviceReduce::Sum(d_cub_temp, temp_bytes, d_temp, d_result, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float unfused_ms;
        cudaEventElapsedTime(&unfused_ms, start, stop);
        unfused_ms /= 100;
        
        // Fused
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            fused_dot_product<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(
                d_a, d_b, d_partial, N);
            cub::DeviceReduce::Sum(d_cub_temp, temp_bytes, d_partial, d_result, num_blocks);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float fused_ms;
        cudaEventElapsedTime(&fused_ms, start, stop);
        fused_ms /= 100;
        
        float result;
        cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        
        printf("   Unfused: %.3f ms\n", unfused_ms);
        printf("   Fused:   %.3f ms\n", fused_ms);
        printf("   Speedup: %.2fx\n", unfused_ms / fused_ms);
        printf("   Result:  %.0f (expected %.0f)\n\n", result, 2.0f * 3.0f * N);
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_temp);
        cudaFree(d_partial);
        cudaFree(d_cub_temp);
        cudaFree(d_result);
        delete[] h_data;
    }
    
    printf("=== Key Points ===\n\n");
    printf("1. Transform-reduce fusion eliminates intermediate array\n");
    printf("2. Dot product: 3 arrays → 2 arrays (saves N reads/writes)\n");
    printf("3. Sum of squares: 2 arrays → 1 array\n");
    printf("4. Softmax: Can reduce from 4 passes to 2-3\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
