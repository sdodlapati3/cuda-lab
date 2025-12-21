/**
 * fence_demo.cu - Memory fences and visibility
 * 
 * Learning objectives:
 * - Memory ordering issues
 * - __threadfence() usage
 * - Producer-consumer patterns
 */

#include <cuda_runtime.h>
#include <cstdio>

// Demo: Inter-block communication without fence (INCORRECT)
__global__ void producer_no_fence(int* data, int* flag) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Write data
        *data = 42;
        // Signal ready (BUG: no fence!)
        *flag = 1;
    }
}

__global__ void consumer_no_fence(int* data, int* flag, int* result) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Spin wait for flag
        while (atomicAdd(flag, 0) == 0);  // Use atomic read
        
        // Read data (BUG: may see stale value!)
        *result = *data;
    }
}

// Demo: Inter-block communication with fence (CORRECT)
__global__ void producer_with_fence(int* data, volatile int* flag) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Write data
        *data = 42;
        
        // Fence ensures data write is visible before flag
        __threadfence();
        
        // Signal ready
        *flag = 1;
    }
}

__global__ void consumer_with_fence(int* data, volatile int* flag, int* result) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Spin wait for flag
        while (*flag == 0);
        
        // Fence ensures we see the data that was written
        __threadfence();
        
        // Read data
        *result = *data;
    }
}

// Practical example: Grid-wide reduction with inter-block sync
__device__ unsigned int block_count = 0;

__global__ void grid_reduce_with_fence(const float* input, float* output, 
                                        float* partial, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and reduce within block
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Block 0's result goes directly to partial
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
        
        // Ensure partial result is visible
        __threadfence();
        
        // Increment counter (acts as barrier)
        unsigned int old = atomicInc(&block_count, gridDim.x);
        
        // Last block to finish does final reduction
        if (old == gridDim.x - 1) {
            // Read all partial sums
            float sum = 0.0f;
            for (int i = 0; i < gridDim.x; i++) {
                sum += partial[i];
            }
            *output = sum;
            
            // Reset for next kernel call
            block_count = 0;
        }
    }
}

// Demo: Block-level fence
__global__ void block_fence_demo(int* shared_data, int* result) {
    __shared__ int sdata[256];
    
    int tid = threadIdx.x;
    
    if (tid < 128) {
        // First half writes
        sdata[tid] = tid;
        
        // Block-level fence (faster than device-level)
        __threadfence_block();
    }
    
    __syncthreads();  // Barrier
    
    if (tid >= 128) {
        // Second half reads
        result[tid - 128] = sdata[tid - 128];
    }
}

// Show fence overhead
__global__ void fence_overhead_test(float* data, int iterations, int fence_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[idx];
    
    for (int i = 0; i < iterations; i++) {
        val += 1.0f;
        
        if (fence_type == 1) {
            __threadfence_block();
        } else if (fence_type == 2) {
            __threadfence();
        } else if (fence_type == 3) {
            __threadfence_system();
        }
    }
    
    data[idx] = val;
}

int main() {
    printf("=== Memory Fences Demo ===\n\n");
    
    int *d_data, *d_flag, *d_result;
    cudaMalloc(&d_data, sizeof(int));
    cudaMalloc(&d_flag, sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    printf("=== Producer-Consumer Pattern ===\n\n");
    
    // Test without fence (may fail)
    printf("Without fence (potential bug):\n");
    cudaMemset(d_data, 0, sizeof(int));
    cudaMemset(d_flag, 0, sizeof(int));
    cudaMemset(d_result, 0, sizeof(int));
    
    producer_no_fence<<<1, 1>>>(d_data, d_flag);
    consumer_no_fence<<<1, 1>>>(d_data, d_flag, d_result);
    cudaDeviceSynchronize();
    
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("  Result: %d (should be 42, but may be 0 due to ordering)\n", result);
    
    // Test with fence (correct)
    printf("\nWith fence (correct):\n");
    cudaMemset(d_data, 0, sizeof(int));
    cudaMemset(d_flag, 0, sizeof(int));
    cudaMemset(d_result, 0, sizeof(int));
    
    producer_with_fence<<<1, 1>>>(d_data, (volatile int*)d_flag);
    consumer_with_fence<<<1, 1>>>(d_data, (volatile int*)d_flag, d_result);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("  Result: %d (correctly 42)\n", result);
    
    // Grid reduction with fence
    printf("\n=== Grid Reduction with Inter-block Sync ===\n");
    
    const int N = 1024;
    float* h_input = new float[N];
    float expected = 0.0f;
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
        expected += 1.0f;
    }
    
    float *d_input, *d_output, *d_partial;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMalloc(&d_partial, 256 * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    grid_reduce_with_fence<<<4, 256, 256 * sizeof(float)>>>(d_input, d_output, d_partial, N);
    
    float gpu_sum;
    cudaMemcpy(&gpu_sum, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU sum: %.0f (expected: %.0f)\n", gpu_sum, expected);
    
    // Fence overhead test
    printf("\n=== Fence Overhead ===\n");
    
    float* d_perf;
    cudaMalloc(&d_perf, 256 * 256 * sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    const char* fence_names[] = {"No fence", "__threadfence_block", 
                                  "__threadfence", "__threadfence_system"};
    int iterations = 1000;
    
    for (int type = 0; type <= 3; type++) {
        cudaEventRecord(start);
        fence_overhead_test<<<256, 256>>>(d_perf, iterations, type);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        
        printf("%-25s: %.3f ms\n", fence_names[type], ms);
    }
    
    printf("\n=== Key Insights ===\n");
    printf("1. Fences ensure ORDER of memory operations\n");
    printf("2. NOT a barrier - other threads don't wait\n");
    printf("3. Use with flag/signal patterns for sync\n");
    printf("4. __threadfence_block() < __threadfence() < __threadfence_system()\n");
    printf("5. Atomics include memory ordering (often no extra fence needed)\n");
    printf("6. 'volatile' prevents compiler reordering, but not hardware\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_flag);
    cudaFree(d_result);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partial);
    cudaFree(d_perf);
    delete[] h_input;
    
    return 0;
}
