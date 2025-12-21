/**
 * lock_free_demo.cu - Lock-free algorithms on GPU
 * 
 * Learning objectives:
 * - CAS-based lock-free patterns
 * - Custom atomic operations
 * - Progress guarantees
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>

// Lock-free atomicMax for float
__device__ float atomicMaxFloat(float* addr, float val) {
    int* addr_int = (int*)addr;
    int old = *addr_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) {
            break;  // Already larger
        }
        old = atomicCAS(addr_int, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}

// Lock-free atomicMin for float
__device__ float atomicMinFloat(float* addr, float val) {
    int* addr_int = (int*)addr;
    int old = *addr_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= val) {
            break;
        }
        old = atomicCAS(addr_int, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}

// Lock-free argmax - stores (max_value, max_index)
__device__ void atomicArgMax(float* max_val, int* max_idx, float val, int idx) {
    // Read current max
    float old_val = *max_val;
    
    while (val > old_val) {
        // Try to update
        int old_int = __float_as_int(old_val);
        int new_int = __float_as_int(val);
        
        int result = atomicCAS((int*)max_val, old_int, new_int);
        
        if (result == old_int) {
            // We updated the value, now update index
            atomicExch(max_idx, idx);
            break;
        }
        
        old_val = __int_as_float(result);
    }
}

// Demo kernel for lock-free max
__global__ void lock_free_max_kernel(const float* input, float* max_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicMaxFloat(max_out, input[idx]);
    }
}

// Demo kernel for lock-free argmax
__global__ void lock_free_argmax_kernel(const float* input, float* max_val, int* max_idx, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicArgMax(max_val, max_idx, input[idx], idx);
    }
}

// Lock-free counter with backoff
__device__ int atomicAddWithBackoff(int* addr, int val) {
    int old, assumed;
    int backoff = 1;
    
    do {
        old = *addr;
        assumed = old;
        old = atomicCAS(addr, assumed, assumed + val);
        
        if (old != assumed) {
            // Contention - exponential backoff
            for (int i = 0; i < backoff; i++) {
                __nanosleep(32);  // SM 7.0+
            }
            backoff = min(backoff * 2, 1024);
        }
    } while (old != assumed);
    
    return old;
}

// Lock-free running average (Welford's algorithm)
struct RunningAvg {
    int count;
    float mean;
};

__device__ void atomicUpdateMean(int* count, float* mean, float val) {
    // Lock-free approximation (not exact Welford due to race)
    int n = atomicAdd(count, 1) + 1;
    float delta = val - *mean;
    atomicAdd(mean, delta / n);
}

__global__ void running_avg_kernel(const float* input, int* count, float* mean, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicUpdateMean(count, mean, input[idx]);
    }
}

// Demonstrate CAS loop pattern
__global__ void cas_pattern_demo() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("=== CAS Loop Pattern ===\n");
        printf("\n");
        printf("Lock-free pattern:\n");
        printf("  do {\n");
        printf("      old = *addr;\n");
        printf("      new_val = compute(old, my_val);\n");
        printf("      result = atomicCAS(addr, old, new_val);\n");
        printf("  } while (result != old);\n");
        printf("\n");
        printf("Key insight: retry if another thread modified\n");
        printf("This is the foundation of lock-free algorithms\n");
    }
}

int main() {
    printf("=== Lock-Free Algorithms Demo ===\n\n");
    
    const int N = 1 << 16;
    const int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;
    
    // Test lock-free max
    float* h_input = new float[N];
    float max_expected = -FLT_MAX;
    int max_idx_expected = 0;
    
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 10000) / 100.0f;
        if (h_input[i] > max_expected) {
            max_expected = h_input[i];
            max_idx_expected = i;
        }
    }
    
    float *d_input, *d_max;
    int *d_max_idx;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_max_idx, sizeof(int));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Lock-free max
    float init = -FLT_MAX;
    cudaMemcpy(d_max, &init, sizeof(float), cudaMemcpyHostToDevice);
    
    lock_free_max_kernel<<<GRID, BLOCK>>>(d_input, d_max, N);
    
    float result_max;
    cudaMemcpy(&result_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("=== Lock-Free Maximum ===\n");
    printf("Expected max: %.2f\n", max_expected);
    printf("Lock-free result: %.2f\n", result_max);
    printf("Match: %s\n\n", (fabsf(result_max - max_expected) < 0.01f) ? "YES" : "NO");
    
    // Lock-free argmax
    cudaMemcpy(d_max, &init, sizeof(float), cudaMemcpyHostToDevice);
    int init_idx = -1;
    cudaMemcpy(d_max_idx, &init_idx, sizeof(int), cudaMemcpyHostToDevice);
    
    lock_free_argmax_kernel<<<GRID, BLOCK>>>(d_input, d_max, d_max_idx, N);
    
    int result_idx;
    cudaMemcpy(&result_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result_idx, d_max_idx, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("=== Lock-Free ArgMax ===\n");
    printf("Expected: value=%.2f at index %d\n", max_expected, max_idx_expected);
    printf("Result: value=%.2f at index %d\n", result_max, result_idx);
    printf("Value at result index: %.2f\n", h_input[result_idx]);
    printf("Note: Index may differ if multiple elements have same max value\n\n");
    
    // Running average (approximate due to races)
    float sum = 0;
    for (int i = 0; i < N; i++) sum += h_input[i];
    float expected_mean = sum / N;
    
    int* d_count;
    float* d_mean;
    cudaMalloc(&d_count, sizeof(int));
    cudaMalloc(&d_mean, sizeof(float));
    
    int zero = 0;
    float zerof = 0.0f;
    cudaMemcpy(d_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, &zerof, sizeof(float), cudaMemcpyHostToDevice);
    
    running_avg_kernel<<<GRID, BLOCK>>>(d_input, d_count, d_mean, N);
    
    float result_mean;
    cudaMemcpy(&result_mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("=== Lock-Free Running Average ===\n");
    printf("Expected mean: %.4f\n", expected_mean);
    printf("Lock-free result: %.4f\n", result_mean);
    printf("Difference: %.4f\n", fabsf(result_mean - expected_mean));
    printf("Note: Some error expected due to parallel race conditions\n\n");
    
    cas_pattern_demo<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    printf("\n=== Progress Guarantees ===\n");
    printf("Wait-free:   ALL threads complete in bounded steps\n");
    printf("Lock-free:   At least ONE thread makes progress\n");
    printf("             (atomicCAS provides this)\n");
    printf("Blocking:    Threads may wait indefinitely\n");
    printf("             (spinlocks are blocking)\n");
    printf("\n");
    printf("GPU atomics are lock-free by design!\n");
    printf("The hardware guarantees at least one thread succeeds.\n");
    
    cudaFree(d_input);
    cudaFree(d_max);
    cudaFree(d_max_idx);
    cudaFree(d_count);
    cudaFree(d_mean);
    delete[] h_input;
    
    return 0;
}
