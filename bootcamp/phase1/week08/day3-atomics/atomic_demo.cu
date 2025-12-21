/**
 * atomic_demo.cu - Basic atomic operations
 * 
 * Learning objectives:
 * - Built-in atomic functions
 * - Implementing custom atomics with CAS
 * - Understanding race conditions
 */

#include <cuda_runtime.h>
#include <cstdio>

// Demo: Race condition without atomics
__global__ void race_condition_demo(int* counter, int n) {
    // Each thread tries to increment
    for (int i = 0; i < n; i++) {
        (*counter)++;  // NOT ATOMIC - race condition!
    }
}

// Demo: Correct with atomics
__global__ void atomic_increment_demo(int* counter, int n) {
    for (int i = 0; i < n; i++) {
        atomicAdd(counter, 1);  // ATOMIC - thread-safe
    }
}

// Demo all atomic operations
__global__ void atomic_operations_demo(int* results) {
    int tid = threadIdx.x;
    
    // Each type of atomic on different result locations
    atomicAdd(&results[0], 1);          // Add
    atomicMax(&results[1], tid);        // Max
    atomicMin(&results[2], tid);        // Min
    atomicOr(&results[3], 1 << tid);    // OR - set bit
    atomicAnd(&results[4], ~(1 << tid)); // AND - clear bit (start from -1)
    atomicXor(&results[5], 1 << tid);   // XOR - toggle bit
}

// Custom atomic multiply using CAS
__device__ int atomicMul(int* address, int val) {
    int old = *address, assumed;
    do {
        assumed = old;
        old = atomicCAS(address, assumed, assumed * val);
    } while (assumed != old);
    return old;
}

// Custom atomic min for float (not natively supported on older GPUs)
__device__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    
    do {
        assumed = old;
        float old_val = __int_as_float(assumed);
        float new_val = fminf(old_val, val);
        old = atomicCAS(address_as_int, assumed, __float_as_int(new_val));
    } while (assumed != old);
    
    return __int_as_float(old);
}

__global__ void custom_atomic_demo(int* mul_result, float* min_result) {
    int tid = threadIdx.x + 1;  // 1, 2, 3, ...
    
    if (tid <= 5) {
        atomicMul(mul_result, tid);  // 1 * 2 * 3 * 4 * 5 = 120
    }
    
    float val = tid * 0.1f;
    atomicMinFloat(min_result, val);
}

// Demo atomicExch for spinlock
__global__ void exchange_demo(int* lock, int* protected_counter) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Simple spinlock pattern
    while (atomicExch(lock, 1) == 1) {
        // Spin until we get the lock (old value was 0)
    }
    
    // Critical section
    (*protected_counter)++;
    
    // Release lock
    atomicExch(lock, 0);
    
    __threadfence();  // Ensure visibility
}

int main() {
    printf("=== Atomic Operations Demo ===\n\n");
    
    int* d_counter;
    cudaMalloc(&d_counter, sizeof(int));
    
    // Test 1: Race condition
    printf("=== Test 1: Race Condition ===\n");
    int threads = 256, iters = 100;
    int expected = threads * iters;
    
    cudaMemset(d_counter, 0, sizeof(int));
    race_condition_demo<<<1, threads>>>(d_counter, iters);
    
    int result;
    cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Without atomic: Expected %d, got %d (%s)\n", 
           expected, result, result == expected ? "correct" : "RACE CONDITION!");
    
    // Test 2: With atomics
    cudaMemset(d_counter, 0, sizeof(int));
    atomic_increment_demo<<<1, threads>>>(d_counter, iters);
    cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("With atomicAdd: Expected %d, got %d (%s)\n\n", 
           expected, result, result == expected ? "correct" : "ERROR");
    
    // Test 3: Various atomics
    printf("=== Test 3: Atomic Operations ===\n");
    int* d_results;
    cudaMalloc(&d_results, 6 * sizeof(int));
    
    int init_results[6] = {0, 0, 256, 0, -1, 0};
    cudaMemcpy(d_results, init_results, 6 * sizeof(int), cudaMemcpyHostToDevice);
    
    atomic_operations_demo<<<1, 32>>>();
    
    int h_results[6];
    cudaMemcpy(h_results, d_results, 6 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("atomicAdd result: %d (32 threads added 1)\n", h_results[0]);
    printf("atomicMax result: %d (max of 0-31)\n", h_results[1]);
    printf("atomicMin result: %d (min of 0-31, started at 256)\n", h_results[2]);
    printf("atomicOr result:  0x%08X (each thread set its bit)\n", h_results[3]);
    printf("atomicAnd result: 0x%08X (each thread cleared its bit from -1)\n", h_results[4]);
    printf("atomicXor result: 0x%08X (each thread toggled its bit)\n\n", h_results[5]);
    
    // Test 4: Custom atomics with CAS
    printf("=== Test 4: Custom Atomics (using CAS) ===\n");
    int* d_mul;
    float* d_min;
    cudaMalloc(&d_mul, sizeof(int));
    cudaMalloc(&d_min, sizeof(float));
    
    int init_mul = 1;
    float init_min = 1000.0f;
    cudaMemcpy(d_mul, &init_mul, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice);
    
    custom_atomic_demo<<<1, 32>>>(d_mul, d_min);
    
    int mul_result;
    float min_result;
    cudaMemcpy(&mul_result, d_mul, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&min_result, d_min, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Custom atomicMul: 1*2*3*4*5 = %d (expected 120)\n", mul_result);
    printf("Custom atomicMinFloat: min value = %.1f (expected 0.1)\n\n", min_result);
    
    // Test 5: Spinlock pattern
    printf("=== Test 5: Spinlock with atomicExch ===\n");
    int* d_lock;
    cudaMalloc(&d_lock, sizeof(int));
    cudaMemset(d_lock, 0, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));
    
    exchange_demo<<<4, 64>>>(d_lock, d_counter);
    
    cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Protected counter with spinlock: %d (expected 256)\n", result);
    
    printf("\n=== atomicCAS Pattern ===\n");
    printf("To implement any custom atomic:\n");
    printf("1. Read current value\n");
    printf("2. Compute new value\n");
    printf("3. CAS: if unchanged, write new; else retry\n");
    printf("\nint old = *addr, assumed;\n");
    printf("do {\n");
    printf("    assumed = old;\n");
    printf("    old = atomicCAS(addr, assumed, new_value(assumed));\n");
    printf("} while (assumed != old);\n");
    
    cudaFree(d_counter);
    cudaFree(d_results);
    cudaFree(d_mul);
    cudaFree(d_min);
    cudaFree(d_lock);
    
    return 0;
}
