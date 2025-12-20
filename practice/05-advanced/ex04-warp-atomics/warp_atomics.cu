#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Naive: every thread does atomic - HIGH CONTENTION
__device__ int naiveAtomicInc(int* counter) {
    return atomicAdd(counter, 1);
}

// TODO: Implement warp-aggregated atomic increment
// Requirements:
// 1. Get coalesced_group of active threads
// 2. Only leader (thread_rank 0) does atomic with active.size()
// 3. Broadcast result to all threads using shfl
// 4. Return (base_value + thread_rank) so each thread gets unique value
__device__ int warpAggregatedAtomicInc(int* counter) {
    // TODO: Get coalesced group
    // cg::coalesced_group active = ...
    
    // TODO: Declare variable for warp result
    int warp_res;
    
    // TODO: Leader thread does atomic
    // if (active.thread_rank() == 0) { ... }
    
    // TODO: Broadcast result to all threads
    // warp_res = active.shfl(warp_res, 0);
    
    // TODO: Return unique value for each thread
    // return warp_res + active.thread_rank();
    
    // Placeholder - replace with your implementation
    return atomicAdd(counter, 1);
}

__global__ void testNaive(int* counter, int* results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        results[idx] = naiveAtomicInc(counter);
    }
}

__global__ void testWarpAgg(int* counter, int* results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        results[idx] = warpAggregatedAtomicInc(counter);
    }
}

bool verifyUnique(int* results, int n) {
    bool* seen = new bool[n]();
    for (int i = 0; i < n; i++) {
        if (results[i] < 0 || results[i] >= n || seen[results[i]]) {
            delete[] seen;
            return false;
        }
        seen[results[i]] = true;
    }
    delete[] seen;
    return true;
}

int main() {
    const int N = 100000;
    int *d_counter, *d_results;
    int *h_results = new int[N];
    
    cudaMalloc(&d_counter, sizeof(int));
    cudaMalloc(&d_results, N * sizeof(int));
    
    // Test correctness of warp-aggregated version
    cudaMemset(d_counter, 0, sizeof(int));
    testWarpAgg<<<(N+255)/256, 256>>>(d_counter, d_results, N);
    cudaDeviceSynchronize();
    
    int final_count;
    cudaMemcpy(&final_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_results, d_results, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    bool count_correct = (final_count == N);
    bool unique_correct = verifyUnique(h_results, N);
    
    printf("Final counter: %d (expected %d) - %s\n", 
           final_count, N, count_correct ? "PASS" : "FAIL");
    printf("Unique values: %s\n", unique_correct ? "PASS" : "FAIL");
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaMemset(d_counter, 0, sizeof(int));
    cudaEventRecord(start);
    testNaive<<<(N+255)/256, 256>>>(d_counter, d_results, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naive_ms;
    cudaEventElapsedTime(&naive_ms, start, stop);
    
    cudaMemset(d_counter, 0, sizeof(int));
    cudaEventRecord(start);
    testWarpAgg<<<(N+255)/256, 256>>>(d_counter, d_results, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float warp_ms;
    cudaEventElapsedTime(&warp_ms, start, stop);
    
    printf("\nPerformance:\n");
    printf("Naive atomics:          %.3f ms\n", naive_ms);
    printf("Warp-aggregated atomics: %.3f ms\n", warp_ms);
    printf("Speedup: %.1fx\n", naive_ms / warp_ms);
    
    delete[] h_results;
    cudaFree(d_counter);
    cudaFree(d_results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return (count_correct && unique_correct) ? 0 : 1;
}
