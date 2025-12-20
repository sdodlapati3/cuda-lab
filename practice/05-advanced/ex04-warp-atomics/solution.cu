#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ int naiveAtomicInc(int* counter) {
    return atomicAdd(counter, 1);
}

__device__ int warpAggregatedAtomicInc(int* counter) {
    cg::coalesced_group active = cg::coalesced_threads();
    
    int warp_res;
    if (active.thread_rank() == 0) {
        warp_res = atomicAdd(counter, active.size());
    }
    
    warp_res = active.shfl(warp_res, 0);
    return warp_res + active.thread_rank();
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
