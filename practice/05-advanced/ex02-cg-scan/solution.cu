#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

namespace cg = cooperative_groups;

__global__ void inclusiveScan(int* input, int* output, int n) {
    cg::thread_block block = cg::this_thread_block();
    int tid = block.thread_rank();
    
    int val = (tid < n) ? input[tid] : 0;
    int result = cg::inclusive_scan(block, val, cg::plus<int>());
    
    if (tid < n) {
        output[tid] = result;
    }
}

__global__ void exclusiveScan(int* input, int* output, int n) {
    cg::thread_block block = cg::this_thread_block();
    int tid = block.thread_rank();
    
    int val = (tid < n) ? input[tid] : 0;
    int result = cg::exclusive_scan(block, val, cg::plus<int>());
    
    if (tid < n) {
        output[tid] = result;
    }
}

int main() {
    const int N = 8;
    int h_input[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    int h_incl[N], h_excl[N];
    int expected_incl[N] = {1, 3, 6, 10, 15, 21, 28, 36};
    int expected_excl[N] = {0, 1, 3, 6, 10, 15, 21, 28};
    
    int *d_input, *d_incl, *d_excl;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_incl, N * sizeof(int));
    cudaMalloc(&d_excl, N * sizeof(int));
    
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    
    inclusiveScan<<<1, N>>>(d_input, d_incl, N);
    exclusiveScan<<<1, N>>>(d_input, d_excl, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_incl, d_incl, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_excl, d_excl, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Input:          ");
    for (int i = 0; i < N; i++) printf("%3d ", h_input[i]);
    printf("\nInclusive scan: ");
    for (int i = 0; i < N; i++) printf("%3d ", h_incl[i]);
    printf("\nExclusive scan: ");
    for (int i = 0; i < N; i++) printf("%3d ", h_excl[i]);
    printf("\n");
    
    bool pass = true;
    for (int i = 0; i < N; i++) {
        if (h_incl[i] != expected_incl[i] || h_excl[i] != expected_excl[i]) {
            pass = false;
            break;
        }
    }
    printf("\nTest %s\n", pass ? "PASSED" : "FAILED");
    
    cudaFree(d_input);
    cudaFree(d_incl);
    cudaFree(d_excl);
    
    return pass ? 0 : 1;
}
