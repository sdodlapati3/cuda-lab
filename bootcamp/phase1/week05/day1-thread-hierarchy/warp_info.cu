/**
 * warp_info.cu - Explore warp-level details
 * 
 * Learning objectives:
 * - Understand warp as scheduling unit
 * - Calculate warp ID and lane ID
 * - See warp-level execution patterns
 */

#include <cuda_runtime.h>
#include <cstdio>

// Kernel that explores warp structure
__global__ void explore_warps(int* warp_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Warp ID within the block
    int warp_id_in_block = threadIdx.x / warpSize;
    
    // Lane ID within the warp (0-31)
    int lane_id = threadIdx.x % warpSize;
    
    // Global warp ID
    int warps_per_block = (blockDim.x + warpSize - 1) / warpSize;
    int global_warp_id = blockIdx.x * warps_per_block + warp_id_in_block;
    
    if (idx < n) {
        // Store global warp ID for each thread
        warp_data[idx] = global_warp_id;
    }
    
    // Print info for first thread of each warp
    if (lane_id == 0 && global_warp_id < 4) {
        printf("Warp %d: Block %d, Warp-in-block %d, First thread: %d\n",
               global_warp_id, blockIdx.x, warp_id_in_block, idx);
    }
}

// Demonstrate that warps execute in lockstep
__global__ void warp_lockstep() {
    int lane = threadIdx.x % 32;
    
    // All threads in a warp execute the same instruction at the same time
    // This is called SIMT (Single Instruction, Multiple Threads)
    
    // Only first warp prints
    if (threadIdx.x < 32) {
        // Threads 0-15 print "A", threads 16-31 print "B"
        // Due to lockstep, we'll see interleaved output
        if (lane < 16) {
            printf("Lane %2d: A\n", lane);
        } else {
            printf("Lane %2d: B\n", lane);
        }
    }
}

int main() {
    printf("=== Warp Structure Exploration ===\n\n");
    
    // Get warp size (always 32 on current NVIDIA GPUs)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Warp size: %d threads\n", prop.warpSize);
    printf("Max warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
    printf("\n");
    
    // Demo 1: Warp IDs
    printf("=== Demo 1: Warp IDs ===\n");
    const int N = 128;
    int* d_warp_data;
    int h_warp_data[N];
    
    cudaMalloc(&d_warp_data, N * sizeof(int));
    
    // Launch: 2 blocks × 64 threads = 4 warps total
    printf("Launch: 2 blocks × 64 threads = 4 warps\n\n");
    explore_warps<<<2, 64>>>(d_warp_data, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_warp_data, d_warp_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("\nWarp assignment for each thread:\n");
    for (int i = 0; i < N; i++) {
        if (i % 32 == 0) printf("Threads %3d-%3d: Warp %d\n", 
                                 i, i+31, h_warp_data[i]);
    }
    
    cudaFree(d_warp_data);
    
    // Demo 2: Lockstep execution
    printf("\n=== Demo 2: SIMT Lockstep ===\n");
    printf("Threads 0-15 print 'A', threads 16-31 print 'B'\n");
    printf("Notice: All execute 'together' (output may be interleaved)\n\n");
    
    warp_lockstep<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    printf("\n=== Key Insights ===\n");
    printf("1. Warp = 32 threads, smallest scheduling unit\n");
    printf("2. All threads in warp execute same instruction (SIMT)\n");
    printf("3. If threads diverge (if/else), both paths execute serially\n");
    printf("4. Warp-level programming can avoid shared memory\n");
    
    return 0;
}
