/**
 * thread_demo.cu - Demonstrate thread hierarchy
 * 
 * Learning objectives:
 * - Understand threadIdx, blockIdx, blockDim, gridDim
 * - Calculate global thread index
 * - See how threads are organized
 */

#include <cuda_runtime.h>
#include <cstdio>

// Kernel that prints thread information
__global__ void print_thread_info() {
    // Calculate global thread index
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate warp ID within block
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    // Only print for first few threads to avoid flood
    if (global_idx < 8) {
        printf("Global ID: %2d | Block: %d | Thread: %2d | Warp: %d | Lane: %2d\n",
               global_idx, blockIdx.x, threadIdx.x, warp_id, lane_id);
    }
}

// Kernel that writes thread indices to array
__global__ void store_indices(int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = idx;
    }
}

// Kernel to demonstrate 2D indexing
__global__ void print_2d_info() {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Print grid info (only thread 0,0)
    if (threadIdx.x == 0 && threadIdx.y == 0 && 
        blockIdx.x == 0 && blockIdx.y == 0) {
        printf("\n2D Grid Info:\n");
        printf("  Grid:  (%d, %d) blocks\n", gridDim.x, gridDim.y);
        printf("  Block: (%d, %d) threads\n", blockDim.x, blockDim.y);
    }
    
    // Print first few elements
    if (row < 2 && col < 4) {
        printf("  Position (%d, %d) = Block (%d, %d), Thread (%d, %d)\n",
               row, col, blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x);
    }
}

int main() {
    printf("=== CUDA Thread Hierarchy Demo ===\n\n");
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp size: %d\n", prop.warpSize);
    printf("\n");
    
    // Demo 1: 1D thread hierarchy
    printf("=== Demo 1: 1D Thread Hierarchy ===\n");
    printf("Launch: 2 blocks × 4 threads = 8 total threads\n\n");
    
    print_thread_info<<<2, 4>>>();
    cudaDeviceSynchronize();
    
    // Demo 2: Store indices
    printf("\n=== Demo 2: Store Indices ===\n");
    const int N = 16;
    int* d_data;
    int h_data[N];
    
    cudaMalloc(&d_data, N * sizeof(int));
    
    // Launch with enough threads
    int block_size = 8;
    int num_blocks = (N + block_size - 1) / block_size;
    printf("N=%d, block_size=%d, num_blocks=%d\n", N, block_size, num_blocks);
    
    store_indices<<<num_blocks, block_size>>>(d_data, N);
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Result: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_data[i]);
    }
    printf("\n");
    
    cudaFree(d_data);
    
    // Demo 3: 2D indexing
    printf("\n=== Demo 3: 2D Thread Hierarchy ===\n");
    printf("Launch: (2,2) blocks × (4,2) threads\n");
    
    dim3 block_dim(4, 2);  // 4 columns, 2 rows per block
    dim3 grid_dim(2, 2);   // 2×2 blocks
    
    print_2d_info<<<grid_dim, block_dim>>>();
    cudaDeviceSynchronize();
    
    printf("\n=== Summary ===\n");
    printf("Thread organization:\n");
    printf("  Grid: Collection of blocks\n");
    printf("  Block: Collection of threads (up to 1024)\n");
    printf("  Warp: 32 threads executing in lockstep\n");
    printf("  Thread: Individual execution unit\n");
    
    return 0;
}
