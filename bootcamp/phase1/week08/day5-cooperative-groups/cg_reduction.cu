/**
 * cg_reduction.cu - Reduction with Cooperative Groups
 * 
 * Learning objectives:
 * - Clean reduction code with tiles
 * - Template-based tile operations
 * - Comparison with intrinsics
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

#define BLOCK_SIZE 256

// Generic tile reduce using cooperative groups
template<int TILE_SIZE>
__device__ float tile_reduce_sum(cg::thread_block_tile<TILE_SIZE> tile, float val) {
    for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
        val += tile.shfl_down(val, offset);
    }
    return val;
}

// Clean reduction using cooperative groups
__global__ void reduce_cg(const float* input, float* output, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    int tid = block.thread_rank();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = warp.thread_rank();
    int warp_id = warp.meta_group_rank();
    
    // Load
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // Warp-level reduction using tile
    val = tile_reduce_sum(warp, val);
    
    // Store warp sums
    __shared__ float warp_sums[8];  // 256 / 32 = 8 warps
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    
    block.sync();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        val = (lane < 8) ? warp_sums[lane] : 0.0f;
        val = tile_reduce_sum(warp, val);
        if (lane == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// Reduction for arbitrary-sized tiles
template<int TILE_SIZE>
__global__ void reduce_arbitrary_tile(const float* input, float* output, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);
    
    int tid = block.thread_rank();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_id = tid / TILE_SIZE;
    int num_tiles = blockDim.x / TILE_SIZE;
    
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // Tile-level reduction
    val = tile_reduce_sum<TILE_SIZE>(tile, val);
    
    // Collect tile sums
    __shared__ float tile_sums[32];  // Up to 32 tiles
    if (tile.thread_rank() == 0) {
        tile_sums[tile_id] = val;
    }
    
    block.sync();
    
    // Final reduction
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < num_tiles; i++) {
            sum += tile_sums[i];
        }
        output[blockIdx.x] = sum;
    }
}

// Compare: Old-style with intrinsics
__global__ void reduce_intrinsics(const float* input, float* output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // Warp reduce with intrinsics
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    
    __shared__ float warp_sums[8];
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    
    __syncthreads();
    
    if (warp_id == 0) {
        val = (lane < 8) ? warp_sums[lane] : 0.0f;
        val += __shfl_down_sync(0xFF, val, 4);
        val += __shfl_down_sync(0xFF, val, 2);
        val += __shfl_down_sync(0xFF, val, 1);
        if (lane == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// Grid-wide reduction (requires cooperative launch)
__global__ void reduce_grid_cg(const float* input, float* partial, float* output, int n) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    int tid = block.thread_rank();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = warp.thread_rank();
    int warp_id = warp.meta_group_rank();
    
    // Block reduction (same as before)
    float val = (idx < n) ? input[idx] : 0.0f;
    val = tile_reduce_sum(warp, val);
    
    __shared__ float warp_sums[8];
    if (lane == 0) warp_sums[warp_id] = val;
    block.sync();
    
    if (warp_id == 0) {
        val = (lane < 8) ? warp_sums[lane] : 0.0f;
        val = tile_reduce_sum(warp, val);
        if (lane == 0) partial[blockIdx.x] = val;
    }
    
    // Grid-wide sync - ALL blocks wait here
    grid.sync();
    
    // First block does final reduction
    if (blockIdx.x == 0) {
        val = 0.0f;
        for (int i = tid; i < gridDim.x; i += blockDim.x) {
            val += partial[i];
        }
        val = tile_reduce_sum(warp, val);
        
        if (lane == 0) warp_sums[warp_id] = val;
        block.sync();
        
        if (warp_id == 0) {
            val = (lane < 8) ? warp_sums[lane] : 0.0f;
            val = tile_reduce_sum(warp, val);
            if (lane == 0) *output = val;
        }
    }
}

int main() {
    printf("=== Reduction with Cooperative Groups ===\n\n");
    
    const int N = 1 << 20;  // 1M elements
    const int TRIALS = 100;
    
    float* h_input = new float[N];
    double expected = 0.0;
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
        expected += 1.0;
    }
    
    float *d_input, *d_partial, *d_output;
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_partial, blocks * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    printf("Array size: %d elements\n", N);
    printf("Expected sum: %.0f\n\n", expected);
    
    printf("%-30s %-12s %-12s\n", "Version", "Time(us)", "Result");
    printf("--------------------------------------------------------\n");
    
    // Cooperative Groups reduction
    float* h_partial = new float[blocks];
    reduce_cg<<<blocks, BLOCK_SIZE>>>(d_input, d_partial, N);
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float sum = 0;
    for (int i = 0; i < blocks; i++) sum += h_partial[i];
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        reduce_cg<<<blocks, BLOCK_SIZE>>>(d_input, d_partial, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("%-30s %-12.1f %-12.0f\n", "CG (warp tiles)", ms / TRIALS * 1000, sum);
    
    // Intrinsics version
    reduce_intrinsics<<<blocks, BLOCK_SIZE>>>(d_input, d_partial, N);
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    sum = 0;
    for (int i = 0; i < blocks; i++) sum += h_partial[i];
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        reduce_intrinsics<<<blocks, BLOCK_SIZE>>>(d_input, d_partial, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("%-30s %-12.1f %-12.0f\n", "Intrinsics", ms / TRIALS * 1000, sum);
    
    // Different tile sizes
    reduce_arbitrary_tile<16><<<blocks, BLOCK_SIZE>>>(d_input, d_partial, N);
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    sum = 0;
    for (int i = 0; i < blocks; i++) sum += h_partial[i];
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        reduce_arbitrary_tile<16><<<blocks, BLOCK_SIZE>>>(d_input, d_partial, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("%-30s %-12.1f %-12.0f\n", "CG (16-thread tiles)", ms / TRIALS * 1000, sum);
    
    printf("\n=== Benefits of Cooperative Groups ===\n");
    printf("1. Cleaner code - tile.shfl_down vs __shfl_down_sync + mask\n");
    printf("2. Generic templates - same code works for any tile size\n");
    printf("3. Meta-groups - easy hierarchical decomposition\n");
    printf("4. Grid sync - true global synchronization\n");
    printf("5. Future-proof - new hardware features via CG API\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_partial);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_partial;
    
    return 0;
}
