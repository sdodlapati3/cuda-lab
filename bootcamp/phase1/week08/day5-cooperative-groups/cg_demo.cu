/**
 * cg_demo.cu - Cooperative Groups basics
 * 
 * Learning objectives:
 * - Thread block and tile groups
 * - Coalesced groups for divergent code
 * - Group operations
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

// Basic thread block usage
__global__ void thread_block_demo() {
    cg::thread_block block = cg::this_thread_block();
    
    int tid = block.thread_rank();  // Like threadIdx.x
    int size = block.size();        // Like blockDim.x
    
    if (tid == 0) {
        printf("=== Thread Block Demo ===\n");
        printf("Block size: %d\n", size);
        printf("Dim: (%d, %d, %d)\n", 
               block.dim_threads().x, block.dim_threads().y, block.dim_threads().z);
    }
    
    block.sync();  // Like __syncthreads()
    
    if (tid == 0) {
        printf("Sync complete!\n");
    }
}

// Tiled partition demo
__global__ void tiled_partition_demo() {
    cg::thread_block block = cg::this_thread_block();
    int tid = block.thread_rank();
    
    // Create warp-sized tiles
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Create half-warp tiles
    cg::thread_block_tile<16> half_warp = cg::tiled_partition<16>(block);
    
    // Properties
    int warp_id = tid / 32;
    int lane = warp.thread_rank();  // Lane within warp
    int half_lane = half_warp.thread_rank();
    
    if (tid == 5) {
        printf("\n=== Tiled Partition Demo ===\n");
        printf("Thread %d:\n", tid);
        printf("  Warp ID: %d, Lane in warp: %d\n", warp_id, lane);
        printf("  Half-warp lane: %d\n", half_lane);
    }
    
    block.sync();
    
    // Shuffle within tile
    float val = (float)lane;
    float from_lane_0 = warp.shfl(val, 0);  // Broadcast from lane 0
    
    if (tid == 5) {
        printf("  My value: %.0f, Broadcast from lane 0: %.0f\n", val, from_lane_0);
    }
    
    // Tile-level reduction
    float sum = val;
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        sum += warp.shfl_down(sum, offset);
    }
    
    if (lane == 0) {
        printf("  Warp %d sum: %.0f (expected: 496)\n", warp_id, sum);
    }
}

// Coalesced groups for divergent code
__global__ void coalesced_group_demo() {
    cg::thread_block block = cg::this_thread_block();
    int tid = block.thread_rank();
    
    if (tid == 0) {
        printf("\n=== Coalesced Group Demo ===\n");
    }
    block.sync();
    
    // Only even threads participate
    if (tid % 2 == 0) {
        cg::coalesced_group active = cg::coalesced_threads();
        
        int active_rank = active.thread_rank();
        int active_size = active.size();
        
        if (tid == 0) {
            printf("Even threads only: %d active threads\n", active_size);
        }
        
        // Reduce only among active threads
        float val = 1.0f;
        float sum = val;
        for (int offset = active.size() / 2; offset > 0; offset /= 2) {
            sum += active.shfl_down(sum, offset);
        }
        
        if (active_rank == 0) {
            printf("Sum of %d values: %.0f\n", active_size, sum);
        }
    }
    
    block.sync();
    
    // Different branch
    if (tid < 10) {
        cg::coalesced_group active = cg::coalesced_threads();
        if (tid == 0) {
            printf("First 10 threads: %d active\n", active.size());
        }
    }
}

// Show meta_group_rank for hierarchical organization
__global__ void meta_group_demo() {
    cg::thread_block block = cg::this_thread_block();
    int tid = block.thread_rank();
    
    // Partition block into warps
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Each warp has a meta_group_rank (which warp am I?)
    int warp_id = warp.meta_group_rank();
    int lane = warp.thread_rank();
    
    // Further partition into 8-thread tiles
    cg::thread_block_tile<8> octet = cg::tiled_partition<8>(block);
    int octet_id = octet.meta_group_rank();
    int octet_lane = octet.thread_rank();
    
    if (tid == 42) {
        printf("\n=== Meta Group Demo ===\n");
        printf("Thread %d:\n", tid);
        printf("  Warp: %d (lane %d)\n", warp_id, lane);
        printf("  Octet: %d (lane %d)\n", octet_id, octet_lane);
    }
}

// Ballot and voting within tiles
__global__ void tile_voting_demo() {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    int tid = block.thread_rank();
    int lane = warp.thread_rank();
    
    bool predicate = (lane < 16);
    
    // Ballot returns bitmask
    unsigned ballot = warp.ballot(predicate);
    
    // Any and all
    bool any_true = warp.any(predicate);
    bool all_true = warp.all(predicate);
    
    if (tid == 0) {
        printf("\n=== Tile Voting Demo ===\n");
        printf("Predicate: lane < 16\n");
        printf("Ballot: 0x%08X\n", ballot);
        printf("Any: %d, All: %d\n", any_true, all_true);
    }
}

int main() {
    printf("=== Cooperative Groups Demo ===\n\n");
    
    thread_block_demo<<<1, 256>>>();
    cudaDeviceSynchronize();
    
    tiled_partition_demo<<<1, 256>>>();
    cudaDeviceSynchronize();
    
    coalesced_group_demo<<<1, 256>>>();
    cudaDeviceSynchronize();
    
    meta_group_demo<<<1, 256>>>();
    cudaDeviceSynchronize();
    
    tile_voting_demo<<<1, 256>>>();
    cudaDeviceSynchronize();
    
    printf("\n=== Key Cooperative Groups Types ===\n");
    printf("thread_block       - All threads in block\n");
    printf("thread_block_tile  - Fixed-size tile (warp, half-warp, etc.)\n");
    printf("coalesced_group    - Currently active threads\n");
    printf("grid_group         - All threads in grid (requires coop launch)\n");
    printf("\n");
    printf("Benefits over intrinsics:\n");
    printf("1. Type safety and clear intent\n");
    printf("2. Portable code patterns\n");
    printf("3. Hierarchical decomposition\n");
    printf("4. Works with divergent code (coalesced_group)\n");
    
    return 0;
}
