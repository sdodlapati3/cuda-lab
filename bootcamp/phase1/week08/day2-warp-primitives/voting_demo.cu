/**
 * voting_demo.cu - Warp voting functions
 * 
 * Learning objectives:
 * - __ballot_sync, __any_sync, __all_sync
 * - Warp-level decision making
 * - Match-any for histogram optimization
 */

#include <cuda_runtime.h>
#include <cstdio>

#define FULL_MASK 0xFFFFFFFF

// Basic voting demo
__global__ void voting_basic_demo() {
    int lane = threadIdx.x % 32;
    bool is_even = (lane % 2 == 0);
    bool is_small = (lane < 5);
    
    if (threadIdx.x < 32) {
        // __ballot_sync: Create bitmask where each bit indicates predicate
        unsigned ballot_even = __ballot_sync(FULL_MASK, is_even);
        unsigned ballot_small = __ballot_sync(FULL_MASK, is_small);
        
        // __any_sync: True if ANY thread has predicate true
        int any_even = __any_sync(FULL_MASK, is_even);
        int any_small = __any_sync(FULL_MASK, is_small);
        
        // __all_sync: True if ALL threads have predicate true
        int all_even = __all_sync(FULL_MASK, is_even);
        int all_small = __all_sync(FULL_MASK, is_small);
        
        // __popc: Count set bits
        int count_even = __popc(ballot_even);
        int count_small = __popc(ballot_small);
        
        // __ffs: Find first set bit (1-indexed, 0 if none)
        int first_small = __ffs(ballot_small) - 1;
        
        if (lane == 0) {
            printf("=== Voting Results ===\n");
            printf("Ballot (even lanes): 0x%08X\n", ballot_even);
            printf("Ballot (lane < 5):   0x%08X\n", ballot_small);
            printf("\n");
            printf("Any even? %d (should be 1)\n", any_even);
            printf("All even? %d (should be 0)\n", all_even);
            printf("Count even: %d (should be 16)\n", count_even);
            printf("Count small: %d (should be 5)\n", count_small);
            printf("First small lane: %d (should be 0)\n", first_small);
        }
    }
}

// Practical use: Stream compaction with voting
__device__ int compact_within_warp(int* output, int val, bool keep) {
    unsigned mask = __ballot_sync(FULL_MASK, keep);
    int lane = threadIdx.x % 32;
    
    // Count how many before me want to write
    unsigned mask_before = mask & ((1U << lane) - 1);
    int write_idx = __popc(mask_before);
    
    if (keep) {
        output[write_idx] = val;
    }
    
    // Return total count for this warp
    return __popc(mask);
}

__global__ void stream_compact_demo() {
    __shared__ int output[32];
    int lane = threadIdx.x % 32;
    
    if (threadIdx.x < 32) {
        int val = lane;
        bool keep = (lane % 3 == 0);  // Keep lanes 0, 3, 6, 9, ...
        
        int count = compact_within_warp(output, val, keep);
        __syncwarp();
        
        if (lane == 0) {
            printf("\n=== Stream Compaction ===\n");
            printf("Keeping lanes divisible by 3\n");
            printf("Input: 0,1,2,3,4,5,6,7,8,9,...\n");
            printf("Output (%d elements): ", count);
            for (int i = 0; i < count; i++) {
                printf("%d ", output[i]);
            }
            printf("\n");
        }
    }
}

// Histogram with warp aggregation
__device__ void histogram_warp_optimized(unsigned int* histogram, int bin) {
    int lane = threadIdx.x % 32;
    
    // Find threads with same bin using __match_any_sync
    unsigned match_mask = __match_any_sync(FULL_MASK, bin);
    
    // Count matching threads
    int count = __popc(match_mask);
    
    // Only leader (first matching lane) does atomic
    int leader = __ffs(match_mask) - 1;
    
    if (lane == leader) {
        atomicAdd(&histogram[bin], count);
    }
}

__global__ void histogram_voting_demo() {
    __shared__ unsigned int histogram[8];
    int lane = threadIdx.x % 32;
    
    // Initialize
    if (lane < 8) histogram[lane] = 0;
    __syncwarp();
    
    if (threadIdx.x < 32) {
        // Each thread has a bin (clustered to show aggregation)
        int bin = lane / 4;  // Bins: 0,0,0,0,1,1,1,1,2,2,2,2,...
        
        histogram_warp_optimized(histogram, bin);
        __syncwarp();
        
        if (lane == 0) {
            printf("\n=== Histogram with Warp Voting ===\n");
            printf("Each group of 4 lanes has same bin\n");
            printf("Histogram: ");
            for (int i = 0; i < 8; i++) {
                printf("[%d]=%d ", i, histogram[i]);
            }
            printf("\n");
            printf("(Each bin should have 4)\n");
        }
    }
}

// Warp-level early exit
__global__ void early_exit_demo(const float* data, float* result, float target, int n) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int warp_start = warp_id * 32;
    
    __shared__ int found_idx;
    if (threadIdx.x == 0) found_idx = -1;
    __syncthreads();
    
    // Each warp searches its portion
    int idx = warp_start + lane;
    float val = (idx < n) ? data[idx] : -1e10f;
    
    bool found = (fabsf(val - target) < 0.001f);
    unsigned ballot = __ballot_sync(FULL_MASK, found);
    
    // If any thread found it, get the first one
    if (ballot != 0) {
        int first_lane = __ffs(ballot) - 1;
        if (lane == first_lane) {
            atomicMin(&found_idx, idx);
        }
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        if (found_idx >= 0) {
            result[0] = (float)found_idx;
        } else {
            result[0] = -1;
        }
    }
}

__global__ void activemask_demo() {
    int lane = threadIdx.x % 32;
    
    if (threadIdx.x < 32) {
        // Divergent code
        if (lane < 16) {
            // Only first 16 lanes execute this
            unsigned active = __activemask();
            if (lane == 0) {
                printf("\n=== Active Mask Demo ===\n");
                printf("In 'if (lane < 16)' branch:\n");
                printf("Active mask: 0x%08X (first 16 bits set)\n", active);
            }
        }
        
        // Back to convergent
        unsigned active_all = __activemask();
        if (lane == 0) {
            printf("After convergence:\n");
            printf("Active mask: 0x%08X (all 32 bits set)\n", active_all);
        }
    }
}

int main() {
    printf("=== Warp Voting Functions ===\n\n");
    
    voting_basic_demo<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    stream_compact_demo<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    histogram_voting_demo<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    activemask_demo<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    printf("\n=== Key Insights ===\n");
    printf("1. __ballot_sync creates bitmask of predicate results\n");
    printf("2. __any_sync / __all_sync for warp-level decisions\n");
    printf("3. __popc counts set bits, __ffs finds first\n");
    printf("4. __match_any_sync finds threads with same value (CC >= 7.0)\n");
    printf("5. Use voting to aggregate before atomics\n");
    printf("6. __activemask() shows which lanes are active in divergent code\n");
    
    return 0;
}
