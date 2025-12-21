/**
 * vote_demo.cu - Warp vote functions
 * 
 * Learning objectives:
 * - Use __any_sync, __all_sync, __ballot_sync
 * - Count with __popc (population count)
 * - Apply for predicate handling
 */

#include <cuda_runtime.h>
#include <cstdio>

#define FULL_MASK 0xffffffff

// ============================================================================
// Demo: Basic Vote Functions
// ============================================================================

__global__ void demo_any_all(int* any_result, int* all_result, int threshold) {
    int lane = threadIdx.x % 32;
    
    // Some lanes match, some don't
    int predicate = (lane >= threshold);
    
    *any_result = __any_sync(FULL_MASK, predicate);
    *all_result = __all_sync(FULL_MASK, predicate);
}

__global__ void demo_ballot(unsigned int* ballot_result, int threshold) {
    int lane = threadIdx.x % 32;
    int predicate = (lane >= threshold);
    
    *ballot_result = __ballot_sync(FULL_MASK, predicate);
}

// ============================================================================
// Practical: Early Exit Pattern
// ============================================================================

__global__ void search_with_early_exit(const int* data, int n, int target, 
                                        int* found_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread found target
    int found = (idx < n && data[idx] == target);
    
    // If ANY thread in warp found it, we can report
    if (__any_sync(FULL_MASK, found)) {
        if (found) {
            atomicMin(found_idx, idx);  // Report first occurrence
        }
    }
}

// ============================================================================
// Practical: Warp-Level Counting
// ============================================================================

__global__ void count_matches(const int* data, int n, int threshold, int* count) {
    __shared__ int warp_counts[32];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    // Check predicate
    int match = (idx < n && data[idx] > threshold);
    
    // Get ballot - bitmap of which lanes match
    unsigned int ballot = __ballot_sync(FULL_MASK, match);
    
    // Count set bits (lanes that match)
    int warp_count = __popc(ballot);
    
    // First lane records warp count
    if (lane == 0) {
        warp_counts[warp_id] = warp_count;
    }
    __syncthreads();
    
    // First warp sums all warp counts
    if (warp_id == 0 && lane < blockDim.x / 32) {
        atomicAdd(count, warp_counts[lane]);
    }
}

// ============================================================================
// Practical: Predicated Execution
// ============================================================================

__global__ void predicated_work(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float val = data[idx];
    int needs_special = (val < 0);  // Negative values need special handling
    
    // Check if entire warp has same condition
    if (__all_sync(FULL_MASK, needs_special)) {
        // ALL negative - fast path for all
        data[idx] = -val * 2.0f;
    } else if (__all_sync(FULL_MASK, !needs_special)) {
        // ALL positive - fast path for all
        data[idx] = val * 2.0f;
    } else {
        // Mixed - handle individually (divergent)
        if (needs_special) {
            data[idx] = -val * 2.0f;
        } else {
            data[idx] = val * 2.0f;
        }
    }
}

int main() {
    printf("=== Vote Functions Demo ===\n\n");
    
    // Part 1: __any_sync and __all_sync
    {
        printf("1. __any_sync and __all_sync\n");
        printf("─────────────────────────────────────────\n");
        
        int *d_any, *d_all;
        cudaMalloc(&d_any, sizeof(int));
        cudaMalloc(&d_all, sizeof(int));
        
        printf("   Predicate: lane >= threshold\n\n");
        printf("   %-10s | %8s | %8s\n", "Threshold", "any?", "all?");
        printf("   ───────────────────────────────\n");
        
        for (int thresh : {0, 16, 31, 32}) {
            demo_any_all<<<1, 32>>>(d_any, d_all, thresh);
            
            int h_any, h_all;
            cudaMemcpy(&h_any, d_any, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_all, d_all, sizeof(int), cudaMemcpyDeviceToHost);
            
            printf("   %-10d | %8s | %8s\n", thresh, 
                   h_any ? "yes" : "no", h_all ? "yes" : "no");
        }
        printf("\n");
        
        cudaFree(d_any);
        cudaFree(d_all);
    }
    
    // Part 2: __ballot_sync
    {
        printf("2. __ballot_sync (returns bitmask)\n");
        printf("─────────────────────────────────────────\n");
        
        unsigned int* d_ballot;
        cudaMalloc(&d_ballot, sizeof(unsigned int));
        
        printf("   Predicate: lane >= threshold\n\n");
        
        for (int thresh : {0, 16, 28}) {
            demo_ballot<<<1, 32>>>(d_ballot, thresh);
            
            unsigned int ballot;
            cudaMemcpy(&ballot, d_ballot, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            
            printf("   Threshold %d: ballot = 0x%08x (%d bits set)\n", 
                   thresh, ballot, __builtin_popcount(ballot));
        }
        printf("\n");
        
        cudaFree(d_ballot);
    }
    
    // Part 3: Counting with ballot
    {
        printf("3. Counting Matches with __ballot + __popc\n");
        printf("─────────────────────────────────────────\n");
        
        const int N = 1024;
        int h_data[N];
        int expected = 0;
        for (int i = 0; i < N; i++) {
            h_data[i] = i % 100;
            if (h_data[i] > 50) expected++;
        }
        
        int *d_data, *d_count;
        cudaMalloc(&d_data, N * sizeof(int));
        cudaMalloc(&d_count, sizeof(int));
        cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(int));
        
        count_matches<<<4, 256>>>(d_data, N, 50, d_count);
        
        int count;
        cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("   Data: 1024 elements with values 0-99\n");
        printf("   Count > 50: %d (expected %d)\n\n", count, expected);
        
        cudaFree(d_data);
        cudaFree(d_count);
    }
    
    // Part 4: Search with early exit
    {
        printf("4. Search with Early Exit\n");
        printf("─────────────────────────────────────────\n");
        
        const int N = 10000;
        int* h_data = new int[N];
        for (int i = 0; i < N; i++) h_data[i] = i;
        
        int target = 4242;
        
        int *d_data, *d_found;
        cudaMalloc(&d_data, N * sizeof(int));
        cudaMalloc(&d_found, sizeof(int));
        cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
        
        int max_int = INT_MAX;
        cudaMemcpy(d_found, &max_int, sizeof(int), cudaMemcpyHostToDevice);
        
        search_with_early_exit<<<(N+255)/256, 256>>>(d_data, N, target, d_found);
        
        int found_at;
        cudaMemcpy(&found_at, d_found, sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("   Searching for %d in array of %d elements\n", target, N);
        printf("   Found at index: %d\n\n", found_at);
        
        cudaFree(d_data);
        cudaFree(d_found);
        delete[] h_data;
    }
    
    printf("=== Vote Function Summary ===\n\n");
    printf("  __any_sync(mask, pred)  : Returns 1 if ANY lane has pred=true\n");
    printf("  __all_sync(mask, pred)  : Returns 1 if ALL lanes have pred=true\n");
    printf("  __ballot_sync(mask,pred): Returns bitmask of lanes with pred=true\n");
    printf("  __popc(ballot)          : Count set bits in ballot\n");
    
    return 0;
}
