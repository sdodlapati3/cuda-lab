/**
 * warp_patterns.cu - Common warp-level patterns
 * 
 * Learning objectives:
 * - Combine primitives for complex ops
 * - Build reusable building blocks
 * - Apply to algorithms
 */

#include <cuda_runtime.h>
#include <cstdio>

#define FULL_MASK 0xffffffff

// ============================================================================
// Pattern 1: Warp Broadcast
// ============================================================================

__device__ __forceinline__ float warp_broadcast(float val, int src_lane) {
    return __shfl_sync(FULL_MASK, val, src_lane);
}

// ============================================================================
// Pattern 2: Warp Reduce (various ops)
// ============================================================================

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(FULL_MASK, val, offset);
        val = max(val, other);
    }
    return val;
}

// ============================================================================
// Pattern 3: Warp All-Reduce (result in all lanes)
// ============================================================================

template<typename T>
__device__ __forceinline__ T warp_allreduce_sum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(FULL_MASK, val, offset);
    }
    return val;  // All lanes have the sum!
}

// ============================================================================
// Pattern 4: Exclusive Scan
// ============================================================================

__device__ __forceinline__ int warp_exclusive_scan(int val) {
    int lane = threadIdx.x % 32;
    
    for (int offset = 1; offset < 32; offset *= 2) {
        int n = __shfl_up_sync(FULL_MASK, val, offset);
        if (lane >= offset) val += n;
    }
    
    // Shift to make exclusive
    int inclusive = val;
    val = __shfl_up_sync(FULL_MASK, inclusive, 1);
    if (lane == 0) val = 0;
    
    return val;
}

// ============================================================================
// Pattern 5: Warp-Level Histogram (simplified)
// ============================================================================

__device__ void warp_histogram(int val, int* histogram, int num_bins) {
    int lane = threadIdx.x % 32;
    
    for (int bin = 0; bin < num_bins; bin++) {
        // Count lanes with this bin value
        unsigned int matches = __ballot_sync(FULL_MASK, val == bin);
        
        // Lane 0 updates histogram
        if (lane == 0) {
            histogram[bin] += __popc(matches);
        }
    }
}

// ============================================================================
// Pattern 6: Segmented Reduction
// ============================================================================

// Reduce within segments marked by 'head' flags
__device__ float warp_segmented_reduce(float val, int head) {
    int lane = threadIdx.x % 32;
    
    // Get segment info via ballot
    unsigned int heads = __ballot_sync(FULL_MASK, head);
    
    // Find which segment we're in
    unsigned int my_segment_mask = 0;
    int segment_start = 0;
    
    for (int i = lane; i >= 0; i--) {
        if ((heads >> i) & 1) {
            segment_start = i;
            break;
        }
    }
    
    // Create mask for lanes in same segment
    unsigned int after_start = ~((1u << segment_start) - 1);
    unsigned int before_next = FULL_MASK;
    
    // Find next segment start after us
    unsigned int later_heads = heads & ~((1u << (lane + 1)) - 1);
    if (later_heads) {
        int next_start = __ffs(later_heads) - 1;
        before_next = (1u << next_start) - 1;
    }
    
    // Our segment lanes
    my_segment_mask = after_start & before_next;
    
    // Simple approach: use reduction within segment bounds
    for (int offset = 1; offset < 32; offset *= 2) {
        float n = __shfl_up_sync(FULL_MASK, val, offset);
        if (lane >= offset && lane >= segment_start) {
            val += n;
        }
    }
    
    return val;
}

// ============================================================================
// Demo Kernels
// ============================================================================

__global__ void demo_all_patterns(float* results) {
    int lane = threadIdx.x % 32;
    float val = (float)(lane + 1);  // 1, 2, 3, ..., 32
    
    // Pattern 1: Broadcast from lane 15
    float broadcast = warp_broadcast(val, 15);
    if (lane == 0) results[0] = broadcast;
    
    // Pattern 2: Reduce sum (result in lane 0)
    float sum = warp_reduce_sum(val);
    if (lane == 0) results[1] = sum;
    
    // Pattern 3: All-reduce (result in ALL lanes)
    float allsum = warp_allreduce_sum(val);
    results[2] = allsum;  // All lanes can write (same value)
    
    // Pattern 4: Exclusive scan
    int scanval = lane + 1;
    int exscan = warp_exclusive_scan(scanval);
    if (lane == 5) results[3] = (float)exscan;  // Lane 5's exclusive prefix
    
    // Pattern 5: Max reduction
    float maxval = warp_reduce_max(val);
    if (lane == 0) results[4] = maxval;
}

__global__ void demo_practical_dot_product(const float* a, const float* b, 
                                            float* result, int n) {
    __shared__ float warp_sums[32];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    // Each thread computes partial product
    float prod = 0.0f;
    if (idx < n) {
        prod = a[idx] * b[idx];
    }
    
    // Warp-level reduce
    float warp_sum = warp_reduce_sum(prod);
    
    if (lane == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // First warp reduces all warp sums
    if (warp_id == 0) {
        float val = (lane < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
        float block_sum = warp_reduce_sum(val);
        
        if (lane == 0) {
            atomicAdd(result, block_sum);
        }
    }
}

int main() {
    printf("=== Warp-Level Patterns Demo ===\n\n");
    
    // Part 1: Basic patterns
    {
        printf("1. Basic Warp Patterns\n");
        printf("─────────────────────────────────────────\n");
        printf("   Input: lanes 0-31 have values 1-32\n\n");
        
        float* d_results;
        float h_results[5];
        cudaMalloc(&d_results, 5 * sizeof(float));
        
        demo_all_patterns<<<1, 32>>>(d_results);
        cudaMemcpy(h_results, d_results, 5 * sizeof(float), cudaMemcpyDeviceToHost);
        
        printf("   Broadcast from lane 15: %.0f\n", h_results[0]);
        printf("   Reduce sum (1+2+...+32): %.0f (expected 528)\n", h_results[1]);
        printf("   All-reduce (all lanes have sum): %.0f\n", h_results[2]);
        printf("   Exclusive scan lane 5: %.0f (1+2+3+4+5=15)\n", h_results[3]);
        printf("   Reduce max: %.0f\n\n", h_results[4]);
        
        cudaFree(d_results);
    }
    
    // Part 2: Practical - Dot product
    {
        printf("2. Practical: Dot Product\n");
        printf("─────────────────────────────────────────\n");
        
        const int N = 1024;
        float* h_a = new float[N];
        float* h_b = new float[N];
        float expected = 0;
        
        for (int i = 0; i < N; i++) {
            h_a[i] = 1.0f;
            h_b[i] = 2.0f;
            expected += h_a[i] * h_b[i];
        }
        
        float *d_a, *d_b, *d_result;
        cudaMalloc(&d_a, N * sizeof(float));
        cudaMalloc(&d_b, N * sizeof(float));
        cudaMalloc(&d_result, sizeof(float));
        
        cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_result, 0, sizeof(float));
        
        demo_practical_dot_product<<<(N+255)/256, 256>>>(d_a, d_b, d_result, N);
        
        float result;
        cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        
        printf("   a = [1, 1, ..., 1] (1024 elements)\n");
        printf("   b = [2, 2, ..., 2] (1024 elements)\n");
        printf("   Dot product: %.0f (expected %.0f)\n\n", result, expected);
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
        delete[] h_a;
        delete[] h_b;
    }
    
    printf("=== Week 13 Summary: Warp Primitives ===\n\n");
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ Primitive          │ Use Case                              │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ __shfl_sync        │ Broadcast, gather specific lanes      │\n");
    printf("│ __shfl_down_sync   │ Reduction (sum to lane 0)             │\n");
    printf("│ __shfl_up_sync     │ Prefix scan                           │\n");
    printf("│ __shfl_xor_sync    │ All-reduce (butterfly pattern)        │\n");
    printf("│ __ballot_sync      │ Get bitmask of matching predicates    │\n");
    printf("│ __any_sync         │ Check if any lane matches             │\n");
    printf("│ __all_sync         │ Check if all lanes match              │\n");
    printf("│ __popc             │ Count set bits in ballot              │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n\n");
    
    printf("Key insight: Warp shuffles are ~15x faster than shared memory!\n");
    printf("Use them for intra-warp communication whenever possible.\n");
    
    return 0;
}
