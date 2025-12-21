/**
 * warp_scan.cu - Prefix scan using warp shuffles
 * 
 * Learning objectives:
 * - Implement inclusive warp scan
 * - Implement exclusive warp scan
 * - Apply scans for stream compaction
 */

#include <cuda_runtime.h>
#include <cstdio>

#define FULL_MASK 0xffffffff

// ============================================================================
// Warp Scan Primitives
// ============================================================================

// Inclusive scan: each lane gets sum of all values up to and including itself
__device__ __forceinline__ int warp_scan_inclusive(int val) {
    int lane = threadIdx.x % 32;
    
    for (int offset = 1; offset < 32; offset *= 2) {
        int n = __shfl_up_sync(FULL_MASK, val, offset);
        if (lane >= offset) {
            val += n;
        }
    }
    return val;
}

// Exclusive scan: each lane gets sum of all values BEFORE itself
__device__ __forceinline__ int warp_scan_exclusive(int val) {
    int inclusive = warp_scan_inclusive(val);
    return inclusive - val;
}

// ============================================================================
// Demo Kernels
// ============================================================================

__global__ void demo_inclusive_scan(const int* in, int* out) {
    int lane = threadIdx.x % 32;
    int val = in[lane];
    
    int scanned = warp_scan_inclusive(val);
    out[lane] = scanned;
}

__global__ void demo_exclusive_scan(const int* in, int* out) {
    int lane = threadIdx.x % 32;
    int val = in[lane];
    
    int scanned = warp_scan_exclusive(val);
    out[lane] = scanned;
}

// ============================================================================
// Practical Application: Stream Compaction
// ============================================================================

// Count elements matching predicate and compact them
__global__ void stream_compact(const int* in, int* out, int* count, 
                                int n, int threshold) {
    __shared__ int warp_totals[32];
    __shared__ int block_offset;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    
    // Check predicate: keep values > threshold
    int val = (idx < n) ? in[idx] : 0;
    int keep = (idx < n && val > threshold) ? 1 : 0;
    
    // Exclusive scan within warp
    int warp_offset = warp_scan_exclusive(keep);
    int warp_total = warp_offset + keep;
    
    // Broadcast warp total from last lane
    warp_total = __shfl_sync(FULL_MASK, warp_total, 31);
    
    // First lane of each warp stores total
    if (lane == 0) {
        warp_totals[warp_id] = warp_total;
    }
    __syncthreads();
    
    // First warp computes prefix of warp totals
    if (warp_id == 0) {
        int wt = (lane < num_warps) ? warp_totals[lane] : 0;
        int warp_prefix = warp_scan_exclusive(wt);
        if (lane < num_warps) {
            warp_totals[lane] = warp_prefix;
        }
        
        // Last lane gets total count
        if (lane == num_warps - 1) {
            block_offset = atomicAdd(count, warp_prefix + wt);
        }
    }
    __syncthreads();
    
    // Write compacted output
    if (keep) {
        int global_offset = block_offset + warp_totals[warp_id] + warp_offset;
        out[global_offset] = val;
    }
}

int main() {
    printf("=== Warp Scans Demo ===\n\n");
    
    // Part 1: Demonstrate inclusive scan
    {
        printf("1. Inclusive Scan\n");
        printf("─────────────────────────────────────────\n");
        
        int h_in[32], h_out[32];
        for (int i = 0; i < 32; i++) h_in[i] = 1;  // All ones
        
        int *d_in, *d_out;
        cudaMalloc(&d_in, 32 * sizeof(int));
        cudaMalloc(&d_out, 32 * sizeof(int));
        cudaMemcpy(d_in, h_in, 32 * sizeof(int), cudaMemcpyHostToDevice);
        
        demo_inclusive_scan<<<1, 32>>>(d_in, d_out);
        cudaMemcpy(h_out, d_out, 32 * sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("   Input:  [");
        for (int i = 0; i < 8; i++) printf("%d ", h_in[i]);
        printf("... ]\n");
        
        printf("   Output: [");
        for (int i = 0; i < 8; i++) printf("%d ", h_out[i]);
        printf("... %d]\n\n", h_out[31]);
        
        cudaFree(d_in);
        cudaFree(d_out);
    }
    
    // Part 2: Demonstrate exclusive scan
    {
        printf("2. Exclusive Scan\n");
        printf("─────────────────────────────────────────\n");
        
        int h_in[32], h_out[32];
        for (int i = 0; i < 32; i++) h_in[i] = 1;
        
        int *d_in, *d_out;
        cudaMalloc(&d_in, 32 * sizeof(int));
        cudaMalloc(&d_out, 32 * sizeof(int));
        cudaMemcpy(d_in, h_in, 32 * sizeof(int), cudaMemcpyHostToDevice);
        
        demo_exclusive_scan<<<1, 32>>>(d_in, d_out);
        cudaMemcpy(h_out, d_out, 32 * sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("   Input:  [");
        for (int i = 0; i < 8; i++) printf("%d ", h_in[i]);
        printf("... ]\n");
        
        printf("   Output: [");
        for (int i = 0; i < 8; i++) printf("%d ", h_out[i]);
        printf("... %d]\n\n", h_out[31]);
        
        cudaFree(d_in);
        cudaFree(d_out);
    }
    
    // Part 3: Stream compaction application
    {
        printf("3. Stream Compaction (keep values > 50)\n");
        printf("─────────────────────────────────────────\n");
        
        const int N = 256;
        int h_in[N], h_out[N];
        
        // Create data with some values > 50
        int expected = 0;
        for (int i = 0; i < N; i++) {
            h_in[i] = i % 100;
            if (h_in[i] > 50) expected++;
        }
        
        int *d_in, *d_out, *d_count;
        cudaMalloc(&d_in, N * sizeof(int));
        cudaMalloc(&d_out, N * sizeof(int));
        cudaMalloc(&d_count, sizeof(int));
        
        cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(int));
        
        stream_compact<<<1, 256>>>(d_in, d_out, d_count, N, 50);
        
        int count;
        cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out, d_out, count * sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("   Input: %d elements with values 0-99\n", N);
        printf("   Output: %d elements > 50 (expected %d)\n", count, expected);
        printf("   First few: [");
        for (int i = 0; i < min(10, count); i++) {
            printf("%d ", h_out[i]);
        }
        printf("...]\n\n");
        
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_count);
    }
    
    printf("=== Warp Scan Pattern ===\n\n");
    printf("__device__ int warp_scan_inclusive(int val) {\n");
    printf("    int lane = threadIdx.x %% 32;\n");
    printf("    for (int offset = 1; offset < 32; offset *= 2) {\n");
    printf("        int n = __shfl_up_sync(0xffffffff, val, offset);\n");
    printf("        if (lane >= offset) val += n;\n");
    printf("    }\n");
    printf("    return val;\n");
    printf("}\n");
    
    return 0;
}
