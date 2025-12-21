/**
 * shuffle_demo.cu - Demonstrate all shuffle variants
 * 
 * Learning objectives:
 * - Use __shfl_sync for broadcast
 * - Use __shfl_down_sync for reduction
 * - Use __shfl_up_sync for scan
 * - Use __shfl_xor_sync for butterfly
 */

#include <cuda_runtime.h>
#include <cstdio>

#define FULL_MASK 0xffffffff

// Demonstrate __shfl_sync - broadcast from one lane
__global__ void demo_shfl_broadcast(int* out) {
    int lane = threadIdx.x % 32;
    int val = lane * 10;  // Each lane has unique value
    
    // Get value from lane 5
    int from_lane_5 = __shfl_sync(FULL_MASK, val, 5);
    
    out[threadIdx.x] = from_lane_5;  // All should be 50
}

// Demonstrate __shfl_down_sync - for reductions
__global__ void demo_shfl_down(int* out) {
    int lane = threadIdx.x % 32;
    int val = lane;  // 0, 1, 2, ..., 31
    
    // Get value from lane + 4
    int from_down = __shfl_down_sync(FULL_MASK, val, 4);
    
    // Store original and received
    out[threadIdx.x * 2] = val;
    out[threadIdx.x * 2 + 1] = from_down;
}

// Demonstrate __shfl_up_sync - for scans
__global__ void demo_shfl_up(int* out) {
    int lane = threadIdx.x % 32;
    int val = lane;  // 0, 1, 2, ..., 31
    
    // Get value from lane - 1
    int from_up = __shfl_up_sync(FULL_MASK, val, 1);
    
    // Store original and received
    out[threadIdx.x * 2] = val;
    out[threadIdx.x * 2 + 1] = from_up;
}

// Demonstrate __shfl_xor_sync - butterfly pattern
__global__ void demo_shfl_xor(int* out) {
    int lane = threadIdx.x % 32;
    int val = lane;
    
    // XOR with 1 = exchange with neighbor
    int from_xor = __shfl_xor_sync(FULL_MASK, val, 1);
    
    out[threadIdx.x * 2] = val;
    out[threadIdx.x * 2 + 1] = from_xor;
}

// Practical: Warp reduction using shuffle
__device__ int warp_reduce_sum(int val) {
    // Butterfly reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(FULL_MASK, val, offset);
    }
    return val;
}

__global__ void shuffle_vs_shared_reduce(const int* in, int* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (idx < n) ? in[idx] : 0;
    
    // Warp-level reduction
    int warp_sum = warp_reduce_sum(val);
    
    // First lane of each warp writes result
    if ((threadIdx.x % 32) == 0) {
        atomicAdd(out, warp_sum);
    }
}

int main() {
    printf("=== Shuffle Instructions Demo ===\n\n");
    
    // Part 1: __shfl_sync (broadcast)
    {
        printf("1. __shfl_sync (Broadcast from lane 5)\n");
        printf("─────────────────────────────────────────\n");
        
        int* d_out;
        int h_out[32];
        cudaMalloc(&d_out, 32 * sizeof(int));
        
        demo_shfl_broadcast<<<1, 32>>>(d_out);
        cudaMemcpy(h_out, d_out, 32 * sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("   All lanes received: %d (lane 5's value = 5 * 10)\n\n", h_out[0]);
        cudaFree(d_out);
    }
    
    // Part 2: __shfl_down_sync
    {
        printf("2. __shfl_down_sync (Get from lane + delta)\n");
        printf("─────────────────────────────────────────\n");
        
        int* d_out;
        int h_out[64];
        cudaMalloc(&d_out, 64 * sizeof(int));
        
        demo_shfl_down<<<1, 32>>>(d_out);
        cudaMemcpy(h_out, d_out, 64 * sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("   Lane 0: had %d, got %d from lane 4\n", h_out[0], h_out[1]);
        printf("   Lane 5: had %d, got %d from lane 9\n", h_out[10], h_out[11]);
        printf("   Lane 28: had %d, got %d from lane 32 (unchanged)\n\n", h_out[56], h_out[57]);
        cudaFree(d_out);
    }
    
    // Part 3: __shfl_up_sync
    {
        printf("3. __shfl_up_sync (Get from lane - delta)\n");
        printf("─────────────────────────────────────────\n");
        
        int* d_out;
        int h_out[64];
        cudaMalloc(&d_out, 64 * sizeof(int));
        
        demo_shfl_up<<<1, 32>>>(d_out);
        cudaMemcpy(h_out, d_out, 64 * sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("   Lane 0: had %d, got %d (unchanged, no source)\n", h_out[0], h_out[1]);
        printf("   Lane 5: had %d, got %d from lane 4\n", h_out[10], h_out[11]);
        printf("   Lane 31: had %d, got %d from lane 30\n\n", h_out[62], h_out[63]);
        cudaFree(d_out);
    }
    
    // Part 4: __shfl_xor_sync
    {
        printf("4. __shfl_xor_sync (XOR pattern exchange)\n");
        printf("─────────────────────────────────────────\n");
        
        int* d_out;
        int h_out[64];
        cudaMalloc(&d_out, 64 * sizeof(int));
        
        demo_shfl_xor<<<1, 32>>>(d_out);
        cudaMemcpy(h_out, d_out, 64 * sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("   XOR with mask=1 (exchange with neighbor):\n");
        printf("   Lane 0: had %d, got %d from lane 1\n", h_out[0], h_out[1]);
        printf("   Lane 1: had %d, got %d from lane 0\n", h_out[2], h_out[3]);
        printf("   Lane 4: had %d, got %d from lane 5\n\n", h_out[8], h_out[9]);
        cudaFree(d_out);
    }
    
    // Part 5: Practical reduction
    {
        printf("5. Practical: Warp Reduction with Shuffle\n");
        printf("─────────────────────────────────────────\n");
        
        const int N = 32;
        int h_in[N];
        for (int i = 0; i < N; i++) h_in[i] = 1;  // Sum should be 32
        
        int *d_in, *d_out;
        cudaMalloc(&d_in, N * sizeof(int));
        cudaMalloc(&d_out, sizeof(int));
        cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_out, 0, sizeof(int));
        
        shuffle_vs_shared_reduce<<<1, 32>>>(d_in, d_out, N);
        
        int result;
        cudaMemcpy(&result, d_out, sizeof(int), cudaMemcpyDeviceToHost);
        printf("   Sum of 32 ones = %d\n\n", result);
        
        cudaFree(d_in);
        cudaFree(d_out);
    }
    
    printf("=== Summary ===\n\n");
    printf("  __shfl_sync:      Get from specific lane (broadcast)\n");
    printf("  __shfl_down_sync: Get from lane + offset (reduction)\n");
    printf("  __shfl_up_sync:   Get from lane - offset (prefix scan)\n");
    printf("  __shfl_xor_sync:  Exchange with XOR pattern (butterfly)\n");
    printf("\n  All are ~15x faster than shared memory!\n");
    
    return 0;
}
