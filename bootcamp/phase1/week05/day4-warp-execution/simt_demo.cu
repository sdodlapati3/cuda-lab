/**
 * simt_demo.cu - Demonstrate SIMT execution model
 * 
 * Learning objectives:
 * - Visualize warp lockstep execution
 * - Understand predicated execution
 * - See how divergence works internally
 */

#include <cuda_runtime.h>
#include <cstdio>

// Show that all threads execute together
__global__ void simt_lockstep() {
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    
    // Only print from first warp
    if (warp == 0 && lane < 8) {
        // All lanes 0-7 will print
        // Order may vary but all execute "together"
        printf("Warp 0, Lane %d executing\n", lane);
    }
}

// Demonstrate predicated execution
__global__ void predicated_execution(int* results, int n) {
    int idx = threadIdx.x;
    if (idx >= n) return;
    
    int value;
    
    // This looks like divergence, but compiler may use predication
    // All threads compute both, then select based on predicate
    if (idx % 2 == 0) {
        value = idx * 2;
    } else {
        value = idx * 3;
    }
    
    results[idx] = value;
}

// Show active mask behavior
__global__ void active_mask_demo() {
    int lane = threadIdx.x % 32;
    
    // Get active mask - shows which threads are active
    unsigned mask = __activemask();
    
    if (lane == 0) {
        printf("Before branch: active mask = 0x%08X (%d threads)\n", 
               mask, __popc(mask));
    }
    __syncwarp();
    
    // Half the threads diverge
    if (lane < 16) {
        mask = __activemask();
        if (lane == 0) {
            printf("In 'if' branch: active mask = 0x%08X (%d threads)\n", 
                   mask, __popc(mask));
        }
    } else {
        mask = __activemask();
        if (lane == 16) {
            printf("In 'else' branch: active mask = 0x%08X (%d threads)\n", 
                   mask, __popc(mask));
        }
    }
    __syncwarp();
    
    // After reconvergence
    mask = __activemask();
    if (lane == 0) {
        printf("After reconvergence: active mask = 0x%08X (%d threads)\n", 
               mask, __popc(mask));
    }
}

// Warp-uniform control flow
__global__ void warp_uniform_control(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.x / 32;
    
    // Good: entire warp takes same path
    if (warp_id == 0) {
        // All 32 threads in warp 0 do this
        if (idx < n) data[idx] *= 2.0f;
    } else if (warp_id == 1) {
        // All 32 threads in warp 1 do this
        if (idx < n) data[idx] += 1.0f;
    }
    // No divergence within warps!
}

int main() {
    printf("=== SIMT Execution Model Demo ===\n\n");
    
    // Demo 1: Lockstep execution
    printf("=== Demo 1: SIMT Lockstep ===\n");
    printf("Threads in a warp execute same instruction together.\n");
    printf("Output order may vary, but execution is synchronized:\n\n");
    
    simt_lockstep<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    // Demo 2: Predicated execution
    printf("\n=== Demo 2: Predicated Execution ===\n");
    printf("Short branches may use predication instead of divergence:\n");
    
    int* d_results;
    int h_results[8];
    cudaMalloc(&d_results, 8 * sizeof(int));
    
    predicated_execution<<<1, 8>>>(d_results, 8);
    cudaMemcpy(h_results, d_results, 8 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Results: ");
    for (int i = 0; i < 8; i++) {
        printf("%d ", h_results[i]);
    }
    printf("\n");
    printf("Even indices: idx*2, Odd indices: idx*3\n");
    
    cudaFree(d_results);
    
    // Demo 3: Active mask
    printf("\n=== Demo 3: Active Mask During Divergence ===\n");
    printf("__activemask() shows which threads are currently active:\n\n");
    
    active_mask_demo<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    printf("\n=== Key Concepts ===\n");
    printf("1. SIMT: All warp threads share instruction pointer\n");
    printf("2. Divergence: Paths execute serially, threads masked out\n");
    printf("3. __activemask(): Query which threads are active\n");
    printf("4. Reconvergence: Threads rejoin after branch\n");
    printf("5. Predication: Compiler optimization for short branches\n");
    
    return 0;
}
