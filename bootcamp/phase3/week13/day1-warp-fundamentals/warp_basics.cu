/**
 * warp_basics.cu - Understanding warp fundamentals
 * 
 * Learning objectives:
 * - See how warps are formed
 * - Measure divergence cost
 * - Understand lane IDs
 */

#include <cuda_runtime.h>
#include <cstdio>

// Demonstrate warp and lane IDs
__global__ void show_warp_structure(int* warp_ids, int* lane_ids) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Lane within warp (0-31)
    int lane = threadIdx.x % 32;
    // Which warp within block
    int warp = threadIdx.x / 32;
    
    warp_ids[tid] = warp;
    lane_ids[tid] = lane;
}

// No divergence - all threads same path
__global__ void no_divergence(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = 1.0f;
        for (int i = 0; i < 100; i++) {
            val = val * 1.01f + 0.01f;
        }
        out[idx] = val;
    }
}

// Mild divergence - warp-aligned branches
__global__ void warp_aligned_divergence(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.x / 32;
    
    if (idx < n) {
        float val = 1.0f;
        
        // Different warps take different paths - OK!
        if (warp_id % 2 == 0) {
            for (int i = 0; i < 100; i++) {
                val = val * 1.01f + 0.01f;
            }
        } else {
            for (int i = 0; i < 100; i++) {
                val = val * 0.99f - 0.01f;
            }
        }
        out[idx] = val;
    }
}

// Bad divergence - threads within warp diverge
__global__ void bad_divergence(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    
    if (idx < n) {
        float val = 1.0f;
        
        // EVERY OTHER THREAD takes different path - BAD!
        if (lane % 2 == 0) {
            for (int i = 0; i < 100; i++) {
                val = val * 1.01f + 0.01f;
            }
        } else {
            for (int i = 0; i < 100; i++) {
                val = val * 0.99f - 0.01f;
            }
        }
        out[idx] = val;
    }
}

// Worst divergence - every thread unique path
__global__ void worst_divergence(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    
    if (idx < n) {
        float val = 1.0f;
        
        // Loop count depends on lane - serialized!
        int iters = 50 + lane * 3;  // Each thread different iterations
        for (int i = 0; i < iters; i++) {
            val = val * 1.01f + 0.01f;
        }
        out[idx] = val;
    }
}

int main() {
    printf("=== Warp Fundamentals Demo ===\n\n");
    
    // Part 1: Show warp structure
    {
        printf("Part 1: Warp Structure\n");
        printf("─────────────────────────────────\n");
        
        const int N = 128;  // 4 warps
        int *d_warp, *d_lane;
        int h_warp[N], h_lane[N];
        
        cudaMalloc(&d_warp, N * sizeof(int));
        cudaMalloc(&d_lane, N * sizeof(int));
        
        show_warp_structure<<<1, 128>>>(d_warp, d_lane);
        
        cudaMemcpy(h_warp, d_warp, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_lane, d_lane, N * sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("ThreadIdx → Warp ID, Lane ID:\n");
        for (int w = 0; w < 4; w++) {
            printf("  Warp %d: threads %d-%d (lanes 0-31)\n", 
                   w, w*32, w*32+31);
        }
        printf("\n");
        
        cudaFree(d_warp);
        cudaFree(d_lane);
    }
    
    // Part 2: Divergence cost
    {
        printf("Part 2: Divergence Cost\n");
        printf("─────────────────────────────────\n");
        
        const int N = 1 << 20;
        float* d_out;
        cudaMalloc(&d_out, N * sizeof(float));
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        
        printf("%-30s | %10s | %10s\n", "Pattern", "Time (ms)", "Slowdown");
        printf("───────────────────────────────────────────────────\n");
        
        float baseline;
        
        // No divergence
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            no_divergence<<<blocks, threads>>>(d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&baseline, start, stop);
        printf("%-30s | %10.4f | %9.2fx\n", "No divergence", baseline/100, 1.0);
        
        // Warp-aligned
        float t1;
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            warp_aligned_divergence<<<blocks, threads>>>(d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t1, start, stop);
        printf("%-30s | %10.4f | %9.2fx\n", "Warp-aligned (OK)", t1/100, t1/baseline);
        
        // Bad - thread divergence
        float t2;
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            bad_divergence<<<blocks, threads>>>(d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t2, start, stop);
        printf("%-30s | %10.4f | %9.2fx\n", "Every-other-thread (BAD)", t2/100, t2/baseline);
        
        // Worst - unique paths
        float t3;
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            worst_divergence<<<blocks, threads>>>(d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t3, start, stop);
        printf("%-30s | %10.4f | %9.2fx\n", "Unique iterations (WORST)", t3/100, t3/baseline);
        
        cudaFree(d_out);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    printf("\n=== Key Takeaways ===\n\n");
    printf("1. Warp = 32 threads executing in lockstep\n");
    printf("2. Lane ID = threadIdx.x %% 32 (position within warp)\n");
    printf("3. Divergence within warp = serialization\n");
    printf("4. Divergence between warps = OK\n");
    printf("5. Design branches to be warp-aligned when possible\n");
    
    return 0;
}
