/**
 * integration_demo.cu - Phase 2 Capstone: Putting It All Together
 * 
 * This demonstrates applying ALL concepts from Phase 2:
 * - Roofline analysis
 * - Occupancy optimization
 * - Profiling metrics
 * - Latency hiding
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// ============================================================================
// EXAMPLE: Optimizing a SAXPY-like kernel through multiple stages
// ============================================================================

// V1: Naive implementation
__global__ void saxpy_v1_naive(const float* x, const float* y, float* z, 
                                float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = a * x[idx] + y[idx];
    }
}

// V2: Vectorized (float4) - better memory throughput
__global__ void saxpy_v2_vectorized(const float4* x, const float4* y, 
                                     float4* z, float a, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 vx = x[idx];
        float4 vy = y[idx];
        z[idx] = make_float4(a * vx.x + vy.x,
                             a * vx.y + vy.y,
                             a * vx.z + vy.z,
                             a * vx.w + vy.w);
    }
}

// V3: ILP - multiple elements per thread
__global__ void saxpy_v3_ilp(const float4* x, const float4* y, 
                              float4* z, float a, int n4) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    
    if (idx + 1 < n4) {
        // Load 2 float4s = 8 elements
        float4 vx0 = x[idx];
        float4 vx1 = x[idx + 1];
        float4 vy0 = y[idx];
        float4 vy1 = y[idx + 1];
        
        // Independent computations
        z[idx] = make_float4(a * vx0.x + vy0.x, a * vx0.y + vy0.y,
                             a * vx0.z + vy0.z, a * vx0.w + vy0.w);
        z[idx+1] = make_float4(a * vx1.x + vy1.x, a * vx1.y + vy1.y,
                               a * vx1.z + vy1.z, a * vx1.w + vy1.w);
    }
}

// ============================================================================
// Analysis functions
// ============================================================================

void analyze_roofline(int n, float time_ms) {
    // SAXPY: 2 FLOPs per element (multiply + add)
    // 3 memory ops: read x, read y, write z = 12 bytes per element
    float flops = 2.0f * n;
    float bytes = 3.0f * n * sizeof(float);
    float ai = flops / bytes;
    
    float achieved_gflops = (flops / 1e9) / (time_ms / 1000);
    float achieved_gbps = (bytes / 1e9) / (time_ms / 1000);
    
    printf("\n  Roofline Analysis:\n");
    printf("    Arithmetic Intensity: %.3f FLOPs/byte\n", ai);
    printf("    Achieved: %.2f GFLOP/s, %.2f GB/s\n", achieved_gflops, achieved_gbps);
    printf("    This kernel is MEMORY-BOUND (AI < 10)\n");
}

void analyze_occupancy(int block_size, int registers_per_thread = 32) {
    // A100 specifics
    const int max_threads_per_sm = 2048;
    const int max_blocks_per_sm = 32;
    const int max_registers_per_sm = 65536;
    
    int warps_per_block = block_size / 32;
    int registers_per_block = registers_per_thread * block_size;
    
    // Calculate limiting factors
    int blocks_by_threads = max_threads_per_sm / block_size;
    int blocks_by_regs = max_registers_per_sm / registers_per_block;
    int blocks_by_limit = max_blocks_per_sm;
    
    int active_blocks = min(blocks_by_threads, min(blocks_by_regs, blocks_by_limit));
    int active_warps = active_blocks * warps_per_block;
    float occupancy = 100.0f * active_warps / (max_threads_per_sm / 32);
    
    printf("\n  Occupancy Analysis (block_size=%d):\n", block_size);
    printf("    Warps per block: %d\n", warps_per_block);
    printf("    Active blocks per SM: %d\n", active_blocks);
    printf("    Active warps per SM: %d\n", active_warps);
    printf("    Theoretical occupancy: %.1f%%\n", occupancy);
}

int main() {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("           PHASE 2 CAPSTONE: INTEGRATION DEMO\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    const int N = 1 << 24;  // 16M elements
    const int bytes = N * sizeof(float);
    
    float *d_x, *d_y, *d_z;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_z, bytes);
    
    // Initialize
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    cudaMemcpy(d_x, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_data, bytes, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const float a = 2.0f;
    const int block_size = 256;
    const int iters = 100;
    
    printf("Data size: %d elements (%.1f MB per array)\n", N, bytes / 1e6);
    printf("Testing %d iterations each\n\n", iters);
    
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("VERSION COMPARISON\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    float baseline_ms = 0;
    
    // V1: Naive
    {
        printf("V1: Naive SAXPY\n");
        printf("───────────────────────────────────────────\n");
        
        int blocks = (N + block_size - 1) / block_size;
        
        cudaEventRecord(start);
        for (int i = 0; i < iters; i++) {
            saxpy_v1_naive<<<blocks, block_size>>>(d_x, d_y, d_z, a, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&baseline_ms, start, stop);
        baseline_ms /= iters;
        
        printf("  Time: %.4f ms\n", baseline_ms);
        printf("  Speedup: 1.00x (baseline)\n");
        
        analyze_roofline(N, baseline_ms);
        analyze_occupancy(block_size);
    }
    
    // V2: Vectorized
    {
        printf("\nV2: Vectorized (float4)\n");
        printf("───────────────────────────────────────────\n");
        
        int n4 = N / 4;
        int blocks = (n4 + block_size - 1) / block_size;
        
        cudaEventRecord(start);
        for (int i = 0; i < iters; i++) {
            saxpy_v2_vectorized<<<blocks, block_size>>>((float4*)d_x, (float4*)d_y, 
                                                         (float4*)d_z, a, n4);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= iters;
        
        printf("  Time: %.4f ms\n", ms);
        printf("  Speedup: %.2fx\n", baseline_ms / ms);
        printf("\n  Why better?\n");
        printf("    → Vectorized loads: 1 transaction for 4 floats\n");
        printf("    → Better memory throughput utilization\n");
    }
    
    // V3: ILP
    {
        printf("\nV3: ILP (8 elements per thread)\n");
        printf("───────────────────────────────────────────\n");
        
        int n4 = N / 4;
        int threads_needed = n4 / 2;
        int blocks = (threads_needed + block_size - 1) / block_size;
        
        cudaEventRecord(start);
        for (int i = 0; i < iters; i++) {
            saxpy_v3_ilp<<<blocks, block_size>>>((float4*)d_x, (float4*)d_y, 
                                                  (float4*)d_z, a, n4);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= iters;
        
        printf("  Time: %.4f ms\n", ms);
        printf("  Speedup: %.2fx\n", baseline_ms / ms);
        printf("\n  Why better?\n");
        printf("    → More independent operations\n");
        printf("    → Better instruction-level parallelism\n");
        printf("    → Memory latency better hidden\n");
    }
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("OPTIMIZATION WORKFLOW SUMMARY\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    printf("1. ROOFLINE FIRST\n");
    printf("   • Calculated AI = 0.167 → Memory-bound\n");
    printf("   • Focus on memory optimizations\n\n");
    
    printf("2. OCCUPANCY CHECK\n");
    printf("   • 256 threads/block → Good occupancy\n");
    printf("   • Not the bottleneck here\n\n");
    
    printf("3. PROFILE WITH NCU\n");
    printf("   • Would show memory bandwidth utilization\n");
    printf("   • L2 cache hit rates\n");
    printf("   • Stall reasons\n\n");
    
    printf("4. APPLY LATENCY HIDING\n");
    printf("   • Vectorization: Better memory access\n");
    printf("   • ILP: More ops in flight per thread\n");
    printf("   • TLP: Enough warps (already good)\n\n");
    
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("YOUR OPTIMIZATION CHECKLIST\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    printf("□ Step 1: Profile to identify bottleneck type\n");
    printf("□ Step 2: Calculate arithmetic intensity\n");
    printf("□ Step 3: Check occupancy (is it limiting?)\n");
    printf("□ Step 4: Apply targeted optimizations:\n");
    printf("    Memory-bound → Coalesce, vectorize, cache\n");
    printf("    Compute-bound → Reduce ops, use fast math, unroll\n");
    printf("    Latency-bound → More warps, ILP, prefetch\n");
    printf("□ Step 5: Re-profile and iterate\n\n");
    
    printf("Commands to profile this code:\n");
    printf("  nsys profile ./build/integration_demo\n");
    printf("  ncu --set full ./build/integration_demo\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    delete[] h_data;
    
    return 0;
}
