/**
 * producer_consumer.cu - Chain operations through registers/shared memory
 * 
 * Learning objectives:
 * - Register-based producer-consumer
 * - Warp shuffle handoff
 * - Shared memory staging
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define FULL_MASK 0xffffffff

// ============================================================================
// Pattern 1: Register-based (same thread)
// ============================================================================

// Unfused: write to memory between stages
__global__ void unfused_pipeline(const float* in, float* temp, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Stage 1: produce to temp
        temp[idx] = in[idx] * 2.0f;
    }
}

__global__ void unfused_pipeline_stage2(const float* temp, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Stage 2: consume from temp
        out[idx] = sqrtf(temp[idx]);
    }
}

// Fused: keep in register
__global__ void fused_pipeline(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];           // Load
        float produced = val * 2.0f;   // Produce (register)
        out[idx] = sqrtf(produced);    // Consume (register) + store
    }
}

// ============================================================================
// Pattern 2: Warp Shuffle Handoff
// ============================================================================

// Unfused: go through shared memory
__global__ void unfused_warp_handoff(const float* in, float* out, int n) {
    __shared__ float buffer[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    
    // Stage 1: Produce
    float produced = (idx < n) ? in[idx] * 2.0f : 0.0f;
    buffer[threadIdx.x] = produced;
    __syncthreads();
    
    // Stage 2: Consume from neighbor (via shared memory)
    int neighbor = (threadIdx.x + 1) % blockDim.x;
    float consumed = buffer[neighbor];
    
    if (idx < n) {
        out[idx] = produced + consumed;
    }
}

// Fused: warp shuffle
__global__ void fused_warp_handoff(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Stage 1: Produce
    float produced = (idx < n) ? in[idx] * 2.0f : 0.0f;
    
    // Stage 2: Consume from neighbor via shuffle (no shared memory!)
    float neighbor_val = __shfl_down_sync(FULL_MASK, produced, 1);
    
    if (idx < n - 1) {  // Last lane has undefined neighbor
        out[idx] = produced + neighbor_val;
    }
}

// ============================================================================
// Pattern 3: Multi-Stage Pipeline in Shared Memory
// ============================================================================

__global__ void three_stage_pipeline(const float* in, float* out, int n) {
    __shared__ float stage1_out[256];
    __shared__ float stage2_out[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Stage 1: Load and transform
    float val = (idx < n) ? in[idx] : 0.0f;
    stage1_out[tid] = val * 2.0f;
    __syncthreads();
    
    // Stage 2: Stencil operation (use neighbors)
    float left = (tid > 0) ? stage1_out[tid - 1] : 0.0f;
    float right = (tid < blockDim.x - 1) ? stage1_out[tid + 1] : 0.0f;
    stage2_out[tid] = (left + stage1_out[tid] + right) / 3.0f;
    __syncthreads();
    
    // Stage 3: Final transform
    if (idx < n) {
        out[idx] = sqrtf(stage2_out[tid]);
    }
}

// Optimized: Reduce shared memory by computing in registers where possible
__global__ void optimized_three_stage(const float* in, float* out, int n) {
    __shared__ float buffer[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    
    // Stage 1: Load and transform (register)
    float val = (idx < n) ? in[idx] : 0.0f;
    float stage1 = val * 2.0f;  // In register!
    
    // For stencil, we need neighbors - use shuffle within warp
    float left_warp = __shfl_up_sync(FULL_MASK, stage1, 1);
    float right_warp = __shfl_down_sync(FULL_MASK, stage1, 1);
    
    // Handle warp boundaries via shared memory
    buffer[tid] = stage1;
    __syncthreads();
    
    float left = (tid > 0) ? buffer[tid - 1] : 0.0f;
    float right = (tid < blockDim.x - 1) ? buffer[tid + 1] : 0.0f;
    
    // Within warp, prefer shuffle results
    if (lane > 0) left = left_warp;
    if (lane < 31) right = right_warp;
    
    // Stage 2: Stencil (register)
    float stage2 = (left + stage1 + right) / 3.0f;
    
    // Stage 3: Final (register)
    if (idx < n) {
        out[idx] = sqrtf(stage2);
    }
}

int main() {
    printf("=== Producer-Consumer Fusion Demo ===\n\n");
    
    const int N = 1 << 20;
    const int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float *d_in, *d_temp, *d_out1, *d_out2;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_temp, N * sizeof(float));
    cudaMalloc(&d_out1, N * sizeof(float));
    cudaMalloc(&d_out2, N * sizeof(float));
    
    // Initialize
    float* h_in = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // ========================================================================
    // Test 1: Register-based Producer-Consumer
    // ========================================================================
    {
        printf("1. Register-based Producer-Consumer\n");
        printf("─────────────────────────────────────────\n");
        
        // Unfused
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            unfused_pipeline<<<GRID, BLOCK>>>(d_in, d_temp, d_out1, N);
            unfused_pipeline_stage2<<<GRID, BLOCK>>>(d_temp, d_out1, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float unfused_ms;
        cudaEventElapsedTime(&unfused_ms, start, stop);
        unfused_ms /= 100;
        
        // Fused
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            fused_pipeline<<<GRID, BLOCK>>>(d_in, d_out2, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float fused_ms;
        cudaEventElapsedTime(&fused_ms, start, stop);
        fused_ms /= 100;
        
        printf("   Unfused (2 kernels + temp): %.3f ms\n", unfused_ms);
        printf("   Fused (1 kernel, registers): %.3f ms\n", fused_ms);
        printf("   Speedup: %.2fx\n\n", unfused_ms / fused_ms);
    }
    
    // ========================================================================
    // Test 2: Warp Shuffle Handoff
    // ========================================================================
    {
        printf("2. Warp Shuffle Handoff\n");
        printf("─────────────────────────────────────────\n");
        
        // Unfused (shared memory)
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            unfused_warp_handoff<<<GRID, BLOCK>>>(d_in, d_out1, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float unfused_ms;
        cudaEventElapsedTime(&unfused_ms, start, stop);
        unfused_ms /= 100;
        
        // Fused (warp shuffle)
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            fused_warp_handoff<<<GRID, BLOCK>>>(d_in, d_out2, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float fused_ms;
        cudaEventElapsedTime(&fused_ms, start, stop);
        fused_ms /= 100;
        
        printf("   Shared memory approach: %.3f ms\n", unfused_ms);
        printf("   Warp shuffle approach:  %.3f ms\n", fused_ms);
        printf("   Speedup: %.2fx\n\n", unfused_ms / fused_ms);
    }
    
    // ========================================================================
    // Test 3: Multi-Stage Pipeline
    // ========================================================================
    {
        printf("3. Multi-Stage Pipeline Optimization\n");
        printf("─────────────────────────────────────────\n");
        
        // Basic 3-stage
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            three_stage_pipeline<<<GRID, BLOCK>>>(d_in, d_out1, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float basic_ms;
        cudaEventElapsedTime(&basic_ms, start, stop);
        basic_ms /= 100;
        
        // Optimized (shuffle + reduced shared mem)
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            optimized_three_stage<<<GRID, BLOCK>>>(d_in, d_out2, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float opt_ms;
        cudaEventElapsedTime(&opt_ms, start, stop);
        opt_ms /= 100;
        
        printf("   Basic (2 shared arrays): %.3f ms\n", basic_ms);
        printf("   Optimized (shuffle + 1): %.3f ms\n", opt_ms);
        printf("   Speedup: %.2fx\n\n", basic_ms / opt_ms);
    }
    
    printf("=== Key Points ===\n\n");
    printf("1. Register-to-register: zero latency, always prefer\n");
    printf("2. Warp shuffle: ~5 cycles, no shared memory needed\n");
    printf("3. Shared memory: ~30 cycles, for cross-warp communication\n");
    printf("4. Pipeline stages that stay in registers are free\n");
    
    // Cleanup
    cudaFree(d_in);
    cudaFree(d_temp);
    cudaFree(d_out1);
    cudaFree(d_out2);
    delete[] h_in;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
