/**
 * fusion_strategies.cu - When to fuse and when not to
 * 
 * Learning objectives:
 * - Identify good fusion candidates
 * - Recognize anti-patterns
 * - Make data-driven decisions
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// ============================================================================
// Example 1: Good Fusion - Element-wise Chain
// ============================================================================

// These SHOULD be fused
__global__ void unfused_chain_1(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = x[idx] + 1.0f;
}

__global__ void unfused_chain_2(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = x[idx] * 2.0f;
}

__global__ void unfused_chain_3(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = sqrtf(x[idx]);
}

__global__ void fused_good(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx] + 1.0f;
        val = val * 2.0f;
        y[idx] = sqrtf(val);
    }
}

// ============================================================================
// Example 2: Register Pressure Anti-Pattern
// ============================================================================

// Too much state - may cause register spilling
__global__ void over_fused(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Accumulate too many intermediate values
        float v1 = x[idx];
        float v2 = v1 * 1.1f;
        float v3 = v2 + v1;
        float v4 = sinf(v3);
        float v5 = cosf(v2);
        float v6 = v4 * v5;
        float v7 = expf(v6);
        float v8 = logf(v7 + 1.0f);
        float v9 = tanhf(v8);
        float v10 = v9 * v3 + v4 * v5;
        float v11 = sqrtf(fabsf(v10));
        float v12 = v11 + v1 + v2 + v3;
        y[idx] = v12;
    }
}

// Better: break into stages if register pressure is high
__global__ void staged_compute(const float* x, float* temp, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v1 = x[idx];
        float v2 = v1 * 1.1f;
        float v3 = v2 + v1;
        float v4 = sinf(v3);
        float v5 = cosf(v2);
        temp[idx] = v4 * v5;  // Checkpoint
    }
}

__global__ void staged_compute_2(const float* temp, const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v6 = temp[idx];
        float v1 = x[idx];
        float v7 = expf(v6);
        float v8 = logf(v7 + 1.0f);
        float v9 = tanhf(v8);
        y[idx] = sqrtf(fabsf(v9 * v1));
    }
}

// ============================================================================
// Example 3: Different Parallelization - Don't Force Fusion
// ============================================================================

// Row-wise operation
__global__ void row_sum(const float* matrix, float* row_sums, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            sum += matrix[row * cols + c];
        }
        row_sums[row] = sum;
    }
}

// Column-wise operation
__global__ void col_normalize(float* matrix, const float* col_means, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        for (int r = 0; r < rows; r++) {
            matrix[r * cols + col] -= col_means[col];
        }
    }
}
// These have DIFFERENT parallelization - don't fuse!

// ============================================================================
// Example 4: Compute-Bound - Fusion Won't Help
// ============================================================================

__global__ void compute_heavy(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        // Heavy compute - already compute bound
        #pragma unroll
        for (int i = 0; i < 100; i++) {
            val = sinf(val) * cosf(val) + tanf(val);
        }
        y[idx] = val;
    }
}

__global__ void light_postprocess(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] + 1.0f;  // Very light
    }
}
// Fusing these won't help much - compute_heavy dominates

int main() {
    printf("=== Fusion Strategies Demo ===\n\n");
    
    const int N = 1 << 20;
    const int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float *d_x, *d_t1, *d_t2, *d_out;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_t1, N * sizeof(float));
    cudaMalloc(&d_t2, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    
    // Initialize
    float* h_x = new float[N];
    for (int i = 0; i < N; i++) h_x[i] = 1.0f;
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // ========================================================================
    // Test 1: Good Fusion Candidate
    // ========================================================================
    {
        printf("1. Good Fusion: Element-wise Chain\n");
        printf("─────────────────────────────────────────\n");
        
        // Unfused
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            unfused_chain_1<<<GRID, BLOCK>>>(d_x, d_t1, N);
            unfused_chain_2<<<GRID, BLOCK>>>(d_t1, d_t2, N);
            unfused_chain_3<<<GRID, BLOCK>>>(d_t2, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float unfused_ms;
        cudaEventElapsedTime(&unfused_ms, start, stop);
        unfused_ms /= 100;
        
        // Fused
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            fused_good<<<GRID, BLOCK>>>(d_x, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float fused_ms;
        cudaEventElapsedTime(&fused_ms, start, stop);
        fused_ms /= 100;
        
        printf("   ✅ SHOULD FUSE\n");
        printf("   Unfused: %.3f ms, Fused: %.3f ms\n", unfused_ms, fused_ms);
        printf("   Speedup: %.2fx\n\n", unfused_ms / fused_ms);
    }
    
    // ========================================================================
    // Test 2: Register Pressure Scenario
    // ========================================================================
    {
        printf("2. Register Pressure Analysis\n");
        printf("─────────────────────────────────────────\n");
        
        // Over-fused (may spill)
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            over_fused<<<GRID, BLOCK>>>(d_x, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float over_ms;
        cudaEventElapsedTime(&over_ms, start, stop);
        over_ms /= 100;
        
        // Staged (controlled)
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            staged_compute<<<GRID, BLOCK>>>(d_x, d_t1, N);
            staged_compute_2<<<GRID, BLOCK>>>(d_t1, d_x, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float staged_ms;
        cudaEventElapsedTime(&staged_ms, start, stop);
        staged_ms /= 100;
        
        printf("   ⚠️  PROFILE BEFORE DECIDING\n");
        printf("   Over-fused: %.3f ms\n", over_ms);
        printf("   Staged:     %.3f ms\n", staged_ms);
        printf("   Note: Check register usage with --ptxas-options=-v\n\n");
    }
    
    // ========================================================================
    // Test 3: Compute-Bound Case
    // ========================================================================
    {
        printf("3. Compute-Bound: Limited Fusion Benefit\n");
        printf("─────────────────────────────────────────\n");
        
        // Heavy compute alone
        cudaEventRecord(start);
        for (int i = 0; i < 10; i++) {
            compute_heavy<<<GRID, BLOCK>>>(d_x, d_t1, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float heavy_ms;
        cudaEventElapsedTime(&heavy_ms, start, stop);
        heavy_ms /= 10;
        
        // Heavy + light (unfused)
        cudaEventRecord(start);
        for (int i = 0; i < 10; i++) {
            compute_heavy<<<GRID, BLOCK>>>(d_x, d_t1, N);
            light_postprocess<<<GRID, BLOCK>>>(d_t1, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float both_ms;
        cudaEventElapsedTime(&both_ms, start, stop);
        both_ms /= 10;
        
        printf("   ❌ LIMITED BENEFIT FROM FUSION\n");
        printf("   Heavy kernel: %.3f ms (dominates runtime)\n", heavy_ms);
        printf("   Heavy + light: %.3f ms\n", both_ms);
        printf("   Light adds only: %.3f ms (%.1f%%)\n", 
               both_ms - heavy_ms, 100 * (both_ms - heavy_ms) / heavy_ms);
        printf("   Fusion would save ~%.3f ms\n\n", both_ms - heavy_ms);
    }
    
    // ========================================================================
    // Decision Matrix
    // ========================================================================
    printf("═══════════════════════════════════════════════════════════\n");
    printf("FUSION DECISION MATRIX\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    
    printf("┌───────────────────────────────────────────────────────────┐\n");
    printf("│ Scenario                          │ Action                │\n");
    printf("├───────────────────────────────────────────────────────────┤\n");
    printf("│ Element-wise chain                │ ✅ Always fuse        │\n");
    printf("│ Transform + Reduce                │ ✅ Fuse when possible │\n");
    printf("│ Same memory access pattern        │ ✅ Strong candidate   │\n");
    printf("│ High register pressure            │ ⚠️  Profile first     │\n");
    printf("│ Different parallelization         │ ❌ Keep separate      │\n");
    printf("│ Compute-bound kernel              │ ❌ Limited benefit    │\n");
    printf("│ Debugging needed                  │ ❌ Keep unfused copy  │\n");
    printf("└───────────────────────────────────────────────────────────┘\n\n");
    
    printf("CHECK REGISTER USAGE:\n");
    printf("  nvcc --ptxas-options=-v -o kernel kernel.cu\n");
    printf("  Look for: Used XX registers, XXXX bytes smem\n\n");
    
    // Cleanup
    cudaFree(d_x);
    cudaFree(d_t1);
    cudaFree(d_t2);
    cudaFree(d_out);
    delete[] h_x;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
