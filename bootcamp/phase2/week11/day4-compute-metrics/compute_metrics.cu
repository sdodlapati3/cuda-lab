/**
 * compute_metrics.cu - Kernels demonstrating compute patterns
 * 
 * Profile with: ncu --section ComputeWorkloadAnalysis ./build/compute_metrics
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// High FMA utilization
__global__ void fma_heavy(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a = 1.0001f, b = 0.9999f;
        float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        
        #pragma unroll
        for (int i = 0; i < 256; i++) {
            c0 = a * c0 + b;
            c1 = a * c1 + b;
            c2 = a * c2 + b;
            c3 = a * c3 + b;
        }
        out[idx] = c0 + c1 + c2 + c3;
    }
}

// Transcendental functions (special function unit)
__global__ void transcendental_heavy(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = (float)idx * 0.001f + 0.1f;
        float result = 0;
        
        for (int i = 0; i < 50; i++) {
            result += sinf(val + (float)i * 0.01f);
            result += cosf(val + (float)i * 0.01f);
            result += expf(-val * 0.01f);
        }
        out[idx] = result;
    }
}

// Integer operations
__global__ void integer_heavy(int* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = idx;
        
        #pragma unroll
        for (int i = 0; i < 200; i++) {
            val = val * 31 + 17;
            val = val ^ (val >> 3);
            val = val | (val << 2);
        }
        out[idx] = val;
    }
}

// Low ILP - serial dependencies
__global__ void serial_deps(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = 1.0f;
        // Each depends on previous - no ILP
        for (int i = 0; i < 500; i++) {
            val = val * 1.0001f + 0.0001f;
        }
        out[idx] = val;
    }
}

// High ILP - independent chains
__global__ void parallel_chains(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 8 independent chains
        float v0 = 1.0f, v1 = 1.1f, v2 = 1.2f, v3 = 1.3f;
        float v4 = 1.4f, v5 = 1.5f, v6 = 1.6f, v7 = 1.7f;
        
        for (int i = 0; i < 62; i++) {  // 62*8 ≈ 500
            v0 = v0 * 1.0001f + 0.0001f;
            v1 = v1 * 1.0002f + 0.0002f;
            v2 = v2 * 1.0003f + 0.0003f;
            v3 = v3 * 1.0004f + 0.0004f;
            v4 = v4 * 1.0005f + 0.0005f;
            v5 = v5 * 1.0006f + 0.0006f;
            v6 = v6 * 1.0007f + 0.0007f;
            v7 = v7 * 1.0008f + 0.0008f;
        }
        out[idx] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
    }
}

// Divergent branches
__global__ void divergent_branches(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = 1.0f;
        int lane = threadIdx.x % 32;
        
        // Each thread in warp takes different path
        if (lane < 8) {
            for (int i = 0; i < 100; i++) val *= 1.01f;
        } else if (lane < 16) {
            for (int i = 0; i < 50; i++) val += 0.01f;
        } else if (lane < 24) {
            for (int i = 0; i < 75; i++) val -= 0.005f;
        } else {
            for (int i = 0; i < 25; i++) val /= 0.99f;
        }
        out[idx] = val;
    }
}

int main() {
    printf("Compute Metrics Demo\n\n");
    printf("Profile with: ncu --section ComputeWorkloadAnalysis ./build/compute_metrics\n\n");
    
    const int N = 1 << 20;  // 1M
    float *d_float_out;
    int *d_int_out;
    
    cudaMalloc(&d_float_out, N * sizeof(float));
    cudaMalloc(&d_int_out, N * sizeof(int));
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    printf("Launching kernels with different compute patterns...\n\n");
    
    printf("1. fma_heavy - high FMA pipe utilization\n");
    fma_heavy<<<blocks, threads>>>(d_float_out, N);
    cudaDeviceSynchronize();
    
    printf("2. transcendental_heavy - SFU (special function) usage\n");
    transcendental_heavy<<<blocks, threads>>>(d_float_out, N);
    cudaDeviceSynchronize();
    
    printf("3. integer_heavy - integer ALU usage\n");
    integer_heavy<<<blocks, threads>>>(d_int_out, N);
    cudaDeviceSynchronize();
    
    printf("4. serial_deps - low ILP due to dependencies\n");
    serial_deps<<<blocks, threads>>>(d_float_out, N);
    cudaDeviceSynchronize();
    
    printf("5. parallel_chains - high ILP with independent ops\n");
    parallel_chains<<<blocks, threads>>>(d_float_out, N);
    cudaDeviceSynchronize();
    
    printf("6. divergent_branches - warp divergence\n");
    divergent_branches<<<blocks, threads>>>(d_float_out, N);
    cudaDeviceSynchronize();
    
    printf("\n=== Compute Metrics Guide ===\n\n");
    printf("Metric                      | What it shows\n");
    printf("------------------------------------------------------\n");
    printf("SM Throughput               | Overall SM utilization\n");
    printf("Pipe FMA Active             | FMA unit utilization\n");
    printf("Pipe ALU Active             | Integer ALU utilization\n");
    printf("Warp Stall: Pipe Busy       | Functional unit contention\n");
    printf("Warp Stall: Short Scoreboard| Data dependency stalls\n");
    printf("Branch Efficiency           | Divergence impact\n");
    printf("\nLook for 'Compute Workload Analysis' section in ncu!\n");
    
    cudaFree(d_float_out);
    cudaFree(d_int_out);
    
    return 0;
}
