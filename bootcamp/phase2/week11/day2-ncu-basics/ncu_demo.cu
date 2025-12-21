/**
 * ncu_demo.cu - Kernels for NCU analysis
 * 
 * Profile with: ncu --set full ./build/ncu_demo
 */

#include <cuda_runtime.h>
#include <cstdio>

// Memory-bound kernel - copy
__global__ void memory_bound(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;  // 1 FLOP, 8 bytes
    }
}

// Compute-bound kernel - heavy math
__global__ void compute_bound(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        // 200 FMAs = 400 FLOPs per 8 bytes = AI of 50
        #pragma unroll
        for (int i = 0; i < 200; i++) {
            val = val * 1.0001f + 0.0001f;
        }
        out[idx] = val;
    }
}

// Balanced kernel - mixed
__global__ void balanced_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        // Moderate compute
        for (int i = 0; i < 20; i++) {
            val = val * 1.01f + 0.01f;
        }
        out[idx] = val;
    }
}

// Inefficient kernel - poor memory pattern
__global__ void inefficient_strided(const float* in, float* out, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int real_idx = (idx * stride) % n;
    if (idx < n / stride) {
        out[real_idx] = in[real_idx] * 2.0f;
    }
}

// Inefficient kernel - divergent branches
__global__ void inefficient_divergent(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        // Different threads in warp take different branches
        if (idx % 32 < 16) {
            for (int i = 0; i < 50; i++) val *= 1.01f;
        } else {
            for (int i = 0; i < 100; i++) val += 0.01f;
        }
        out[idx] = val;
    }
}

int main() {
    printf("NCU Demo - Profile Kernels with Different Characteristics\n\n");
    printf("Profile with:\n");
    printf("  ncu --set full ./build/ncu_demo\n");
    printf("  ncu --set roofline ./build/ncu_demo\n\n");
    
    const int N = 1 << 22;  // 4M elements
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    cudaMemcpy(d_in, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    printf("Launching kernels for profiling...\n\n");
    
    // Launch each kernel once for profiling
    printf("1. memory_bound - expect high BW, low compute\n");
    memory_bound<<<blocks, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    
    printf("2. compute_bound - expect low BW, high compute\n");
    compute_bound<<<blocks, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    
    printf("3. balanced_kernel - expect moderate both\n");
    balanced_kernel<<<blocks, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    
    printf("4. inefficient_strided - expect low BW efficiency\n");
    inefficient_strided<<<blocks/32, threads>>>(d_in, d_out, N, 32);
    cudaDeviceSynchronize();
    
    printf("5. inefficient_divergent - expect warp divergence\n");
    inefficient_divergent<<<blocks, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    
    printf("\n=== NCU Metric Interpretation Guide ===\n\n");
    printf("Metric Category          | Good Value | Problem If...\n");
    printf("---------------------------------------------------------\n");
    printf("DRAM Throughput          | Near peak  | Far below peak\n");
    printf("Compute Throughput       | Near peak  | Far below peak\n");
    printf("Memory %%                 | < Compute  | You're memory-bound\n");
    printf("Warp Stall (LG Throttle) | Low        | Memory waiting\n");
    printf("Warp Stall (Not Selected)| Moderate   | Normal scheduling\n");
    printf("Branch Efficiency        | 100%%       | Divergence issues\n");
    printf("Global Load Efficiency   | 100%%       | Uncoalesced access\n");
    
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_data;
    
    return 0;
}
