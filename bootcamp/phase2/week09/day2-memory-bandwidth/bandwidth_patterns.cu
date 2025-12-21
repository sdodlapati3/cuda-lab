/**
 * bandwidth_patterns.cu - How access patterns affect bandwidth
 * 
 * Learning objectives:
 * - See how non-coalesced access kills bandwidth
 * - Measure strided access patterns
 * - Understand practical bandwidth implications
 */

#include <cuda_runtime.h>
#include <cstdio>

// Coalesced access (stride = 1)
__global__ void coalesced_read(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

// Strided access (bad for coalescing)
template<int STRIDE>
__global__ void strided_read(const float* in, float* out, int n, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strided_idx = idx * STRIDE;
    if (strided_idx < total && idx < n) {
        out[idx] = in[strided_idx];
    }
}

// Random access (worst case)
__global__ void random_read(const float* in, float* out, const int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[indices[idx]];
    }
}

int main() {
    printf("=== Access Pattern Effects on Bandwidth ===\n\n");
    
    const int N = 1 << 24;  // 16M elements
    const int TRIALS = 20;
    
    float *d_in, *d_out;
    int *d_indices;
    cudaMalloc(&d_in, N * sizeof(float) * 32);  // Extra space for strided
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMalloc(&d_indices, N * sizeof(int));
    
    // Initialize random indices
    int* h_indices = new int[N];
    for (int i = 0; i < N; i++) {
        h_indices[i] = rand() % N;
    }
    cudaMemcpy(d_indices, h_indices, N * sizeof(int), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    printf("%-25s %12s %12s\n", "Access Pattern", "GB/s", "Efficiency");
    printf("----------------------------------------------------\n");
    
    // Get peak for reference
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        coalesced_read<<<blocks, threads>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float peak_bw = (2.0 * N * sizeof(float) * TRIALS) / (ms / 1000) / 1e9;
    printf("%-25s %12.1f %12s\n", "Coalesced (stride=1)", peak_bw, "100%");
    
    // Stride 2
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        strided_read<2><<<blocks, threads>>>(d_in, d_out, N, N * 2);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float bw = (2.0 * N * sizeof(float) * TRIALS) / (ms / 1000) / 1e9;
    printf("%-25s %12.1f %11.0f%%\n", "Stride 2", bw, 100 * bw / peak_bw);
    
    // Stride 4
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        strided_read<4><<<blocks, threads>>>(d_in, d_out, N, N * 4);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    bw = (2.0 * N * sizeof(float) * TRIALS) / (ms / 1000) / 1e9;
    printf("%-25s %12.1f %11.0f%%\n", "Stride 4", bw, 100 * bw / peak_bw);
    
    // Stride 8
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        strided_read<8><<<blocks, threads>>>(d_in, d_out, N, N * 8);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    bw = (2.0 * N * sizeof(float) * TRIALS) / (ms / 1000) / 1e9;
    printf("%-25s %12.1f %11.0f%%\n", "Stride 8", bw, 100 * bw / peak_bw);
    
    // Stride 16
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        strided_read<16><<<blocks, threads>>>(d_in, d_out, N, N * 16);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    bw = (2.0 * N * sizeof(float) * TRIALS) / (ms / 1000) / 1e9;
    printf("%-25s %12.1f %11.0f%%\n", "Stride 16", bw, 100 * bw / peak_bw);
    
    // Stride 32 (worst for 32-wide warp)
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        strided_read<32><<<blocks, threads>>>(d_in, d_out, N, N * 32);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    bw = (2.0 * N * sizeof(float) * TRIALS) / (ms / 1000) / 1e9;
    printf("%-25s %12.1f %11.0f%%\n", "Stride 32 (warp width)", bw, 100 * bw / peak_bw);
    
    // Random access
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        random_read<<<blocks, threads>>>(d_in, d_out, d_indices, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    bw = (2.0 * N * sizeof(float) * TRIALS) / (ms / 1000) / 1e9;
    printf("%-25s %12.1f %11.0f%%\n", "Random access", bw, 100 * bw / peak_bw);
    
    printf("\n=== Key Insights ===\n");
    printf("1. Strided access dramatically reduces effective bandwidth\n");
    printf("2. Stride 32 (warp width) is especially bad - each thread gets separate transaction\n");
    printf("3. Random access is even worse - almost never coalesced\n");
    printf("4. This is why SoA > AoS for GPU data layouts\n");
    printf("5. Memory coalescing is often the #1 optimization opportunity\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_indices);
    delete[] h_indices;
    
    return 0;
}
