/**
 * aos_vs_soa.cu - Array of Structures vs Structure of Arrays
 * 
 * Learning objectives:
 * - Understand data layout impact on coalescing
 * - Convert AoS to SoA for better performance
 */

#include <cuda_runtime.h>
#include <cstdio>

// Array of Structures (AoS) - Bad for coalescing
struct ParticleAoS {
    float x, y, z, w;
};

// Structure of Arrays (SoA) - Good for coalescing
struct ParticlesSoA {
    float* x;
    float* y;
    float* z;
    float* w;
};

// Kernel with AoS access pattern (strided by 16 bytes!)
__global__ void process_aos(ParticleAoS* particles, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        // Reading x, y, z, w causes strided access within warp
        float x = particles[i].x;  // Thread i reads at offset i*16 + 0
        float y = particles[i].y;  // Thread i reads at offset i*16 + 4
        float z = particles[i].z;  // Thread i reads at offset i*16 + 8
        float w = particles[i].w;  // Thread i reads at offset i*16 + 12
        
        particles[i].x = x + y + z + w;
    }
}

// Kernel with SoA access pattern (coalesced!)
__global__ void process_soa(float* x, float* y, float* z, float* w, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        // All threads read consecutive x values, then y, etc.
        float px = x[i];  // Coalesced!
        float py = y[i];  // Coalesced!
        float pz = z[i];  // Coalesced!
        float pw = w[i];  // Coalesced!
        
        x[i] = px + py + pz + pw;
    }
}

int main() {
    printf("=== AoS vs SoA Data Layout ===\n\n");
    
    const int N = 16 * 1024 * 1024;  // 16M particles
    const int TRIALS = 100;
    
    printf("Number of particles: %d (%.1f M)\n\n", N, N / 1e6);
    
    // Allocate AoS
    ParticleAoS* d_aos;
    cudaMalloc(&d_aos, N * sizeof(ParticleAoS));
    cudaMemset(d_aos, 0, N * sizeof(ParticleAoS));
    
    // Allocate SoA
    float *d_x, *d_y, *d_z, *d_w;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_z, N * sizeof(float));
    cudaMalloc(&d_w, N * sizeof(float));
    cudaMemset(d_x, 0, N * sizeof(float));
    cudaMemset(d_y, 0, N * sizeof(float));
    cudaMemset(d_z, 0, N * sizeof(float));
    cudaMemset(d_w, 0, N * sizeof(float));
    
    int block_size = 256;
    int num_blocks = 512;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    process_aos<<<num_blocks, block_size>>>(d_aos, N);
    process_soa<<<num_blocks, block_size>>>(d_x, d_y, d_z, d_w, N);
    cudaDeviceSynchronize();
    
    // Benchmark AoS
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        process_aos<<<num_blocks, block_size>>>(d_aos, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float aos_ms;
    cudaEventElapsedTime(&aos_ms, start, stop);
    aos_ms /= TRIALS;
    
    size_t aos_bytes = N * sizeof(ParticleAoS) * 2;  // Read + write
    float aos_bw = aos_bytes / aos_ms / 1e6;
    
    // Benchmark SoA
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        process_soa<<<num_blocks, block_size>>>(d_x, d_y, d_z, d_w, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float soa_ms;
    cudaEventElapsedTime(&soa_ms, start, stop);
    soa_ms /= TRIALS;
    
    size_t soa_bytes = N * 4 * sizeof(float) * 2;  // 4 arrays, read + write
    float soa_bw = soa_bytes / soa_ms / 1e6;
    
    printf("=== Results ===\n\n");
    printf("Layout   Memory (MB)   Time (ms)   Bandwidth (GB/s)\n");
    printf("------------------------------------------------------\n");
    printf("AoS      %8zu     %8.3f    %8.1f\n", 
           N * sizeof(ParticleAoS) / (1024*1024), aos_ms, aos_bw);
    printf("SoA      %8zu     %8.3f    %8.1f\n",
           N * 4 * sizeof(float) / (1024*1024), soa_ms, soa_bw);
    printf("\n");
    printf("Speedup: %.2fx\n\n", aos_ms / soa_ms);
    
    // Memory layout visualization
    printf("=== Memory Layout Visualization ===\n\n");
    printf("AoS (Array of Structures):\n");
    printf("  [x0|y0|z0|w0][x1|y1|z1|w1][x2|y2|z2|w2]...\n");
    printf("  Thread 0 reads x0, Thread 1 reads x1 → Stride of 16 bytes!\n\n");
    
    printf("SoA (Structure of Arrays):\n");
    printf("  x: [x0|x1|x2|x3|x4|x5|x6|...]\n");
    printf("  y: [y0|y1|y2|y3|y4|y5|y6|...]\n");
    printf("  Thread 0 reads x0, Thread 1 reads x1 → Consecutive (coalesced)!\n\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_aos);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_w);
    
    printf("=== Key Takeaway ===\n");
    printf("Always prefer SoA layout for GPU data structures!\n");
    printf("AoS causes strided access, killing memory bandwidth.\n");
    
    return 0;
}
