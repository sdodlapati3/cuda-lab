/**
 * occupancy_demo.cu - Use CUDA occupancy calculator
 * 
 * Learning objectives:
 * - Use cudaOccupancyMaxPotentialBlockSize
 * - Understand occupancy calculation
 * - Apply launch bounds
 */

#include <cuda_runtime.h>
#include <cstdio>

// Kernel without launch bounds
__global__ void kernel_auto(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        data[i] = data[i] * 2.0f + 1.0f;
    }
}

// Kernel with launch bounds hint
__global__ void __launch_bounds__(256, 4)
kernel_bounded(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        data[i] = data[i] * 2.0f + 1.0f;
    }
}

// Kernel with high register usage
__global__ void kernel_heavy_regs(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Force high register usage
    float a = data[idx % n];
    float b = sinf(a);
    float c = cosf(a);
    float d = tanf(a);
    float e = expf(a);
    float f = logf(fabsf(a) + 1.0f);
    float g = sqrtf(fabsf(a) + 1.0f);
    float h = powf(fabsf(a), 0.5f);
    
    for (int i = idx; i < n; i += stride) {
        data[i] = a + b + c + d + e + f + g + h;
    }
}

// Kernel using shared memory
__global__ void kernel_smem(float* data, int n) {
    extern __shared__ float smem[];  // Dynamic shared memory
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < n) {
        smem[tid] = data[idx];
        __syncthreads();
        data[idx] = smem[tid] * 2.0f;
    }
}

template<typename KernelFunc>
void analyze_occupancy(const char* name, KernelFunc kernel, size_t smem = 0) {
    int min_grid_size, block_size;
    
    cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &block_size,
        kernel,
        smem,
        0  // No block size limit
    );
    
    int num_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks,
        kernel,
        block_size,
        smem
    );
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int warps_per_block = block_size / 32;
    int active_warps = num_blocks * warps_per_block;
    int max_warps = prop.maxThreadsPerMultiProcessor / 32;
    float occupancy = 100.0f * active_warps / max_warps;
    
    printf("%-20s:\n", name);
    printf("  Suggested block size:  %d threads (%d warps)\n", block_size, warps_per_block);
    printf("  Min grid size:         %d blocks\n", min_grid_size);
    printf("  Max blocks per SM:     %d\n", num_blocks);
    printf("  Active warps per SM:   %d / %d (%.1f%% occupancy)\n", 
           active_warps, max_warps, occupancy);
    printf("\n");
}

int main() {
    printf("=== CUDA Occupancy Calculator Demo ===\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Max threads per SM: %d (%d warps)\n", 
           prop.maxThreadsPerMultiProcessor,
           prop.maxThreadsPerMultiProcessor / 32);
    printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Shared memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
    printf("\n");
    
    printf("=== Occupancy Analysis ===\n\n");
    
    analyze_occupancy("kernel_auto", kernel_auto);
    analyze_occupancy("kernel_bounded", kernel_bounded);
    analyze_occupancy("kernel_heavy_regs", kernel_heavy_regs);
    
    // Analyze with different shared memory sizes
    printf("=== Shared Memory Impact ===\n\n");
    
    size_t smem_sizes[] = {0, 4096, 16384, 49152};
    for (size_t smem : smem_sizes) {
        printf("Shared memory = %zu bytes:\n", smem);
        
        int min_grid, block_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_size, kernel_smem, smem, 0);
        
        int num_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, kernel_smem, block_size, smem);
        
        printf("  Block size: %d, Blocks per SM: %d\n\n", block_size, num_blocks);
    }
    
    // Practical example
    printf("=== Practical: Auto-tuned Launch ===\n\n");
    
    const int N = 1 << 20;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    
    int min_grid, block_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_size, kernel_auto, 0, 0);
    
    int grid_size = (N + block_size - 1) / block_size;
    
    printf("For N = %d elements:\n", N);
    printf("  Auto-tuned block size: %d\n", block_size);
    printf("  Required grid size: %d blocks\n", grid_size);
    printf("  Launch: kernel<<<N=%d, blockSize=%d>>>\n", grid_size, block_size);
    
    kernel_auto<<<grid_size, block_size>>>(d_data, N);
    cudaDeviceSynchronize();
    
    cudaFree(d_data);
    
    printf("\n=== Key Takeaways ===\n");
    printf("1. Use cudaOccupancyMaxPotentialBlockSize for auto-tuning\n");
    printf("2. __launch_bounds__ helps compiler optimize register usage\n");
    printf("3. Shared memory usage reduces max blocks per SM\n");
    printf("4. High occupancy != always best performance\n");
    printf("5. Profile to find actual optimal configuration\n");
    
    return 0;
}
