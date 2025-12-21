/**
 * occupancy_basics.cu - Understanding GPU occupancy
 * 
 * Learning objectives:
 * - Query device occupancy limits
 * - Calculate theoretical occupancy
 * - See relationship between resources and occupancy
 */

#include <cuda_runtime.h>
#include <cstdio>

// Simple kernel - we'll analyze its occupancy
__global__ void simple_kernel(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (float)idx * 2.0f;
    }
}

// Kernel with more register usage
__global__ void register_heavy(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Declare many variables to use registers
    float r0 = (float)idx, r1 = r0 * 2, r2 = r1 * 2, r3 = r2 * 2;
    float r4 = r3 * 2, r5 = r4 * 2, r6 = r5 * 2, r7 = r6 * 2;
    float r8 = r7 * 2, r9 = r8 * 2, r10 = r9 * 2, r11 = r10 * 2;
    float r12 = r11 * 2, r13 = r12 * 2, r14 = r13 * 2, r15 = r14 * 2;
    
    if (idx < n) {
        out[idx] = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 +
                   r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15;
    }
}

// Kernel with shared memory
template<int SMEM_SIZE>
__global__ void shared_mem_kernel(float* out, int n) {
    __shared__ float smem[SMEM_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (tid < SMEM_SIZE) smem[tid] = (float)tid;
    __syncthreads();
    
    if (idx < n) {
        out[idx] = smem[tid % SMEM_SIZE];
    }
}

void print_device_limits() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("=== Device Occupancy Limits: %s ===\n\n", prop.name);
    
    printf("Per-SM Limits:\n");
    printf("  Max threads per SM:         %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max warps per SM:           %d\n", prop.maxThreadsPerMultiProcessor / 32);
    printf("  Max blocks per SM:          %d\n", prop.maxBlocksPerMultiProcessor);
    printf("  Registers per SM:           %d\n", prop.regsPerMultiprocessor);
    printf("  Shared memory per SM:       %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("\n");
    
    printf("Per-Block Limits:\n");
    printf("  Max threads per block:      %d\n", prop.maxThreadsPerBlock);
    printf("  Registers per block:        %d\n", prop.regsPerBlock);
    printf("  Shared memory per block:    %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("\n");
    
    printf("Thread/Warp Info:\n");
    printf("  Warp size:                  %d\n", prop.warpSize);
    printf("  Max threads dim:            (%d, %d, %d)\n", 
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("\n");
}

template<typename Func>
void analyze_occupancy(const char* name, Func kernel, int block_size, 
                       size_t dynamic_smem = 0) {
    int max_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, kernel, block_size, dynamic_smem);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int warps_per_block = block_size / 32;
    int active_warps = max_blocks * warps_per_block;
    int max_warps = prop.maxThreadsPerMultiProcessor / 32;
    float occupancy = 100.0f * active_warps / max_warps;
    
    printf("%-25s | Block=%4d | Blocks/SM=%2d | Warps=%2d/%2d | Occ=%.0f%%\n",
           name, block_size, max_blocks, active_warps, max_warps, occupancy);
}

int main() {
    print_device_limits();
    
    printf("=== Occupancy Analysis ===\n\n");
    printf("%-25s | %-10s | %-11s | %-11s | %s\n",
           "Kernel", "BlockSize", "Blocks/SM", "ActiveWarps", "Occupancy");
    printf("-----------------------------------------------------------------------------\n");
    
    // Simple kernel with different block sizes
    analyze_occupancy("simple (64 threads)", simple_kernel, 64);
    analyze_occupancy("simple (128 threads)", simple_kernel, 128);
    analyze_occupancy("simple (256 threads)", simple_kernel, 256);
    analyze_occupancy("simple (512 threads)", simple_kernel, 512);
    analyze_occupancy("simple (1024 threads)", simple_kernel, 1024);
    printf("\n");
    
    // Register-heavy kernel
    analyze_occupancy("register_heavy (256)", register_heavy, 256);
    analyze_occupancy("register_heavy (512)", register_heavy, 512);
    printf("\n");
    
    // Shared memory kernels
    analyze_occupancy("shared_1KB", shared_mem_kernel<256>, 256, 0);
    analyze_occupancy("shared_4KB", shared_mem_kernel<1024>, 256, 0);
    analyze_occupancy("shared_16KB", shared_mem_kernel<4096>, 256, 0);
    analyze_occupancy("shared_32KB", shared_mem_kernel<8192>, 256, 0);
    printf("\n");
    
    // Demonstrate occupancy API for dynamic calculation
    printf("=== Using cudaOccupancyMaxPotentialBlockSize ===\n\n");
    
    int min_grid, block_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_size, 
                                        simple_kernel, 0, 0);
    printf("Suggested block size for simple_kernel: %d\n", block_size);
    printf("Minimum grid size for full occupancy:   %d\n", min_grid);
    
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_size,
                                        register_heavy, 0, 0);
    printf("Suggested block size for register_heavy: %d\n", block_size);
    printf("\n");
    
    printf("=== Key Takeaways ===\n\n");
    printf("1. Occupancy = Active Warps / Max Warps per SM\n");
    printf("2. Limited by: registers, shared memory, block size, max blocks\n");
    printf("3. Use cudaOccupancyMaxActiveBlocksPerMultiprocessor() to query\n");
    printf("4. Use cudaOccupancyMaxPotentialBlockSize() for suggestions\n");
    printf("5. Higher occupancy helps hide latency (but isn't always needed)\n");
    
    return 0;
}
