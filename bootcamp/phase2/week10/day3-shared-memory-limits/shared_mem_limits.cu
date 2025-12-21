/**
 * shared_mem_limits.cu - Shared memory and occupancy
 * 
 * Learning objectives:
 * - See how shared memory limits concurrent blocks
 * - Configure dynamic shared memory size
 * - Trade off shared memory vs occupancy
 */

#include <cuda_runtime.h>
#include <cstdio>

// Kernel with configurable static shared memory
template<int SMEM_FLOATS>
__global__ void static_smem_kernel(float* out, int n) {
    __shared__ float smem[SMEM_FLOATS];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Use shared memory
    if (tid < SMEM_FLOATS) {
        smem[tid] = (float)tid;
    }
    __syncthreads();
    
    if (idx < n) {
        out[idx] = smem[tid % SMEM_FLOATS];
    }
}

// Kernel with dynamic shared memory
__global__ void dynamic_smem_kernel(float* out, int n, int smem_size) {
    extern __shared__ float smem[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (tid < smem_size) {
        smem[tid] = (float)tid;
    }
    __syncthreads();
    
    if (idx < n) {
        out[idx] = smem[tid % smem_size];
    }
}

// Large shared memory kernel (needs carveout configuration)
__global__ void large_smem_kernel(float* out, int n) {
    extern __shared__ float smem[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Use large shared memory buffer
    for (int i = tid; i < 24576; i += blockDim.x) {  // 96KB / 4 = 24K floats
        smem[i] = (float)i;
    }
    __syncthreads();
    
    if (idx < n) {
        out[idx] = smem[tid];
    }
}

template<typename Func>
void analyze_smem(const char* name, Func kernel, int block_size, size_t smem_bytes) {
    int max_blocks;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, kernel, block_size, smem_bytes);
    
    if (err != cudaSuccess) {
        printf("%-25s | Smem=%6zu B | ERROR: %s\n", 
               name, smem_bytes, cudaGetErrorString(err));
        return;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int active_warps = max_blocks * (block_size / 32);
    int max_warps = prop.maxThreadsPerMultiProcessor / 32;
    float occupancy = 100.0f * active_warps / max_warps;
    
    // Calculate smem-limited blocks
    int smem_limited_blocks = (smem_bytes > 0) ?
        prop.sharedMemPerMultiprocessor / smem_bytes : 
        prop.maxBlocksPerMultiProcessor;
    
    printf("%-25s | Smem=%6zu B | Blocks/SM=%2d (smem limit=%2d) | Occ=%.0f%%\n",
           name, smem_bytes, max_blocks, 
           (int)(smem_limited_blocks > prop.maxBlocksPerMultiProcessor ? 
                 prop.maxBlocksPerMultiProcessor : smem_limited_blocks),
           occupancy);
}

int main() {
    printf("=== Shared Memory and Occupancy ===\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device: %s\n", prop.name);
    printf("Shared memory per SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Shared memory per block (default): %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Max blocks per SM: %d\n\n", prop.maxBlocksPerMultiProcessor);
    
    const int BLOCK = 256;
    
    printf("=== Static Shared Memory Impact ===\n\n");
    
    analyze_smem("1KB static smem", static_smem_kernel<256>, BLOCK, 256 * sizeof(float));
    analyze_smem("4KB static smem", static_smem_kernel<1024>, BLOCK, 1024 * sizeof(float));
    analyze_smem("16KB static smem", static_smem_kernel<4096>, BLOCK, 4096 * sizeof(float));
    analyze_smem("32KB static smem", static_smem_kernel<8192>, BLOCK, 8192 * sizeof(float));
    analyze_smem("48KB static smem", static_smem_kernel<12288>, BLOCK, 12288 * sizeof(float));
    printf("\n");
    
    printf("=== Dynamic Shared Memory Impact ===\n\n");
    
    for (int kb = 1; kb <= 48; kb *= 2) {
        char name[32];
        snprintf(name, sizeof(name), "%dKB dynamic smem", kb);
        analyze_smem(name, dynamic_smem_kernel, BLOCK, kb * 1024);
    }
    printf("\n");
    
    printf("=== Large Shared Memory (needs opt-in) ===\n\n");
    
    // Default: limited shared memory
    analyze_smem("96KB (default limit)", large_smem_kernel, BLOCK, 96 * 1024);
    
    // Opt-in for larger shared memory
    cudaFuncSetAttribute(large_smem_kernel, 
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 
                         98304);  // 96KB
    
    analyze_smem("96KB (with opt-in)", large_smem_kernel, BLOCK, 96 * 1024);
    printf("\n");
    
    printf("=== Shared Memory vs Occupancy Trade-off ===\n\n");
    printf("More shared memory per block → fewer blocks per SM → lower occupancy\n");
    printf("BUT: shared memory reduces global memory traffic!\n\n");
    
    printf("Decision process:\n");
    printf("1. Calculate shared memory needed for your algorithm\n");
    printf("2. Check resulting occupancy\n");
    printf("3. If occupancy is very low, consider:\n");
    printf("   - Reducing shared memory usage\n");
    printf("   - Using smaller tile sizes\n");
    printf("   - Multi-stage loading\n");
    printf("4. Profile to verify performance impact\n");
    printf("\n");
    
    printf("Key insight: 50%% occupancy with great cache reuse often beats\n");
    printf("             100%% occupancy with constant memory traffic!\n");
    
    return 0;
}
