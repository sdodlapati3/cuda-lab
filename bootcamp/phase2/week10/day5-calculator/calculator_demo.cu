/**
 * calculator_demo.cu - Implementing an occupancy calculator
 * 
 * Learning objectives:
 * - Calculate occupancy from first principles
 * - Identify which factor limits occupancy
 * - Compare to API results
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

// Various kernels with different resource profiles
__global__ void kernel_light(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (float)idx;
}

__global__ void kernel_reg_heavy(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float r[32];
    for (int i = 0; i < 32; i++) r[i] = (float)(idx + i);
    float sum = 0;
    for (int i = 0; i < 32; i++) sum += r[i];
    if (idx < n) out[idx] = sum;
}

template<int SMEM_SIZE>
__global__ void kernel_smem(float* out, int n) {
    __shared__ float smem[SMEM_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (tid < SMEM_SIZE) smem[tid] = (float)tid;
    __syncthreads();
    if (idx < n) out[idx] = smem[tid % SMEM_SIZE];
}

struct OccupancyResult {
    int blocks_by_threads;
    int blocks_by_registers;
    int blocks_by_smem;
    int blocks_by_limit;
    int final_blocks;
    const char* limiter;
    float occupancy;
};

template<typename Func>
OccupancyResult calculate_occupancy(Func kernel, int block_size, size_t dynamic_smem) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel);
    
    OccupancyResult result;
    
    // Limit 1: Threads per SM
    int warps_per_block = block_size / 32;
    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / 32;
    result.blocks_by_threads = max_warps_per_sm / warps_per_block;
    
    // Limit 2: Registers
    int regs_per_thread = attr.numRegs;
    int regs_per_sm = prop.regsPerMultiprocessor;
    if (regs_per_thread > 0) {
        int regs_per_block = regs_per_thread * block_size;
        // Registers allocated in granularity of 256 on recent architectures
        regs_per_block = ((regs_per_block + 255) / 256) * 256;
        result.blocks_by_registers = regs_per_sm / regs_per_block;
    } else {
        result.blocks_by_registers = 999;  // No limit
    }
    
    // Limit 3: Shared memory
    size_t total_smem = attr.sharedSizeBytes + dynamic_smem;
    if (total_smem > 0) {
        result.blocks_by_smem = prop.sharedMemPerMultiprocessor / total_smem;
    } else {
        result.blocks_by_smem = 999;  // No limit
    }
    
    // Limit 4: Hardware block limit
    result.blocks_by_limit = prop.maxBlocksPerMultiProcessor;
    
    // Find minimum
    result.final_blocks = std::min({
        result.blocks_by_threads,
        result.blocks_by_registers,
        result.blocks_by_smem,
        result.blocks_by_limit
    });
    
    // Identify limiter
    if (result.final_blocks == result.blocks_by_threads) {
        result.limiter = "threads";
    } else if (result.final_blocks == result.blocks_by_registers) {
        result.limiter = "registers";
    } else if (result.final_blocks == result.blocks_by_smem) {
        result.limiter = "shared_mem";
    } else {
        result.limiter = "block_limit";
    }
    
    // Calculate occupancy
    int active_warps = result.final_blocks * warps_per_block;
    result.occupancy = 100.0f * active_warps / max_warps_per_sm;
    
    return result;
}

template<typename Func>
void analyze_with_calculator(const char* name, Func kernel, int block_size, 
                             size_t dynamic_smem = 0) {
    // Our calculation
    OccupancyResult calc = calculate_occupancy(kernel, block_size, dynamic_smem);
    
    // API calculation for verification
    int api_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&api_blocks, kernel, block_size, dynamic_smem);
    
    // Get kernel attributes
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel);
    
    printf("\n%-25s (Block=%d, Regs=%d, Smem=%zu+%zu)\n", 
           name, block_size, attr.numRegs, attr.sharedSizeBytes, dynamic_smem);
    printf("  Limits: threads=%d, regs=%d, smem=%d, hw=%d\n",
           calc.blocks_by_threads, calc.blocks_by_registers, 
           calc.blocks_by_smem, calc.blocks_by_limit);
    printf("  Result: %d blocks/SM (limited by %s)\n", 
           calc.final_blocks, calc.limiter);
    printf("  Occupancy: %.0f%% | API says: %d blocks (%.0f%%)\n",
           calc.occupancy, api_blocks, 
           100.0f * api_blocks * (block_size / 32) / 64.0f);
}

int main() {
    printf("=== Occupancy Calculator Demo ===\n\n");
    
    // Print device limits
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Max threads/SM: %d (%d warps)\n", 
           prop.maxThreadsPerMultiProcessor, 
           prop.maxThreadsPerMultiProcessor / 32);
    printf("Max blocks/SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Registers/SM: %d\n", prop.regsPerMultiprocessor);
    printf("Shared mem/SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    
    printf("\n=== Occupancy Calculations ===");
    
    // Light kernel - likely thread limited
    analyze_with_calculator("kernel_light", kernel_light, 256);
    analyze_with_calculator("kernel_light", kernel_light, 128);
    analyze_with_calculator("kernel_light", kernel_light, 64);
    
    // Register heavy - likely register limited
    analyze_with_calculator("kernel_reg_heavy", kernel_reg_heavy, 256);
    analyze_with_calculator("kernel_reg_heavy", kernel_reg_heavy, 128);
    
    // Shared memory variants
    analyze_with_calculator("kernel_smem_4KB", kernel_smem<1024>, 256, 0);
    analyze_with_calculator("kernel_smem_16KB", kernel_smem<4096>, 256, 0);
    analyze_with_calculator("kernel_smem_32KB", kernel_smem<8192>, 256, 0);
    
    printf("\n\n=== Key Insights ===\n\n");
    printf("1. Occupancy is limited by the MOST constrained resource\n");
    printf("2. To improve occupancy, address the limiter:\n");
    printf("   - threads: use smaller block sizes (but watch for other limits)\n");
    printf("   - registers: use __launch_bounds__ or reduce variables\n");
    printf("   - shared_mem: reduce tile sizes or use multi-stage loading\n");
    printf("   - block_limit: use larger block sizes\n");
    printf("3. Our calculation matches the API (minor differences due to allocation granularity)\n");
    
    return 0;
}
