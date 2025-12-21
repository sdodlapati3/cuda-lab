/**
 * register_analysis.cu - Understanding register pressure
 * 
 * Learning objectives:
 * - See how register usage affects occupancy
 * - Use launch bounds to control registers
 * - Trade off registers vs occupancy
 */

#include <cuda_runtime.h>
#include <cstdio>

// Low register usage - will have high occupancy
__global__ void low_registers(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (float)idx;
    }
}

// High register usage - will have lower occupancy
__global__ void high_registers(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Force many registers
    float r[32];
    for (int i = 0; i < 32; i++) {
        r[i] = (float)(idx + i);
    }
    
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        sum += r[i] * r[31-i];
    }
    
    if (idx < n) out[idx] = sum;
}

// Very high register usage
__global__ void very_high_registers(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float r[64];
    for (int i = 0; i < 64; i++) {
        r[i] = (float)(idx + i) * 0.1f;
    }
    
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        sum += r[i] * r[63-i] + sinf(r[i]);
    }
    
    if (idx < n) out[idx] = sum;
}

// Using __launch_bounds__ to limit registers
__global__ __launch_bounds__(256, 4)  // 256 threads, 4 min blocks/SM
void bounded_kernel(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float r[32];
    for (int i = 0; i < 32; i++) {
        r[i] = (float)(idx + i);
    }
    
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        sum += r[i] * r[31-i];
    }
    
    if (idx < n) out[idx] = sum;
}

// Aggressive register limiting
__global__ __launch_bounds__(256, 8)  // Force higher occupancy
void aggressive_bounds(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Same code as high_registers, but compiler must spill to achieve bounds
    float r[32];
    for (int i = 0; i < 32; i++) {
        r[i] = (float)(idx + i);
    }
    
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        sum += r[i] * r[31-i];
    }
    
    if (idx < n) out[idx] = sum;
}

template<typename Func>
void analyze_kernel(const char* name, Func kernel, int block_size) {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel);
    
    int max_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, block_size, 0);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int warps_per_block = block_size / 32;
    int active_warps = max_blocks * warps_per_block;
    int max_warps = prop.maxThreadsPerMultiProcessor / 32;
    float occupancy = 100.0f * active_warps / max_warps;
    
    // Calculate register-limited threads
    int regs_per_sm = prop.regsPerMultiprocessor;
    int max_threads_by_regs = (attr.numRegs > 0) ? 
                               regs_per_sm / attr.numRegs : 
                               prop.maxThreadsPerMultiProcessor;
    
    printf("%-20s | Regs=%3d | LocalMem=%5zu | Blocks/SM=%2d | Occ=%5.1f%%\n",
           name, attr.numRegs, attr.localSizeBytes, max_blocks, occupancy);
    printf("                     | Max threads by regs: %d (%.0f%% of %d)\n",
           max_threads_by_regs, 
           100.0f * max_threads_by_regs / prop.maxThreadsPerMultiProcessor,
           prop.maxThreadsPerMultiProcessor);
    printf("\n");
}

int main() {
    printf("=== Register Pressure Analysis ===\n\n");
    printf("(Build with --ptxas-options=-v to see register counts at compile time)\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Regs for 100%% occupancy: %d per thread\n\n", 
           prop.regsPerMultiprocessor / prop.maxThreadsPerMultiProcessor);
    
    const int BLOCK = 256;
    
    printf("%-20s | %-8s | %-10s | %-11s | %s\n",
           "Kernel", "Regs", "LocalMem", "Blocks/SM", "Occupancy");
    printf("------------------------------------------------------------------------\n");
    
    analyze_kernel("low_registers", low_registers, BLOCK);
    analyze_kernel("high_registers", high_registers, BLOCK);
    analyze_kernel("very_high_registers", very_high_registers, BLOCK);
    analyze_kernel("bounded_kernel", bounded_kernel, BLOCK);
    analyze_kernel("aggressive_bounds", aggressive_bounds, BLOCK);
    
    printf("=== Key Insights ===\n\n");
    printf("1. More registers per thread → fewer threads → lower occupancy\n");
    printf("2. __launch_bounds__(threads, blocks) hints to compiler\n");
    printf("3. If compiler can't fit in bounds, it 'spills' to local memory\n");
    printf("4. Local memory spills hurt performance (goes to DRAM)\n");
    printf("5. Check localSizeBytes - non-zero means spilling occurred\n");
    printf("\n");
    printf("Trade-off: Fewer registers = higher occupancy but maybe slower per-thread\n");
    printf("           More registers = faster per-thread but lower occupancy\n");
    
    return 0;
}
