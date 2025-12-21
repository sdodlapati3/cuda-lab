/**
 * register_demo.cu - Understanding register usage and occupancy
 * 
 * Learning objectives:
 * - See register allocation in practice
 * - Understand occupancy impact
 * - Use __launch_bounds__
 */

#include <cuda_runtime.h>
#include <cstdio>

// Low register kernel
__global__ void low_registers(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

// Medium register kernel
__global__ void medium_registers(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a = data[idx];
        float b = sinf(a);
        float c = cosf(a);
        float d = tanf(a);
        float e = expf(a * 0.01f);
        float f = logf(fabsf(a) + 1.0f);
        data[idx] = a + b + c + d + e + f;
    }
}

// High register kernel (many intermediates)
__global__ void high_registers(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a = data[idx];
        float b = a * 2.0f;
        float c = b + 1.0f;
        float d = sinf(a);
        float e = cosf(b);
        float f = tanf(c);
        float g = expf(d * 0.01f);
        float h = logf(fabsf(e) + 1.0f);
        float i = sqrtf(fabsf(f) + 1.0f);
        float j = powf(fabsf(g), 0.5f);
        float k = a + b + c + d;
        float l = e + f + g + h;
        float m = i + j + k + l;
        float n_val = sinf(m);
        float o = cosf(n_val);
        float p = tanf(o);
        data[idx] = a + b + c + d + e + f + g + h + i + j + k + l + m + n_val + o + p;
    }
}

// With launch bounds (hint to compiler)
__global__ void __launch_bounds__(256, 4)  // 256 threads/block, 4 blocks/SM
bounded_registers(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a = data[idx];
        float b = sinf(a);
        float c = cosf(a);
        float d = tanf(a);
        float e = expf(a * 0.01f);
        float f = logf(fabsf(a) + 1.0f);
        data[idx] = a + b + c + d + e + f;
    }
}

template<typename KernelFunc>
void analyze_kernel(const char* name, KernelFunc kernel, float* d_data, int n) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    
    // Get occupancy
    int max_active_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel, block_size, 0);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    
    int active_threads = max_active_blocks * block_size;
    float occupancy = 100.0f * active_threads / max_threads_per_sm;
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    kernel<<<num_blocks, block_size>>>(d_data, n);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        kernel<<<num_blocks, block_size>>>(d_data, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 100;
    
    printf("%-20s: Blocks/SM: %d/%d, Occupancy: %.0f%%, Time: %.3f ms\n",
           name, max_active_blocks, max_blocks_per_sm, occupancy, ms);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("=== Register Usage and Occupancy ===\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max blocks per SM: %d\n\n", prop.maxBlocksPerMultiProcessor);
    
    printf("Register limit per thread for full occupancy:\n");
    printf("  %d regs/SM ÷ %d threads/SM = %d regs/thread\n\n",
           prop.regsPerMultiprocessor, prop.maxThreadsPerMultiProcessor,
           prop.regsPerMultiprocessor / prop.maxThreadsPerMultiProcessor);
    
    const int N = 1 << 20;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    
    printf("=== Kernel Analysis ===\n");
    printf("(Compile with --ptxas-options=-v to see actual register counts)\n\n");
    
    analyze_kernel("Low registers", low_registers, d_data, N);
    analyze_kernel("Medium registers", medium_registers, d_data, N);
    analyze_kernel("High registers", high_registers, d_data, N);
    analyze_kernel("With __launch_bounds__", bounded_registers, d_data, N);
    
    cudaFree(d_data);
    
    printf("\n=== Key Points ===\n");
    printf("1. More registers → lower occupancy → fewer concurrent threads\n");
    printf("2. Use __launch_bounds__(threads, min_blocks) to hint compiler\n");
    printf("3. Check register usage: nvcc --ptxas-options=-v\n");
    printf("4. Trade-off: More registers can mean faster kernels despite lower occupancy\n");
    
    return 0;
}
