/**
 * memory_latency.cu - Memory latency hiding techniques
 * 
 * Learning objectives:
 * - Double buffering
 * - Async memory copies
 * - Prefetching patterns
 */

#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cstdio>
#include <cmath>

// Simple kernel - no latency hiding
__global__ void naive_tiled(const float* in, float* out, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Load
        smem[tid] = in[idx];
        __syncthreads();
        
        // Simple computation using shared memory
        float val = smem[tid];
        for (int i = 0; i < 10; i++) {
            val = val * 1.01f + 0.01f;
        }
        
        out[idx] = val;
    }
}

// Double buffering - overlap load and compute
__global__ void double_buffer_kernel(const float* in, float* out, 
                                      int n, int tiles_per_block) {
    __shared__ float smem[2][256];  // Two buffers
    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x * tiles_per_block;
    
    // Initial load into buffer 0
    int idx = block_start + tid;
    if (idx < n) smem[0][tid] = in[idx];
    
    for (int tile = 0; tile < tiles_per_block; tile++) {
        int current = tile % 2;
        int next = 1 - current;
        
        // Start loading next tile into other buffer
        int next_idx = block_start + (tile + 1) * blockDim.x + tid;
        if (tile + 1 < tiles_per_block && next_idx < n) {
            smem[next][tid] = in[next_idx];
        }
        
        __syncthreads();
        
        // Process current tile
        float val = smem[current][tid];
        for (int i = 0; i < 10; i++) {
            val = val * 1.01f + 0.01f;
        }
        
        // Write output
        int out_idx = block_start + tile * blockDim.x + tid;
        if (out_idx < n) out[out_idx] = val;
        
        __syncthreads();
    }
}

// Using async memcpy (Ampere+)
__global__ void async_copy_kernel(const float* in, float* out, 
                                   int n, int tiles_per_block) {
    __shared__ float smem[2][256];
    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x * tiles_per_block;
    
    // Initial async load
    int idx = block_start + tid;
    if (idx < n) {
        __pipeline_memcpy_async(&smem[0][tid], &in[idx], sizeof(float));
    }
    __pipeline_commit();
    
    for (int tile = 0; tile < tiles_per_block; tile++) {
        int current = tile % 2;
        int next = 1 - current;
        
        // Start async load for next tile
        int next_idx = block_start + (tile + 1) * blockDim.x + tid;
        if (tile + 1 < tiles_per_block && next_idx < n) {
            __pipeline_memcpy_async(&smem[next][tid], &in[next_idx], sizeof(float));
        }
        __pipeline_commit();
        
        // Wait for current tile to be ready
        __pipeline_wait_prior(1);  // Wait for all but the most recent
        __syncthreads();
        
        // Process current tile
        float val = smem[current][tid];
        for (int i = 0; i < 10; i++) {
            val = val * 1.01f + 0.01f;
        }
        
        // Write output
        int out_idx = block_start + tile * blockDim.x + tid;
        if (out_idx < n) out[out_idx] = val;
        
        __syncthreads();
    }
}

// Prefetch pattern - load ahead before compute
__global__ void prefetch_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Prefetch first element
    float next_val = 0;
    if (idx < n) next_val = in[idx];
    
    for (int i = idx; i < n; i += stride) {
        float current = next_val;
        
        // Prefetch next iteration's data
        if (i + stride < n) {
            next_val = in[i + stride];
        }
        
        // Compute while load is in flight
        for (int j = 0; j < 10; j++) {
            current = current * 1.01f + 0.01f;
        }
        
        out[i] = current;
    }
}

int main() {
    printf("=== Memory Latency Hiding Demo ===\n\n");
    
    const int N = 1 << 22;  // 4M
    const int bytes = N * sizeof(float);
    const int block_size = 256;
    const int tiles = 8;  // Tiles per block for multi-tile kernels
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    cudaMemcpy(d_in, h_data, bytes, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("%-25s | %10s | %10s\n", "Method", "Time (ms)", "Speedup");
    printf("--------------------------------------------------\n");
    
    float baseline = 0;
    
    // Naive
    {
        int blocks = (N + block_size - 1) / block_size;
        
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            naive_tiled<<<blocks, block_size>>>(d_in, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        baseline = ms;
        printf("%-25s | %10.4f | %9.2fx\n", "Naive (no hiding)", ms/100, 1.0);
    }
    
    // Double buffer
    {
        int blocks = (N + block_size * tiles - 1) / (block_size * tiles);
        
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            double_buffer_kernel<<<blocks, block_size>>>(d_in, d_out, N, tiles);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("%-25s | %10.4f | %9.2fx\n", "Double Buffer", ms/100, baseline/ms);
    }
    
    // Async copy
    {
        int blocks = (N + block_size * tiles - 1) / (block_size * tiles);
        
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            async_copy_kernel<<<blocks, block_size>>>(d_in, d_out, N, tiles);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("%-25s | %10.4f | %9.2fx\n", "Async Copy (cp.async)", ms/100, baseline/ms);
    }
    
    // Prefetch
    {
        int blocks = (N + block_size - 1) / block_size;
        blocks = min(blocks, 256);  // Limit blocks for strided access
        
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            prefetch_kernel<<<blocks, block_size>>>(d_in, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("%-25s | %10.4f | %9.2fx\n", "Prefetch Pattern", ms/100, baseline/ms);
    }
    
    printf("\n=== Key Techniques ===\n\n");
    printf("1. Double Buffering:\n");
    printf("   - Use 2 buffers: load into one while processing the other\n");
    printf("   - Classic technique for overlap\n\n");
    
    printf("2. Async Copy (Ampere+):\n");
    printf("   - Hardware-accelerated memory copies\n");
    printf("   - Bypasses register file\n");
    printf("   - Use __pipeline_memcpy_async()\n\n");
    
    printf("3. Prefetching:\n");
    printf("   - Load next iteration's data during compute\n");
    printf("   - Works well with strided access patterns\n\n");
    
    printf("Profile command:\n");
    printf("  ncu --metrics l1tex__t_sector_hit_rate.pct ./build/memory_latency\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_data;
    
    return 0;
}
