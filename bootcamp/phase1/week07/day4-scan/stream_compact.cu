/**
 * stream_compact.cu - Scan application: stream compaction
 * 
 * Learning objectives:
 * - Apply scan to real problem
 * - Filter array in parallel
 * - Build output indices
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 256
#define FULL_MASK 0xFFFFFFFF

// Warp-level inclusive scan
__device__ int warp_inclusive_scan(int val) {
    for (int offset = 1; offset < 32; offset *= 2) {
        int n = __shfl_up_sync(FULL_MASK, val, offset);
        if (threadIdx.x % 32 >= offset) {
            val += n;
        }
    }
    return val;
}

// Step 1: Generate flags (1 if keep, 0 if discard)
__global__ void generate_flags(int* flags, const float* input, int n, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        flags[idx] = (input[idx] > threshold) ? 1 : 0;
    }
}

// Step 2: Exclusive scan on flags to get output indices
// Simple block scan for demo (use CUB for production)
__global__ void scan_flags(int* output_indices, const int* flags, int* block_sums, int n) {
    __shared__ int sdata[BLOCK_SIZE];
    __shared__ int warp_sums[8];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    // Load
    int val = (idx < n) ? flags[idx] : 0;
    
    // Warp inclusive scan
    val = warp_inclusive_scan(val);
    
    // Store warp totals
    if (lane == 31) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();
    
    // Scan warp sums in first warp
    if (warp_id == 0 && lane < 8) {
        int warp_val = warp_sums[lane];
        for (int offset = 1; offset < 8; offset *= 2) {
            int n = __shfl_up_sync(0xFF, warp_val, offset);
            if (lane >= offset) warp_val += n;
        }
        warp_sums[lane] = warp_val;
    }
    __syncthreads();
    
    // Get block total
    int block_total = warp_sums[7];
    
    // Store block sum for inter-block scan
    if (tid == 0) {
        block_sums[blockIdx.x] = block_total;
    }
    
    // Exclusive scan: shift right
    int exclusive = (lane == 0) ? 0 : __shfl_up_sync(FULL_MASK, val, 1);
    if (warp_id > 0) {
        exclusive += warp_sums[warp_id - 1];
    }
    
    if (idx < n) {
        output_indices[idx] = exclusive;
    }
}

// Step 3: Add block prefixes
__global__ void add_block_offsets(int* output_indices, const int* block_prefixes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && blockIdx.x > 0) {
        output_indices[idx] += block_prefixes[blockIdx.x - 1];
    }
}

// Step 4: Scatter using computed indices
__global__ void scatter(float* output, const float* input, const int* output_indices, 
                        const int* flags, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && flags[idx]) {
        output[output_indices[idx]] = input[idx];
    }
}

// Simple CPU scan for block sums
void scan_cpu(int* output, const int* input, int n) {
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = output[i-1] + input[i];
    }
}

int stream_compact_cpu(float* output, const float* input, int n, float threshold) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (input[i] > threshold) {
            output[count++] = input[i];
        }
    }
    return count;
}

int main() {
    printf("=== Stream Compaction using Scan ===\n\n");
    
    const int N = 1 << 20;  // 1M elements
    const float threshold = 0.5f;
    
    printf("Array size: %d elements\n", N);
    printf("Keeping elements > %.2f\n\n", threshold);
    
    // Allocate host memory
    float* h_input = new float[N];
    float* h_output = new float[N];
    float* h_output_cpu = new float[N];
    
    // Initialize with random values [0, 1]
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = rand() / (float)RAND_MAX;
    }
    
    // CPU reference
    int cpu_count = stream_compact_cpu(h_output_cpu, h_input, N, threshold);
    printf("CPU result: kept %d elements (%.1f%%)\n", cpu_count, 100.0f * cpu_count / N);
    
    // Allocate device memory
    float *d_input, *d_output;
    int *d_flags, *d_indices, *d_block_sums, *d_block_prefixes;
    
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_flags, N * sizeof(int));
    cudaMalloc(&d_indices, N * sizeof(int));
    cudaMalloc(&d_block_sums, num_blocks * sizeof(int));
    cudaMalloc(&d_block_prefixes, num_blocks * sizeof(int));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Step 1: Generate flags
    generate_flags<<<num_blocks, BLOCK_SIZE>>>(d_flags, d_input, N, threshold);
    
    // Step 2: Scan flags within blocks
    scan_flags<<<num_blocks, BLOCK_SIZE>>>(d_indices, d_flags, d_block_sums, N);
    
    // Step 2b: Scan block sums on CPU (for simplicity)
    int* h_block_sums = new int[num_blocks];
    int* h_block_prefixes = new int[num_blocks];
    cudaMemcpy(h_block_sums, d_block_sums, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
    scan_cpu(h_block_prefixes, h_block_sums, num_blocks);
    cudaMemcpy(d_block_prefixes, h_block_prefixes, num_blocks * sizeof(int), cudaMemcpyHostToDevice);
    
    // Step 3: Add block offsets
    add_block_offsets<<<num_blocks, BLOCK_SIZE>>>(d_indices, d_block_prefixes, N);
    
    // Step 4: Scatter
    scatter<<<num_blocks, BLOCK_SIZE>>>(d_output, d_input, d_indices, d_flags, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    // Get result count
    int gpu_count = h_block_prefixes[num_blocks - 1];
    
    printf("GPU result: kept %d elements (%.1f%%)\n", gpu_count, 100.0f * gpu_count / N);
    printf("Time: %.3f ms\n\n", ms);
    
    // Verify
    cudaMemcpy(h_output, d_output, gpu_count * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool passed = true;
    if (gpu_count != cpu_count) {
        printf("Count mismatch: GPU=%d, CPU=%d\n", gpu_count, cpu_count);
        passed = false;
    } else {
        for (int i = 0; i < cpu_count && passed; i++) {
            if (fabsf(h_output[i] - h_output_cpu[i]) > 1e-6) {
                printf("Value mismatch at %d: GPU=%.6f, CPU=%.6f\n", 
                       i, h_output[i], h_output_cpu[i]);
                passed = false;
            }
        }
    }
    
    if (passed) {
        printf("Verification: PASSED\n");
    }
    
    printf("\n=== Stream Compaction Algorithm ===\n");
    printf("1. Generate flags: flag[i] = (input[i] > threshold) ? 1 : 0\n");
    printf("2. Exclusive scan on flags → output indices\n");
    printf("3. Scatter: if flag[i], output[index[i]] = input[i]\n");
    printf("\nExample:\n");
    printf("  Input:   [0.3, 0.7, 0.2, 0.8, 0.1, 0.9]\n");
    printf("  Flags:   [0,   1,   0,   1,   0,   1  ]\n");
    printf("  Indices: [0,   0,   1,   1,   2,   2  ] (exclusive scan)\n");
    printf("  Output:  [0.7, 0.8, 0.9]\n");
    printf("\nThis pattern is fundamental to many GPU algorithms!\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_flags);
    cudaFree(d_indices);
    cudaFree(d_block_sums);
    cudaFree(d_block_prefixes);
    delete[] h_input;
    delete[] h_output;
    delete[] h_output_cpu;
    delete[] h_block_sums;
    delete[] h_block_prefixes;
    
    return 0;
}
