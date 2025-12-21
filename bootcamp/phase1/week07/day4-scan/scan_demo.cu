/**
 * scan_demo.cu - Prefix sum (scan) algorithms
 * 
 * Learning objectives:
 * - Inclusive vs exclusive scan
 * - Warp-level and block-level scan
 * - Work-efficient algorithms
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 256
#define FULL_MASK 0xFFFFFFFF

// Warp-level inclusive scan using shuffle
__device__ float warp_inclusive_scan(float val) {
    for (int offset = 1; offset < 32; offset *= 2) {
        float n = __shfl_up_sync(FULL_MASK, val, offset);
        if (threadIdx.x % 32 >= offset) {
            val += n;
        }
    }
    return val;
}

// Block-level inclusive scan (Hillis-Steele in shared memory)
__device__ void block_inclusive_scan_hillis_steele(float* sdata, int tid) {
    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        float temp = 0.0f;
        if (tid >= offset) {
            temp = sdata[tid - offset];
        }
        __syncthreads();
        sdata[tid] += temp;
        __syncthreads();
    }
}

// Blelloch scan (work-efficient): Up-sweep then down-sweep
__device__ void block_exclusive_scan_blelloch(float* sdata, int tid) {
    int n = BLOCK_SIZE;
    
    // Up-sweep (reduce) phase
    for (int stride = 1; stride < n; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < n) {
            sdata[index] += sdata[index - stride];
        }
        __syncthreads();
    }
    
    // Clear the last element
    if (tid == 0) {
        sdata[n - 1] = 0.0f;
    }
    __syncthreads();
    
    // Down-sweep phase
    for (int stride = n / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < n) {
            float temp = sdata[index - stride];
            sdata[index - stride] = sdata[index];
            sdata[index] += temp;
        }
        __syncthreads();
    }
}

// Kernel: Warp scan
__global__ void warp_scan_kernel(float* output, const float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (idx < n) ? input[idx] : 0.0f;
    val = warp_inclusive_scan(val);
    
    if (idx < n) {
        output[idx] = val;
    }
}

// Kernel: Block scan (Hillis-Steele)
__global__ void block_scan_hillis_steele(float* output, const float* input, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    block_inclusive_scan_hillis_steele(sdata, tid);
    
    if (idx < n) {
        output[idx] = sdata[tid];
    }
}

// Kernel: Block scan (Blelloch - work efficient)
__global__ void block_scan_blelloch(float* output, const float* input, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    float original = sdata[tid];
    
    block_exclusive_scan_blelloch(sdata, tid);
    
    // Convert exclusive to inclusive
    if (idx < n) {
        output[idx] = sdata[tid] + original;
    }
}

// Efficient warp-based block scan
__global__ void block_scan_warp_based(float* output, const float* input, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    __shared__ float warp_sums[8];  // For 256 threads = 8 warps
    
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // Warp-level inclusive scan
    val = warp_inclusive_scan(val);
    
    // Last thread in each warp writes sum to shared memory
    if (lane == 31) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();
    
    // First warp scans the warp sums
    float warp_prefix = 0.0f;
    if (warp_id == 0 && lane < 8) {
        float warp_val = warp_sums[lane];
        // Scan warp sums
        for (int offset = 1; offset < 8; offset *= 2) {
            float n = __shfl_up_sync(0xFF, warp_val, offset);
            if (lane >= offset) {
                warp_val += n;
            }
        }
        // Convert to exclusive for prefix
        warp_sums[lane] = (lane == 0) ? 0.0f : __shfl_up_sync(0xFF, warp_val, 1);
    }
    __syncthreads();
    
    // Add warp prefix to all elements
    if (warp_id > 0) {
        val += warp_sums[warp_id];
    }
    
    if (idx < n) {
        output[idx] = val;
    }
}

void inclusive_scan_cpu(const float* input, float* output, int n) {
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = output[i-1] + input[i];
    }
}

bool verify(const float* gpu, const float* cpu, int n, const char* name) {
    float max_err = 0.0f;
    int err_idx = -1;
    
    for (int i = 0; i < n; i++) {
        float err = fabsf(gpu[i] - cpu[i]);
        if (err > max_err) {
            max_err = err;
            err_idx = i;
        }
    }
    
    float tolerance = 1e-3f * n;  // Accumulation error grows with n
    if (max_err < tolerance) {
        printf("  %s: PASSED (max error: %.2e at index %d)\n", name, max_err, err_idx);
        return true;
    } else {
        printf("  %s: FAILED (max error: %.2e at index %d)\n", name, max_err, err_idx);
        printf("    GPU[%d] = %.6f, CPU[%d] = %.6f\n", err_idx, gpu[err_idx], err_idx, cpu[err_idx]);
        return false;
    }
}

int main() {
    printf("=== Prefix Sum (Scan) Algorithms ===\n\n");
    
    // Small test for correctness
    const int N_SMALL = 256;
    printf("Correctness test (N = %d):\n", N_SMALL);
    
    float h_input_small[N_SMALL];
    float h_output_cpu[N_SMALL];
    float h_output_gpu[N_SMALL];
    
    for (int i = 0; i < N_SMALL; i++) {
        h_input_small[i] = i + 1;  // 1, 2, 3, ...
    }
    
    inclusive_scan_cpu(h_input_small, h_output_cpu, N_SMALL);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, N_SMALL * sizeof(float));
    cudaMalloc(&d_output, N_SMALL * sizeof(float));
    cudaMemcpy(d_input, h_input_small, N_SMALL * sizeof(float), cudaMemcpyHostToDevice);
    
    // Test Hillis-Steele
    block_scan_hillis_steele<<<1, N_SMALL>>>(d_output, d_input, N_SMALL);
    cudaMemcpy(h_output_gpu, d_output, N_SMALL * sizeof(float), cudaMemcpyDeviceToHost);
    verify(h_output_gpu, h_output_cpu, N_SMALL, "Hillis-Steele");
    
    // Test Blelloch
    block_scan_blelloch<<<1, N_SMALL>>>(d_output, d_input, N_SMALL);
    cudaMemcpy(h_output_gpu, d_output, N_SMALL * sizeof(float), cudaMemcpyDeviceToHost);
    verify(h_output_gpu, h_output_cpu, N_SMALL, "Blelloch");
    
    // Test Warp-based
    block_scan_warp_based<<<1, N_SMALL>>>(d_output, d_input, N_SMALL);
    cudaMemcpy(h_output_gpu, d_output, N_SMALL * sizeof(float), cudaMemcpyDeviceToHost);
    verify(h_output_gpu, h_output_cpu, N_SMALL, "Warp-based");
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    // Performance test
    printf("\n=== Performance Test ===\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    
    const int N = 1 << 24;
    const int TRIALS = 100;
    size_t bytes = N * sizeof(float);
    
    printf("Array size: %d elements (%.1f MB)\n\n", N, bytes / 1e6);
    
    float* h_input = new float[N];
    for (int i = 0; i < N; i++) {
        h_input[i] = (rand() / (float)RAND_MAX) * 0.001f;  // Small to avoid overflow
    }
    
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    printf("Note: These are single-block algorithms, so they scan\n");
    printf("each block independently (not full array scan).\n\n");
    
    printf("%-25s %-12s %-15s\n", "Algorithm", "Time(ms)", "Bandwidth(GB/s)");
    printf("----------------------------------------------------\n");
    
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Hillis-Steele
    block_scan_hillis_steele<<<blocks, BLOCK_SIZE>>>(d_output, d_input, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        block_scan_hillis_steele<<<blocks, BLOCK_SIZE>>>(d_output, d_input, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    ms /= TRIALS;
    printf("%-25s %-12.3f %-15.1f\n", "Hillis-Steele", ms, 2.0f * bytes / ms / 1e6);
    
    // Blelloch
    block_scan_blelloch<<<blocks, BLOCK_SIZE>>>(d_output, d_input, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        block_scan_blelloch<<<blocks, BLOCK_SIZE>>>(d_output, d_input, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    ms /= TRIALS;
    printf("%-25s %-12.3f %-15.1f\n", "Blelloch", ms, 2.0f * bytes / ms / 1e6);
    
    // Warp-based
    block_scan_warp_based<<<blocks, BLOCK_SIZE>>>(d_output, d_input, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        block_scan_warp_based<<<blocks, BLOCK_SIZE>>>(d_output, d_input, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    ms /= TRIALS;
    printf("%-25s %-12.3f %-15.1f\n", "Warp-based", ms, 2.0f * bytes / ms / 1e6);
    
    printf("\n=== Algorithm Comparison ===\n");
    printf("Hillis-Steele:\n");
    printf("  - Simple, low span O(log n)\n");
    printf("  - Work-inefficient: O(n log n)\n");
    printf("  - Good for small arrays\n");
    printf("\nBlelloch:\n");
    printf("  - Work-efficient: O(n)\n");
    printf("  - Two phases: up-sweep + down-sweep\n");
    printf("  - Better for large arrays\n");
    printf("\nWarp-based:\n");
    printf("  - Uses shuffle for warp scan\n");
    printf("  - Hierarchical: warp → block\n");
    printf("  - Modern approach, minimal shared memory\n");
    printf("\nFor production: Use CUB's DeviceScan (uses decoupled look-back)\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    
    return 0;
}
