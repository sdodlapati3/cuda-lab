/**
 * histogram.cu - Parallel histogram computation
 * 
 * Learning objectives:
 * - Atomic operations for scatter pattern
 * - Shared memory privatization
 * - Reducing atomic contention
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 256

// Version 1: Naive global atomics (high contention)
__global__ void histogram_global_atomic(unsigned int* histogram, 
                                         const unsigned char* data, 
                                         int n, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        int bin = data[i] % num_bins;
        atomicAdd(&histogram[bin], 1);
    }
}

// Version 2: Shared memory privatization (256 bins max)
__global__ void histogram_shared_atomic(unsigned int* histogram,
                                         const unsigned char* data,
                                         int n, int num_bins) {
    // Private histogram in shared memory
    extern __shared__ unsigned int s_hist[];
    
    int tid = threadIdx.x;
    
    // Initialize shared histogram to 0
    for (int i = tid; i < num_bins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();
    
    // Count in shared memory (much less contention)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        int bin = data[i] % num_bins;
        atomicAdd(&s_hist[bin], 1);
    }
    __syncthreads();
    
    // Merge to global histogram
    for (int i = tid; i < num_bins; i += blockDim.x) {
        atomicAdd(&histogram[i], s_hist[i]);
    }
}

// Version 3: Multiple sub-histograms per block to reduce contention
#define NUM_SUB_HISTOGRAMS 4

__global__ void histogram_multi_private(unsigned int* histogram,
                                         const unsigned char* data,
                                         int n, int num_bins) {
    // Multiple private histograms per block
    extern __shared__ unsigned int s_hist[];
    
    int tid = threadIdx.x;
    int sub_id = tid % NUM_SUB_HISTOGRAMS;
    
    // Initialize all sub-histograms
    for (int i = tid; i < num_bins * NUM_SUB_HISTOGRAMS; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();
    
    // Each thread uses its assigned sub-histogram
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        int bin = data[i] % num_bins;
        // Write to thread's private sub-histogram
        atomicAdd(&s_hist[sub_id * num_bins + bin], 1);
    }
    __syncthreads();
    
    // Merge sub-histograms within block, then to global
    for (int bin = tid; bin < num_bins; bin += blockDim.x) {
        unsigned int sum = 0;
        for (int s = 0; s < NUM_SUB_HISTOGRAMS; s++) {
            sum += s_hist[s * num_bins + bin];
        }
        if (sum > 0) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

// Version 4: Warp-level aggregation before atomic
__global__ void histogram_warp_aggregate(unsigned int* histogram,
                                          const unsigned char* data,
                                          int n, int num_bins) {
    extern __shared__ unsigned int s_hist[];
    
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    // Initialize shared histogram
    for (int i = tid; i < num_bins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        int bin = data[i] % num_bins;
        
        // Warp-level vote: which threads have same bin?
        unsigned int mask = __match_any_sync(0xFFFFFFFF, bin);
        
        // Count matching threads
        int count = __popc(mask);
        
        // Leader thread (lowest lane in match) does atomic
        int leader = __ffs(mask) - 1;
        if (lane == leader) {
            atomicAdd(&s_hist[bin], count);
        }
    }
    __syncthreads();
    
    // Merge to global
    for (int i = tid; i < num_bins; i += blockDim.x) {
        if (s_hist[i] > 0) {
            atomicAdd(&histogram[i], s_hist[i]);
        }
    }
}

void histogram_cpu(unsigned int* histogram, const unsigned char* data, 
                   int n, int num_bins) {
    for (int i = 0; i < num_bins; i++) {
        histogram[i] = 0;
    }
    for (int i = 0; i < n; i++) {
        histogram[data[i] % num_bins]++;
    }
}

bool verify(const unsigned int* gpu, const unsigned int* cpu, int num_bins) {
    for (int i = 0; i < num_bins; i++) {
        if (gpu[i] != cpu[i]) {
            printf("Mismatch at bin %d: GPU=%u, CPU=%u\n", i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("=== Parallel Histogram ===\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n\n", prop.name);
    
    const int N = 1 << 24;  // 16M elements
    const int NUM_BINS = 256;
    const int TRIALS = 100;
    
    printf("Data size: %d elements\n", N);
    printf("Number of bins: %d\n\n", NUM_BINS);
    
    // Allocate host memory
    unsigned char* h_data = new unsigned char[N];
    unsigned int* h_hist_cpu = new unsigned int[NUM_BINS];
    unsigned int* h_hist_gpu = new unsigned int[NUM_BINS];
    
    // Generate random data (uniform distribution)
    srand(42);
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % NUM_BINS;
    }
    
    // CPU reference
    histogram_cpu(h_hist_cpu, h_data, N, NUM_BINS);
    
    // Allocate device memory
    unsigned char* d_data;
    unsigned int* d_histogram;
    
    cudaMalloc(&d_data, N);
    cudaMalloc(&d_histogram, NUM_BINS * sizeof(unsigned int));
    
    cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int num_blocks = 256;
    float ms;
    
    printf("%-25s %-12s %-12s %-10s\n", "Version", "Time(ms)", "Throughput", "Verified");
    printf("--------------------------------------------------------------\n");
    
    auto benchmark = [&](const char* name, auto kernel_fn, int shared_size) {
        cudaMemset(d_histogram, 0, NUM_BINS * sizeof(unsigned int));
        kernel_fn<<<num_blocks, BLOCK_SIZE, shared_size>>>(d_histogram, d_data, N, NUM_BINS);
        cudaDeviceSynchronize();
        
        cudaEventRecord(start);
        for (int t = 0; t < TRIALS; t++) {
            cudaMemset(d_histogram, 0, NUM_BINS * sizeof(unsigned int));
            kernel_fn<<<num_blocks, BLOCK_SIZE, shared_size>>>(d_histogram, d_data, N, NUM_BINS);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        ms /= TRIALS;
        
        cudaMemcpy(h_hist_gpu, d_histogram, NUM_BINS * sizeof(unsigned int), 
                   cudaMemcpyDeviceToHost);
        
        bool passed = verify(h_hist_gpu, h_hist_cpu, NUM_BINS);
        float throughput = N / ms / 1e6;  // GB/s (1 byte per element)
        
        printf("%-25s %-12.3f %-12.1f %-10s\n", name, ms, throughput, 
               passed ? "PASSED" : "FAILED");
    };
    
    benchmark("V1: Global atomic", histogram_global_atomic, 0);
    benchmark("V2: Shared atomic", histogram_shared_atomic, NUM_BINS * sizeof(unsigned int));
    benchmark("V3: Multi-private", histogram_multi_private, 
              NUM_BINS * NUM_SUB_HISTOGRAMS * sizeof(unsigned int));
    benchmark("V4: Warp aggregate", histogram_warp_aggregate, NUM_BINS * sizeof(unsigned int));
    
    printf("\n=== Analysis ===\n");
    printf("V1 (Global atomic):\n");
    printf("  - Simple but high contention on popular bins\n");
    printf("  - All threads compete for same global memory\n");
    printf("\nV2 (Shared atomic):\n");
    printf("  - Block-private histogram in shared memory\n");
    printf("  - Much faster atomics (shared vs global)\n");
    printf("  - Only one global merge per block\n");
    printf("\nV3 (Multi-private):\n");
    printf("  - Multiple sub-histograms reduce contention\n");
    printf("  - Threads interleaved across sub-histograms\n");
    printf("  - Extra merge step but less waiting\n");
    printf("\nV4 (Warp aggregate):\n");
    printf("  - Use __match_any_sync to find threads with same bin\n");
    printf("  - One thread does atomic for all matches\n");
    printf("  - Great for clustered data\n");
    
    printf("\n=== Key Takeaways ===\n");
    printf("1. Atomics are slow when contended\n");
    printf("2. Privatization (private copies) reduces contention\n");
    printf("3. Shared memory atomics >> global memory atomics\n");
    printf("4. Warp-level primitives can aggregate before atomic\n");
    printf("5. Best approach depends on data distribution\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_histogram);
    delete[] h_data;
    delete[] h_hist_cpu;
    delete[] h_hist_gpu;
    
    return 0;
}
