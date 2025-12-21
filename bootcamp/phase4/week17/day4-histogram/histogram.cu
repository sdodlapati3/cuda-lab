/*
 * Day 4: Histogram & Equalization
 * 
 * GPU histogram computation with atomic optimizations.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define NUM_BINS 256
#define BLOCK_SIZE 256

// Naive histogram with global atomics
__global__ void histogram_global_atomic(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        atomicAdd(&histogram[input[i]], 1);
    }
}

// Histogram with shared memory privatization
__global__ void histogram_shared_atomic(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    int size)
{
    __shared__ unsigned int localHist[NUM_BINS];
    
    // Initialize shared histogram
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();
    
    // Accumulate to shared memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        atomicAdd(&localHist[input[i]], 1);
    }
    __syncthreads();
    
    // Write back to global memory
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&histogram[i], localHist[i]);
    }
}

// Histogram with warp aggregation
__global__ void histogram_warp_aggregated(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    int size)
{
    __shared__ unsigned int localHist[NUM_BINS];
    
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        unsigned char val = input[i];
        
        // Warp-level aggregation: count duplicates within warp
        unsigned int mask = __match_any_sync(0xFFFFFFFF, val);
        int leader = __ffs(mask) - 1;
        
        if ((threadIdx.x % 32) == leader) {
            atomicAdd(&localHist[val], __popc(mask));
        }
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&histogram[i], localHist[i]);
    }
}

// Compute CDF for histogram equalization
__global__ void compute_cdf(
    const unsigned int* __restrict__ histogram,
    float* __restrict__ cdf,
    int totalPixels)
{
    __shared__ unsigned int temp[NUM_BINS];
    
    int tid = threadIdx.x;
    temp[tid] = histogram[tid];
    __syncthreads();
    
    // Inclusive scan (Blelloch)
    for (int stride = 1; stride < NUM_BINS; stride *= 2) {
        int val = 0;
        if (tid >= stride) {
            val = temp[tid - stride];
        }
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }
    
    // Normalize
    cdf[tid] = (float)temp[tid] / totalPixels;
}

// Apply histogram equalization
__global__ void apply_equalization(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    const float* __restrict__ cdf,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        output[i] = (unsigned char)(cdf[input[i]] * 255.0f);
    }
}

// CPU reference histogram
void histogram_cpu(const unsigned char* input, unsigned int* histogram, int size) {
    for (int i = 0; i < NUM_BINS; i++) histogram[i] = 0;
    for (int i = 0; i < size; i++) histogram[input[i]]++;
}

int main() {
    printf("=== Day 4: Histogram & Equalization ===\n\n");
    
    const int width = 1920;
    const int height = 1080;
    const int size = width * height;
    const int iterations = 100;
    
    printf("Image size: %d x %d (%d pixels)\n\n", width, height, size);
    
    // Host memory
    unsigned char* h_input = new unsigned char[size];
    unsigned char* h_output = new unsigned char[size];
    unsigned int h_histogram[NUM_BINS] = {0};
    unsigned int h_histogram_gpu[NUM_BINS];
    
    // Random input with bias (to test equalization)
    for (int i = 0; i < size; i++) {
        // Biased toward darker values
        h_input[i] = (unsigned char)(rand() % 128 + (rand() % 64));
    }
    
    // Device memory
    unsigned char *d_input, *d_output;
    unsigned int *d_histogram;
    float *d_cdf;
    
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output, size));
    CHECK_CUDA(cudaMalloc(&d_histogram, NUM_BINS * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_cdf, NUM_BINS * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    // CPU reference
    histogram_cpu(h_input, h_histogram, size);
    
    // Verify correctness
    printf("--- Correctness Check ---\n");
    
    CHECK_CUDA(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(unsigned int)));
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    numBlocks = min(numBlocks, 256);  // Limit blocks
    
    histogram_shared_atomic<<<numBlocks, BLOCK_SIZE>>>(d_input, d_histogram, size);
    CHECK_CUDA(cudaMemcpy(h_histogram_gpu, d_histogram, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (int i = 0; i < NUM_BINS; i++) {
        if (h_histogram[i] != h_histogram_gpu[i]) {
            printf("Mismatch at bin %d: CPU=%u, GPU=%u\n", i, h_histogram[i], h_histogram_gpu[i]);
            correct = false;
        }
    }
    printf("Histogram %s\n\n", correct ? "PASSED" : "FAILED");
    
    // Benchmark different implementations
    printf("--- Performance Comparison ---\n");
    printf("%-25s %12s %12s\n", "Method", "Time (ms)", "Throughput");
    printf("%-25s %12s %12s\n", "------", "---------", "----------");
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Global atomic
    CHECK_CUDA(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(unsigned int)));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(unsigned int)));
        histogram_global_atomic<<<numBlocks, BLOCK_SIZE>>>(d_input, d_histogram, size);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float msGlobal;
    CHECK_CUDA(cudaEventElapsedTime(&msGlobal, start, stop));
    msGlobal /= iterations;
    float gbsGlobal = (size / (1024.0f * 1024.0f * 1024.0f)) / (msGlobal / 1000.0f);
    printf("%-25s %12.3f %10.1f GB/s\n", "Global Atomics", msGlobal, gbsGlobal);
    
    // Shared atomic
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(unsigned int)));
        histogram_shared_atomic<<<numBlocks, BLOCK_SIZE>>>(d_input, d_histogram, size);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float msShared;
    CHECK_CUDA(cudaEventElapsedTime(&msShared, start, stop));
    msShared /= iterations;
    float gbsShared = (size / (1024.0f * 1024.0f * 1024.0f)) / (msShared / 1000.0f);
    printf("%-25s %12.3f %10.1f GB/s\n", "Shared Memory", msShared, gbsShared);
    
    // Warp aggregated
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(unsigned int)));
        histogram_warp_aggregated<<<numBlocks, BLOCK_SIZE>>>(d_input, d_histogram, size);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float msWarp;
    CHECK_CUDA(cudaEventElapsedTime(&msWarp, start, stop));
    msWarp /= iterations;
    float gbsWarp = (size / (1024.0f * 1024.0f * 1024.0f)) / (msWarp / 1000.0f);
    printf("%-25s %12.3f %10.1f GB/s\n", "Warp Aggregation", msWarp, gbsWarp);
    
    // Full equalization pipeline
    printf("\n--- Histogram Equalization Pipeline ---\n");
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(unsigned int)));
        histogram_shared_atomic<<<numBlocks, BLOCK_SIZE>>>(d_input, d_histogram, size);
        compute_cdf<<<1, NUM_BINS>>>(d_histogram, d_cdf, size);
        apply_equalization<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, d_cdf, size);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float msPipeline;
    CHECK_CUDA(cudaEventElapsedTime(&msPipeline, start, stop));
    msPipeline /= iterations;
    printf("Full pipeline time: %.3f ms\n", msPipeline);
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_histogram));
    CHECK_CUDA(cudaFree(d_cdf));
    
    delete[] h_input;
    delete[] h_output;
    
    printf("\n=== Day 4 Complete ===\n");
    return 0;
}
