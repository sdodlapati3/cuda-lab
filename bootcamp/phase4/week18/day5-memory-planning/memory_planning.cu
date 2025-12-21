/*
 * Day 5: Memory Planning
 * 
 * Efficient buffer allocation and reuse for inference.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(call) do { cudaError_t e = call; if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } } while(0)

class InferenceMemoryPool {
public:
    float* bufferA;
    float* bufferB;
    size_t bufferSize;
    
    InferenceMemoryPool(const std::vector<int>& layerSizes, int batchSize) {
        int maxSize = *std::max_element(layerSizes.begin(), layerSizes.end());
        bufferSize = maxSize * batchSize * sizeof(float);
        
        CHECK_CUDA(cudaMalloc(&bufferA, bufferSize));
        CHECK_CUDA(cudaMalloc(&bufferB, bufferSize));
        
        printf("Memory pool: 2 x %.2f MB = %.2f MB total\n",
               bufferSize / (1024.0f * 1024.0f),
               2 * bufferSize / (1024.0f * 1024.0f));
    }
    
    ~InferenceMemoryPool() {
        cudaFree(bufferA);
        cudaFree(bufferB);
    }
    
    // Get ping-pong buffers for layer i
    void getBuffers(int layerIdx, float*& input, float*& output) {
        if (layerIdx % 2 == 0) {
            input = bufferA;
            output = bufferB;
        } else {
            input = bufferB;
            output = bufferA;
        }
    }
};

// Compare with naive allocation (allocate per layer)
void compareAllocationStrategies(int numLayers, int batchSize) {
    std::vector<int> layerSizes;
    for (int i = 0; i <= numLayers; i++) {
        layerSizes.push_back(512);  // All same size for simplicity
    }
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    const int iterations = 100;
    size_t layerBytes = 512 * batchSize * sizeof(float);
    
    // Naive: allocate/free each inference
    CHECK_CUDA(cudaEventRecord(start));
    for (int iter = 0; iter < iterations; iter++) {
        std::vector<float*> buffers(numLayers);
        for (int i = 0; i < numLayers; i++) {
            CHECK_CUDA(cudaMalloc(&buffers[i], layerBytes));
        }
        for (int i = 0; i < numLayers; i++) {
            CHECK_CUDA(cudaFree(buffers[i]));
        }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_naive;
    CHECK_CUDA(cudaEventElapsedTime(&ms_naive, start, stop));
    
    // Pool: preallocated
    InferenceMemoryPool pool(layerSizes, batchSize);
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < numLayers; i++) {
            float *in, *out;
            pool.getBuffers(i, in, out);
            // Would run kernel here
        }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_pool;
    CHECK_CUDA(cudaEventElapsedTime(&ms_pool, start, stop));
    
    printf("\n--- Allocation Strategy Comparison ---\n");
    printf("Naive (alloc/free per inference): %.3f ms\n", ms_naive / iterations);
    printf("Pool (preallocated):              %.3f ms\n", ms_pool / iterations);
    printf("Speedup: %.0fx\n", ms_naive / ms_pool);
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    printf("=== Day 5: Memory Planning ===\n\n");
    
    compareAllocationStrategies(10, 64);
    
    printf("\n=== Day 5 Complete ===\n");
    return 0;
}
