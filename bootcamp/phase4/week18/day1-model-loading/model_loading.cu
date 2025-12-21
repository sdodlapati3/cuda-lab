/*
 * Day 1: Model Loading
 * 
 * Efficient weight loading and memory layout for inference.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", \
                    __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Simple MLP layer descriptor
struct LinearLayer {
    int inputSize;
    int outputSize;
    float* weights;   // Device pointer
    float* bias;      // Device pointer
    size_t weightOffset;
    size_t biasOffset;
};

// Model container
class SimpleModel {
public:
    std::vector<LinearLayer> layers;
    float* weightBuffer;      // Unified weight buffer
    size_t totalWeightBytes;
    cublasHandle_t cublas;
    
    SimpleModel() : weightBuffer(nullptr), totalWeightBytes(0) {
        CHECK_CUBLAS(cublasCreate(&cublas));
    }
    
    ~SimpleModel() {
        if (weightBuffer) cudaFree(weightBuffer);
        cublasDestroy(cublas);
    }
    
    void addLayer(int inputSize, int outputSize) {
        LinearLayer layer;
        layer.inputSize = inputSize;
        layer.outputSize = outputSize;
        layer.weightOffset = totalWeightBytes;
        totalWeightBytes += inputSize * outputSize * sizeof(float);
        layer.biasOffset = totalWeightBytes;
        totalWeightBytes += outputSize * sizeof(float);
        
        // Align to 256 bytes
        totalWeightBytes = (totalWeightBytes + 255) & ~255;
        
        layers.push_back(layer);
    }
    
    void allocateWeights() {
        CHECK_CUDA(cudaMalloc(&weightBuffer, totalWeightBytes));
        
        // Set pointers for each layer
        for (auto& layer : layers) {
            layer.weights = (float*)((char*)weightBuffer + layer.weightOffset);
            layer.bias = (float*)((char*)weightBuffer + layer.biasOffset);
        }
    }
    
    void loadRandomWeights() {
        // Simulate loading from file
        std::vector<float> hostWeights(totalWeightBytes / sizeof(float));
        
        for (size_t i = 0; i < hostWeights.size(); i++) {
            // Xavier initialization approximation
            hostWeights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        
        CHECK_CUDA(cudaMemcpy(weightBuffer, hostWeights.data(), 
                              totalWeightBytes, cudaMemcpyHostToDevice));
    }
    
    void loadFromPinned(float* pinnedBuffer, size_t size, cudaStream_t stream) {
        CHECK_CUDA(cudaMemcpyAsync(weightBuffer, pinnedBuffer, size,
                                    cudaMemcpyHostToDevice, stream));
    }
};

// ReLU activation kernel
__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// Bias add kernel
__global__ void add_bias_kernel(float* output, const float* bias, 
                                 int batchSize, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize * features) {
        int featureIdx = idx % features;
        output[idx] += bias[featureIdx];
    }
}

// Forward pass through MLP
void forward(SimpleModel& model, float* d_input, float* d_output,
             float* d_temp, int batchSize) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    float* currentInput = d_input;
    float* currentOutput = d_temp;
    
    for (size_t i = 0; i < model.layers.size(); i++) {
        auto& layer = model.layers[i];
        
        // Use output buffer for last layer
        if (i == model.layers.size() - 1) {
            currentOutput = d_output;
        }
        
        // GEMM: output = weights^T * input
        // cuBLAS uses column-major, so we compute: output^T = input^T * weights
        CHECK_CUBLAS(cublasSgemm(model.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                  layer.outputSize, batchSize, layer.inputSize,
                                  &alpha,
                                  layer.weights, layer.inputSize,
                                  currentInput, layer.inputSize,
                                  &beta,
                                  currentOutput, layer.outputSize));
        
        // Add bias
        int totalElements = batchSize * layer.outputSize;
        int blockSize = 256;
        int numBlocks = (totalElements + blockSize - 1) / blockSize;
        add_bias_kernel<<<numBlocks, blockSize>>>(currentOutput, layer.bias,
                                                   batchSize, layer.outputSize);
        
        // ReLU (except for last layer)
        if (i < model.layers.size() - 1) {
            relu_kernel<<<numBlocks, blockSize>>>(currentOutput, totalElements);
        }
        
        // Swap buffers
        std::swap(currentInput, currentOutput);
        currentOutput = (currentInput == d_input) ? d_temp : d_input;
    }
}

int main() {
    printf("=== Day 1: Model Loading ===\n\n");
    
    // Define model architecture
    const int batchSize = 32;
    const std::vector<int> layerSizes = {784, 512, 256, 128, 10};  // MNIST-like
    
    printf("Model architecture:\n");
    for (size_t i = 0; i < layerSizes.size() - 1; i++) {
        printf("  Layer %zu: %d -> %d\n", i, layerSizes[i], layerSizes[i+1]);
    }
    printf("\n");
    
    // Create model
    SimpleModel model;
    for (size_t i = 0; i < layerSizes.size() - 1; i++) {
        model.addLayer(layerSizes[i], layerSizes[i+1]);
    }
    
    printf("Total weight memory: %.2f MB\n", model.totalWeightBytes / (1024.0f * 1024.0f));
    printf("Number of parameters: %zu\n\n", model.totalWeightBytes / sizeof(float));
    
    // Allocate and load weights
    model.allocateWeights();
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Measure weight loading time
    printf("--- Weight Loading Performance ---\n");
    
    // Regular loading (pageable memory)
    float* h_weights = new float[model.totalWeightBytes / sizeof(float)];
    for (size_t i = 0; i < model.totalWeightBytes / sizeof(float); i++) {
        h_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(model.weightBuffer, h_weights, model.totalWeightBytes,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float msPageable;
    CHECK_CUDA(cudaEventElapsedTime(&msPageable, start, stop));
    printf("Pageable memory: %.3f ms\n", msPageable);
    
    // Pinned memory loading
    float* h_pinnedWeights;
    CHECK_CUDA(cudaMallocHost(&h_pinnedWeights, model.totalWeightBytes));
    memcpy(h_pinnedWeights, h_weights, model.totalWeightBytes);
    
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(model.weightBuffer, h_pinnedWeights, model.totalWeightBytes,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float msPinned;
    CHECK_CUDA(cudaEventElapsedTime(&msPinned, start, stop));
    printf("Pinned memory: %.3f ms (%.2fx faster)\n", msPinned, msPageable/msPinned);
    
    // Allocate activation buffers
    int maxSize = *std::max_element(layerSizes.begin(), layerSizes.end());
    float *d_input, *d_output, *d_temp;
    CHECK_CUDA(cudaMalloc(&d_input, batchSize * layerSizes[0] * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, batchSize * layerSizes.back() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_temp, batchSize * maxSize * sizeof(float)));
    
    // Initialize input
    std::vector<float> h_input(batchSize * layerSizes[0]);
    for (auto& v : h_input) v = (float)rand() / RAND_MAX;
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    
    // Warmup
    forward(model, d_input, d_output, d_temp, batchSize);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark inference
    printf("\n--- Inference Performance ---\n");
    const int iterations = 1000;
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        forward(model, d_input, d_output, d_temp, batchSize);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float msTotal;
    CHECK_CUDA(cudaEventElapsedTime(&msTotal, start, stop));
    float msPerBatch = msTotal / iterations;
    float samplesPerSec = batchSize * 1000.0f / msPerBatch;
    
    printf("Batch size: %d\n", batchSize);
    printf("Latency: %.3f ms/batch\n", msPerBatch);
    printf("Throughput: %.0f samples/sec\n", samplesPerSec);
    
    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFreeHost(h_pinnedWeights));
    delete[] h_weights;
    
    printf("\n=== Day 1 Complete ===\n");
    return 0;
}
