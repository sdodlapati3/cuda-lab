/*
 * Day 6: Inference Pipeline
 * 
 * Complete optimized inference pipeline.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <vector>

#define CHECK_CUDA(call) do { cudaError_t e = call; if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } } while(0)
#define CHECK_CUBLAS(call) do { cublasStatus_t s = call; if (s != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS error\n"); exit(1); } } while(0)

__global__ void fused_bias_relu(float* x, const float* bias, int n, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = fmaxf(0.0f, x[idx] + bias[idx % features]);
}

__global__ void softmax(float* x, int batchSize, int numClasses) {
    int batch = blockIdx.x;
    if (batch >= batchSize) return;
    
    float* row = x + batch * numClasses;
    float maxVal = row[0];
    for (int i = 1; i < numClasses; i++) maxVal = fmaxf(maxVal, row[i]);
    
    float sum = 0.0f;
    for (int i = 0; i < numClasses; i++) {
        row[i] = expf(row[i] - maxVal);
        sum += row[i];
    }
    for (int i = 0; i < numClasses; i++) row[i] /= sum;
}

class OptimizedMLP {
    cublasHandle_t cublas;
    std::vector<float*> weights, biases;
    std::vector<int> sizes;
    float *bufA, *bufB;
    int maxSize, batchSize;
    
public:
    OptimizedMLP(const std::vector<int>& layerSizes, int bs) : sizes(layerSizes), batchSize(bs) {
        CHECK_CUBLAS(cublasCreate(&cublas));
        
        maxSize = 0;
        for (int s : sizes) maxSize = std::max(maxSize, s);
        
        CHECK_CUDA(cudaMalloc(&bufA, maxSize * batchSize * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&bufB, maxSize * batchSize * sizeof(float)));
        
        for (size_t i = 0; i < sizes.size() - 1; i++) {
            float *w, *b;
            CHECK_CUDA(cudaMalloc(&w, sizes[i] * sizes[i+1] * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&b, sizes[i+1] * sizeof(float)));
            weights.push_back(w);
            biases.push_back(b);
        }
    }
    
    ~OptimizedMLP() {
        cudaFree(bufA); cudaFree(bufB);
        for (auto w : weights) cudaFree(w);
        for (auto b : biases) cudaFree(b);
        cublasDestroy(cublas);
    }
    
    void forward(float* input, float* output) {
        float *in = input, *out = bufA;
        const float alpha = 1.0f, beta = 0.0f;
        
        for (size_t i = 0; i < weights.size(); i++) {
            CHECK_CUBLAS(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                sizes[i+1], batchSize, sizes[i], &alpha,
                weights[i], sizes[i+1], in, sizes[i], &beta, out, sizes[i+1]));
            
            int n = batchSize * sizes[i+1];
            if (i < weights.size() - 1) {
                fused_bias_relu<<<(n+255)/256, 256>>>(out, biases[i], n, sizes[i+1]);
            }
            
            in = out;
            out = (out == bufA) ? bufB : bufA;
        }
        
        softmax<<<batchSize, 1>>>(in, batchSize, sizes.back());
        CHECK_CUDA(cudaMemcpy(output, in, batchSize * sizes.back() * sizeof(float), cudaMemcpyDeviceToDevice));
    }
};

int main() {
    printf("=== Day 6: Inference Pipeline ===\n\n");
    
    const int batchSize = 64;
    std::vector<int> arch = {784, 512, 256, 10};
    
    OptimizedMLP model(arch, batchSize);
    
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, batchSize * 784 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, batchSize * 10 * sizeof(float)));
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Warmup
    model.forward(d_input, d_output);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    const int iterations = 1000;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        model.forward(d_input, d_output);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float msPerBatch = ms / iterations;
    
    printf("Architecture: 784 -> 512 -> 256 -> 10\n");
    printf("Batch size: %d\n", batchSize);
    printf("Latency: %.3f ms/batch\n", msPerBatch);
    printf("Throughput: %.0f samples/sec\n", batchSize * 1000.0f / msPerBatch);
    
    printf("\n--- Week 18 Complete ---\n");
    printf("Covered: Model loading, batching, quantization, fusion, memory, pipelines\n");
    
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return 0;
}
