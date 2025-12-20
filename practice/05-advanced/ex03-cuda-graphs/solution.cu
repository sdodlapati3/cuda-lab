#include <stdio.h>
#include <cuda_runtime.h>

__global__ void initKernel(float* data, int n, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = val;
}

__global__ void squareKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = data[idx] * data[idx];
}

__global__ void addKernel(float* data, int n, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += val;
}

__global__ void reduceKernel(float* input, float* output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

void runWithoutGraph(float* d_data, float* d_partial, int n, int blocks) {
    initKernel<<<blocks, 256>>>(d_data, n, 2.0f);
    squareKernel<<<blocks, 256>>>(d_data, n);
    addKernel<<<blocks, 256>>>(d_data, n, 1.0f);
    reduceKernel<<<blocks, 256>>>(d_data, d_partial, n);
    cudaDeviceSynchronize();
}

cudaGraphExec_t createGraph(float* d_data, float* d_partial, int n, int blocks, cudaStream_t stream) {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    // Begin stream capture
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    // Launch kernels (captured, not executed)
    initKernel<<<blocks, 256, 0, stream>>>(d_data, n, 2.0f);
    squareKernel<<<blocks, 256, 0, stream>>>(d_data, n);
    addKernel<<<blocks, 256, 0, stream>>>(d_data, n, 1.0f);
    reduceKernel<<<blocks, 256, 0, stream>>>(d_data, d_partial, n);
    
    // End capture
    cudaStreamEndCapture(stream, &graph);
    
    // Create executable graph
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphDestroy(graph);
    
    return graphExec;
}

int main() {
    const int N = 1 << 20;
    const int blocks = (N + 255) / 256;
    
    float *d_data, *d_partial;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_partial, blocks * sizeof(float));
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Warmup
    runWithoutGraph(d_data, d_partial, N, blocks);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int ITERATIONS = 100;
    
    // Benchmark without graph
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        runWithoutGraph(d_data, d_partial, N, blocks);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float msWithout;
    cudaEventElapsedTime(&msWithout, start, stop);
    
    // Create graph once
    cudaGraphExec_t graphExec = createGraph(d_data, d_partial, N, blocks, stream);
    
    // Benchmark with graph
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float msWith;
    cudaEventElapsedTime(&msWith, start, stop);
    
    printf("Without CUDA Graph: %.3f ms (%d iterations)\n", msWithout, ITERATIONS);
    printf("With CUDA Graph:    %.3f ms (%d iterations)\n", msWith, ITERATIONS);
    printf("Speedup: %.2fx\n", msWithout / msWith);
    
    cudaGraphExecDestroy(graphExec);
    cudaFree(d_data);
    cudaFree(d_partial);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
