/**
 * streams_demo.cu - Demonstrate CUDA streams for concurrency
 * 
 * Learning objectives:
 * - Create and use streams
 * - Run kernels concurrently
 * - Overlap compute and data transfer
 */

#include <cuda_runtime.h>
#include <cstdio>

// Kernel that does some work (easy to see in profiler)
__global__ void work_kernel(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        float val = data[i];
        for (int iter = 0; iter < iterations; iter++) {
            val = sinf(val) + 1.0f;
        }
        data[i] = val;
    }
}

int main() {
    printf("=== CUDA Streams Demo ===\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Async engines: %d\n", prop.asyncEngineCount);
    printf("Concurrent kernels: %s\n\n", prop.concurrentKernels ? "Yes" : "No");
    
    const int N = 1 << 20;
    const int ITERATIONS = 100;
    const int NUM_STREAMS = 4;
    
    // Allocate memory for each stream
    float* d_data[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMalloc(&d_data[i], N * sizeof(float));
        cudaMemset(d_data[i], 0, N * sizeof(float));
    }
    
    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    
    // Demo 1: Sequential execution (default stream)
    printf("=== Demo 1: Sequential (Default Stream) ===\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < NUM_STREAMS; i++) {
        work_kernel<<<num_blocks, block_size>>>(d_data[i], N, ITERATIONS);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float sequential_ms;
    cudaEventElapsedTime(&sequential_ms, start, stop);
    printf("Sequential time: %.3f ms\n\n", sequential_ms);
    
    // Reset data
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMemset(d_data[i], 0, N * sizeof(float));
    }
    
    // Demo 2: Concurrent execution with streams
    printf("=== Demo 2: Concurrent (Multiple Streams) ===\n");
    
    cudaEventRecord(start);
    for (int i = 0; i < NUM_STREAMS; i++) {
        work_kernel<<<num_blocks, block_size, 0, streams[i]>>>(d_data[i], N, ITERATIONS);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float concurrent_ms;
    cudaEventElapsedTime(&concurrent_ms, start, stop);
    printf("Concurrent time: %.3f ms\n", concurrent_ms);
    printf("Speedup: %.2fx\n\n", sequential_ms / concurrent_ms);
    
    // Demo 3: Stream synchronization
    printf("=== Demo 3: Stream Synchronization ===\n");
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        work_kernel<<<num_blocks, block_size, 0, streams[i]>>>(d_data[i], N, ITERATIONS);
    }
    
    // Sync only stream 0
    printf("Syncing stream 0...\n");
    cudaStreamSynchronize(streams[0]);
    printf("Stream 0 done. Other streams may still be running.\n");
    
    // Sync all
    printf("Syncing all...\n");
    cudaDeviceSynchronize();
    printf("All streams done.\n\n");
    
    // Demo 4: Overlapping copy and compute (requires pinned memory)
    printf("=== Demo 4: Overlap Copy and Compute ===\n");
    
    float* h_pinned;
    cudaMallocHost(&h_pinned, N * sizeof(float));  // Pinned memory!
    
    for (int i = 0; i < N; i++) h_pinned[i] = 1.0f;
    
    cudaEventRecord(start);
    
    // Stream 0: Copy data
    cudaMemcpyAsync(d_data[0], h_pinned, N * sizeof(float), 
                    cudaMemcpyHostToDevice, streams[0]);
    
    // Stream 1: Compute (can run while stream 0 copies)
    work_kernel<<<num_blocks, block_size, 0, streams[1]>>>(d_data[1], N, ITERATIONS);
    
    // Wait for copy, then compute
    cudaStreamSynchronize(streams[0]);
    work_kernel<<<num_blocks, block_size, 0, streams[0]>>>(d_data[0], N, ITERATIONS);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float overlap_ms;
    cudaEventElapsedTime(&overlap_ms, start, stop);
    printf("Overlapped copy+compute time: %.3f ms\n", overlap_ms);
    
    // Cleanup
    cudaFreeHost(h_pinned);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFree(d_data[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\n=== Key Points ===\n");
    printf("1. Default stream: All operations sequential\n");
    printf("2. Named streams: Enable concurrent execution\n");
    printf("3. cudaStreamSynchronize: Wait for specific stream\n");
    printf("4. cudaDeviceSynchronize: Wait for all streams\n");
    printf("5. Async copy requires pinned host memory\n");
    printf("\n");
    printf("Profile with: nsys profile ./build/streams_demo\n");
    
    return 0;
}
