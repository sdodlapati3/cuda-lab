/**
 * cub_basics.cu - CUB device-wide primitives
 * 
 * Learning objectives:
 * - Use CUB reduce, scan, sort
 * - Understand temp storage pattern
 * - See performance benefits
 */

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <algorithm>

int main() {
    printf("=== CUB Primitives Demo ===\n\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========================================================================
    // Part 1: Device Reduce
    // ========================================================================
    {
        printf("1. DeviceReduce::Sum\n");
        printf("─────────────────────────────────────────\n");
        
        const int N = 1 << 24;  // 16M
        
        // Host data
        int* h_in = new int[N];
        long long expected = 0;
        for (int i = 0; i < N; i++) {
            h_in[i] = 1;
            expected += h_in[i];
        }
        
        // Device data
        int* d_in;
        long long* d_out;
        cudaMalloc(&d_in, N * sizeof(int));
        cudaMalloc(&d_out, sizeof(long long));
        cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
        
        // Determine temp storage
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, N);
        cudaMalloc(&d_temp, temp_bytes);
        
        printf("   Temp storage: %zu bytes\n", temp_bytes);
        
        // Run reduce
        cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, N);
        
        long long result;
        cudaMemcpy(&result, d_out, sizeof(long long), cudaMemcpyDeviceToHost);
        printf("   Sum of %d elements: %lld (expected %lld)\n", N, result, expected);
        
        // Benchmark
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        float gbps = (float)N * sizeof(int) / (ms / 100 / 1000) / 1e9;
        printf("   Bandwidth: %.2f GB/s\n\n", gbps);
        
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_temp);
        delete[] h_in;
    }
    
    // ========================================================================
    // Part 2: Device Scan (Prefix Sum)
    // ========================================================================
    {
        printf("2. DeviceScan::ExclusiveSum\n");
        printf("─────────────────────────────────────────\n");
        
        const int N = 1 << 20;  // 1M
        
        int* h_in = new int[N];
        int* h_out = new int[N];
        for (int i = 0; i < N; i++) h_in[i] = 1;
        
        int *d_in, *d_out;
        cudaMalloc(&d_in, N * sizeof(int));
        cudaMalloc(&d_out, N * sizeof(int));
        cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
        
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out, N);
        cudaMalloc(&d_temp, temp_bytes);
        
        cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out, N);
        
        cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("   Input:  [1, 1, 1, 1, ...]\n");
        printf("   Output: [%d, %d, %d, %d, ... %d]\n", 
               h_out[0], h_out[1], h_out[2], h_out[3], h_out[N-1]);
        printf("   (Exclusive prefix sum)\n\n");
        
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_temp);
        delete[] h_in;
        delete[] h_out;
    }
    
    // ========================================================================
    // Part 3: Device RadixSort
    // ========================================================================
    {
        printf("3. DeviceRadixSort::SortKeys\n");
        printf("─────────────────────────────────────────\n");
        
        const int N = 1 << 20;  // 1M
        
        unsigned int* h_in = new unsigned int[N];
        for (int i = 0; i < N; i++) h_in[i] = rand();
        
        unsigned int *d_in, *d_out;
        cudaMalloc(&d_in, N * sizeof(unsigned int));
        cudaMalloc(&d_out, N * sizeof(unsigned int));
        cudaMemcpy(d_in, h_in, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
        
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, N);
        cudaMalloc(&d_temp, temp_bytes);
        
        printf("   Temp storage: %zu bytes (%.2f MB)\n", temp_bytes, temp_bytes / 1e6);
        
        cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, N);
        
        unsigned int* h_out = new unsigned int[N];
        cudaMemcpy(h_out, d_out, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        
        // Verify sorted
        bool sorted = true;
        for (int i = 1; i < N; i++) {
            if (h_out[i] < h_out[i-1]) {
                sorted = false;
                break;
            }
        }
        printf("   Sorted %d elements: %s\n", N, sorted ? "SUCCESS" : "FAILED");
        
        // Benchmark
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("   Time: %.4f ms\n", ms / 100);
        printf("   Rate: %.2f M keys/sec\n\n", N / (ms / 100 / 1000) / 1e6);
        
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_temp);
        delete[] h_in;
        delete[] h_out;
    }
    
    // ========================================================================
    // Part 4: Device Select (Stream Compaction)
    // ========================================================================
    {
        printf("4. DeviceSelect::If\n");
        printf("─────────────────────────────────────────\n");
        
        const int N = 1 << 20;
        
        int* h_in = new int[N];
        int expected = 0;
        for (int i = 0; i < N; i++) {
            h_in[i] = rand() % 100;
            if (h_in[i] > 50) expected++;
        }
        
        int *d_in, *d_out, *d_num_selected;
        cudaMalloc(&d_in, N * sizeof(int));
        cudaMalloc(&d_out, N * sizeof(int));
        cudaMalloc(&d_num_selected, sizeof(int));
        cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
        
        // Select elements > 50
        auto select_op = [] __device__ (int x) { return x > 50; };
        
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        cub::DeviceSelect::If(d_temp, temp_bytes, d_in, d_out, d_num_selected, N, select_op);
        cudaMalloc(&d_temp, temp_bytes);
        
        cub::DeviceSelect::If(d_temp, temp_bytes, d_in, d_out, d_num_selected, N, select_op);
        
        int num_selected;
        cudaMemcpy(&num_selected, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("   Selected %d elements > 50 (expected ~%d)\n\n", num_selected, expected);
        
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_num_selected);
        cudaFree(d_temp);
        delete[] h_in;
    }
    
    printf("=== CUB Pattern ===\n\n");
    printf("1. Query temp storage size (pass nullptr)\n");
    printf("2. Allocate temp storage\n");
    printf("3. Run algorithm\n");
    printf("4. Reuse temp storage for same-size problems\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
