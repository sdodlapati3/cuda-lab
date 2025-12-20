#include <stdio.h>
#include <cuda_runtime.h>
#include <algorithm>

__device__ void insertionSort(int* data, int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        int key = data[i];
        int j = i - 1;
        while (j >= left && data[j] > key) {
            data[j + 1] = data[j];
            j--;
        }
        data[j + 1] = key;
    }
}

__device__ int partition(int* data, int left, int right) {
    int pivot = data[right];
    int i = left - 1;
    
    for (int j = left; j < right; j++) {
        if (data[j] <= pivot) {
            i++;
            int temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
    
    int temp = data[i + 1];
    data[i + 1] = data[right];
    data[right] = temp;
    
    return i + 1;
}

__global__ void cdpQuicksort(int* data, int left, int right, int depth) {
    if (right - left <= 32 || depth >= 16) {
        if (threadIdx.x == 0 && left < right) {
            insertionSort(data, left, right);
        }
        return;
    }
    
    int pivotIdx = partition(data, left, right);
    
    if (threadIdx.x == 0) {
        cudaStream_t s1, s2;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
        
        if (left < pivotIdx - 1) {
            cdpQuicksort<<<1, 1, 0, s1>>>(data, left, pivotIdx - 1, depth + 1);
        }
        if (pivotIdx + 1 < right) {
            cdpQuicksort<<<1, 1, 0, s2>>>(data, pivotIdx + 1, right, depth + 1);
        }
        
        cudaDeviceSynchronize();
        cudaStreamDestroy(s1);
        cudaStreamDestroy(s2);
    }
}

int main() {
    const int N = 10000;
    int* h_data = new int[N];
    int* h_sorted = new int[N];
    
    srand(42);
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 100000;
        h_sorted[i] = h_data[i];
    }
    
    std::sort(h_sorted, h_sorted + N);
    
    int* d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 16);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
    
    cdpQuicksort<<<1, 1>>>(d_data, 0, N - 1, 0);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    bool sorted = true;
    for (int i = 0; i < N; i++) {
        if (h_data[i] != h_sorted[i]) {
            sorted = false;
            printf("Mismatch at index %d: got %d, expected %d\n", 
                   i, h_data[i], h_sorted[i]);
            break;
        }
    }
    
    printf("Array size: %d\n", N);
    printf("Test %s\n", sorted ? "PASSED" : "FAILED");
    
    delete[] h_data;
    delete[] h_sorted;
    cudaFree(d_data);
    
    return sorted ? 0 : 1;
}
