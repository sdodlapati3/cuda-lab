#include <stdio.h>
#include <cuda_runtime.h>
#include <algorithm>

// Base case: insertion sort for small arrays
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

// TODO: Implement partition function
// Returns pivot index after partitioning
__device__ int partition(int* data, int left, int right) {
    int pivot = data[right];
    int i = left - 1;
    
    for (int j = left; j < right; j++) {
        if (data[j] <= pivot) {
            i++;
            // Swap data[i] and data[j]
            int temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
    
    // Place pivot in correct position
    int temp = data[i + 1];
    data[i + 1] = data[right];
    data[right] = temp;
    
    return i + 1;
}

// TODO: Implement CDP quicksort
// Requirements:
// 1. Base case: use insertionSort for arrays <= 32 elements
// 2. Partition the array
// 3. Launch child kernels for left and right partitions
// 4. Synchronize children
__global__ void cdpQuicksort(int* data, int left, int right, int depth) {
    // Base case or depth limit
    if (right - left <= 32 || depth >= 16) {
        if (threadIdx.x == 0 && left < right) {
            insertionSort(data, left, right);
        }
        return;
    }
    
    // TODO: Partition
    // int pivotIdx = partition(data, left, right);
    
    // TODO: Launch child kernels
    // if (threadIdx.x == 0) {
    //     if (left < pivotIdx - 1) {
    //         cdpQuicksort<<<1, 1>>>(data, left, pivotIdx - 1, depth + 1);
    //     }
    //     if (pivotIdx + 1 < right) {
    //         cdpQuicksort<<<1, 1>>>(data, pivotIdx + 1, right, depth + 1);
    //     }
    //     cudaDeviceSynchronize();
    // }
    
    // Placeholder - uses insertion sort for now
    if (threadIdx.x == 0) {
        insertionSort(data, left, right);
    }
}

int main() {
    const int N = 10000;
    int* h_data = new int[N];
    int* h_sorted = new int[N];
    
    // Generate random data
    srand(42);
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 100000;
        h_sorted[i] = h_data[i];
    }
    
    // CPU sort for verification
    std::sort(h_sorted, h_sorted + N);
    
    int* d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Set recursion limit
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 16);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
    
    cdpQuicksort<<<1, 1>>>(d_data, 0, N - 1, 0);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Verify
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
