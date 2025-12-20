#include <stdio.h>
#include <cuda_runtime.h>

__device__ int baseSum(int* data, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

__global__ void recursiveSum(int* data, int* result, int n) {
    if (n <= 32) {
        if (threadIdx.x == 0) {
            *result = baseSum(data, n);
        }
        return;
    }
    
    int mid = n / 2;
    
    // Use managed memory or device malloc for child results
    int *left_result, *right_result;
    cudaMalloc(&left_result, sizeof(int));
    cudaMalloc(&right_result, sizeof(int));
    
    if (threadIdx.x == 0) {
        recursiveSum<<<1, 1>>>(data, left_result, mid);
        recursiveSum<<<1, 1>>>(data + mid, right_result, n - mid);
    }
    
    cudaDeviceSynchronize();
    
    if (threadIdx.x == 0) {
        *result = *left_result + *right_result;
    }
    
    cudaFree(left_result);
    cudaFree(right_result);
}

int main() {
    const int N = 1024;
    int h_data[N];
    int expected_sum = 0;
    
    for (int i = 0; i < N; i++) {
        h_data[i] = i + 1;
        expected_sum += h_data[i];
    }
    
    int *d_data, *d_result;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Increase device heap for CDP allocations
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
    
    recursiveSum<<<1, 1>>>(d_data, d_result, N);
    cudaDeviceSynchronize();
    
    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Sum of 1 to %d = %d (expected %d)\n", N, h_result, expected_sum);
    printf("Test %s\n", (h_result == expected_sum) ? "PASSED" : "FAILED");
    
    cudaFree(d_data);
    cudaFree(d_result);
    
    return (h_result == expected_sum) ? 0 : 1;
}
