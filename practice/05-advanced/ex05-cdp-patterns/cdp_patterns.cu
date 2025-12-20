#include <stdio.h>
#include <cuda_runtime.h>

// Base case: single element or small array - compute directly
__device__ int baseSum(int* data, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

// TODO: Implement recursive parallel sum using CDP
// Requirements:
// 1. Base case: if n <= 32, compute sum directly
// 2. Recursive case: split array, launch child kernels
// 3. Synchronize children with cudaDeviceSynchronize()
// 4. Combine results
__global__ void recursiveSum(int* data, int* result, int n) {
    if (n <= 32) {
        // Base case: compute directly
        if (threadIdx.x == 0) {
            *result = baseSum(data, n);
        }
        return;
    }
    
    // TODO: Recursive case
    // 1. Calculate midpoint
    // int mid = n / 2;
    
    // 2. Allocate space for partial results (or use result array cleverly)
    // __shared__ int left_result, right_result;
    
    // 3. Launch child kernels for left and right halves
    // if (threadIdx.x == 0) {
    //     recursiveSum<<<1, 1>>>(data, &left_result, mid);
    //     recursiveSum<<<1, 1>>>(data + mid, &right_result, n - mid);
    // }
    
    // 4. Wait for children
    // cudaDeviceSynchronize();
    
    // 5. Combine results
    // if (threadIdx.x == 0) {
    //     *result = left_result + right_result;
    // }
    
    // Placeholder - replace with CDP implementation
    if (threadIdx.x == 0) {
        *result = baseSum(data, n);
    }
}

int main() {
    const int N = 1024;
    int h_data[N];
    int expected_sum = 0;
    
    for (int i = 0; i < N; i++) {
        h_data[i] = i + 1;  // 1 to N
        expected_sum += h_data[i];
    }
    
    int *d_data, *d_result;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
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
