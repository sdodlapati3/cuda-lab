/**
 * index_1d.cu - 1D thread indexing patterns
 * 
 * Learning objectives:
 * - Master 1D indexing for arrays and vectors
 * - Handle arrays larger than grid size
 * - Understand bounds checking
 */

#include <cuda_runtime.h>
#include <cstdio>

// Basic 1D indexing
__global__ void fill_1d(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx;
    }
}

// 1D with stride (grid-stride loop preview)
__global__ void fill_with_stride(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread handles multiple elements
    for (int i = idx; i < n; i += stride) {
        data[i] = i;
    }
}

// Offset-based indexing for segments
__global__ void fill_segment(int* data, int offset, int segment_size, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < segment_size) {
        data[offset + idx] = value;
    }
}

void print_array(const char* name, int* data, int n, int max_print = 20) {
    printf("%s: [", name);
    for (int i = 0; i < n && i < max_print; i++) {
        printf("%d", data[i]);
        if (i < n - 1 && i < max_print - 1) printf(", ");
    }
    if (n > max_print) printf(", ...");
    printf("]\n");
}

int main() {
    printf("=== 1D Thread Indexing ===\n\n");
    
    const int N = 100;
    int* d_data;
    int* h_data = new int[N];
    
    cudaMalloc(&d_data, N * sizeof(int));
    
    // Demo 1: Basic 1D indexing
    printf("=== Demo 1: Basic 1D Indexing ===\n");
    int block_size = 32;
    int num_blocks = (N + block_size - 1) / block_size;  // Ceiling division
    
    printf("N = %d, block_size = %d, num_blocks = %d\n", N, block_size, num_blocks);
    printf("Total threads = %d (>= N, some threads do nothing)\n\n", 
           num_blocks * block_size);
    
    fill_1d<<<num_blocks, block_size>>>(d_data, N);
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    print_array("Result", h_data, N);
    
    // Demo 2: Stride-based (for large arrays)
    printf("\n=== Demo 2: Grid-Stride Loop ===\n");
    const int LARGE_N = 1000;
    int* d_large;
    int* h_large = new int[LARGE_N];
    
    cudaMalloc(&d_large, LARGE_N * sizeof(int));
    
    // Use fewer blocks than elements
    int few_blocks = 4;
    printf("N = %d, but only %d blocks × %d threads = %d threads\n",
           LARGE_N, few_blocks, block_size, few_blocks * block_size);
    printf("Each thread processes %d elements\n\n",
           (LARGE_N + few_blocks * block_size - 1) / (few_blocks * block_size));
    
    fill_with_stride<<<few_blocks, block_size>>>(d_large, LARGE_N);
    cudaMemcpy(h_large, d_large, LARGE_N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Verify
    bool correct = true;
    for (int i = 0; i < LARGE_N; i++) {
        if (h_large[i] != i) {
            correct = false;
            break;
        }
    }
    printf("Grid-stride result: %s\n", correct ? "CORRECT" : "INCORRECT");
    
    cudaFree(d_large);
    delete[] h_large;
    
    // Demo 3: Segment processing
    printf("\n=== Demo 3: Segment Processing ===\n");
    cudaMemset(d_data, 0, N * sizeof(int));
    
    // Fill different segments with different values
    fill_segment<<<1, 32>>>(d_data, 0, 25, 1);   // First 25 elements = 1
    fill_segment<<<1, 32>>>(d_data, 25, 25, 2);  // Next 25 elements = 2
    fill_segment<<<1, 32>>>(d_data, 50, 25, 3);  // Next 25 elements = 3
    fill_segment<<<1, 32>>>(d_data, 75, 25, 4);  // Last 25 elements = 4
    
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Segment 0-24:  value = %d\n", h_data[0]);
    printf("Segment 25-49: value = %d\n", h_data[25]);
    printf("Segment 50-74: value = %d\n", h_data[50]);
    printf("Segment 75-99: value = %d\n", h_data[75]);
    
    cudaFree(d_data);
    delete[] h_data;
    
    printf("\n=== Key Formulas ===\n");
    printf("Global index:  idx = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("Total stride:  stride = blockDim.x * gridDim.x\n");
    printf("Num blocks:    (N + blockSize - 1) / blockSize  (ceiling division)\n");
    
    return 0;
}
