/**
 * grid_stride_demo.cu - Grid-stride loop pattern
 * 
 * Learning objectives:
 * - Understand why grid-stride loops are preferred
 * - Write kernels that handle any data size
 * - See the memory access pattern
 */

#include <cuda_runtime.h>
#include <cstdio>

// Naive approach: one thread per element (requires launching N threads)
__global__ void naive_increment(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1;
    }
}

// Grid-stride loop: each thread handles multiple elements
__global__ void grid_stride_increment(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        data[i] += 1;
    }
}

// Debug version that tracks which thread processes which element
__global__ void grid_stride_debug(int* data, int* thread_map, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int global_id = idx;  // This thread's global ID
    
    for (int i = idx; i < n; i += stride) {
        data[i] += 1;
        thread_map[i] = global_id;  // Record which thread processed this
    }
}

void verify(int* h_data, int n, int expected) {
    bool pass = true;
    for (int i = 0; i < n; i++) {
        if (h_data[i] != expected) {
            printf("FAIL at index %d: got %d, expected %d\n", 
                   i, h_data[i], expected);
            pass = false;
            break;
        }
    }
    if (pass) printf("PASS: All %d elements equal %d\n", n, expected);
}

int main() {
    printf("=== Grid-Stride Loop Pattern ===\n\n");
    
    const int N = 100;  // Small for demonstration
    int* d_data;
    int* h_data = new int[N];
    
    cudaMalloc(&d_data, N * sizeof(int));
    
    // Demo 1: Naive approach
    printf("=== Demo 1: Naive (1 thread per element) ===\n");
    
    cudaMemset(d_data, 0, N * sizeof(int));
    
    int block_size = 32;
    int num_blocks = (N + block_size - 1) / block_size;
    
    printf("N = %d\n", N);
    printf("Blocks = %d, Threads/block = %d\n", num_blocks, block_size);
    printf("Total threads = %d\n\n", num_blocks * block_size);
    
    naive_increment<<<num_blocks, block_size>>>(d_data, N);
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    verify(h_data, N, 1);
    
    // Demo 2: Grid-stride with FEWER threads than elements
    printf("\n=== Demo 2: Grid-Stride (fewer threads than elements) ===\n");
    
    cudaMemset(d_data, 0, N * sizeof(int));
    
    int few_blocks = 2;  // Only 2 blocks × 32 threads = 64 threads
    
    printf("N = %d\n", N);
    printf("Blocks = %d, Threads/block = %d\n", few_blocks, block_size);
    printf("Total threads = %d (< N!)\n", few_blocks * block_size);
    printf("Elements per thread = %.1f\n\n", (float)N / (few_blocks * block_size));
    
    grid_stride_increment<<<few_blocks, block_size>>>(d_data, N);
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    verify(h_data, N, 1);
    
    // Demo 3: See which thread processes which element
    printf("\n=== Demo 3: Thread Assignment Visualization ===\n");
    
    int* d_thread_map;
    int* h_thread_map = new int[N];
    
    cudaMalloc(&d_thread_map, N * sizeof(int));
    cudaMemset(d_data, 0, N * sizeof(int));
    cudaMemset(d_thread_map, -1, N * sizeof(int));
    
    // Use 8 threads total (1 block × 8 threads)
    printf("Using 8 threads for %d elements:\n", N);
    grid_stride_debug<<<1, 8>>>(d_data, d_thread_map, N);
    cudaMemcpy(h_thread_map, d_thread_map, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Element -> Thread mapping (first 32):\n  ");
    for (int i = 0; i < 32 && i < N; i++) {
        printf("%d", h_thread_map[i]);
    }
    printf("\n");
    printf("Notice: Thread 0 handles [0, 8, 16, 24, ...]\n");
    printf("        Thread 1 handles [1, 9, 17, 25, ...]\n");
    printf("This is coalesced! Adjacent threads access adjacent memory.\n");
    
    cudaFree(d_thread_map);
    delete[] h_thread_map;
    
    cudaFree(d_data);
    delete[] h_data;
    
    printf("\n=== Grid-Stride Pattern ===\n");
    printf("```cuda\n");
    printf("int idx = blockIdx.x * blockDim.x + threadIdx.x;\n");
    printf("int stride = blockDim.x * gridDim.x;\n");
    printf("for (int i = idx; i < n; i += stride) {\n");
    printf("    data[i] = process(data[i]);\n");
    printf("}\n");
    printf("```\n");
    
    return 0;
}
