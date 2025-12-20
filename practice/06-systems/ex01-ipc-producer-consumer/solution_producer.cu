/*
 * IPC Producer - Solution
 */

#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define N 1024
#define HANDLE_FILE "ipc_handle.bin"
#define DONE_FILE "consumer_done"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

__global__ void fillData(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 2 + 1;
    }
}

int main() {
    printf("=== IPC Producer ===\n\n");
    
    // Remove any stale files
    remove(HANDLE_FILE);
    remove(DONE_FILE);
    
    // Allocate GPU memory
    int* d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(int)));
    printf("Producer: Allocated %zu bytes\n", N * sizeof(int));
    
    // Fill with pattern
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    fillData<<<numBlocks, blockSize>>>(d_data, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Producer: Filled with pattern (data[i] = i * 2 + 1)\n");
    
    // Get IPC handle
    cudaIpcMemHandle_t handle;
    CHECK_CUDA(cudaIpcGetMemHandle(&handle, d_data));
    
    // Write handle to file
    FILE* f = fopen(HANDLE_FILE, "wb");
    if (!f) {
        printf("Failed to open handle file\n");
        return 1;
    }
    fwrite(&handle, sizeof(handle), 1, f);
    fclose(f);
    printf("Producer: IPC handle written to %s\n", HANDLE_FILE);
    
    // Wait for consumer
    printf("Producer: Waiting for consumer...\n");
    while (access(DONE_FILE, F_OK) != 0) {
        usleep(100000);  // 100ms
    }
    
    // Cleanup
    printf("Producer: Consumer finished, cleaning up\n");
    remove(DONE_FILE);
    remove(HANDLE_FILE);
    CHECK_CUDA(cudaFree(d_data));
    
    printf("Producer: Done\n");
    return 0;
}
