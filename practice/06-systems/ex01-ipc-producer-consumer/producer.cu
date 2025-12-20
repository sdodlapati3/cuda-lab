/*
 * IPC Producer - Starter Code
 * 
 * TODO: Implement the producer side of IPC memory sharing
 * 
 * Steps:
 * 1. Allocate GPU memory
 * 2. Fill with a known pattern
 * 3. Get IPC handle
 * 4. Write handle to file
 * 5. Wait for consumer
 * 6. Cleanup
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define HANDLE_FILE "ipc_handle.bin"
#define DONE_FILE "consumer_done"

// TODO: Implement check macro
#define CHECK_CUDA(call) /* Your code here */

// TODO: Implement kernel to fill data
__global__ void fillData(int* data, int n) {
    // Fill with pattern: data[i] = i * 2 + 1
}

int main() {
    printf("=== IPC Producer ===\n\n");
    
    // TODO: Allocate GPU memory
    int* d_data = nullptr;
    
    // TODO: Fill with pattern using kernel
    
    // TODO: Get IPC handle
    cudaIpcMemHandle_t handle;
    
    // TODO: Write handle to file
    
    // TODO: Wait for consumer (check for DONE_FILE)
    printf("Producer: Waiting for consumer...\n");
    
    // TODO: Cleanup
    
    return 0;
}
