/*
 * IPC Consumer - Starter Code
 * 
 * TODO: Implement the consumer side of IPC memory sharing
 * 
 * Steps:
 * 1. Read IPC handle from file
 * 2. Open shared memory
 * 3. Verify data correctness
 * 4. Close handle
 * 5. Signal producer we're done
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define HANDLE_FILE "ipc_handle.bin"
#define DONE_FILE "consumer_done"

// TODO: Implement check macro
#define CHECK_CUDA(call) /* Your code here */

int main() {
    printf("=== IPC Consumer ===\n\n");
    
    // TODO: Read IPC handle from file
    cudaIpcMemHandle_t handle;
    
    // TODO: Open shared memory
    int* d_data = nullptr;
    
    // TODO: Copy to host and verify
    int* h_data = new int[N];
    
    // Verify pattern: data[i] should equal i * 2 + 1
    int errors = 0;
    for (int i = 0; i < N; i++) {
        int expected = i * 2 + 1;
        if (h_data[i] != expected) {
            errors++;
            if (errors <= 5) {
                printf("Mismatch at %d: got %d, expected %d\n", 
                       i, h_data[i], expected);
            }
        }
    }
    
    if (errors == 0) {
        printf("SUCCESS - All %d values correct!\n", N);
    } else {
        printf("FAILED - %d errors found\n", errors);
    }
    
    // TODO: Close IPC handle
    
    // TODO: Signal producer we're done (create DONE_FILE)
    
    delete[] h_data;
    return errors == 0 ? 0 : 1;
}
