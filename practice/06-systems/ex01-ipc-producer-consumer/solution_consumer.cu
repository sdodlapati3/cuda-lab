/*
 * IPC Consumer - Solution
 */

#include <stdio.h>
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

int main() {
    printf("=== IPC Consumer ===\n\n");
    
    // Read IPC handle from file
    cudaIpcMemHandle_t handle;
    FILE* f = fopen(HANDLE_FILE, "rb");
    if (!f) {
        printf("Handle file not found. Run producer first.\n");
        return 1;
    }
    fread(&handle, sizeof(handle), 1, f);
    fclose(f);
    printf("Consumer: Read IPC handle from %s\n", HANDLE_FILE);
    
    // Open shared memory
    int* d_data = nullptr;
    CHECK_CUDA(cudaIpcOpenMemHandle((void**)&d_data, handle, 
                                     cudaIpcMemLazyEnablePeerAccess));
    printf("Consumer: Opened shared memory\n");
    
    // Copy to host
    int* h_data = new int[N];
    CHECK_CUDA(cudaMemcpy(h_data, d_data, N * sizeof(int), 
                          cudaMemcpyDeviceToHost));
    
    // Verify pattern
    printf("Consumer: Verifying data...\n");
    int errors = 0;
    for (int i = 0; i < N; i++) {
        int expected = i * 2 + 1;
        if (h_data[i] != expected) {
            errors++;
            if (errors <= 5) {
                printf("  Mismatch at %d: got %d, expected %d\n", 
                       i, h_data[i], expected);
            }
        }
    }
    
    if (errors == 0) {
        printf("Consumer: SUCCESS - All %d values correct!\n", N);
    } else {
        printf("Consumer: FAILED - %d errors found\n", errors);
    }
    
    // Close IPC handle (NOT cudaFree!)
    CHECK_CUDA(cudaIpcCloseMemHandle(d_data));
    
    // Signal producer we're done
    f = fopen(DONE_FILE, "w");
    if (f) {
        fprintf(f, "done\n");
        fclose(f);
    }
    
    delete[] h_data;
    return errors == 0 ? 0 : 1;
}
