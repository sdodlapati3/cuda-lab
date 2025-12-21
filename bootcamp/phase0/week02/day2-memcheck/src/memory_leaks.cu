/**
 * Day 2: Memory Leaks
 * 
 * Run: compute-sanitizer --tool memcheck --leak-check full ./memory_leaks
 */

#include <cstdio>
#include <cuda_runtime.h>

// ============================================================================
// Leak 1: Simple leak - forgot cudaFree
// ============================================================================
void simple_leak() {
    int* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(int));
    // BUG: No cudaFree!
}

// ============================================================================
// Leak 2: Exception path leak
// ============================================================================
void exception_leak() {
    int* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(int));
    
    // Simulate an error condition
    if (true) {  // In real code this might be an error check
        // BUG: Early return without free
        printf("  Error path taken, memory leaked!\n");
        return;
    }
    
    cudaFree(d_data);  // Never reached
}

// ============================================================================
// Leak 3: Overwritten pointer
// ============================================================================
void overwrite_leak() {
    int* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(int));
    
    // BUG: Overwrite pointer before freeing!
    cudaMalloc(&d_data, 2048 * sizeof(int));  // First allocation leaked
    
    cudaFree(d_data);  // Only frees the second allocation
}

// ============================================================================
// Fixed versions
// ============================================================================
void no_leak_raii() {
    // Using RAII pattern (see cuda_utils.cuh from Week 1)
    struct DevicePtr {
        int* ptr = nullptr;
        DevicePtr(size_t bytes) { cudaMalloc(&ptr, bytes); }
        ~DevicePtr() { if (ptr) cudaFree(ptr); }
    };
    
    DevicePtr data(1024 * sizeof(int));
    // Automatically freed when data goes out of scope
}

void no_leak_goto() {
    int* d_data1 = nullptr;
    int* d_data2 = nullptr;
    
    if (cudaMalloc(&d_data1, 1024) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_data2, 2048) != cudaSuccess) goto cleanup;
    
    // Use the data...
    
cleanup:
    if (d_data1) cudaFree(d_data1);
    if (d_data2) cudaFree(d_data2);
}

int main() {
    printf("Memory Leak Examples\n");
    printf("====================\n");
    printf("Run with: compute-sanitizer --tool memcheck --leak-check full ./memory_leaks\n\n");
    
    printf("Creating leaks for demonstration...\n\n");
    
    printf("Leak 1: Simple leak (no cudaFree)...\n");
    simple_leak();
    
    printf("Leak 2: Exception path leak...\n");
    exception_leak();
    
    printf("Leak 3: Overwritten pointer...\n");
    overwrite_leak();
    
    printf("\nNow showing fixed versions (no leaks)...\n\n");
    
    printf("Fixed 1: RAII pattern...\n");
    no_leak_raii();
    printf("  ✓ No leak\n");
    
    printf("Fixed 2: Goto cleanup pattern...\n");
    no_leak_goto();
    printf("  ✓ No leak\n");
    
    printf("\nCheck sanitizer output for leak reports!\n");
    printf("You should see 3 leaks from the buggy functions.\n");
    
    return 0;
}
