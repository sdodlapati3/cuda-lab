/**
 * Day 6: Debugged Application
 * 
 * The fixed version of buggy_app.cu with all bugs corrected.
 */

#include <cstdio>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define N 1000
#define BLOCK_SIZE 256

// ============================================================================
// Fixed reduction - no race conditions
// ============================================================================
__global__ void fixed_sum(int* data, int* block_sums, int n) {
    __shared__ int sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    sdata[tid] = (idx < n) ? data[idx] : 0;
    __syncthreads();  // FIX: Added sync
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // FIX: Added sync in loop
    }
    
    // FIX: Each block writes to its own location
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

// Final reduction on CPU (simple, correct)
int reduce_block_sums(int* block_sums, int num_blocks) {
    int total = 0;
    for (int i = 0; i < num_blocks; i++) {
        total += block_sums[i];
    }
    return total;
}

// ============================================================================
// Fixed transform - bounds checked
// ============================================================================
__global__ void fixed_transform(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // FIX: Bounds check
    if (idx >= n) return;
    
    output[idx] = input[idx] * 2.0f;
    
    // FIX: Check bounds for neighbor access
    if (idx + 1 < n) {
        output[idx] += input[idx + 1];
    }
}

// ============================================================================
// RAII wrapper for automatic cleanup
// ============================================================================
class CudaBuffer {
public:
    CudaBuffer(size_t size) : ptr_(nullptr), size_(size) {
        CUDA_CHECK(cudaMalloc(&ptr_, size));
    }
    ~CudaBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    void* get() { return ptr_; }
    template<typename T> T* as() { return static_cast<T*>(ptr_); }
    
    // No copy
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    
private:
    void* ptr_;
    size_t size_;
};

// ============================================================================
// Main - properly debugged
// ============================================================================
int main() {
    printf("Debugged Application\n");
    printf("====================\n\n");
    
    // Setup with RAII for automatic cleanup
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Host memory
    int* h_data = (int*)malloc(N * sizeof(int));
    int* h_block_sums = (int*)malloc(num_blocks * sizeof(int));
    
    for (int i = 0; i < N; i++) h_data[i] = 1;
    
    // Device memory with RAII
    CudaBuffer d_data(N * sizeof(int));
    CudaBuffer d_block_sums(num_blocks * sizeof(int));
    
    CUDA_CHECK(cudaMemcpy(d_data.get(), h_data, N * sizeof(int), 
               cudaMemcpyHostToDevice));
    
    // Run fixed sum
    printf("Running fixed_sum...\n");
    fixed_sum<<<num_blocks, BLOCK_SIZE>>>(d_data.as<int>(), 
                                          d_block_sums.as<int>(), N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_block_sums, d_block_sums.get(), 
               num_blocks * sizeof(int), cudaMemcpyDeviceToHost));
    
    int result = reduce_block_sums(h_block_sums, num_blocks);
    printf("Sum result: %d (expected: %d) %s\n", 
           result, N, result == N ? "✓" : "✗");
    
    // Run fixed transform
    printf("\nRunning fixed_transform...\n");
    float* h_float = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_float[i] = 1.0f;
    
    CudaBuffer d_input(N * sizeof(float));
    CudaBuffer d_output(N * sizeof(float));
    
    CUDA_CHECK(cudaMemcpy(d_input.get(), h_float, N * sizeof(float), 
               cudaMemcpyHostToDevice));
    
    fixed_transform<<<num_blocks, BLOCK_SIZE>>>(d_input.as<float>(), 
                                                d_output.as<float>(), N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_float, d_output.get(), N * sizeof(float), 
               cudaMemcpyDeviceToHost));
    
    // Verify
    bool correct = true;
    for (int i = 0; i < N; i++) {
        float expected = 2.0f + (i + 1 < N ? 1.0f : 0.0f);
        if (h_float[i] != expected) {
            correct = false;
            break;
        }
    }
    printf("Transform result: %s\n", correct ? "✓" : "✗");
    
    // Free host memory
    free(h_data);
    free(h_block_sums);
    free(h_float);
    
    // Device memory is automatically freed by RAII
    
    printf("\nAll bugs fixed!\n");
    printf("Run with sanitizer to verify:\n");
    printf("  compute-sanitizer ./build/debugged_app\n");
    
    return 0;
}
