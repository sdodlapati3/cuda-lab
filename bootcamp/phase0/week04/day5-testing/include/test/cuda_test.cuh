#pragma once
/**
 * cuda_test.cuh - CUDA-specific test utilities
 */

#include <cuda_runtime.h>
#include <cstdio>

namespace test {
namespace cuda {

// Check for CUDA errors
#define CUDA_ASSERT(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::ostringstream ss; \
        ss << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__; \
        TestRegistry::instance().mark_failed(ss.str()); \
        return; \
    } \
} while(0)

// Check for kernel errors after launch
inline bool check_kernel_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
        return false;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Sync error: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

// Vector operations for testing
void vector_add(float* c, const float* a, const float* b, int n);
void vector_scale(float* out, const float* in, float scalar, int n);
float reduce_sum(const float* data, int n);

// Initialize test environment
void init();
void cleanup();

}  // namespace cuda
}  // namespace test
