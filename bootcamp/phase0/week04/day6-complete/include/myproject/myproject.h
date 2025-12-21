#pragma once
/**
 * myproject.h - Main library interface
 * 
 * This is the public API for MyProject.
 */

#include <cstddef>

namespace myproject {

// Library version
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

/**
 * Initialize the library.
 * @param device_id CUDA device to use
 * @return true on success
 */
bool initialize(int device_id = 0);

/**
 * Cleanup library resources.
 */
void cleanup();

/**
 * Get device information string.
 */
const char* get_device_info();

/**
 * Vector addition: c = a + b
 * @param c Output array
 * @param a First input array
 * @param b Second input array
 * @param n Number of elements
 */
void vector_add(float* c, const float* a, const float* b, size_t n);

/**
 * Vector scale: out = in * scalar
 */
void vector_scale(float* out, const float* in, float scalar, size_t n);

/**
 * Reduce sum
 * @return Sum of all elements
 */
float reduce_sum(const float* data, size_t n);

/**
 * Matrix multiply: C = A * B
 * @param C Output matrix (M x N)
 * @param A Input matrix (M x K)
 * @param B Input matrix (K x N)
 */
void matmul(float* C, const float* A, const float* B, 
            int M, int N, int K);

}  // namespace myproject
