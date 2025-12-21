#pragma once
/**
 * mylib.h - Public API for MyCudaLib
 * 
 * This header provides the public interface. 
 * Users only need to include this file.
 */

#include <cstddef>

namespace mylib {

/**
 * Initialize the library. Call once at startup.
 * @return true on success
 */
bool initialize();

/**
 * Cleanup library resources. Call before exit.
 */
void cleanup();

/**
 * Vector addition: c = a + b
 * @param a First input array (device or host pointer)
 * @param b Second input array
 * @param c Output array
 * @param n Number of elements
 * @param on_device true if pointers are device memory
 */
void vector_add(const float* a, const float* b, float* c, size_t n, 
                bool on_device = false);

/**
 * Vector scale: b = alpha * a
 */
void vector_scale(const float* a, float* b, float alpha, size_t n,
                  bool on_device = false);

/**
 * Dot product: result = sum(a[i] * b[i])
 */
float dot_product(const float* a, const float* b, size_t n,
                  bool on_device = false);

/**
 * Get last error message (if any)
 */
const char* get_last_error();

}  // namespace mylib
