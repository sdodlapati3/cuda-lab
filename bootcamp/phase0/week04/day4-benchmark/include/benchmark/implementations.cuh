#pragma once
/**
 * implementations.cuh - Kernel implementations to benchmark
 */

#include <cstddef>

namespace impl {

// Setup/teardown
void setup(size_t n);
void cleanup();

// Different reduction implementations
float reduce_v0_global();    // Global atomics
float reduce_v1_shared();    // Shared memory
float reduce_v2_warp();      // Warp shuffle
float reduce_v3_vector();    // Vectorized load

// Get result (for verification)
float get_result();

// Get data size
size_t get_data_size();

}  // namespace impl
