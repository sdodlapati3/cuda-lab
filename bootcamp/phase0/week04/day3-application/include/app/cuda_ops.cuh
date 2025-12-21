#pragma once
/**
 * cuda_ops.cuh - CUDA operations interface
 */

#include <cstddef>

namespace app {
namespace cuda {

// Initialize CUDA device
bool init(int device_id);

// Cleanup CUDA resources
void cleanup();

// Process data on GPU
// Returns time in milliseconds
float process(float* data, size_t n, int iterations);

// Verify results
bool verify(const float* data, size_t n);

// Get device info string
const char* get_device_info();

}  // namespace cuda
}  // namespace app
