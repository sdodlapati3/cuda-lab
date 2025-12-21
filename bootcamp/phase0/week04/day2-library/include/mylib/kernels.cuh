#pragma once
/**
 * kernels.cuh - Internal kernel declarations
 * 
 * Not part of public API. Used internally by the library.
 */

namespace mylib {
namespace kernels {

__global__ void add_kernel(const float* a, const float* b, float* c, int n);
__global__ void scale_kernel(const float* a, float* b, float alpha, int n);
__global__ void dot_kernel(const float* a, const float* b, float* partial, int n);

}  // namespace kernels
}  // namespace mylib
