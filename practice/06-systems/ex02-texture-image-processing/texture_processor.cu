/*
 * Texture Image Processing - Starter Code
 * 
 * TODO: Implement texture-based image processing
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// TODO: Implement bilinear upscale kernel using texture
__global__ void upscaleTexture(cudaTextureObject_t tex, float* output,
                                int outWidth, int outHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < outWidth && y < outHeight) {
        // TODO: Calculate normalized coordinates
        // TODO: Sample texture
        // TODO: Write output
    }
}

// TODO: Implement global memory version for comparison
__global__ void upscaleGlobal(const float* input, float* output,
                               int inWidth, int inHeight,
                               int outWidth, int outHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < outWidth && y < outHeight) {
        // TODO: Implement bilinear interpolation manually
        // Hint: Find 4 nearest source pixels and blend
    }
}

int main() {
    printf("=== Texture Image Processing ===\n\n");
    
    const int IN_WIDTH = 256;
    const int IN_HEIGHT = 256;
    const int OUT_WIDTH = 512;
    const int OUT_HEIGHT = 512;
    
    // Create test image
    float* h_input = new float[IN_WIDTH * IN_HEIGHT];
    for (int y = 0; y < IN_HEIGHT; y++) {
        for (int x = 0; x < IN_WIDTH; x++) {
            h_input[y * IN_WIDTH + x] = (float)(x + y);
        }
    }
    
    // TODO: Create CUDA array and texture object
    cudaArray_t cuArray;
    cudaTextureObject_t tex;
    
    // TODO: Allocate output
    float* d_output;
    
    // TODO: Run kernels and benchmark
    
    // TODO: Cleanup
    
    delete[] h_input;
    return 0;
}
