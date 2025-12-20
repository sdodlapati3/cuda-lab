/*
 * Texture Image Processing - Solution
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

// Texture-based upscale (bilinear filtering done by hardware)
__global__ void upscaleTexture(cudaTextureObject_t tex, float* output,
                                int outWidth, int outHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < outWidth && y < outHeight) {
        // Normalized coordinates with half-pixel offset
        float u = (x + 0.5f) / outWidth;
        float v = (y + 0.5f) / outHeight;
        
        // Hardware bilinear filtering!
        output[y * outWidth + x] = tex2D<float>(tex, u, v);
    }
}

// Global memory version (manual bilinear interpolation)
__global__ void upscaleGlobal(const float* input, float* output,
                               int inWidth, int inHeight,
                               int outWidth, int outHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < outWidth && y < outHeight) {
        // Map to source coordinates
        float srcX = (x + 0.5f) * inWidth / outWidth - 0.5f;
        float srcY = (y + 0.5f) * inHeight / outHeight - 0.5f;
        
        // Get integer and fractional parts
        int x0 = (int)floorf(srcX);
        int y0 = (int)floorf(srcY);
        float fx = srcX - x0;
        float fy = srcY - y0;
        
        // Clamp coordinates
        int x1 = min(x0 + 1, inWidth - 1);
        int y1 = min(y0 + 1, inHeight - 1);
        x0 = max(x0, 0);
        y0 = max(y0, 0);
        
        // Fetch 4 neighbors
        float v00 = input[y0 * inWidth + x0];
        float v10 = input[y0 * inWidth + x1];
        float v01 = input[y1 * inWidth + x0];
        float v11 = input[y1 * inWidth + x1];
        
        // Bilinear interpolation
        float v0 = v00 * (1 - fx) + v10 * fx;
        float v1 = v01 * (1 - fx) + v11 * fx;
        output[y * outWidth + x] = v0 * (1 - fy) + v1 * fy;
    }
}

int main() {
    printf("=== Texture Image Processing ===\n\n");
    
    const int IN_WIDTH = 256;
    const int IN_HEIGHT = 256;
    const int OUT_WIDTH = 512;
    const int OUT_HEIGHT = 512;
    const int ITERATIONS = 100;
    
    printf("Input: %dx%d image\n", IN_WIDTH, IN_HEIGHT);
    printf("Output: %dx%d (2x upscale)\n\n", OUT_WIDTH, OUT_HEIGHT);
    
    // Create test image
    float* h_input = new float[IN_WIDTH * IN_HEIGHT];
    for (int y = 0; y < IN_HEIGHT; y++) {
        for (int x = 0; x < IN_WIDTH; x++) {
            h_input[y * IN_WIDTH + x] = (float)(x + y);
        }
    }
    
    // Create CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray;
    CHECK_CUDA(cudaMallocArray(&cuArray, &channelDesc, IN_WIDTH, IN_HEIGHT));
    CHECK_CUDA(cudaMemcpy2DToArray(cuArray, 0, 0, h_input,
                                    IN_WIDTH * sizeof(float),
                                    IN_WIDTH * sizeof(float), IN_HEIGHT,
                                    cudaMemcpyHostToDevice));
    
    // Create texture object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.normalizedCoords = true;
    
    cudaTextureObject_t tex;
    CHECK_CUDA(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
    
    // Allocate device memory for global memory version
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, IN_WIDTH * IN_HEIGHT * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, OUT_WIDTH * OUT_HEIGHT * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, IN_WIDTH * IN_HEIGHT * sizeof(float),
                          cudaMemcpyHostToDevice));
    
    dim3 block(16, 16);
    dim3 grid((OUT_WIDTH + 15) / 16, (OUT_HEIGHT + 15) / 16);
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Warmup
    upscaleTexture<<<grid, block>>>(tex, d_output, OUT_WIDTH, OUT_HEIGHT);
    upscaleGlobal<<<grid, block>>>(d_input, d_output, IN_WIDTH, IN_HEIGHT,
                                    OUT_WIDTH, OUT_HEIGHT);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark texture version
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; i++) {
        upscaleTexture<<<grid, block>>>(tex, d_output, OUT_WIDTH, OUT_HEIGHT);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float texMs;
    CHECK_CUDA(cudaEventElapsedTime(&texMs, start, stop));
    
    // Benchmark global version
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; i++) {
        upscaleGlobal<<<grid, block>>>(d_input, d_output, IN_WIDTH, IN_HEIGHT,
                                        OUT_WIDTH, OUT_HEIGHT);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float globalMs;
    CHECK_CUDA(cudaEventElapsedTime(&globalMs, start, stop));
    
    printf("Bilinear Upscale (%d iterations):\n", ITERATIONS);
    printf("  Texture: %.2f ms\n", texMs);
    printf("  Global:  %.2f ms\n", globalMs);
    printf("  Speedup: %.2fx\n", globalMs / texMs);
    
    // Cleanup
    CHECK_CUDA(cudaDestroyTextureObject(tex));
    CHECK_CUDA(cudaFreeArray(cuArray));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    delete[] h_input;
    
    return 0;
}
