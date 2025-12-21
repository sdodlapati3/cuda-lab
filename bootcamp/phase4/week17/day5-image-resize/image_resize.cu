/*
 * Day 5: Image Resizing
 * 
 * Bilinear and bicubic interpolation for image scaling.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Nearest neighbor interpolation
__global__ void resize_nearest(
    const float* __restrict__ input,
    float* __restrict__ output,
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight)
{
    int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dstX >= dstWidth || dstY >= dstHeight) return;
    
    float scaleX = (float)srcWidth / dstWidth;
    float scaleY = (float)srcHeight / dstHeight;
    
    int srcX = (int)(dstX * scaleX + 0.5f);
    int srcY = (int)(dstY * scaleY + 0.5f);
    
    srcX = min(max(srcX, 0), srcWidth - 1);
    srcY = min(max(srcY, 0), srcHeight - 1);
    
    output[dstY * dstWidth + dstX] = input[srcY * srcWidth + srcX];
}

// Bilinear interpolation
__global__ void resize_bilinear(
    const float* __restrict__ input,
    float* __restrict__ output,
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight)
{
    int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dstX >= dstWidth || dstY >= dstHeight) return;
    
    float scaleX = (float)srcWidth / dstWidth;
    float scaleY = (float)srcHeight / dstHeight;
    
    // Source coordinates (floating point)
    float srcXf = dstX * scaleX;
    float srcYf = dstY * scaleY;
    
    // Integer parts
    int x0 = (int)srcXf;
    int y0 = (int)srcYf;
    int x1 = min(x0 + 1, srcWidth - 1);
    int y1 = min(y0 + 1, srcHeight - 1);
    
    // Fractional parts
    float fx = srcXf - x0;
    float fy = srcYf - y0;
    
    // Bilinear interpolation
    float p00 = input[y0 * srcWidth + x0];
    float p10 = input[y0 * srcWidth + x1];
    float p01 = input[y1 * srcWidth + x0];
    float p11 = input[y1 * srcWidth + x1];
    
    float top = p00 * (1.0f - fx) + p10 * fx;
    float bottom = p01 * (1.0f - fx) + p11 * fx;
    
    output[dstY * dstWidth + dstX] = top * (1.0f - fy) + bottom * fy;
}

// Cubic interpolation helper
__device__ float cubicWeight(float t) {
    // Catmull-Rom spline
    float tt = t * t;
    float ttt = tt * t;
    
    if (t < 0) t = -t;
    
    if (t < 1.0f) {
        return 0.5f * (2.0f - 5.0f * tt + 3.0f * ttt);
    } else if (t < 2.0f) {
        return 0.5f * (4.0f - 8.0f * t + 5.0f * tt - ttt);
    }
    return 0.0f;
}

// Bicubic interpolation
__global__ void resize_bicubic(
    const float* __restrict__ input,
    float* __restrict__ output,
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight)
{
    int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dstX >= dstWidth || dstY >= dstHeight) return;
    
    float scaleX = (float)srcWidth / dstWidth;
    float scaleY = (float)srcHeight / dstHeight;
    
    float srcXf = dstX * scaleX;
    float srcYf = dstY * scaleY;
    
    int x0 = (int)srcXf;
    int y0 = (int)srcYf;
    
    float fx = srcXf - x0;
    float fy = srcYf - y0;
    
    float result = 0.0f;
    float weightSum = 0.0f;
    
    // 4x4 neighborhood
    for (int j = -1; j <= 2; j++) {
        for (int i = -1; i <= 2; i++) {
            int sx = max(0, min(srcWidth - 1, x0 + i));
            int sy = max(0, min(srcHeight - 1, y0 + j));
            
            float wx = cubicWeight(fx - i);
            float wy = cubicWeight(fy - j);
            float w = wx * wy;
            
            result += input[sy * srcWidth + sx] * w;
            weightSum += w;
        }
    }
    
    output[dstY * dstWidth + dstX] = result / weightSum;
}

// Texture-based bilinear (using CUDA textures)
cudaTextureObject_t createTexture(float* d_data, int width, int height) {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    
    cudaArray* cuArray;
    CHECK_CUDA(cudaMallocArray(&cuArray, &channelDesc, width, height));
    CHECK_CUDA(cudaMemcpy2DToArray(cuArray, 0, 0, d_data, width * sizeof(float),
                                    width * sizeof(float), height, cudaMemcpyDeviceToDevice));
    
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;
    
    cudaTextureObject_t tex;
    CHECK_CUDA(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
    
    return tex;
}

__global__ void resize_texture(
    cudaTextureObject_t tex,
    float* __restrict__ output,
    int dstWidth, int dstHeight)
{
    int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dstX >= dstWidth || dstY >= dstHeight) return;
    
    // Normalized coordinates with 0.5 offset for texel centers
    float u = (dstX + 0.5f) / dstWidth;
    float v = (dstY + 0.5f) / dstHeight;
    
    output[dstY * dstWidth + dstX] = tex2D<float>(tex, u, v);
}

int main() {
    printf("=== Day 5: Image Resizing ===\n\n");
    
    const int srcWidth = 1920;
    const int srcHeight = 1080;
    const int dstWidth = 3840;   // 2x upscale
    const int dstHeight = 2160;
    const int iterations = 100;
    
    printf("Source: %d x %d\n", srcWidth, srcHeight);
    printf("Destination: %d x %d (%.1fx scale)\n\n", dstWidth, dstHeight, 
           (float)dstWidth / srcWidth);
    
    // Host memory
    float* h_input = new float[srcWidth * srcHeight];
    float* h_output = new float[dstWidth * dstHeight];
    
    // Test pattern: gradient
    for (int y = 0; y < srcHeight; y++) {
        for (int x = 0; x < srcWidth; x++) {
            h_input[y * srcWidth + x] = (float)x / srcWidth;
        }
    }
    
    // Device memory
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, srcWidth * srcHeight * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, dstWidth * dstHeight * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, srcWidth * srcHeight * sizeof(float), 
                          cudaMemcpyHostToDevice));
    
    dim3 block(16, 16);
    dim3 grid((dstWidth + 15) / 16, (dstHeight + 15) / 16);
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    printf("--- Performance Comparison ---\n");
    printf("%-20s %12s %12s\n", "Method", "Time (ms)", "Mpix/s");
    printf("%-20s %12s %12s\n", "------", "---------", "------");
    
    // Nearest neighbor
    resize_nearest<<<grid, block>>>(d_input, d_output, srcWidth, srcHeight, dstWidth, dstHeight);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        resize_nearest<<<grid, block>>>(d_input, d_output, srcWidth, srcHeight, dstWidth, dstHeight);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float msNearest;
    CHECK_CUDA(cudaEventElapsedTime(&msNearest, start, stop));
    msNearest /= iterations;
    float mpixNearest = (dstWidth * dstHeight / 1e6f) / (msNearest / 1000.0f);
    printf("%-20s %12.3f %12.1f\n", "Nearest Neighbor", msNearest, mpixNearest);
    
    // Bilinear
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        resize_bilinear<<<grid, block>>>(d_input, d_output, srcWidth, srcHeight, dstWidth, dstHeight);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float msBilinear;
    CHECK_CUDA(cudaEventElapsedTime(&msBilinear, start, stop));
    msBilinear /= iterations;
    float mpixBilinear = (dstWidth * dstHeight / 1e6f) / (msBilinear / 1000.0f);
    printf("%-20s %12.3f %12.1f\n", "Bilinear", msBilinear, mpixBilinear);
    
    // Bicubic
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        resize_bicubic<<<grid, block>>>(d_input, d_output, srcWidth, srcHeight, dstWidth, dstHeight);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float msBicubic;
    CHECK_CUDA(cudaEventElapsedTime(&msBicubic, start, stop));
    msBicubic /= iterations;
    float mpixBicubic = (dstWidth * dstHeight / 1e6f) / (msBicubic / 1000.0f);
    printf("%-20s %12.3f %12.1f\n", "Bicubic", msBicubic, mpixBicubic);
    
    // Texture-based bilinear
    cudaTextureObject_t tex = createTexture(d_input, srcWidth, srcHeight);
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        resize_texture<<<grid, block>>>(tex, d_output, dstWidth, dstHeight);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float msTexture;
    CHECK_CUDA(cudaEventElapsedTime(&msTexture, start, stop));
    msTexture /= iterations;
    float mpixTexture = (dstWidth * dstHeight / 1e6f) / (msTexture / 1000.0f);
    printf("%-20s %12.3f %12.1f\n", "Texture Bilinear", msTexture, mpixTexture);
    
    printf("\n--- Key Insights ---\n");
    printf("- Texture memory: Hardware interpolation is faster\n");
    printf("- Bicubic: 16 samples vs 4 for bilinear\n");
    printf("- Quality vs speed tradeoff\n");
    
    CHECK_CUDA(cudaDestroyTextureObject(tex));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    
    delete[] h_input;
    delete[] h_output;
    
    printf("\n=== Day 5 Complete ===\n");
    return 0;
}
