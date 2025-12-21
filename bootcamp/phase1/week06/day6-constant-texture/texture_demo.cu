/**
 * texture_demo.cu - Demonstrate texture memory usage
 * 
 * Learning objectives:
 * - Create texture objects
 * - Use hardware interpolation
 * - Understand texture memory benefits
 */

#include <cuda_runtime.h>
#include <cstdio>

// Read using texture object
__global__ void read_texture_1d(cudaTextureObject_t tex, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tex1Dfetch<float>(tex, idx);
    }
}

// Read with interpolation (floating point coordinates)
__global__ void sample_texture_1d(cudaTextureObject_t tex, float* output, 
                                   int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Sample at fractional position - hardware interpolation!
        float coord = idx * scale;
        output[idx] = tex1D<float>(tex, coord);
    }
}

// Regular global memory read (for comparison)
__global__ void read_global(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

cudaTextureObject_t createTextureObject(float* d_data, int n, bool normalized) {
    // Resource descriptor
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_data;
    resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
    resDesc.res.linear.sizeInBytes = n * sizeof(float);
    
    // Texture descriptor
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.filterMode = normalized ? cudaFilterModeLinear : cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = normalized;
    
    cudaTextureObject_t tex;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
    
    return tex;
}

int main() {
    printf("=== Texture Memory Demo ===\n\n");
    
    printf("Texture memory features:\n");
    printf("- Optimized for 2D spatial locality\n");
    printf("- Hardware interpolation (linear, bilinear)\n");
    printf("- Automatic boundary handling (clamp, wrap)\n");
    printf("- Read-only cache\n\n");
    
    const int N = 1 << 20;
    
    // Allocate and initialize
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = sinf((float)i * 0.01f);
    }
    
    float* d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create texture object
    cudaTextureObject_t tex = createTextureObject(d_input, N, false);
    
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    printf("=== Read Performance Comparison ===\n\n");
    
    // Warmup
    read_global<<<num_blocks, block_size>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Global memory
    cudaEventRecord(start);
    for (int t = 0; t < 100; t++) {
        read_global<<<num_blocks, block_size>>>(d_input, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float global_time = ms / 100;
    printf("Global memory: %.3f ms\n", global_time);
    
    // Texture memory
    cudaEventRecord(start);
    for (int t = 0; t < 100; t++) {
        read_texture_1d<<<num_blocks, block_size>>>(tex, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Texture memory: %.3f ms (%.2fx)\n", ms / 100, global_time / (ms / 100));
    
    cudaDestroyTextureObject(tex);
    
    printf("\n=== Hardware Interpolation Demo ===\n\n");
    
    // Small array for interpolation demo
    const int SMALL_N = 10;
    float small_data[SMALL_N];
    for (int i = 0; i < SMALL_N; i++) {
        small_data[i] = (float)i * 10.0f;  // [0, 10, 20, 30, ...]
    }
    
    float* d_small;
    cudaMalloc(&d_small, SMALL_N * sizeof(float));
    cudaMemcpy(d_small, small_data, SMALL_N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create texture with linear interpolation
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_small;
    resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
    resDesc.res.linear.sizeInBytes = SMALL_N * sizeof(float);
    
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;  // Linear interpolation
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = true;  // 0.0 to 1.0 coordinates
    
    cudaTextureObject_t texInterp;
    cudaCreateTextureObject(&texInterp, &resDesc, &texDesc, NULL);
    
    printf("Source data: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]\n\n");
    
    // Sample at fractional positions
    const int SAMPLES = 20;
    float* d_samples;
    cudaMalloc(&d_samples, SAMPLES * sizeof(float));
    
    // Scale to sample 2× more points than data
    sample_texture_1d<<<1, SAMPLES>>>(texInterp, d_samples, SAMPLES, 1.0f / SAMPLES);
    
    float h_samples[SAMPLES];
    cudaMemcpy(h_samples, d_samples, SAMPLES * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Interpolated samples (hardware bilinear):\n");
    for (int i = 0; i < SAMPLES; i++) {
        float coord = (float)i / SAMPLES;
        printf("  coord=%.2f → %.1f\n", coord, h_samples[i]);
    }
    
    cudaDestroyTextureObject(texInterp);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_small);
    cudaFree(d_samples);
    delete[] h_data;
    
    printf("\n=== Texture Memory Use Cases ===\n");
    printf("✓ Image processing (2D spatial locality)\n");
    printf("✓ Volume rendering (3D textures)\n");
    printf("✓ Lookup tables with interpolation\n");
    printf("✓ Data with 2D/3D access patterns\n");
    printf("✗ Random 1D access (use global memory)\n");
    printf("✗ Write operations (texture is read-only)\n");
    
    return 0;
}
