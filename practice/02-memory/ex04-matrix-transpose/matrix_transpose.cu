#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define TILE_DIM 32
#define BLOCK_ROWS 8

// =============================================================================
// TODO 1: Naive transpose (no shared memory)
// Each thread directly reads from input and writes to transposed output
// =============================================================================
__global__ void transpose_naive(float* out, const float* in, int width, int height) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // TODO: Each thread transposes multiple elements (TILE_DIM/BLOCK_ROWS per thread)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            // TODO: Write transposed element
        }
    }
}

// =============================================================================
// TODO 2: Tiled transpose with shared memory (has bank conflicts)
// =============================================================================
__global__ void transpose_tiled(float* out, const float* in, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // TODO: Load tile to shared memory (coalesced)
    // TODO: __syncthreads()
    // TODO: Write transposed (from shared memory)
}

// =============================================================================
// TODO 3: Tiled transpose without bank conflicts
// =============================================================================
__global__ void transpose_optimized(float* out, const float* in, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 padding
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // TODO: Same as tiled but with padding
}

int main() {
    const int WIDTH = 4096;
    const int HEIGHT = 4096;
    const int iterations = 100;
    
    printf("Matrix Transpose Optimization\n");
    printf("==============================\n");
    printf("Matrix size: %dx%d\n\n", WIDTH, HEIGHT);
    
    size_t bytes = WIDTH * HEIGHT * sizeof(float);
    
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    
    // Initialize
    float* h_in = (float*)malloc(bytes);
    for (int i = 0; i < WIDTH * HEIGHT; i++) h_in[i] = (float)i;
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid(WIDTH / TILE_DIM, HEIGHT / TILE_DIM);
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Benchmark each version
    const char* names[] = {"Naive", "Tiled", "Optimized"};
    void (*kernels[])(float*, const float*, int, int) = {
        transpose_naive, transpose_tiled, transpose_optimized
    };
    
    for (int k = 0; k < 3; k++) {
        kernels[k]<<<grid, block>>>(d_out, d_in, WIDTH, HEIGHT);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iterations; i++) {
            kernels[k]<<<grid, block>>>(d_out, d_in, WIDTH, HEIGHT);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        ms /= iterations;
        float bw = 2 * bytes / (ms * 1e6);
        
        printf("%s: %.3f ms, %.1f GB/s\n", names[k], ms, bw);
    }
    
    free(h_in);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    
    return 0;
}
