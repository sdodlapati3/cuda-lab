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

// SOLUTION: Naive transpose
__global__ void transpose_naive(float* out, const float* in, int width, int height) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            out[x * height + (y + j)] = in[(y + j) * width + x];
        }
    }
}

// SOLUTION: Tiled transpose (has bank conflicts)
__global__ void transpose_tiled(float* out, const float* in, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
        }
    }
    
    __syncthreads();
    
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// SOLUTION: Optimized transpose (no bank conflicts)
__global__ void transpose_optimized(float* out, const float* in, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
        }
    }
    
    __syncthreads();
    
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

int main() {
    const int WIDTH = 4096;
    const int HEIGHT = 4096;
    const int iterations = 100;
    
    printf("Matrix Transpose - SOLUTION\n");
    printf("===========================\n");
    printf("Matrix size: %dx%d\n\n", WIDTH, HEIGHT);
    
    size_t bytes = WIDTH * HEIGHT * sizeof(float);
    
    float *h_in, *h_out, *h_ref;
    float *d_in, *d_out;
    
    h_in = (float*)malloc(bytes);
    h_out = (float*)malloc(bytes);
    h_ref = (float*)malloc(bytes);
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    
    for (int i = 0; i < WIDTH * HEIGHT; i++) h_in[i] = (float)i;
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            h_ref[x * HEIGHT + y] = h_in[y * WIDTH + x];
        }
    }
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid(WIDTH / TILE_DIM, HEIGHT / TILE_DIM);
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
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
        
        CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        bool correct = true;
        for (int i = 0; i < WIDTH * HEIGHT; i++) {
            if (h_out[i] != h_ref[i]) { correct = false; break; }
        }
        
        printf("%s: %.3f ms, %.1f GB/s [%s]\n", names[k], ms, bw, correct ? "PASS" : "FAIL");
    }
    
    free(h_in); free(h_out); free(h_ref);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    
    printf("\n✓ Solution complete!\n");
    return 0;
}
