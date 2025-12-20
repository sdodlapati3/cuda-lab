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

// SOLUTION: Transpose with bank conflicts
__global__ void transpose_bank_conflicts(float* out, const float* in, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    
    __syncthreads();
    
    // Transposed coordinates
    int out_x = blockIdx.y * TILE_DIM + threadIdx.x;
    int out_y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    if (out_x < height && out_y < width) {
        // Reading column causes bank conflicts!
        out[out_y * height + out_x] = tile[threadIdx.x][threadIdx.y];
    }
}

// SOLUTION: Transpose without bank conflicts
__global__ void transpose_no_conflicts(float* out, const float* in, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 padding!
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    
    __syncthreads();
    
    int out_x = blockIdx.y * TILE_DIM + threadIdx.x;
    int out_y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    if (out_x < height && out_y < width) {
        // Padding shifts bank indices - no conflicts!
        out[out_y * height + out_x] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    const int WIDTH = 4096;
    const int HEIGHT = 4096;
    const int iterations = 100;
    
    printf("Bank Conflicts - SOLUTION\n");
    printf("=========================\n");
    printf("Matrix size: %dx%d\n\n", WIDTH, HEIGHT);
    
    size_t bytes = WIDTH * HEIGHT * sizeof(float);
    
    float *h_in, *h_out, *h_ref;
    float *d_in, *d_out;
    
    h_in = (float*)malloc(bytes);
    h_out = (float*)malloc(bytes);
    h_ref = (float*)malloc(bytes);
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_in[i] = (float)i;
    }
    // CPU reference transpose
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            h_ref[x * HEIGHT + y] = h_in[y * WIDTH + x];
        }
    }
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(WIDTH / TILE_DIM, HEIGHT / TILE_DIM);
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Benchmark with conflicts
    transpose_bank_conflicts<<<grid, block>>>(d_out, d_in, WIDTH, HEIGHT);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        transpose_bank_conflicts<<<grid, block>>>(d_out, d_in, WIDTH, HEIGHT);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float time_conflicts;
    CHECK_CUDA(cudaEventElapsedTime(&time_conflicts, start, stop));
    time_conflicts /= iterations;
    float bw_conflicts = 2 * bytes / (time_conflicts * 1e6);
    
    printf("With Bank Conflicts:\n");
    printf("  Time: %.3f ms, Bandwidth: %.1f GB/s\n", time_conflicts, bw_conflicts);
    
    // Verify
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < 1000; i++) {
        if (h_out[i] != h_ref[i]) { correct = false; break; }
    }
    printf("  Verification: %s\n\n", correct ? "PASSED" : "FAILED");
    
    // Benchmark without conflicts
    transpose_no_conflicts<<<grid, block>>>(d_out, d_in, WIDTH, HEIGHT);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        transpose_no_conflicts<<<grid, block>>>(d_out, d_in, WIDTH, HEIGHT);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float time_no_conflicts;
    CHECK_CUDA(cudaEventElapsedTime(&time_no_conflicts, start, stop));
    time_no_conflicts /= iterations;
    float bw_no_conflicts = 2 * bytes / (time_no_conflicts * 1e6);
    
    printf("Without Bank Conflicts:\n");
    printf("  Time: %.3f ms, Bandwidth: %.1f GB/s\n", time_no_conflicts, bw_no_conflicts);
    
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    correct = true;
    for (int i = 0; i < 1000; i++) {
        if (h_out[i] != h_ref[i]) { correct = false; break; }
    }
    printf("  Verification: %s\n\n", correct ? "PASSED" : "FAILED");
    
    printf("Speedup: %.2fx\n", time_conflicts / time_no_conflicts);
    
    free(h_in); free(h_out); free(h_ref);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    
    printf("\n✓ Solution complete!\n");
    return 0;
}
