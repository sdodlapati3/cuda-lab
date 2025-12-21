/**
 * transpose.cu - Matrix transpose with bank conflict optimization
 * 
 * Learning objectives:
 * - Implement tiled matrix transpose
 * - Apply shared memory padding
 * - Achieve high memory bandwidth
 */

#include <cuda_runtime.h>
#include <cstdio>

#define TILE_DIM 32
#define BLOCK_ROWS 8

// Naive transpose (non-coalesced writes)
__global__ void transpose_naive(const float* input, float* output, int width, int height) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            // Read coalesced, write strided (BAD)
            output[x * height + (y + j)] = input[(y + j) * width + x];
        }
    }
}

// Tiled transpose with shared memory (has bank conflicts)
__global__ void transpose_shared(const float* input, float* output, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM];  // Bank conflicts!
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile (coalesced read)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
        }
    }
    __syncthreads();
    
    // Write transposed tile (coalesced write, but bank conflicts on read!)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            output[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Tiled transpose with padding (NO bank conflicts)
__global__ void transpose_padded(const float* input, float* output, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 padding eliminates conflicts!
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile (coalesced read)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
        }
    }
    __syncthreads();
    
    // Write transposed tile (coalesced write, NO bank conflicts on read!)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            output[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

bool verify_transpose(const float* input, const float* output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (input[y * width + x] != output[x * height + y]) {
                printf("Mismatch at (%d, %d): %.1f != %.1f\n", 
                       x, y, input[y * width + x], output[x * height + y]);
                return false;
            }
        }
    }
    return true;
}

int main() {
    printf("=== Matrix Transpose with Bank Conflict Optimization ===\n\n");
    
    const int WIDTH = 2048;
    const int HEIGHT = 2048;
    const size_t bytes = WIDTH * HEIGHT * sizeof(float);
    const int TRIALS = 100;
    
    printf("Matrix: %d × %d (%.1f MB)\n\n", WIDTH, HEIGHT, bytes / 1e6);
    
    // Allocate
    float* h_input = new float[WIDTH * HEIGHT];
    float* h_output = new float[WIDTH * HEIGHT];
    
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_input[i] = (float)i;
    }
    
    float* d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((WIDTH + TILE_DIM - 1) / TILE_DIM,
              (HEIGHT + TILE_DIM - 1) / TILE_DIM);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    printf("=== Benchmark Results ===\n");
    printf("%-25s %12s %15s\n", "Implementation", "Time (ms)", "Bandwidth (GB/s)");
    printf("-------------------------------------------------------------\n");
    
    // Warmup
    transpose_naive<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    
    // Naive
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        transpose_naive<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    ms /= TRIALS;
    printf("%-25s %12.3f %15.1f\n", "Naive (strided write)", ms, 2 * bytes / ms / 1e6);
    
    // Verify
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    printf("  Verification: %s\n", verify_transpose(h_input, h_output, WIDTH, HEIGHT) ? "PASS" : "FAIL");
    
    // Shared (with conflicts)
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        transpose_shared<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    ms /= TRIALS;
    printf("%-25s %12.3f %15.1f\n", "Shared (bank conflicts)", ms, 2 * bytes / ms / 1e6);
    
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    printf("  Verification: %s\n", verify_transpose(h_input, h_output, WIDTH, HEIGHT) ? "PASS" : "FAIL");
    
    // Padded (no conflicts)
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        transpose_padded<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    ms /= TRIALS;
    printf("%-25s %12.3f %15.1f\n", "Padded (no conflicts)", ms, 2 * bytes / ms / 1e6);
    
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    printf("  Verification: %s\n", verify_transpose(h_input, h_output, WIDTH, HEIGHT) ? "PASS" : "FAIL");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
    
    printf("\n=== Transpose Optimization Summary ===\n");
    printf("1. Naive: Strided writes kill bandwidth\n");
    printf("2. Shared: Tiling helps, but bank conflicts remain\n");
    printf("3. Padded: __shared__[32][33] eliminates conflicts\n");
    printf("\n");
    printf("Profile bank conflicts with:\n");
    printf("  ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./build/transpose\n");
    
    return 0;
}
