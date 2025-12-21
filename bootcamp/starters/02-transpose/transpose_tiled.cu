/**
 * Tiled Matrix Transpose: The Shared Memory Pattern
 * 
 * This kernel demonstrates THE critical shared memory pattern:
 * - Load a tile from global memory (coalesced)
 * - Process in shared memory
 * - Store to global memory (coalesced, different layout)
 * 
 * Master this pattern → Apply to: GEMM, convolution, any algorithm
 * that needs to change data layout.
 * 
 * Progression:
 *   V1: Naive (uncoalesced writes)           → ~15% of peak bandwidth
 *   V2: Tiled (coalesced, but bank conflicts) → ~50% of peak bandwidth
 *   V3: Tiled + padding (no bank conflicts)   → ~85% of peak bandwidth
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define TILE_DIM 32
#define BLOCK_ROWS 8

// ============================================================================
// V1: Naive Transpose
// 
// Problem: Writes are NOT coalesced
// - Threads in a warp write to column (stride = width)
// - Memory transactions are scattered → terrible bandwidth
// ============================================================================
__global__ void transpose_naive(const float* __restrict__ input,
                                float* __restrict__ output,
                                int width, int height) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    if (x < width && y < height) {
        // Read: input[y][x] - coalesced (threads read consecutive x)
        // Write: output[x][y] - NOT coalesced! (threads write to different rows)
        output[x * height + y] = input[y * width + x];
    }
}

// ============================================================================
// V2: Tiled Transpose (Coalesced, but with Bank Conflicts)
// 
// Key insight: Use shared memory as a staging area
// 1. Load tile from input (coalesced reads)
// 2. __syncthreads()
// 3. Store transposed tile to output (coalesced writes)
// 
// BUT: We still have bank conflicts in shared memory!
// ============================================================================
__global__ void transpose_tiled(const float* __restrict__ input,
                                float* __restrict__ output,
                                int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile (each thread loads TILE_DIM/BLOCK_ROWS elements)
    // This is coalesced: threads in a warp access consecutive x values
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
        }
    }
    
    __syncthreads();
    
    // Calculate output position (swapped block indices for transposition)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Store transposed tile
    // Reading tile[threadIdx.x][threadIdx.y + j] causes bank conflicts!
    // 32 threads reading column = 32 accesses to same bank
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            output[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ============================================================================
// V3: Tiled Transpose with Padding (No Bank Conflicts!)
// 
// The trick: Add 1 to shared memory width
// - tile[32][32] → 32 banks, column access = conflict
// - tile[32][33] → column access spreads across different banks!
// 
// Visual:
//   Without padding:     With padding:
//   Bank 0  1  2  3      Bank 0  1  2  3  0  1  2  3
//   [0,0] [0,1] [0,2]    [0,0] [0,1] [0,2] ... [0,32(pad)]
//   [1,0] [1,1] [1,2]    [1,0] [1,1] [1,2] ... [1,32(pad)]
//     ↓     ↓     ↓        ↓     ↓     ↓
//   Column access:       Column access:
//   All same bank!       Different banks!
// ============================================================================
__global__ void transpose_tiled_padded(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int width, int height) {
    // The magic: TILE_DIM + 1 padding eliminates bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 padding!
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile (coalesced)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
        }
    }
    
    __syncthreads();
    
    // Output position (transposed)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Store transposed tile (now conflict-free!)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            output[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ============================================================================
// Verification and Benchmarking
// ============================================================================
bool verify_transpose(const float* input, const float* output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float expected = input[y * width + x];
            float actual = output[x * height + y];
            if (fabsf(expected - actual) > 1e-5f) {
                printf("Mismatch at (%d, %d): expected %f, got %f\n", 
                       x, y, expected, actual);
                return false;
            }
        }
    }
    return true;
}

typedef void (*TransposeKernel)(const float*, float*, int, int);

void benchmark_transpose(const char* name, TransposeKernel kernel,
                         const float* d_input, float* d_output,
                         int width, int height, float peak_bw,
                         int warmup, int timed) {
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM, 
              (height + TILE_DIM - 1) / TILE_DIM);
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel<<<grid, block>>>(d_input, d_output, width, height);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed runs
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < timed; i++) {
        kernel<<<grid, block>>>(d_input, d_output, width, height);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / timed;
    
    // Bandwidth: read + write = 2 * matrix size
    size_t bytes = 2 * (size_t)width * height * sizeof(float);
    double gb_per_s = (bytes / 1e9) / (avg_ms / 1e3);
    double efficiency = 100.0 * gb_per_s / peak_bw;
    
    printf("%-30s: %8.2f μs | %7.1f GB/s | %5.1f%% peak\n",
           name, avg_ms * 1000, gb_per_s, efficiency);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char** argv) {
    int width = (argc > 1) ? atoi(argv[1]) : 4096;
    int height = (argc > 2) ? atoi(argv[2]) : 4096;
    
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║         TILED TRANSPOSE BENCHMARK                              ║\n");
    printf("╠════════════════════════════════════════════════════════════════╣\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    float peak_bw = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
    
    printf("║ Device: %-54s ║\n", prop.name);
    printf("║ Peak Bandwidth: %7.1f GB/s                                   ║\n", peak_bw);
    printf("║ Matrix: %d × %d (%d MB)                                  ║\n",
           width, height, (int)(width * height * sizeof(float) / (1024 * 1024)));
    printf("╠════════════════════════════════════════════════════════════════╣\n");
    
    // Allocate
    size_t size = (size_t)width * height * sizeof(float);
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    
    // Initialize
    for (int i = 0; i < width * height; i++) {
        h_input[i] = (float)(rand() % 1000) / 1000.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    int warmup = 5;
    int timed = 20;
    
    // Benchmark
    benchmark_transpose("V1: Naive (uncoalesced)", transpose_naive,
                        d_input, d_output, width, height, peak_bw, warmup, timed);
    
    benchmark_transpose("V2: Tiled (bank conflicts)", transpose_tiled,
                        d_input, d_output, width, height, peak_bw, warmup, timed);
    
    benchmark_transpose("V3: Tiled + padding", transpose_tiled_padded,
                        d_input, d_output, width, height, peak_bw, warmup, timed);
    
    // Verify V3
    CUDA_CHECK(cudaMemset(d_output, 0, size));
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM, 
              (height + TILE_DIM - 1) / TILE_DIM);
    transpose_tiled_padded<<<grid, block>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    bool correct = verify_transpose(h_input, h_output, width, height);
    
    printf("╠════════════════════════════════════════════════════════════════╣\n");
    printf("║ Verification: %s                                            ║\n",
           correct ? "PASSED ✓" : "FAILED ✗");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    // Cleanup
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return 0;
}

/*
 * YOUR TURN: Exercises
 * 
 * 1. Profile V2 and V3 with Nsight Compute
 *    - Look at "Shared Memory Bank Conflicts" metric
 *    - Quantify the difference
 * 
 * 2. What happens with TILE_DIM = 16? Does padding still help?
 *    - Remember: 32 banks, 4 bytes per bank
 * 
 * 3. Implement transpose with vectorized loads (float4)
 *    - Each thread loads 4 elements at once
 *    - How does this affect shared memory layout?
 * 
 * 4. Handle non-square matrices correctly
 *    - Current code assumes width = height for simplicity
 * 
 * 5. Implement "swizzled" shared memory access
 *    - Alternative to padding that wastes no memory
 *    - XOR the row and column indices
 */
