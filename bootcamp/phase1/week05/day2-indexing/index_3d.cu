/**
 * index_3d.cu - 3D thread indexing for volumes and tensors
 * 
 * Learning objectives:
 * - Map 3D thread grid to 3D data
 * - Handle batched operations
 * - Understand memory layout for 3D data
 */

#include <cuda_runtime.h>
#include <cstdio>

// 3D kernel for volume initialization
__global__ void init_volume(int* volume, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        // 3D to 1D index: z * (width * height) + y * width + x
        int idx = z * (width * height) + y * width + x;
        volume[idx] = z * 100 + y * 10 + x;  // Encode position
    }
}

// Process batched 2D matrices (batch, row, col)
__global__ void batched_matrix_scale(float* data, int batch_size, 
                                      int rows, int cols, float scale) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;  // One block per batch item
    
    if (col < cols && row < rows && batch < batch_size) {
        int idx = batch * (rows * cols) + row * cols + col;
        data[idx] *= scale;
    }
}

// 3D stencil operation (simplified - center value only)
__global__ void stencil_3d(const int* input, int* output, 
                           int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Skip boundaries
    if (x > 0 && x < width - 1 &&
        y > 0 && y < height - 1 &&
        z > 0 && z < depth - 1) {
        
        int idx = z * (width * height) + y * width + x;
        
        // 6-point stencil (neighbors in each direction)
        int left   = z * (width * height) + y * width + (x - 1);
        int right  = z * (width * height) + y * width + (x + 1);
        int up     = z * (width * height) + (y - 1) * width + x;
        int down   = z * (width * height) + (y + 1) * width + x;
        int front  = (z - 1) * (width * height) + y * width + x;
        int back   = (z + 1) * (width * height) + y * width + x;
        
        output[idx] = (input[left] + input[right] + 
                       input[up] + input[down] +
                       input[front] + input[back]) / 6;
    }
}

void print_slice(const char* name, int* volume, int width, int height, int z) {
    printf("%s (z=%d):\n", name, z);
    for (int y = 0; y < height; y++) {
        printf("  [");
        for (int x = 0; x < width; x++) {
            int idx = z * (width * height) + y * width + x;
            printf("%3d", volume[idx]);
            if (x < width - 1) printf(", ");
        }
        printf("]\n");
    }
}

int main() {
    printf("=== 3D Thread Indexing ===\n\n");
    
    // Demo 1: Volume initialization
    printf("=== Demo 1: Initialize 3D Volume ===\n");
    const int WIDTH = 4, HEIGHT = 3, DEPTH = 2;
    int total = WIDTH * HEIGHT * DEPTH;
    
    int* d_volume;
    int* h_volume = new int[total];
    
    cudaMalloc(&d_volume, total * sizeof(int));
    
    // 3D block and grid
    dim3 block_dim(4, 4, 4);  // 64 threads per block
    dim3 grid_dim(
        (WIDTH + block_dim.x - 1) / block_dim.x,
        (HEIGHT + block_dim.y - 1) / block_dim.y,
        (DEPTH + block_dim.z - 1) / block_dim.z
    );
    
    printf("Volume: %d × %d × %d = %d elements\n", WIDTH, HEIGHT, DEPTH, total);
    printf("Block: (%d, %d, %d) threads\n", block_dim.x, block_dim.y, block_dim.z);
    printf("Grid:  (%d, %d, %d) blocks\n\n", grid_dim.x, grid_dim.y, grid_dim.z);
    
    init_volume<<<grid_dim, block_dim>>>(d_volume, WIDTH, HEIGHT, DEPTH);
    cudaMemcpy(h_volume, d_volume, total * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Values encoded as z*100 + y*10 + x:\n");
    for (int z = 0; z < DEPTH; z++) {
        print_slice("Slice", h_volume, WIDTH, HEIGHT, z);
        printf("\n");
    }
    
    // Demo 2: Batched matrix operation
    printf("=== Demo 2: Batched Matrix Scaling ===\n");
    const int BATCH = 2, ROWS = 2, COLS = 3;
    int batch_total = BATCH * ROWS * COLS;
    
    float* d_batch;
    float* h_batch = new float[batch_total];
    
    // Initialize
    for (int i = 0; i < batch_total; i++) {
        h_batch[i] = (float)(i + 1);
    }
    
    cudaMalloc(&d_batch, batch_total * sizeof(float));
    cudaMemcpy(d_batch, h_batch, batch_total * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("Before scaling (batch=0): [");
    for (int i = 0; i < ROWS * COLS; i++) {
        printf("%.0f ", h_batch[i]);
    }
    printf("]\n");
    
    // Launch: blockIdx.z handles batch dimension
    dim3 batch_block(4, 4, 1);
    dim3 batch_grid(
        (COLS + batch_block.x - 1) / batch_block.x,
        (ROWS + batch_block.y - 1) / batch_block.y,
        BATCH  // One z-block per batch
    );
    
    batched_matrix_scale<<<batch_grid, batch_block>>>(d_batch, BATCH, ROWS, COLS, 2.0f);
    cudaMemcpy(h_batch, d_batch, batch_total * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("After scaling by 2 (batch=0): [");
    for (int i = 0; i < ROWS * COLS; i++) {
        printf("%.0f ", h_batch[i]);
    }
    printf("]\n");
    
    printf("After scaling by 2 (batch=1): [");
    for (int i = ROWS * COLS; i < 2 * ROWS * COLS; i++) {
        printf("%.0f ", h_batch[i]);
    }
    printf("]\n");
    
    // Cleanup
    cudaFree(d_volume);
    cudaFree(d_batch);
    delete[] h_volume;
    delete[] h_batch;
    
    printf("\n=== Key Formulas (3D) ===\n");
    printf("X index: x = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("Y index: y = blockIdx.y * blockDim.y + threadIdx.y\n");
    printf("Z index: z = blockIdx.z * blockDim.z + threadIdx.z\n");
    printf("Linear:  idx = z * (width * height) + y * width + x\n");
    printf("\nBatch layout: data[batch][row][col]\n");
    printf("  idx = batch * (rows * cols) + row * cols + col\n");
    
    return 0;
}
