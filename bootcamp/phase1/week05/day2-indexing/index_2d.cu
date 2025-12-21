/**
 * index_2d.cu - 2D thread indexing for matrices and images
 * 
 * Learning objectives:
 * - Map 2D thread grid to 2D data
 * - Understand row-major vs column-major
 * - Handle 2D bounds checking
 */

#include <cuda_runtime.h>
#include <cstdio>

// 2D kernel for matrix initialization
__global__ void init_matrix(int* matrix, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows && col < cols) {
        // Row-major indexing: idx = row * width + col
        int idx = row * cols + col;
        matrix[idx] = row * 10 + col;  // Store row*10+col for visibility
    }
}

// 2D kernel with explicit row-major access
__global__ void matrix_add(const int* A, const int* B, int* C, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

// Process image-like data (2D with channels)
__global__ void process_rgb(unsigned char* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Each pixel has RGB channels
        int pixel_idx = (y * width + x) * 3;  // 3 channels
        
        // Invert colors
        image[pixel_idx + 0] = 255 - image[pixel_idx + 0];  // R
        image[pixel_idx + 1] = 255 - image[pixel_idx + 1];  // G
        image[pixel_idx + 2] = 255 - image[pixel_idx + 2];  // B
    }
}

void print_matrix(const char* name, int* matrix, int rows, int cols) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int r = 0; r < rows; r++) {
        printf("  [");
        for (int c = 0; c < cols; c++) {
            printf("%3d", matrix[r * cols + c]);
            if (c < cols - 1) printf(", ");
        }
        printf("]\n");
    }
}

int main() {
    printf("=== 2D Thread Indexing ===\n\n");
    
    // Demo 1: Matrix initialization
    printf("=== Demo 1: Initialize Matrix ===\n");
    const int ROWS = 4, COLS = 6;
    int* d_matrix;
    int* h_matrix = new int[ROWS * COLS];
    
    cudaMalloc(&d_matrix, ROWS * COLS * sizeof(int));
    
    // 2D block and grid
    dim3 block_dim(4, 4);  // 4x4 = 16 threads per block
    dim3 grid_dim(
        (COLS + block_dim.x - 1) / block_dim.x,
        (ROWS + block_dim.y - 1) / block_dim.y
    );
    
    printf("Matrix: %d rows × %d cols\n", ROWS, COLS);
    printf("Block: (%d, %d) threads\n", block_dim.x, block_dim.y);
    printf("Grid:  (%d, %d) blocks\n\n", grid_dim.x, grid_dim.y);
    
    init_matrix<<<grid_dim, block_dim>>>(d_matrix, ROWS, COLS);
    cudaMemcpy(h_matrix, d_matrix, ROWS * COLS * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Values are row*10 + col:\n");
    print_matrix("Matrix", h_matrix, ROWS, COLS);
    
    // Demo 2: Matrix addition
    printf("\n=== Demo 2: Matrix Addition ===\n");
    int* d_A, *d_B, *d_C;
    int* h_A = new int[ROWS * COLS];
    int* h_B = new int[ROWS * COLS];
    int* h_C = new int[ROWS * COLS];
    
    // Initialize host matrices
    for (int i = 0; i < ROWS * COLS; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }
    
    cudaMalloc(&d_A, ROWS * COLS * sizeof(int));
    cudaMalloc(&d_B, ROWS * COLS * sizeof(int));
    cudaMalloc(&d_C, ROWS * COLS * sizeof(int));
    
    cudaMemcpy(d_A, h_A, ROWS * COLS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, ROWS * COLS * sizeof(int), cudaMemcpyHostToDevice);
    
    matrix_add<<<grid_dim, block_dim>>>(d_A, d_B, d_C, ROWS, COLS);
    cudaMemcpy(h_C, d_C, ROWS * COLS * sizeof(int), cudaMemcpyDeviceToHost);
    
    print_matrix("A", h_A, ROWS, COLS);
    printf("+\n");
    print_matrix("B", h_B, ROWS, COLS);
    printf("=\n");
    print_matrix("C", h_C, ROWS, COLS);
    
    // Cleanup
    cudaFree(d_matrix);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_matrix;
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    printf("\n=== Key Formulas (Row-Major) ===\n");
    printf("Global row:    row = blockIdx.y * blockDim.y + threadIdx.y\n");
    printf("Global col:    col = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("Linear index:  idx = row * width + col\n");
    printf("Grid size X:   (cols + blockDim.x - 1) / blockDim.x\n");
    printf("Grid size Y:   (rows + blockDim.y - 1) / blockDim.y\n");
    
    return 0;
}
