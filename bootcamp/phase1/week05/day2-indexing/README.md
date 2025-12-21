# Day 2: Thread Indexing Patterns

## Learning Objectives

- Master 1D, 2D, and 3D indexing
- Calculate global indices for any dimension
- Map thread indices to data elements

## Key Concepts

### 1D Indexing (Arrays, Vectors)

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
```

### 2D Indexing (Matrices, Images)

```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * width + col;  // Row-major layout
```

### 3D Indexing (Volumes, Tensors)

```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
int idx = z * (width * height) + y * width + x;
```

### Common Patterns

| Data Structure | Dimensionality | Index Formula |
|----------------|----------------|---------------|
| 1D array | 1D | `blockIdx.x * blockDim.x + threadIdx.x` |
| Matrix | 2D | Row-major: `row * width + col` |
| 3D volume | 3D | `z * (width * height) + y * width + x` |
| Batched matrices | 2D + batch | `batch * (rows * cols) + row * cols + col` |

## Exercises

1. **1D Array Access**: Fill array with thread indices
2. **2D Matrix Access**: Initialize matrix with row+column values
3. **3D Volume Access**: Process a 3D volume

## Build & Run

```bash
./build.sh
./build/index_1d
./build/index_2d
./build/index_3d
```
