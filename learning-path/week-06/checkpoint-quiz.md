# Week 6 Checkpoint Quiz: Matrix Operations

Test your understanding of matrix operations on GPU.

---

## Part 1: Conceptual Questions (12 points)

### Q1. Memory Access Pattern (3 points)
In naive matrix multiplication C = A × B:
a) How does thread (i, j) access matrix A?
b) How does thread (i, j) access matrix B?
c) Which access pattern is coalesced?

### Q2. Tiling Benefits (3 points)
a) Why does tiling improve performance?
b) What is the optimal tile size?
c) How much does tiling reduce global memory access?

### Q3. Shared Memory (3 points)
For a 32×32 tile of float32:
a) How much shared memory is needed?
b) Is this within typical limits?
c) How many tiles can fit simultaneously?

### Q4. Matrix Transpose (3 points)
a) Why is naive transpose inefficient?
b) How does tiled transpose help?
c) What's the role of padding in bank conflicts?

---

## Part 2: Code Analysis (10 points)

### Q5. Identify the Bug (5 points)
```cpp
__global__ void matmul_naive(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];  // Line 7
    }
    C[row * N + col] = sum;  // Line 9
}
```
What happens when row >= M or col >= N? How would you fix it?

### Q6. Complete the Tiled Code (5 points)
```cpp
__global__ void matmul_tiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Load tiles - YOUR CODE HERE
        As[threadIdx.y][threadIdx.x] = /* ??? */;
        Bs[threadIdx.y][threadIdx.x] = /* ??? */;
        
        __syncthreads();
        
        // Compute partial product - YOUR CODE HERE
        for (int k = 0; k < TILE; k++) {
            sum += /* ??? */;
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

---

## Part 3: Performance Analysis (8 points)

### Q7. FLOPS Calculation (4 points)
For matrix multiplication C = A × B where A is 1024×1024 and B is 1024×1024:
a) How many floating point operations?
b) If a GPU has 10 TFLOPS peak, what's the theoretical minimum time?
c) If measured time is 0.5ms, what's the achieved TFLOPS?
d) What's the efficiency?

### Q8. Memory Bandwidth (4 points)
For the same 1024×1024 × 1024×1024 multiplication:
a) How many bytes are read (naive, no caching)?
b) How many bytes are read with 32×32 tiling?
c) What's the reduction factor?
d) Is this operation compute-bound or memory-bound?

---

## Answers

<details>
<summary>Click to reveal answers</summary>

### Q1 Answers
a) A[row][k] - reads entire row, strided by K (coalesced along k)
b) B[k][col] - reads entire column, strided by 1 (coalesced)
c) B access is coalesced (consecutive threads read consecutive memory)

### Q2 Answers
a) Reduces global memory reads by reusing data in shared memory
b) Typically 16×16 or 32×32, balance occupancy and shared memory
c) Reduces by factor of TILE_SIZE (e.g., 32x less with 32×32 tiles)

### Q3 Answers
a) 32 × 32 × 4 bytes = 4KB
b) Yes, typical limit is 48KB-100KB per block
c) Could fit 12+ tiles (48KB / 4KB)

### Q4 Answers
a) Non-coalesced writes (row→column or column→row)
b) Tile transpose allows coalesced reads and writes
c) Adds padding to avoid bank conflicts in shared memory

### Q5 Answer
Bug: No bounds checking. When row >= M or col >= N, accesses are out of bounds.
Fix:
```cpp
if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
```

### Q6 Answer
```cpp
// Load tiles with bounds checking
int a_col = t * TILE + threadIdx.x;
int b_row = t * TILE + threadIdx.y;
As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

// Compute
for (int k = 0; k < TILE; k++) {
    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
}
```

### Q7 Answers
a) 2 × 1024³ = 2.15 billion FLOPs (multiply + add for each)
b) 2.15e9 / 10e12 = 0.215 ms
c) 2.15e9 / 0.5e-3 = 4.3 TFLOPS
d) 4.3 / 10 = 43%

### Q8 Answers
a) Naive: Each C element reads 1024 A's + 1024 B's × 4 bytes = 8KB
   Total: 1024² × 8KB = 8GB
b) Tiled: Each tile loaded once per tile-row/col traversal
   A: 1024 × 1024 × 4 × (1024/32) = 128MB
   B: Same = 128MB, Total ≈ 256MB
c) 8GB / 256MB = 32x reduction
d) Compute-bound (lots of compute per byte loaded)

</details>

---

## Scoring

| Part | Points | Your Score |
|------|--------|------------|
| Conceptual | 12 | |
| Code Analysis | 10 | |
| Performance | 8 | |
| **Total** | **30** | |

**Passing Score: 25/30**
