# Week 6: Matrix Operations

## 📜 Learning Philosophy
> **CUDA C++ First, Python/Numba as Optional Backup**

---

## 🎯 Learning Goals

By the end of this week, you will:
- Implement optimized matrix multiplication
- Master tiling strategies with shared memory
- Optimize matrix transpose
- Achieve high performance relative to cuBLAS

---

## 📅 Daily Schedule

| Day | Topic | Notebook |
|-----|-------|----------|
| 1 | Naive Matrix Multiply | [day-1-naive-matmul.ipynb](day-1-naive-matmul.ipynb) |
| 2 | Tiled Matrix Multiply | [day-2-tiled-matmul.ipynb](day-2-tiled-matmul.ipynb) |
| 3 | Matrix Transpose | [day-3-transpose.ipynb](day-3-transpose.ipynb) |
| 4 | cuBLAS Comparison | [day-4-cublas.ipynb](day-4-cublas.ipynb) |
| 5 | Practice & Quiz | [checkpoint-quiz.md](checkpoint-quiz.md) |

---

## 🔑 Key Concepts

### Matrix Multiplication
```
C[i,j] = Σ A[i,k] * B[k,j]  for k = 0 to K-1

For MxK @ KxN = MxN:
• Each output element: K multiply-adds
• Total operations: M * N * K MADs
```

### Tiling Strategy
```
Instead of loading entire rows/columns:
1. Load TILE_SIZE x TILE_SIZE tiles into shared memory
2. Compute partial products
3. Accumulate across tiles
4. Reduces global memory access by TILE_SIZE factor
```

### Performance Targets
- Naive: ~10% of peak
- Tiled: ~50-80% of peak
- cuBLAS: ~90%+ of peak

---

## 📋 Deliverables

- [ ] Complete all notebook exercises
- [ ] Quiz score ≥ 25/30
- [ ] Implement matrix library
- [ ] Achieve 80%+ cuBLAS performance

---

## 🔗 Resources

- [CUDA C Programming Guide - Matrix Multiplication](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
