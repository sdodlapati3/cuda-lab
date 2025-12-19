# Week 5: Prefix Sum (Scan)

## 📜 Learning Philosophy
> **CUDA C++ First, Python/Numba as Optional Backup**

All notebooks show CUDA C++ code as the PRIMARY learning material. Python/Numba code is provided as an OPTIONAL alternative for quick interactive testing in Google Colab.

---

## 🎯 Learning Goals

By the end of this week, you will:
- Understand scan as a fundamental parallel primitive
- Implement work-efficient scan algorithms
- Handle arrays larger than block size
- Apply scan to real problems (stream compaction, radix sort)

---

## 📅 Daily Schedule

| Day | Topic | Notebook |
|-----|-------|----------|
| 1 | Inclusive vs Exclusive Scan | [day-1-scan-basics.ipynb](day-1-scan-basics.ipynb) |
| 2 | Hillis-Steele Algorithm | [day-2-hillis-steele.ipynb](day-2-hillis-steele.ipynb) |
| 3 | Blelloch Algorithm | [day-3-blelloch.ipynb](day-3-blelloch.ipynb) |
| 4 | Large Array Scan & Applications | [day-4-large-scan.ipynb](day-4-large-scan.ipynb) |
| 5 | Practice & Quiz | [checkpoint-quiz.md](checkpoint-quiz.md) |

---

## 🔑 Key Concepts

### Scan Operations
```
Inclusive Scan (prefix sum):
Input:  [3, 1, 7, 0, 4, 1, 6, 3]
Output: [3, 4, 11, 11, 15, 16, 22, 25]

Exclusive Scan:
Input:  [3, 1, 7, 0, 4, 1, 6, 3]
Output: [0, 3, 4, 11, 11, 15, 16, 22]
```

### Algorithms
- **Hillis-Steele**: O(n log n) work, O(log n) steps - simple but more work
- **Blelloch**: O(n) work, O(2 log n) steps - work-efficient

### Applications
- Stream compaction (filter arrays)
- Radix sort
- Polynomial evaluation
- Histograms
- Tree operations

---

## 📋 Deliverables

- [ ] Complete all notebook exercises
- [ ] Quiz score ≥ 25/30
- [ ] Implement stream compaction project
- [ ] Can write scan from scratch without reference

---

## 🔗 Resources

- [GPU Gems 3: Parallel Prefix Sum](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
- [CUB Scan Documentation](https://nvlabs.github.io/cub/structcub_1_1_device_scan.html)
