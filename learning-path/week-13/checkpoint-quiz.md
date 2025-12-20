# Week 13 Checkpoint Quiz: Tensor Cores & Mixed Precision

**Total Points: 30** | **Passing Score: 24 (80%)**

---

## Part 1: Tensor Core Fundamentals (10 points)

### Question 1 (2 points)
What is the minimum compute capability required for Tensor Cores?

- A) SM 5.0
- B) SM 6.0
- C) SM 7.0
- D) SM 8.0

### Question 2 (2 points)
What is the standard WMMA fragment size for matrix multiplication?

- A) 8×8×8
- B) 16×16×16
- C) 32×32×32
- D) 64×64×64

### Question 3 (3 points)
Why do Tensor Cores require warp-level programming?

- A) For better cache utilization
- B) All 32 threads cooperate to form fragments
- C) To reduce register usage
- D) For automatic memory coalescing

### Question 4 (3 points)
Match the fragment type to its role:

| Fragment | Role |
|----------|------|
| 1. `matrix_a` | A. Result accumulator |
| 2. `matrix_b` | B. Left input matrix |
| 3. `accumulator` | C. Right input matrix |

---

## Part 2: WMMA Programming (10 points)

### Question 5 (2 points)
Which header must be included for WMMA operations?

- A) `<cuda_wmma.h>`
- B) `<mma.h>`
- C) `<tensor_core.h>`
- D) `<cuda_fp16.h>`

### Question 6 (3 points)
What is the correct order of WMMA operations?

- A) mma_sync → load_matrix_sync → store_matrix_sync
- B) load_matrix_sync → mma_sync → store_matrix_sync
- C) store_matrix_sync → load_matrix_sync → mma_sync
- D) fill_fragment → mma_sync → load_matrix_sync

### Question 7 (2 points)
Why must `fill_fragment` be called on the accumulator before use?

- A) To allocate memory
- B) To initialize to zero (or desired value)
- C) To set the fragment size
- D) To select the compute type

### Question 8 (3 points)
In the following code, identify the error:

```cpp
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::load_matrix_sync(a_frag, A, 8);  // A is 16×16
```

- A) Wrong fragment size
- B) Wrong leading dimension (should be 16)
- C) Wrong data type
- D) Wrong layout specification

---

## Part 3: Mixed Precision Training (10 points)

### Question 9 (2 points)
What is the primary reason for using loss scaling in mixed precision training?

- A) Faster computation
- B) Less memory usage
- C) Prevent gradient underflow
- D) Better model accuracy

### Question 10 (3 points)
Which operations should remain in FP32 for numerical stability?

- A) Matrix multiplications
- B) Convolutions
- C) Softmax and layer normalization
- D) ReLU activations

### Question 11 (2 points)
What does dynamic loss scaling do when overflow is detected?

- A) Skip the training step and reduce scale
- B) Increase the scale factor
- C) Convert to FP64
- D) Restart training

### Question 12 (3 points)
In cuBLAS, which compute type uses Tensor Cores with FP32 inputs?

- A) `CUBLAS_COMPUTE_32F`
- B) `CUBLAS_COMPUTE_16F`
- C) `CUBLAS_COMPUTE_32F_FAST_16F`
- D) `CUBLAS_COMPUTE_64F`

---

## Answer Key

| Question | Answer | Explanation |
|----------|--------|-------------|
| 1 | C | Tensor Cores were introduced with Volta (SM 7.0) |
| 2 | B | Standard WMMA uses 16×16×16 fragments |
| 3 | B | WMMA is a warp-collective operation requiring all 32 threads |
| 4 | 1-B, 2-C, 3-A | matrix_a is left input, matrix_b is right, accumulator stores result |
| 5 | B | WMMA is in the `<mma.h>` header |
| 6 | B | Load inputs, compute, store output |
| 7 | B | Fragments contain garbage until initialized |
| 8 | B | Leading dimension must match matrix width (16, not 8) |
| 9 | C | FP16 has limited range; small gradients underflow to zero |
| 10 | C | Reduction operations need FP32 to maintain precision |
| 11 | A | Overflow means scale is too high; skip and reduce |
| 12 | C | FAST_16F allows FP32 inputs with FP16 Tensor Core compute |

---

## Scoring Guide

- **30 points**: Excellent! Ready for real-world Tensor Core applications
- **24-29 points**: Great understanding, review any missed concepts
- **18-23 points**: Good foundation, revisit mixed precision section
- **Below 18 points**: Re-study the week's material before proceeding

---

## Next Steps

After passing this quiz:
1. Complete the cuBLAS exercises
2. Profile your WMMA kernel with Nsight Compute
3. Move on to Week 14: Real-World Applications
