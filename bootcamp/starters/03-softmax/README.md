# Starter 03: Online Softmax

**The FlashAttention Foundation** - Understand this, and you understand Flash Attention's core insight.

## Why This Matters

Softmax appears in:
- **Every attention layer** in transformers
- **Classification heads** 
- **Mixture of Experts** gating

The online algorithm enables:
- **FlashAttention:** 2-4× speedup, O(N) memory instead of O(N²)
- **Kernel fusion:** Combine with QK^T computation
- **Streaming computation:** Process sequences without storing full matrix

## The Key Insight

**Naive Softmax (3 passes):**
```
Pass 1: max = max(x)           → Read input
Pass 2: exp_sum = Σexp(x-max)  → Read input again
Pass 3: output = exp/exp_sum   → Read input again, write output
```

**Online Softmax (1 pass + 1 write):**
```
Pass 1: Compute max AND sum simultaneously using correction factor
Pass 2: Write output (unavoidable)
```

**The Math:**
When max changes from `m` to `m'`:
```
sum' = sum × exp(m - m') + exp(x - m')
```

This "running correction" is the key to FlashAttention!

## Build & Run

```bash
make
./softmax

# Custom size
./softmax 2048 8192  # 2048 batch, 8192 dim (like GPT hidden size)
```

## Expected Output

```
╔════════════════════════════════════════════════════════════════╗
║           ONLINE SOFTMAX BENCHMARK                             ║
╠════════════════════════════════════════════════════════════════╣
║ Device: NVIDIA A100-SXM4-80GB                                  ║
║ Peak Bandwidth: 2039.0 GB/s                                    ║
║ Batch: 1024, Dim: 4096 (16 MB)                                 ║
╠════════════════════════════════════════════════════════════════╣
V1: Naive (3-pass)       :   123.45 μs |   259.3 GB/s |  12.7% peak
V2: 2-pass (fused)       :    82.34 μs |   388.7 GB/s |  19.1% peak
V3: Online (1-pass)      :    52.12 μs |   614.2 GB/s |  30.1% peak
╠════════════════════════════════════════════════════════════════╣
║ Verification: Max error = 2.38e-07 ✓                           ║
╚════════════════════════════════════════════════════════════════╝
```

## The OnlineSoftmax Struct

```cuda
struct OnlineSoftmax {
    float max_val;
    float sum;
    
    __device__ void update(float x) {
        float new_max = fmaxf(max_val, x);
        // THE KEY: Adjust previous sum for new max
        sum = sum * expf(max_val - new_max) + expf(x - new_max);
        max_val = new_max;
    }
    
    __device__ void merge(const OnlineSoftmax& other) {
        float new_max = fmaxf(max_val, other.max_val);
        sum = sum * expf(max_val - new_max) + 
              other.sum * expf(other.max_val - new_max);
        max_val = new_max;
    }
};
```

## FlashAttention Connection

FlashAttention uses this exact pattern for attention:
```
Instead of:
  S = Q @ K^T       # O(N²) memory
  P = softmax(S)    # O(N²) memory  
  O = P @ V         # O(N²) memory

FlashAttention:
  For each block:
    S_block = Q_block @ K_block^T
    Update online softmax state
    O = update(O, V_block)
  # Never materialize full S or P!
```

## Exercises

1. **Implement attention with online softmax**
   - Input: Q, K, V
   - Output: Attention(Q, K, V) without storing full NxN matrix

2. **Add causal masking**
   - Set elements above diagonal to -inf before softmax

3. **Fuse with QK^T**
   - Compute QK^T and softmax together

4. **Profile memory traffic**
   - Verify 1-pass vs 3-pass with Nsight Compute

5. **Half precision**
   - Implement for FP16/BF16 with numerical stability

## What You Learn Here Applies To

| Concept | Used In |
|---------|---------|
| Online algorithms | FlashAttention, streaming |
| Numerical stability | LayerNorm, log-sum-exp |
| Warp reduction with state | Complex reductions |
| Memory vs compute trade-off | All kernel fusion |

## Next Steps

After mastering this:
1. → Study FlashAttention paper and code
2. → Implement LayerNorm (similar reduction pattern)
3. → Build fused attention kernel
