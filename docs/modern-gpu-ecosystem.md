# Modern GPU Programming Ecosystem

> **After completing this CUDA curriculum, you understand the fundamentals that ALL these tools build upon.**

## The Reality Check

You may have heard that "nobody writes raw CUDA anymore" because of tools like Triton and torch.compile. Let's be honest about what this means:

| Statement | Reality |
|-----------|---------|
| "Use Triton instead of CUDA" | Triton generates CUDA (PTX). Understanding CUDA makes you a better Triton programmer. |
| "torch.compile does it automatically" | It does—until it doesn't. Understanding why it fails requires CUDA knowledge. |
| "cuBLAS/cuDNN are faster than custom kernels" | Usually true! But you need to know *when* custom kernels win (see Week 14). |
| "XLA/TVM compile everything" | Compilers optimize common patterns. Novel algorithms still need manual optimization. |

**Bottom line:** The 18 weeks you've completed teach the concepts that make ALL these tools work. You can now learn any of them in days, not months.

---

## Quick Reference: When to Use What

```
┌─────────────────────────────────────────────────────────────────┐
│                    CHOOSING YOUR TOOL                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  "I need maximum performance for a custom operation"            │
│       │                                                         │
│       ├─→ Is it a standard operation (matmul, conv, attention)? │
│       │       YES → Use cuBLAS / cuDNN / Flash Attention        │
│       │       NO  ↓                                             │
│       │                                                         │
│       ├─→ Am I prototyping or need to iterate quickly?          │
│       │       YES → Use Triton (Python-like, fast iteration)    │
│       │       NO  ↓                                             │
│       │                                                         │
│       ├─→ Do I need maximum control (warp-level, specific ISA)? │
│       │       YES → Write CUDA (what you learned here!)         │
│       │       NO  → Triton is probably fine                     │
│       │                                                         │
│  "I just want my PyTorch model to be faster"                    │
│       → torch.compile(model) and you're done                    │
│                                                                 │
│  "I'm building production inference"                            │
│       → TensorRT or torch.compile with max-autotune             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tool-by-Tool Overview

### 1. Triton (OpenAI)

**What it is:** A Python-based language for writing GPU kernels.

**The pitch:** "Write GPU kernels at 80% of CUDA performance with 20% of the effort."

**How your CUDA knowledge helps:**
| CUDA Concept | Triton Equivalent |
|--------------|-------------------|
| `__shared__` memory | `tl.load()` with `cache=True` (automatic) |
| Thread blocks | `@triton.jit` with `BLOCK_SIZE` parameter |
| Memory coalescing | Still matters! Triton autotunes but doesn't fix bad access patterns |
| Bank conflicts | Abstracted away (mostly) |
| Occupancy | Controlled via `num_warps`, `num_stages` |

**Learning time for you:** 1-2 days. You already understand the concepts.

**Example - Vector Add in Triton:**
```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # Like blockIdx.x
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # Like threadIdx.x
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)  # Coalesced load
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

**When to use Triton over CUDA:**
- ✅ Prototyping custom operations
- ✅ Research kernels that need quick iteration
- ✅ Operations that don't need warp-level control
- ❌ When you need PTX-level control
- ❌ When interfacing with CUDA libraries

---

### 2. torch.compile / Inductor (PyTorch 2.0+)

**What it is:** A JIT compiler that optimizes PyTorch models.

**Your interaction with it:**
```python
model = torch.compile(model)  # That's it
```

**What happens under the hood:**
1. **TorchDynamo** traces your Python code
2. **AOTAutograd** handles backward pass
3. **Inductor** generates Triton kernels (which become CUDA)

**When it matters that you know CUDA:**
- Debugging why `torch.compile` is slow for your operation
- Writing custom `torch.autograd.Function` with CUDA backend
- Understanding performance profiles in Nsight Systems

**The honest truth:** For 95% of use cases, you don't need to understand Inductor internals. Just use it.

---

### 3. cuBLAS / cuDNN / cuFFT / cuSPARSE

**What they are:** NVIDIA's hand-optimized libraries.

**The rule:** If your operation is covered by these libraries, use them. They're faster than anything you'll write.

| Library | Use For |
|---------|---------|
| cuBLAS | Matrix multiply, BLAS operations |
| cuDNN | Convolutions, batch norm, attention, RNNs |
| cuFFT | Fast Fourier Transforms |
| cuSPARSE | Sparse matrix operations |

**What you learned in Week 6 & 13:** How to use cuBLAS and when custom kernels win (non-standard operations, fused kernels).

---

### 4. CUTLASS

**What it is:** NVIDIA's template library for building custom matrix multiply kernels.

**Who needs it:** 
- NVIDIA engineers building cuBLAS/cuDNN
- Library authors (PyTorch, TensorFlow internals)
- **Not typical CUDA programmers**

**The reality:** CUTLASS uses C++ template metaprogramming that's an order of magnitude more complex than regular CUDA. It's designed for building libraries, not applications.

**Learning time:** Weeks to months. Only pursue if you're building a deep learning framework or GPU library.

---

### 5. TensorRT

**What it is:** NVIDIA's inference optimizer.

**Use case:** Deploying trained models at maximum speed.

**How it works:**
1. Takes a trained model (ONNX, PyTorch, TensorFlow)
2. Applies optimizations (layer fusion, precision calibration, kernel selection)
3. Outputs an optimized inference engine

**Your CUDA knowledge helps:** Understanding why certain optimizations work (fusion, memory layout, precision tradeoffs).

---

### 6. XLA / JAX / TVM

**What they are:** Compiler-based approaches to GPU programming.

| Tool | Ecosystem | Notes |
|------|-----------|-------|
| XLA | TensorFlow, JAX | Google's compiler, great for TPUs too |
| JAX | Standalone | NumPy-like API with automatic differentiation + XLA |
| TVM | Framework-agnostic | Research compiler, good for edge deployment |

**Do you need to learn these?** 
- If you use JAX → Yes, learn JAX
- If you use TensorFlow → XLA is automatic
- If you're a compiler researcher → TVM is interesting
- For CUDA programming → Not required

---

## How Your CUDA Knowledge Transfers

| CUDA Concept (You Know) | Why It Matters Everywhere |
|-------------------------|---------------------------|
| Memory hierarchy | All tools must respect L2/L1/shared/registers |
| Occupancy | Triton has `num_warps`; compilers optimize for occupancy |
| Coalescing | Bad access patterns are slow in ANY tool |
| Bank conflicts | Shared memory is shared memory, regardless of abstraction |
| Warp divergence | SIMT execution model is fundamental |
| Streams & async | All frameworks use CUDA streams underneath |
| Profiling (Nsight) | You can profile ANY GPU code with the same tools |

---

## The Career Perspective

**Junior GPU Engineer:** 
- Uses torch.compile, cuBLAS, maybe Triton for custom ops
- Your curriculum covers this

**Senior GPU Engineer:**
- Writes custom CUDA when needed
- Optimizes with Nsight, understands roofline
- Your curriculum covers this

**GPU Library Developer (NVIDIA, Meta, Google):**
- Knows CUTLASS, writes Triton backends, contributes to compilers
- Your curriculum is the foundation; CUTLASS/compiler work is the specialization

---

## Recommended Next Steps

After completing the 18-week curriculum:

1. **For ML Practitioners:**
   - Try Triton for a custom operation you wrote in CUDA
   - Compare performance with your hand-written kernel
   - You'll appreciate both the convenience and limitations

2. **For Systems Engineers:**
   - Profile a torch.compile'd model with Nsight Systems
   - See how Inductor generates kernels
   - Identify where custom CUDA could help

3. **For Researchers:**
   - Implement a paper's novel operation in Triton
   - When Triton isn't enough, you know CUDA
   - This is the 80/20 balance

4. **For Library Developers:**
   - Study CUTLASS (but budget weeks, not days)
   - Contribute to Triton's kernel library
   - Your CUDA foundation makes this possible

---

## Resources

| Topic | Resource |
|-------|----------|
| Triton | [triton-lang.org](https://triton-lang.org), OpenAI Triton tutorials |
| torch.compile | [PyTorch 2.0 docs](https://pytorch.org/docs/stable/torch.compiler.html) |
| CUTLASS | [NVIDIA CUTLASS GitHub](https://github.com/NVIDIA/cutlass) |
| TensorRT | [NVIDIA TensorRT docs](https://developer.nvidia.com/tensorrt) |
| Flash Attention | [GitHub - FlashAttention](https://github.com/Dao-AILab/flash-attention) |

---

## Final Thought

> **You didn't learn CUDA to write CUDA forever. You learned CUDA to understand GPUs.**
> 
> That understanding transfers to every tool in this ecosystem. The concepts don't change; only the syntax does.
> 
> Congratulations on completing the curriculum! 🎉
