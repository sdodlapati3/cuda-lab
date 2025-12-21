# Day 3: PTX vs SASS

## What You'll Learn

- PTX: Portable intermediate representation
- SASS: Actual GPU machine code
- How to extract and read both
- Why this matters for optimization

## The Compilation Pipeline

```
your_kernel.cu
      ↓
   [nvcc frontend]
      ↓
    PTX (portable)  ← Human-readable, GPU-agnostic
      ↓
   [ptxas]
      ↓
    SASS (native)   ← Actual instructions, architecture-specific
```

## PTX vs SASS

| Aspect | PTX | SASS |
|--------|-----|------|
| Portable? | Yes, across GPU generations | No, architecture-specific |
| Human-readable? | Somewhat | Requires expertise |
| When generated? | Compile time | Compile time or JIT |
| Performance? | Template for optimization | What actually runs |

## Quick Start

```bash
# Extract PTX
./scripts/extract_ptx.sh

# Extract SASS
./scripts/extract_sass.sh

# Compare SASS across architectures
./scripts/compare_versions.sh
```

## Understanding PTX

PTX looks like assembly but isn't what runs on the GPU. Example:

```ptx
.visible .entry vector_add(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<4>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [c];
    ld.param.u32 %r2, [n];
    
    // Thread indexing
    mov.u32 %r3, %ctaid.x;      // blockIdx.x
    mov.u32 %r4, %ntid.x;       // blockDim.x
    mov.u32 %r5, %tid.x;        // threadIdx.x
    mad.lo.s32 %r1, %r3, %r4, %r5;  // idx = blockIdx.x * blockDim.x + threadIdx.x
    
    // Bounds check
    setp.ge.s32 %p1, %r1, %r2;  // if idx >= n
    @%p1 bra EXIT;
    
    // Load, add, store
    ld.global.f32 %f1, [%rd4];
    ld.global.f32 %f2, [%rd5];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd6], %f3;
    
EXIT:
    ret;
}
```

Key PTX concepts:
- `.reg` declares virtual registers (unlimited!)
- `%p<n>` are predicate registers (for branching)
- `@%p1 bra` is predicated branch
- `ld.global`, `st.global` are memory operations

## Understanding SASS

SASS is the real GPU assembly. Here's what vector_add looks like on sm_80 (A100):

```sass
// Function: vector_add
// sm_80 SASS

        IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]     // Load param
        S2R R0, SR_CTAID.X                         // blockIdx.x
        S2R R3, SR_TID.X                           // threadIdx.x
        IMAD R0, R0, c[0x0][0x0], R3               // idx = blockIdx.x * blockDim.x + threadIdx.x
        ISETP.GE.AND P0, PT, R0, c[0x0][0x170], PT // if idx >= n
@P0     EXIT                                        // predicated exit
        
        IMAD.WIDE.U32 R4, R0, 0x4, c[0x0][0x160]   // addr_a = a + idx*4
        IMAD.WIDE.U32 R6, R0, 0x4, c[0x0][0x168]   // addr_b = b + idx*4
        LDG.E.SYS R2, [R4]                          // load a[idx]
        LDG.E.SYS R3, [R6]                          // load b[idx]
        IMAD.WIDE.U32 R4, R0, 0x4, c[0x0][0x178]   // addr_c = c + idx*4
        FADD R2, R2, R3                             // a + b
        STG.E.SYS [R4], R2                          // store c[idx]
        EXIT
```

Key SASS concepts:
- Real registers: R0-R255 (limited by architecture)
- `LDG.E.SYS`: Global load, `.E` = extended, `.SYS` = system coherent
- `IMAD`: Integer multiply-add (very common)
- `@P0`: Predicated execution (no branch divergence for simple if)

## What to Look For

### 1. Register Count
```bash
cuobjdump -res-usage your_kernel | grep "REG"
```
More registers = fewer warps = lower occupancy.

### 2. Memory Instructions
- `LDG` / `STG` = global memory (slow)
- `LDS` / `STS` = shared memory (fast)
- `LDSM` = shared memory matrix load (tensor cores)

### 3. Compute Instructions
- `FFMA` = fused multiply-add (1 clock)
- `HMMA` = half-precision matrix multiply (tensor cores)
- `IMAD` = integer multiply-add

### 4. Control Flow
- `BRA` = branch
- `@P0 INST` = predicated (no branch, just mask)
- `SYNC` = syncthreads

## Exercises

1. Extract PTX for vector_add and find the load/store instructions
2. Compare sm_70 vs sm_90 SASS—what changed?
3. Compile with `-O0` and `-O3`, compare SASS instruction count
4. Find the register count for a kernel and calculate max occupancy
