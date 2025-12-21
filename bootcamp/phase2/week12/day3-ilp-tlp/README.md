# Day 3: ILP and TLP

## Learning Objectives

- Understand Instruction-Level Parallelism (ILP)
- Understand Thread-Level Parallelism (TLP)
- Learn when to use each approach

## Key Concepts

### Two Dimensions of Parallelism

```
        ▲ ILP (per thread)
        │
   High │  ┌─────────────────┐
        │  │ Unrolled loops  │
        │  │ Multiple ops    │
        │  │ per thread      │
        │  └─────────────────┘
   Low  │  ┌─────────────────┐
        │  │ Simple kernels  │
        │  │ 1 op per thread │
        └──┴─────────────────┴──► TLP (threads)
              Low      High
```

### ILP: Do more per thread

```cpp
// Low ILP - one operation per thread
out[idx] = in[idx] * 2.0f;

// High ILP - multiple independent operations
float a = in[idx] * 2.0f;
float b = in[idx + stride] * 2.0f;  // Independent!
float c = in[idx + 2*stride] * 2.0f;  // Independent!
out[idx] = a; out[idx+stride] = b; out[idx+2*stride] = c;
```

### TLP: Use more threads

- More threads = more warps
- More warps = better latency hiding
- But uses more resources

## Tradeoffs

| Strategy | Pros | Cons |
|----------|------|------|
| High ILP | Better register use, fewer threads needed | More registers per thread |
| High TLP | Simpler code, good latency hiding | More overhead |

## Build & Run

```bash
./build.sh
./build/ilp_tlp
```
