# Day 2: Batched Inference

## Learning Objectives
- Implement dynamic batching
- Handle variable-length sequences with padding
- Optimize batch size for throughput vs latency

## Key Concepts

### Batching Strategies
```
Static Batching: Fixed batch size, wait for full batch
Dynamic Batching: Variable batch, timeout-based
Continuous Batching: Add/remove sequences mid-batch
```

### Padding for Variable Lengths
- Pad shorter sequences to max length
- Use attention masks to ignore padding
- Memory overhead vs. compute overhead tradeoff

### Batch Size Selection
- Larger batch → higher throughput
- Smaller batch → lower latency
- Sweet spot depends on model and hardware
