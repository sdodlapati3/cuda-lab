# Week 24, Day 3: Async Copy with cp.async

## Objective
Use CUDA's asynchronous copy instructions for overlapping loads with compute.

## Key Concepts
- cp.async PTX instruction (sm_80+)
- cuda_pipeline.h API
- Overlapping memory transfers with computation
- Pipeline stages for latency hiding

## Why Async Copy?
- Global → shared memory bypass L1 cache
- Hardware-managed copy operations
- Overlaps with ALU operations
- Critical for achieving peak performance

## Implementation
- __pipeline_memcpy_async() intrinsic
- __pipeline_commit() to submit operations
- __pipeline_wait_prior() to synchronize

## Expected Results
- 15-25% improvement with async loading
- Better GPU utilization
