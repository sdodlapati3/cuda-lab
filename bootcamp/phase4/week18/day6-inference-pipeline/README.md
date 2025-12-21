# Day 6: Inference Pipeline

## Learning Objectives
- Build complete inference pipeline
- Combine all optimizations
- Measure end-to-end performance

## Key Concepts

### Pipeline Components
1. Weight loading (pinned memory)
2. Input preprocessing
3. Batched inference
4. Output postprocessing

### Optimization Checklist
- [ ] Weights preloaded to GPU
- [ ] Pinned memory for transfers
- [ ] Layer fusion where possible
- [ ] Buffer reuse (memory pool)
- [ ] Optimal batch size
