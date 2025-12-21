# Day 5: Memory Planning

## Learning Objectives
- Plan buffer allocation for inference
- Implement buffer reuse across layers
- Use memory pools for efficiency

## Key Concepts

### Activation Buffer Strategy
- Only need current and next layer activations
- "Ping-pong" between two buffers
- Size = max(layer_sizes) × batch_size × 2

### Memory Pool Benefits
- Avoid allocation overhead during inference
- Preallocate all needed memory
- Reduce fragmentation
