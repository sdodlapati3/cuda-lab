# Practice: Optimization Techniques

Hands-on exercises for CUDA performance optimization.

## Exercises

| Exercise | Topic | Difficulty | Week |
|----------|-------|------------|------|
| [ex01-occupancy](ex01-occupancy/) | Occupancy analysis & tuning | ⭐⭐ | Week 7 |
| [ex02-streams](ex02-streams/) | CUDA streams & overlap | ⭐⭐ | Week 9 |
| [ex03-events](ex03-events/) | Events for timing & sync | ⭐⭐ | Week 9 |

## Learning Objectives

After completing these exercises, you will be able to:

1. **Analyze occupancy** - Use occupancy calculator, understand limiting factors
2. **Overlap computation and transfers** - Hide latency with streams
3. **Measure performance accurately** - Use CUDA events for timing

## Key Concepts

### Occupancy
- Active warps / maximum warps per SM
- Limited by registers, shared memory, or block size
- Higher occupancy doesn't always mean faster (balance with ILP)

### Streams
- Asynchronous execution queues
- Enable overlapping H2D, kernel, D2H
- Use pinned memory for async transfers

### Events
- Synchronization points within streams
- Accurate GPU timing without CPU overhead
- Inter-stream dependencies

## Prerequisites

- Weeks 1-6 fundamentals
- Understanding of GPU resource limits
