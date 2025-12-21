# Day 5: Finding Bottlenecks

## Learning Objectives

- Systematic bottleneck identification
- Use NCU's guided analysis
- Map symptoms to solutions

## Bottleneck Diagnosis Flow

```
1. Run ncu --set full
2. Check Speed of Light (SOL) section
3. Identify: Memory-bound or Compute-bound?
4. Drill into specific limiters
5. Map to optimization actions
```

## Common Bottleneck Patterns

### Memory-Bound Patterns
| Symptom | Cause | Solution |
|---------|-------|----------|
| High DRAM, low L2 hit | No reuse | Shared memory tiling |
| Low load efficiency | Uncoalesced | Fix access patterns |
| High L2, low DRAM | Good caching | Might be optimal |

### Compute-Bound Patterns
| Symptom | Cause | Solution |
|---------|-------|----------|
| Low occupancy | Resource use | Reduce regs/smem |
| High stall: pipe busy | Unit contention | Different ops |
| Low ILP | Dependencies | Unroll, reorder |

## Build & Run

```bash
./build.sh
./build/bottleneck_demo
ncu --set full ./build/bottleneck_demo
```
