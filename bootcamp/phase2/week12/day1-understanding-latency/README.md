# Day 1: Understanding Latency

## Learning Objectives

- Understand GPU latency sources
- Learn typical latency values
- See why hiding latency matters

## Key Concepts

### Memory Latency

| Memory Type | Latency (cycles) | Bandwidth |
|-------------|------------------|-----------|
| Registers | 0-1 | Very High |
| Shared Memory | ~20-30 | High |
| L1 Cache | ~30 | High |
| L2 Cache | ~200 | Medium |
| Global (DRAM) | ~400-800 | Peak: 1.5 TB/s |

### Instruction Latency

| Instruction | Latency (cycles) |
|-------------|------------------|
| FMA | ~4 |
| Integer | ~4 |
| Special (sin/cos) | ~16 |
| Double precision | ~4-8 |

### Why Hiding Matters

At 400 cycle memory latency:
- 1 warp waiting → 400 idle cycles
- 16 warps → 25 cycles each, fully hidden!

## Build & Run

```bash
./build.sh
./build/latency_demo
```
