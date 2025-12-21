# Day 4: Building Rooflines

## Learning Objectives

- Construct complete roofline plots
- Understand ridge point significance
- Use roofline for kernel analysis

## Key Concepts

### The Roofline Model

```
Performance = min(Peak FLOPS, AI × Peak Bandwidth)
```

Where:
- **AI** = Arithmetic Intensity (FLOPs/byte)
- **Peak FLOPS** = compute ceiling (horizontal line)
- **Peak Bandwidth** = memory ceiling (sloped line)

### Ridge Point

The AI where memory and compute ceilings meet:

```
Ridge Point AI = Peak FLOPS / Peak Bandwidth

A100:
  Ridge ≈ 19.5 TFLOPS / 2 TB/s ≈ 10 FLOPS/byte
```

### Interpreting Position

| Position | Meaning | Action |
|----------|---------|--------|
| Below slope | Memory-bound, inefficient | Improve access patterns |
| On slope | Memory-bound, efficient | Add compute or use shared mem |
| At ceiling | Compute-bound, efficient | Optimal! |
| Below ceiling | Compute-bound, inefficient | Improve occupancy/ILP |

## Build & Run

```bash
./build.sh
./build/roofline_data
python3 plot_roofline.py
```
