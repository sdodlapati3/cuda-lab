# Day 3: Quantization Basics

## Learning Objectives
- Understand FP16 and INT8 quantization
- Implement simple quantization/dequantization
- Measure speedup vs. accuracy tradeoff

## Key Concepts

### Precision Hierarchy
```
FP32: 32-bit float (baseline)
FP16: 16-bit float (2x throughput on tensor cores)
INT8: 8-bit integer (4x throughput, needs calibration)
```

### Quantization Formula
```
q = round((x - zero_point) / scale)
x_approx = q * scale + zero_point
```

### Calibration
- Collect activation statistics
- Determine scale and zero_point
- Minimize quantization error
