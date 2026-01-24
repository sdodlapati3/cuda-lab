# Energy Profiling Module

> **NESAP Relevance:** "Science per joule matters at HPC scale. Can you measure energy efficiency?"

## Overview

Energy profiling is critical for:
- HPC center allocations (measured in node-hours)
- Carbon footprint awareness
- Cost optimization
- Identifying inefficient code patterns

## Key Metrics

| Metric | Unit | Meaning |
|--------|------|---------|
| Power Draw | Watts | Instantaneous power consumption |
| Energy | Joules | Power × Time |
| GPU Utilization | % | SM activity |
| Energy Efficiency | TFLOPS/Watt | Performance per power |

## Tools

### nvidia-smi
```bash
# Real-time power monitoring
nvidia-smi --query-gpu=power.draw --format=csv -l 1

# Detailed power info
nvidia-smi -q -d POWER
```

### NVML (Python)
```python
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
power_w = power_mw / 1000
```

### dcgm-exporter
For cluster-wide monitoring with Prometheus/Grafana.

## Exercises

1. [ex01-power-measurement](./exercises/ex01-power-measurement/) - Measure power during kernels
2. [ex02-energy-efficiency](./exercises/ex02-energy-efficiency/) - Compare implementations by energy

## Quick Start

```python
from energy_benchmark import EnergyBenchmark

bench = EnergyBenchmark()

@bench.measure
def my_workload():
    # Your GPU code
    pass

my_workload()
bench.print_results()
```

## Energy Efficiency Formula

```
Energy Efficiency = Work Done / Energy Consumed

For ML:
  - TFLOPS/Watt (compute efficiency)
  - Samples/Joule (training efficiency)
  - Tokens/Joule (inference efficiency)
```

## Reference Values (A100-80GB)

| Workload | Power | TFLOPS | Efficiency |
|----------|-------|--------|------------|
| Idle | 50W | 0 | 0 |
| Light | 150W | 8 | 0.05 |
| Medium | 250W | 15 | 0.06 |
| Heavy | 400W | 19 | 0.05 |
| TDP | 400W | - | - |

## Best Practices

1. **Measure at steady state** - Skip warmup period
2. **Average over time** - Power fluctuates rapidly
3. **Include all GPUs** - Multi-GPU workloads
4. **Report both power AND time** - Either can be optimized
