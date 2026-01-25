# Exercise 01: Python Backtrace Profiling

## Learning Objectives
- Enable Python backtrace collection in Nsight Systems
- Correlate GPU kernels with Python source code
- Identify which Python functions trigger expensive GPU operations
- Debug performance issues in PyTorch training loops

## Background

When profiling PyTorch code, the default Nsight Systems view shows CUDA kernels but not which Python code triggered them. Python backtrace profiling bridges this gap.

### The Problem
```
Timeline shows:
  [ampere_sgemm_128x64_tn]  <- What Python code called this?
  [volta_fp16_gemm]         <- Which layer triggered this?
```

### The Solution
With Python backtraces:
```
Timeline shows:
  [ampere_sgemm_128x64_tn]
    └── torch/nn/modules/linear.py:forward()
        └── model.py:attention_layer()
            └── train.py:training_step()
```

---

## Exercise Files

```
ex01-python-backtrace/
├── README.md
├── train_model.py      # Training script to profile
├── model.py            # Model with various layers
├── profile_basic.sh    # Profile without backtraces
├── profile_python.sh   # Profile with Python backtraces
├── analyze_results.py  # Python script to analyze
└── solutions/
    └── analysis_notes.md
```

---

## Part 1: Profile Without Python Backtraces

First, let's see what the timeline looks like without Python information:

```bash
# Basic profiling (no Python info)
nsys profile \
    --trace=cuda,nvtx \
    -o basic_profile \
    python train_model.py --epochs 2
```

### Observe:
1. Open in `nsys-ui` or use `nsys stats basic_profile.nsys-rep`
2. Notice kernels are identified only by CUDA names
3. Hard to tell which model layer triggered each kernel

---

## Part 2: Enable Python Backtraces

Now profile with Python backtrace collection:

```bash
# With Python backtraces
nsys profile \
    --trace=cuda,nvtx \
    --python-backtrace=cuda \
    --python-sampling=true \
    --python-sampling-frequency=1000 \
    -o python_profile \
    python train_model.py --epochs 2
```

### Key Flags:
| Flag | Purpose |
|------|---------|
| `--python-backtrace=cuda` | Capture Python stack on CUDA API calls |
| `--python-sampling=true` | Enable Python stack sampling |
| `--python-sampling-frequency=1000` | Sample every 1ms |

---

## Part 3: Analyze Python Correlation

### In Nsight Systems GUI:
1. Click on any CUDA kernel
2. Look at the "Python Backtrace" panel
3. See the full Python call stack

### Using CLI:
```bash
# View summary with Python info
nsys stats --report cuda_gpu_kern_sum python_profile.nsys-rep

# Export for custom analysis
nsys export --type=json python_profile.nsys-rep -o python_profile.json
```

---

## Part 4: Hands-On Exercise

### Task: Find the Slow Layer

The provided `train_model.py` has a performance issue in one layer. Use Python backtraces to find it.

```python
# train_model.py creates a model with:
# - Embedding layer
# - Multiple transformer blocks  
# - A "heavy" custom layer (intentionally slow)
# - Output projection

# Your task: Identify which layer is the bottleneck
```

### Steps:
1. Profile with Python backtraces
2. Find the longest-running kernel
3. Trace it back to Python code
4. Identify the problematic layer

### Expected Output:
```
Top CUDA Kernels by Time:
1. some_kernel (45% of GPU time)
   └── Python: model.py:HeavyLayer.forward() line 87
       └── Python: train_model.py:training_step() line 42

Conclusion: HeavyLayer is the bottleneck
```

---

## Part 5: Profile-Guided Optimization

Once you identify the slow layer:

1. **Document the finding**: Which layer? What kernel?
2. **Hypothesize the cause**: Memory-bound? Compute-bound? Wrong dtype?
3. **Propose optimization**: Fuse ops? Change algorithm? Use torch.compile?
4. **Re-profile to validate**

---

## Code Reference

### train_model.py
```python
"""Training script with intentional performance issues."""

import torch
import torch.nn as nn
from model import TransformerWithHeavyLayer


def train_epoch(model, dataloader, optimizer, device):
    """Single training epoch."""
    model.train()
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx >= 10:  # Short run for profiling
            break


def main():
    device = torch.device('cuda')
    model = TransformerWithHeavyLayer(
        vocab_size=10000,
        d_model=512,
        nhead=8,
        num_layers=4
    ).to(device)
    
    # Dummy data
    dataloader = create_dummy_dataloader(batch_size=32, seq_len=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(2):
        train_epoch(model, dataloader, optimizer, device)


if __name__ == "__main__":
    main()
```

---

## Common Issues

### "Python backtraces not showing"
- Ensure `--python-backtrace=cuda` flag is set
- Check Python version compatibility (3.8+)
- May need `--python-sampling=true` for better coverage

### "Too many frames"
- Use `--python-backtrace-depth=10` to limit depth
- Filter in GUI by module name

### "Performance overhead"
- Python backtraces add ~5-10% overhead
- Disable for production benchmarks
- Use for debugging/analysis only

---

## Success Criteria

You've completed this exercise when you can:
- [ ] Profile with Python backtraces enabled
- [ ] Correlate a CUDA kernel to its Python source
- [ ] Identify the bottleneck layer in the example model
- [ ] Explain why Python backtraces are useful for debugging

---

## Further Reading

- [Nsight Systems Python Profiling](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#python-profiling)
- [PyTorch Profiler Integration](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
