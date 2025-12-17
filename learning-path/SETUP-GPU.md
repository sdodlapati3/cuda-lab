# Using T4 GPU for Hands-On CUDA Learning

This guide explains how to run the CUDA notebooks we created using your T4 GPU.

---

## 🎯 Option 1: Google Colab (Easiest - Free T4 GPU) ⭐ RECOMMENDED

**Start learning in 60 seconds - no setup required!**

### One-Click Launch Links

Click these links to open notebooks directly in Colab:

| Day | Notebook | Open in Colab |
|-----|----------|---------------|
| 1 | GPU Basics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapa/cuda-lab/blob/main/learning-path/week-01/day-1-gpu-basics.ipynb) |
| 2 | Thread Indexing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapa/cuda-lab/blob/main/learning-path/week-01/day-2-thread-indexing.ipynb) |
| 3 | Memory Basics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapa/cuda-lab/blob/main/learning-path/week-01/day-3-memory-basics.ipynb) |
| 4 | Error Handling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapa/cuda-lab/blob/main/learning-path/week-01/day-4-error-handling.ipynb) |

### Quick Start (3 Steps)

**Step 1:** Click any Colab link above

**Step 2:** Enable GPU: **Runtime → Change runtime type → T4 GPU → Save**

**Step 3:** Run this setup cell first:
```python
# Run this cell first in any notebook!
!pip install numba -q
import numpy as np
from numba import cuda

# Verify GPU
print("✅ GPU:", cuda.get_current_device().name.decode())
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

### Colab Pro Tips
- **Save work**: File → Save a copy in Drive
- **Sessions timeout** after ~90 min idle (free tier)
- **GPU limits**: ~12 hrs/day free, then may get CPU-only
- **Reconnect**: If disconnected, Runtime → Reconnect
- **Keep alive**: Keep browser tab active

---

## 🎯 Option 2: Your ODU HPC System

If you have access to GPU nodes on ODU's HPC:

### Step 1: Request a GPU Node

```bash
# Interactive session with T4 GPU
srun --partition=gpu --gres=gpu:t4:1 --time=04:00:00 --pty bash

# Or submit a job
sbatch --partition=gpu --gres=gpu:1 your_script.sh
```

### Step 2: Load CUDA Module

```bash
module load cuda/12.0
module load anaconda3/2023.09  # or your Python module
```

### Step 3: Create Conda Environment

```bash
conda create -n cuda-learning python=3.10 -y
conda activate cuda-learning
conda install numba cudatoolkit numpy jupyter -c conda-forge -y
```

### Step 4: Run Jupyter on GPU Node

```bash
# On the GPU node
jupyter notebook --no-browser --port=8888

# Then tunnel from your local machine:
ssh -L 8888:localhost:8888 your_username@turing.hpc.odu.edu
```

### Step 5: Open Notebooks

Navigate to `~/cuda-lab/learning-path/week-01/` and open notebooks.

---

## 🎯 Option 3: Local Machine with NVIDIA GPU

If you have a local NVIDIA GPU:

### Step 1: Install CUDA Toolkit

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit

# Verify
nvidia-smi
nvcc --version
```

### Step 2: Create Python Environment

```bash
python3 -m venv cuda-env
source cuda-env/bin/activate
pip install numba numpy jupyter matplotlib
```

### Step 3: Run Notebooks

```bash
cd ~/cuda-lab/learning-path/week-01
jupyter notebook
```

---

## 🎯 Option 4: Cloud GPU (AWS, GCP, Lambda Labs)

### AWS EC2 with T4

1. Launch `g4dn.xlarge` instance (1× T4 GPU, ~$0.50/hr)
2. Use Deep Learning AMI (CUDA pre-installed)
3. SSH and clone your repo

```bash
git clone https://github.com/sdodlapa/cuda-lab.git
cd cuda-lab
pip install numba numpy jupyter
jupyter notebook --no-browser --port=8888
```

### Lambda Labs (Recommended for simplicity)

1. Sign up at [lambdalabs.com](https://lambdalabs.com/)
2. Launch T4 instance (~$0.50/hr)
3. CUDA and Python pre-installed!

---

## 📋 Quick Verification Script

Run this to verify your CUDA setup works:

```python
#!/usr/bin/env python3
"""Verify CUDA setup is working"""

import numpy as np
from numba import cuda
import math

print("=" * 50)
print("CUDA SETUP VERIFICATION")
print("=" * 50)

# Check CUDA
if not cuda.is_available():
    print("❌ CUDA is NOT available!")
    print("   Make sure you have an NVIDIA GPU and drivers installed")
    exit(1)

print("✅ CUDA is available")

# Get device info
device = cuda.get_current_device()
print(f"✅ GPU: {device.name.decode()}")
print(f"   Compute Capability: {device.compute_capability}")
print(f"   Max threads/block: {device.MAX_THREADS_PER_BLOCK}")

# Test a simple kernel
@cuda.jit
def test_kernel(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] = idx * 2

# Run test
n = 1000
arr = cuda.device_array(n, dtype=np.float32)
threads = 256
blocks = math.ceil(n / threads)

test_kernel[blocks, threads](arr)
result = arr.copy_to_host()

if result[10] == 20.0:
    print("✅ Kernel execution successful")
else:
    print("❌ Kernel execution failed")

# Memory test
ctx = cuda.current_context()
free, total = ctx.get_memory_info()
print(f"✅ GPU Memory: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

print("=" * 50)
print("🚀 Ready for CUDA learning!")
print("=" * 50)
```

Save as `verify_cuda.py` and run:
```bash
python verify_cuda.py
```

---

## 🎓 Recommended Learning Workflow

### Daily Session (4-6 hours)

1. **Start GPU session** (Colab/HPC/Cloud)

2. **Open today's notebook**
   ```
   learning-path/week-01/day-X-topic.ipynb
   ```

3. **Work through sections:**
   - Read markdown explanations
   - Run code cells (Shift+Enter)
   - Complete TODO exercises
   - Experiment with modifications

4. **Save your work**
   - In Colab: Save to Drive
   - On HPC: Files persist in home directory
   - Cloud: Git commit regularly

5. **End of week: Take the quiz**
   ```
   learning-path/week-01/checkpoint-quiz.md
   ```

---

## 💡 Pro Tips

### For Colab Users
- Bookmark the notebooks in Drive for quick access
- Use `%%time` magic to time cells
- Mount Drive to save large datasets: 
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

### For HPC Users
- Request longer sessions for project work: `--time=08:00:00`
- Use screen/tmux to keep sessions alive
- Store large data in scratch, not home

### For Everyone
- **Commit progress daily** to GitHub
- Take notes on tricky concepts
- If kernel crashes, restart and run all cells
- GPU memory doesn't auto-clear - use `cuda.close()` or restart kernel

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No CUDA GPU" | Check GPU runtime (Colab) or `nvidia-smi` |
| "Out of memory" | Restart kernel, use smaller arrays |
| Numba import error | `pip install numba` |
| Kernel hangs | Add `cuda.synchronize()` calls |
| Slow first run | Normal - JIT compilation. Second run is fast |

---

## 📚 Next Steps

1. **Pick your platform** (Colab recommended to start)
2. **Run verification script** 
3. **Start Day 1 notebook**
4. **Complete all 4 days + quiz**
5. **Move to Week 2!**

Good luck! 🚀
