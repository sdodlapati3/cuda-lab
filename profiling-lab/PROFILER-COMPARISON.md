# Profiling Tools Comparison: Single GPU to Multi-Node

## Quick Decision Matrix

```
┌─────────────────────┬───────────────────┬─────────────────────────────────┐
│ Question            │ Best Tool         │ Why                             │
├─────────────────────┼───────────────────┼─────────────────────────────────┤
│ "Where is the       │ Nsight Systems    │ Timeline shows CPU/GPU/IO gaps  │
│  bottleneck?"       │                   │                                 │
├─────────────────────┼───────────────────┼─────────────────────────────────┤
│ "Which operation    │ PyTorch Profiler  │ Operator-level breakdown        │
│  is slow?"          │                   │                                 │
├─────────────────────┼───────────────────┼─────────────────────────────────┤
│ "Is data loading    │ Nsight Systems    │ See DataLoader vs GPU overlap   │
│  the bottleneck?"   │                   │                                 │
├─────────────────────┼───────────────────┼─────────────────────────────────┤
│ "Is communication   │ Nsight Systems    │ NCCL trace + NVLink viz         │
│  the bottleneck?"   │                   │                                 │
├─────────────────────┼───────────────────┼─────────────────────────────────┤
│ "Memory issues?"    │ PyTorch Profiler  │ Tensor-level memory tracking    │
├─────────────────────┼───────────────────┼─────────────────────────────────┤
│ "Production         │ DCGM + Prometheus │ Real-time cluster metrics       │
│  monitoring?"       │                   │                                 │
└─────────────────────┴───────────────────┴─────────────────────────────────┘
```

## Tool Comparison by Scale

### Single GPU

| Tool | Pros | Cons | When to Use |
|------|------|------|-------------|
| **Nsight Systems** | Full system view, zero code changes | Learning curve for GUI | Initial bottleneck hunting |
| **PyTorch Profiler** | Python-native, TensorBoard integration | Less system-level detail | Optimizing specific ops |
| **Nsight Compute** | Kernel-level detail, roofline analysis | One kernel at a time | Deep CUDA optimization |

### Multi-GPU (Single Node)

| Tool | Pros | Cons | When to Use |
|------|------|------|-------------|
| **Nsight Systems** | NCCL tracing, NVLink visualization | Large trace files | Communication optimization |
| **PyTorch Profiler** | Rank-aware profiling | Limited comm. detail | Per-GPU op analysis |
| **DCGM** | Real-time metrics, GPU health | No timeline view | Production monitoring |

### Multi-Node

| Tool | Pros | Cons | When to Use |
|------|------|------|-------------|
| **Nsight Systems** | Per-node traces, MPI support | Need to correlate manually | Deep analysis |
| **DCGM Exporter** | Cluster-wide Prometheus metrics | Coarse granularity | Production/monitoring |
| **HPCToolkit** | Multi-node correlation built-in | Complex setup | HPC research |

---

## 1. Nsight Systems

### Single GPU
```bash
nsys profile -o single_gpu \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    python train.py
```

### Multi-GPU (Single Node)
```bash
nsys profile -o multi_gpu \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    torchrun --nproc_per_node=4 train.py
```

### Multi-Node (SLURM)
```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8

# Profile each node separately
srun --ntasks-per-node=1 bash -c '
    nsys profile -o profile_node_${SLURM_NODEID} \
        --trace=cuda,nvtx,osrt,mpi \
        python -m torch.distributed.run \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=8 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train.py
'
```

### Key Nsight Systems Flags
```bash
--trace=cuda           # CUDA API calls
--trace=nvtx           # NVTX annotations (custom markers)
--trace=osrt           # OS runtime (I/O, syscalls)
--trace=cudnn          # cuDNN operations
--trace=cublas         # cuBLAS operations
--trace=mpi            # MPI calls (multi-node)
--stats=true           # Generate summary stats
--capture-range=cudaProfilerApi  # Use cudaProfilerStart/Stop
```

---

## 2. PyTorch Profiler

### Single GPU
```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for batch in dataloader:
        train_step(batch)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Multi-GPU / Multi-Node (Distributed)
```python
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

# Same code works for all distributed configurations!
profiler = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=3, active=10, repeat=1),
    on_trace_ready=tensorboard_trace_handler(
        f"./logs/rank_{torch.distributed.get_rank()}"
    ),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
)

profiler.start()
for step, batch in enumerate(dataloader):
    train_step(batch)
    profiler.step()
profiler.stop()
```

### View Results
```bash
tensorboard --logdir=./logs
```

---

## 3. Combined Approach (Recommended)

### Using unified_profiler.py

```python
# In your training script, add NVTX markers
from profiling_lab.utils.unified_profiler import NVTXAnnotator, ProfilerWrapper

annotator = NVTXAnnotator()
profiler = ProfilerWrapper()

for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        with profiler.profile_step(step):
            # NVTX markers show up in Nsight Systems
            with annotator.range("data_transfer"):
                batch = {k: v.cuda() for k, v in batch.items()}
            
            with annotator.range("forward"):
                output = model(batch)
            
            with annotator.range("backward"):
                loss.backward()
            
            with annotator.range("optimizer"):
                optimizer.step()

profiler.finish()
```

### Run with profiling
```bash
# Profiles with both Nsight Systems AND PyTorch Profiler
python unified_profiler.py --mode multi-gpu --gpus 4 train.py

# View results:
# - Nsight Systems: nsys-ui ./profiling_results/*.nsys-rep
# - PyTorch: tensorboard --logdir=./profiling_results/
```

---

## 4. Specialized Tools

### DCGM (Data Center GPU Manager) - Production Monitoring

```bash
# Install
pip install pynvml

# Enable DCGM metrics
dcgmi dmon -e 100,101,140,150,155,156,203,204,1001,1002,1003,1004

# With Prometheus
docker run -d --gpus all \
    -p 9400:9400 \
    nvidia/dcgm-exporter
```

### Nsight Compute - Kernel Deep Dive

```bash
# Profile specific kernels
ncu --set full -o kernel_report python train.py

# Target specific kernel
ncu --kernel-name "volta_sgemm" -o gemm_report python train.py
```

---

## 5. End-to-End Pipeline Profiling

For profiling the complete pipeline (data → inference → save):

```python
import torch
from torch.profiler import profile, ProfilerActivity

def profile_full_pipeline():
    """Profile entire pipeline from data staging to result saving."""
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        # 1. Data staging
        torch.cuda.nvtx.range_push("data_staging")
        data = load_data()
        data = preprocess(data)
        data = data.cuda()
        torch.cuda.nvtx.range_pop()
        
        # 2. Inference
        torch.cuda.nvtx.range_push("inference")
        with torch.no_grad():
            output = model(data)
        torch.cuda.synchronize()  # Ensure timing accuracy
        torch.cuda.nvtx.range_pop()
        
        # 3. Postprocess & Save
        torch.cuda.nvtx.range_push("postprocess_save")
        results = postprocess(output.cpu())
        save_results(results)
        torch.cuda.nvtx.range_pop()
    
    # Print breakdown
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    ))
```

---

## Summary Recommendation

| Scale | Primary Tool | Secondary Tool | What They Show Together |
|-------|--------------|----------------|-------------------------|
| **Single GPU** | Nsight Systems | PyTorch Profiler | WHERE + WHAT is slow |
| **Multi-GPU** | Nsight Systems | PyTorch Profiler | Comm overhead + Op breakdown |
| **Multi-Node** | Nsight Systems | DCGM | Per-node detail + Cluster health |

**Pro tip**: Always start with Nsight Systems to find WHERE the bottleneck is, then use PyTorch Profiler to understand WHAT operations within that region are slow.
