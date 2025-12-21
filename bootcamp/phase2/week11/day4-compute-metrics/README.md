# Day 4: Compute Metrics

## Learning Objectives

- Analyze compute utilization
- Understand instruction mix
- Diagnose compute inefficiencies

## Key Compute Metrics

### Throughput Metrics
```bash
ncu --metrics sm__throughput.avg_pct_of_peak_sustained_elapsed ./app
ncu --metrics sm__inst_executed.sum ./app
```

### Instruction Analysis
```bash
ncu --metrics sm__sass_thread_inst_executed_op_fadd_pred_on.sum ./app
ncu --metrics sm__sass_thread_inst_executed_op_fmul_pred_on.sum ./app
```

### Warp Execution
```bash
ncu --metrics smsp__warps_launched.sum,smsp__warps_eligible.avg ./app
```

## Key Metrics Table

| Metric | Meaning |
|--------|---------|
| sm__throughput | Overall SM utilization |
| sm__inst_executed | Instructions executed |
| sm__pipe_fma_cycles_active | FMA pipe activity |
| smsp__warp_issue_stalled* | Stall reasons |

## Build & Run

```bash
./build.sh
./build/compute_metrics
ncu --section ComputeWorkloadAnalysis ./build/compute_metrics
```
