# Exercise 03: nsys stats CLI Analysis

## Learning Objectives
- Master `nsys stats` command-line analysis
- Generate reports without GUI
- Extract key metrics programmatically
- Create automated profiling pipelines

## Background

The Nsight Systems GUI is powerful but not always available (remote servers, CI/CD). The `nsys stats` CLI provides comprehensive analysis capabilities.

---

## Part 1: Basic Stats Report

```bash
# Generate summary statistics
nsys stats report.nsys-rep

# This produces:
# - CUDA API Summary
# - CUDA Kernel Summary  
# - CUDA Memory Operation Summary
# - OS Runtime Summary
```

### Output Structure:
```
** CUDA API Summary (cuda_api_sum) **

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)    Med (ns)  Name
 --------  ---------------  ---------  -----------  ---------  ----
     45.2      1234567890       1000     1234567     1200000   cudaLaunchKernel
     30.1       823456789        500     1646913     1500000   cudaMemcpyAsync
...
```

---

## Part 2: Specific Report Types

```bash
# List available reports
nsys stats --help-reports

# Key reports:
nsys stats --report cuda_gpu_kern_sum report.nsys-rep   # GPU kernels
nsys stats --report cuda_api_sum report.nsys-rep        # CUDA API calls
nsys stats --report cuda_gpu_mem_time_sum report.nsys-rep  # Memory ops
nsys stats --report osrt_sum report.nsys-rep            # OS runtime
nsys stats --report nvtx_sum report.nsys-rep            # NVTX ranges
```

---

## Part 3: Output Formats

### CSV for Scripting
```bash
nsys stats --report cuda_gpu_kern_sum --format csv -o kernels.csv report.nsys-rep
```

### JSON for Parsing
```bash
nsys stats --report cuda_gpu_kern_sum --format json -o kernels.json report.nsys-rep
```

### Column Selection
```bash
nsys stats --report cuda_gpu_kern_sum \
    --format csv \
    --columns "Name,Instances,Total Time (%)" \
    report.nsys-rep
```

---

## Part 4: Filtering and Sorting

```bash
# Top 10 kernels by time
nsys stats --report cuda_gpu_kern_sum \
    --format table \
    --timeunit ms \
    report.nsys-rep | head -20

# Filter by kernel name
nsys stats --report cuda_gpu_kern_sum \
    --filter "Name LIKE '%gemm%'" \
    report.nsys-rep
```

---

## Part 5: Multiple Reports at Once

```bash
# Generate comprehensive analysis
nsys stats \
    --report cuda_api_sum \
    --report cuda_gpu_kern_sum \
    --report cuda_gpu_mem_time_sum \
    --report nvtx_sum \
    --format csv \
    -o analysis \
    report.nsys-rep

# Creates: analysis_cuda_api_sum.csv, analysis_cuda_gpu_kern_sum.csv, etc.
```

---

## Part 6: Hands-On Exercise

### Task: Create an Automated Analysis Script

Create `analyze_profile.sh`:

```bash
#!/bin/bash
# Automated profile analysis script

REPORT=$1

if [ -z "$REPORT" ]; then
    echo "Usage: $0 <report.nsys-rep>"
    exit 1
fi

echo "=== Profile Analysis: $REPORT ==="
echo ""

# 1. Top GPU Kernels
echo ">>> Top 5 GPU Kernels by Time:"
nsys stats --report cuda_gpu_kern_sum \
    --format table \
    --timeunit ms \
    "$REPORT" 2>/dev/null | head -12

echo ""

# 2. CUDA API breakdown
echo ">>> CUDA API Summary:"
nsys stats --report cuda_api_sum \
    --format table \
    --timeunit ms \
    "$REPORT" 2>/dev/null | head -12

echo ""

# 3. Memory operations
echo ">>> Memory Operations:"
nsys stats --report cuda_gpu_mem_time_sum \
    --format table \
    --timeunit ms \
    "$REPORT" 2>/dev/null | head -10

echo ""

# 4. Calculate key metrics
echo ">>> Key Metrics:"

# Extract total GPU kernel time
KERNEL_TIME=$(nsys stats --report cuda_gpu_kern_sum --format csv "$REPORT" 2>/dev/null | \
    tail -n +2 | cut -d',' -f3 | paste -sd+ | bc 2>/dev/null || echo "N/A")

# Extract total memory time
MEM_TIME=$(nsys stats --report cuda_gpu_mem_time_sum --format csv "$REPORT" 2>/dev/null | \
    tail -n +2 | cut -d',' -f3 | paste -sd+ | bc 2>/dev/null || echo "N/A")

echo "Total GPU Kernel Time: $KERNEL_TIME ns"
echo "Total Memory Op Time: $MEM_TIME ns"

# 5. Export for detailed analysis
echo ""
echo ">>> Exporting detailed CSVs..."
nsys stats --report cuda_gpu_kern_sum --format csv -o kernels "$REPORT" 2>/dev/null
nsys stats --report cuda_api_sum --format csv -o api "$REPORT" 2>/dev/null
echo "Exported to kernels_cuda_gpu_kern_sum.csv, api_cuda_api_sum.csv"
```

### Run It:
```bash
chmod +x analyze_profile.sh
./analyze_profile.sh my_profile.nsys-rep
```

---

## Part 7: CI/CD Integration

### GitHub Actions Example:
```yaml
- name: Profile Performance
  run: |
    nsys profile -o ci_profile python benchmark.py
    
- name: Extract Metrics
  run: |
    # Extract total kernel time
    KERNEL_TIME=$(nsys stats --report cuda_gpu_kern_sum --format csv ci_profile.nsys-rep | \
      awk -F',' 'NR>1 {sum+=$3} END {print sum}')
    echo "kernel_time=$KERNEL_TIME" >> $GITHUB_OUTPUT
    
- name: Check Regression
  run: |
    if [ "${{ steps.metrics.outputs.kernel_time }}" -gt "$THRESHOLD" ]; then
      echo "Performance regression detected!"
      exit 1
    fi
```

---

## Reference: All Report Types

| Report | Description |
|--------|-------------|
| `cuda_api_sum` | CUDA API call summary |
| `cuda_gpu_kern_sum` | GPU kernel summary |
| `cuda_gpu_mem_time_sum` | GPU memory operation time |
| `cuda_gpu_mem_size_sum` | GPU memory operation size |
| `nvtx_sum` | NVTX range summary |
| `osrt_sum` | OS runtime summary |
| `dx11_pix_sum` | DirectX 11 summary |
| `dx12_gpu_marker_sum` | DirectX 12 markers |
| `vulkan_gpu_marker_sum` | Vulkan markers |
| `opengl_gpu_marker_sum` | OpenGL markers |

---

## Success Criteria

- [ ] Can generate stats reports from CLI
- [ ] Can export to CSV/JSON formats
- [ ] Created automated analysis script
- [ ] Understand all major report types
