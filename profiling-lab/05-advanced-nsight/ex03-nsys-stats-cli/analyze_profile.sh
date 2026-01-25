#!/bin/bash
# Automated Nsight Systems profile analysis
# Usage: ./analyze_profile.sh <report.nsys-rep>

set -e

REPORT=$1

if [ -z "$REPORT" ]; then
    echo "Usage: $0 <report.nsys-rep>"
    echo ""
    echo "Generates comprehensive analysis from Nsight Systems report."
    exit 1
fi

if [ ! -f "$REPORT" ]; then
    echo "Error: Report file not found: $REPORT"
    exit 1
fi

BASENAME=$(basename "$REPORT" .nsys-rep)
OUTPUT_DIR="${BASENAME}_analysis"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Nsight Systems Profile Analysis"
echo "Report: $REPORT"
echo "Output: $OUTPUT_DIR/"
echo "=============================================="
echo ""

# 1. GPU Kernel Summary
echo ">>> GPU Kernel Summary (Top 10 by time):"
echo "-------------------------------------------"
nsys stats --report cuda_gpu_kern_sum \
    --format table \
    --timeunit ms \
    "$REPORT" 2>/dev/null | head -17
echo ""

# Export full kernel data
nsys stats --report cuda_gpu_kern_sum \
    --format csv \
    -o "$OUTPUT_DIR/kernels" \
    "$REPORT" 2>/dev/null

# 2. CUDA API Summary
echo ">>> CUDA API Summary:"
echo "-------------------------------------------"
nsys stats --report cuda_api_sum \
    --format table \
    --timeunit ms \
    "$REPORT" 2>/dev/null | head -15
echo ""

# Export API data
nsys stats --report cuda_api_sum \
    --format csv \
    -o "$OUTPUT_DIR/api" \
    "$REPORT" 2>/dev/null

# 3. Memory Operations
echo ">>> GPU Memory Operations:"
echo "-------------------------------------------"
nsys stats --report cuda_gpu_mem_time_sum \
    --format table \
    --timeunit ms \
    "$REPORT" 2>/dev/null | head -12
echo ""

# Export memory data
nsys stats --report cuda_gpu_mem_time_sum \
    --format csv \
    -o "$OUTPUT_DIR/memory" \
    "$REPORT" 2>/dev/null

# 4. NVTX Ranges (if present)
echo ">>> NVTX Ranges:"
echo "-------------------------------------------"
nsys stats --report nvtx_sum \
    --format table \
    --timeunit ms \
    "$REPORT" 2>/dev/null | head -12 || echo "No NVTX ranges found"
echo ""

# 5. OS Runtime (if traced)
echo ">>> OS Runtime Events:"
echo "-------------------------------------------"
nsys stats --report osrt_sum \
    --format table \
    --timeunit ms \
    "$REPORT" 2>/dev/null | head -12 || echo "OS runtime not traced"
echo ""

# 6. Calculate Summary Metrics
echo "=============================================="
echo "Summary Metrics"
echo "=============================================="

# Total kernel time
KERNEL_CSV="$OUTPUT_DIR/kernels_cuda_gpu_kern_sum.csv"
if [ -f "$KERNEL_CSV" ]; then
    TOTAL_KERNEL_NS=$(tail -n +2 "$KERNEL_CSV" | cut -d',' -f3 | paste -sd+ | bc 2>/dev/null || echo "0")
    TOTAL_KERNEL_MS=$(echo "scale=2; $TOTAL_KERNEL_NS / 1000000" | bc 2>/dev/null || echo "N/A")
    NUM_KERNELS=$(tail -n +2 "$KERNEL_CSV" | wc -l)
    
    echo "Total GPU Kernel Time: ${TOTAL_KERNEL_MS} ms"
    echo "Number of Kernel Types: $NUM_KERNELS"
fi

# Total memory time
MEM_CSV="$OUTPUT_DIR/memory_cuda_gpu_mem_time_sum.csv"
if [ -f "$MEM_CSV" ]; then
    TOTAL_MEM_NS=$(tail -n +2 "$MEM_CSV" | cut -d',' -f3 | paste -sd+ | bc 2>/dev/null || echo "0")
    TOTAL_MEM_MS=$(echo "scale=2; $TOTAL_MEM_NS / 1000000" | bc 2>/dev/null || echo "N/A")
    
    echo "Total Memory Op Time: ${TOTAL_MEM_MS} ms"
fi

# Top kernel
if [ -f "$KERNEL_CSV" ]; then
    TOP_KERNEL=$(tail -n +2 "$KERNEL_CSV" | sort -t',' -k3 -rn | head -1 | cut -d',' -f1)
    TOP_TIME=$(tail -n +2 "$KERNEL_CSV" | sort -t',' -k3 -rn | head -1 | cut -d',' -f3)
    TOP_TIME_MS=$(echo "scale=2; $TOP_TIME / 1000000" | bc 2>/dev/null || echo "N/A")
    
    echo ""
    echo "Hottest Kernel: $TOP_KERNEL"
    echo "Hottest Kernel Time: ${TOP_TIME_MS} ms"
fi

echo ""
echo "=============================================="
echo "Analysis complete!"
echo "Detailed CSVs saved in: $OUTPUT_DIR/"
echo "=============================================="
