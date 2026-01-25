#!/bin/bash
# Generate comparison reports between two profiles
# Usage: ./compare_profiles.sh before.nsys-rep after.nsys-rep

set -e

BEFORE="${1:-before.nsys-rep}"
AFTER="${2:-after.nsys-rep}"
REPORT_DIR="comparison_report"

echo "=================================================="
echo "Nsight Systems Profile Comparison"
echo "=================================================="
echo "Before: $BEFORE"
echo "After:  $AFTER"
echo ""

# Validate inputs
if [ ! -f "$BEFORE" ]; then
    echo "Error: Before profile not found: $BEFORE"
    exit 1
fi

if [ ! -f "$AFTER" ]; then
    echo "Error: After profile not found: $AFTER"
    exit 1
fi

# Create report directory
mkdir -p "$REPORT_DIR"

echo "Generating reports..."

# Export stats for both profiles
echo "1. Extracting kernel statistics..."
nsys stats -r gpukernsum "$BEFORE" --format csv > "$REPORT_DIR/before_kernels.csv" 2>/dev/null || true
nsys stats -r gpukernsum "$AFTER" --format csv > "$REPORT_DIR/after_kernels.csv" 2>/dev/null || true

echo "2. Extracting memory transfer statistics..."
nsys stats -r gpumemtimesum "$BEFORE" --format csv > "$REPORT_DIR/before_memcpy.csv" 2>/dev/null || true
nsys stats -r gpumemtimesum "$AFTER" --format csv > "$REPORT_DIR/after_memcpy.csv" 2>/dev/null || true

echo "3. Extracting CUDA API statistics..."
nsys stats -r cudaapisum "$BEFORE" --format csv > "$REPORT_DIR/before_api.csv" 2>/dev/null || true
nsys stats -r cudaapisum "$AFTER" --format csv > "$REPORT_DIR/after_api.csv" 2>/dev/null || true

echo "4. Generating comparison analysis..."

# Python comparison script
python3 << 'EOF'
import pandas as pd
import os
from pathlib import Path

report_dir = Path("comparison_report")

def safe_read_csv(path):
    """Read CSV with error handling."""
    try:
        df = pd.read_csv(path, on_bad_lines='skip')
        return df
    except Exception as e:
        return pd.DataFrame()

def compare_kernels():
    """Compare kernel performance."""
    before = safe_read_csv(report_dir / "before_kernels.csv")
    after = safe_read_csv(report_dir / "after_kernels.csv")
    
    if before.empty or after.empty:
        print("⚠️  Kernel comparison skipped (missing data)")
        return
    
    print("\n📊 KERNEL COMPARISON")
    print("=" * 60)
    
    # Find common time column
    time_col = None
    for col in ['Time (%)', 'Total Time (ns)', 'Avg (ns)']:
        if col in before.columns and col in after.columns:
            time_col = col
            break
    
    if time_col is None:
        print("Could not find comparable time columns")
        return
    
    # Find name column  
    name_col = None
    for col in ['Name', 'Kernel Name', 'Operation']:
        if col in before.columns:
            name_col = col
            break
    
    if name_col:
        before_total = before[time_col].sum() if time_col in before.columns else 0
        after_total = after[time_col].sum() if time_col in after.columns else 0
        
        if before_total > 0:
            speedup = before_total / after_total if after_total > 0 else float('inf')
            print(f"Overall kernel speedup: {speedup:.2f}x")
            
            if speedup > 1:
                print("✅ Performance IMPROVED")
            elif speedup < 1:
                print("❌ Performance DEGRADED")
            else:
                print("➖ No change")

def compare_memory():
    """Compare memory transfer performance."""
    before = safe_read_csv(report_dir / "before_memcpy.csv")
    after = safe_read_csv(report_dir / "after_memcpy.csv")
    
    if before.empty or after.empty:
        print("\n⚠️  Memory comparison skipped (missing data)")
        return
    
    print("\n📊 MEMORY TRANSFER COMPARISON")
    print("=" * 60)
    
    for df, name in [(before, "Before"), (after, "After")]:
        time_col = next((c for c in df.columns if 'Time' in c or 'time' in c), None)
        if time_col:
            total = df[time_col].sum()
            print(f"{name}: {total/1e6:.2f} ms total transfer time")

def main():
    print("\n" + "=" * 60)
    print("COMPARISON REPORT ANALYSIS")
    print("=" * 60)
    
    compare_kernels()
    compare_memory()
    
    print("\n" + "=" * 60)
    print("Reports saved to:", report_dir)
    print("=" * 60)

if __name__ == "__main__":
    main()
EOF

echo ""
echo "Reports generated in: $REPORT_DIR/"
ls -la "$REPORT_DIR/"
