# Exercise 07: Comparison Reports

## Learning Objectives
- Generate comparison reports between profiles
- Validate optimization improvements
- Track performance regressions in CI/CD
- Create before/after analysis workflows

## Background

When optimizing code, you need to validate that changes actually improve performance. Nsight Systems comparison features help quantify improvements.

---

## Part 1: Basic Comparison

### Step 1: Profile Baseline
```bash
git checkout main
nsys profile -o baseline python train.py
```

### Step 2: Profile Optimized Version
```bash
git checkout optimization-branch
nsys profile -o optimized python train.py
```

### Step 3: Compare
```bash
# GUI comparison
nsys-ui baseline.nsys-rep optimized.nsys-rep

# CLI comparison (newer nsys versions)
nsys compare baseline.nsys-rep optimized.nsys-rep
```

---

## Part 2: Metrics to Compare

### Timing Metrics
| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Total GPU Time | 1000 ms | 800 ms | -20% ✅ |
| Top Kernel Time | 400 ms | 300 ms | -25% ✅ |
| Memory Ops Time | 200 ms | 100 ms | -50% ✅ |

### Utilization Metrics
| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| GPU Utilization | 65% | 85% | +20% ✅ |
| GPU Idle Time | 350 ms | 150 ms | -57% ✅ |

---

## Part 3: Automated Comparison Script

```python
#!/usr/bin/env python3
"""compare_profiles.py - Automated profile comparison"""

import sqlite3
import pandas as pd
import argparse
import sys


def load_kernel_times(db_path: str) -> pd.DataFrame:
    """Load kernel timing data from SQLite export."""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        s.value AS kernel_name,
        COUNT(*) AS invocations,
        SUM(k.end - k.start) AS total_time_ns,
        AVG(k.end - k.start) AS avg_time_ns
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName = s.id
    GROUP BY s.value
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def compare_profiles(baseline_db: str, optimized_db: str, threshold: float = 0.05):
    """
    Compare two profiles and report changes.
    
    Args:
        baseline_db: Baseline SQLite database
        optimized_db: Optimized SQLite database  
        threshold: Significance threshold (0.05 = 5%)
    
    Returns:
        exit_code: 0 if no regressions, 1 if regressions found
    """
    baseline = load_kernel_times(baseline_db)
    optimized = load_kernel_times(optimized_db)
    
    # Merge on kernel name
    merged = pd.merge(
        baseline, optimized,
        on='kernel_name',
        suffixes=('_baseline', '_optimized'),
        how='outer'
    )
    
    # Fill NaN (new or removed kernels)
    merged = merged.fillna(0)
    
    # Calculate changes
    merged['time_change'] = merged['total_time_ns_optimized'] - merged['total_time_ns_baseline']
    merged['time_change_pct'] = (merged['time_change'] / 
                                  merged['total_time_ns_baseline'].replace(0, 1)) * 100
    
    # Summary
    total_baseline = merged['total_time_ns_baseline'].sum()
    total_optimized = merged['total_time_ns_optimized'].sum()
    overall_change = (total_optimized - total_baseline) / total_baseline * 100
    
    print("=" * 70)
    print("PROFILE COMPARISON REPORT")
    print("=" * 70)
    print(f"Baseline total GPU time:  {total_baseline/1e6:.2f} ms")
    print(f"Optimized total GPU time: {total_optimized/1e6:.2f} ms")
    print(f"Overall change: {overall_change:+.1f}%")
    print()
    
    # Find regressions (slower by more than threshold)
    regressions = merged[merged['time_change_pct'] > threshold * 100].copy()
    regressions = regressions.sort_values('time_change', ascending=False)
    
    # Find improvements
    improvements = merged[merged['time_change_pct'] < -threshold * 100].copy()
    improvements = improvements.sort_values('time_change')
    
    if len(improvements) > 0:
        print(">>> IMPROVEMENTS")
        print("-" * 70)
        for _, row in improvements.head(10).iterrows():
            print(f"  {row['kernel_name'][:50]}: {row['time_change_pct']:+.1f}%")
        print()
    
    if len(regressions) > 0:
        print(">>> REGRESSIONS ⚠️")
        print("-" * 70)
        for _, row in regressions.head(10).iterrows():
            print(f"  {row['kernel_name'][:50]}: {row['time_change_pct']:+.1f}%")
        print()
    
    # Verdict
    print("=" * 70)
    if overall_change < -threshold * 100:
        print(f"✅ PASSED: Overall {overall_change:.1f}% improvement")
        return 0
    elif overall_change > threshold * 100:
        print(f"❌ FAILED: Overall {overall_change:.1f}% regression")
        return 1
    else:
        print(f"⚡ NEUTRAL: Change within threshold ({overall_change:.1f}%)")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Nsight Systems profiles")
    parser.add_argument("baseline", help="Baseline SQLite database")
    parser.add_argument("optimized", help="Optimized SQLite database")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Significance threshold (default: 0.05 = 5%%)")
    args = parser.parse_args()
    
    sys.exit(compare_profiles(args.baseline, args.optimized, args.threshold))
```

---

## Part 4: CI/CD Integration

### GitHub Actions Example:
```yaml
name: Performance Regression Check

on:
  pull_request:
    branches: [main]

jobs:
  perf-check:
    runs-on: [self-hosted, gpu]
    
    steps:
    - uses: actions/checkout@v3
      with:
        fetch-depth: 0
    
    - name: Profile Baseline (main)
      run: |
        git checkout main
        nsys profile -o baseline.nsys-rep python benchmark.py
        nsys export --type=sqlite -o baseline.sqlite baseline.nsys-rep
    
    - name: Profile PR Branch
      run: |
        git checkout ${{ github.head_ref }}
        nsys profile -o pr.nsys-rep python benchmark.py
        nsys export --type=sqlite -o pr.sqlite pr.nsys-rep
    
    - name: Compare Profiles
      run: |
        python compare_profiles.py baseline.sqlite pr.sqlite --threshold 0.05
```

---

## Part 5: Hands-On Exercise

### Task: Validate an Optimization

1. **Profile slow version:**
```bash
python train_slow.py &  # Use slow model
nsys profile -o slow python train_slow.py
```

2. **Profile optimized version:**
```bash
python train_fast.py &  # Use optimized model
nsys profile -o fast python train_fast.py
```

3. **Export to SQLite:**
```bash
nsys export --type=sqlite -o slow.sqlite slow.nsys-rep
nsys export --type=sqlite -o fast.sqlite fast.nsys-rep
```

4. **Run comparison:**
```bash
python compare_profiles.py slow.sqlite fast.sqlite
```

### Expected Output:
```
PROFILE COMPARISON REPORT
======================================================================
Baseline total GPU time:  1234.56 ms
Optimized total GPU time: 890.12 ms
Overall change: -27.9%

>>> IMPROVEMENTS
----------------------------------------------------------------------
  ampere_sgemm_128x64_tn: -35.2%
  volta_fp16_gemm: -22.1%

======================================================================
✅ PASSED: Overall -27.9% improvement
```

---

## Part 6: Best Practices

### 1. Use Consistent Conditions
- Same batch size
- Same number of iterations
- Same GPU temperature (warmup!)
- Same data (deterministic)

### 2. Multiple Runs
```bash
for i in 1 2 3; do
    nsys profile -o baseline_$i python train.py
done
# Average results
```

### 3. Statistical Significance
- Run multiple times
- Calculate mean and stddev
- Check if difference > 2σ

---

## Success Criteria

- [ ] Can generate comparison reports
- [ ] Created automated comparison script
- [ ] Integrated into CI/CD workflow
- [ ] Understand statistical significance
