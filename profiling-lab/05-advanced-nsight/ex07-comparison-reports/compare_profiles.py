#!/usr/bin/env python3
"""
Python script for comprehensive profile comparison.
Generates detailed reports comparing before/after optimization profiles.

Usage:
    python compare_profiles.py before.sqlite after.sqlite [--output report.md]
"""

import sqlite3
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List


def connect_db(path: str) -> sqlite3.Connection:
    """Connect to SQLite database."""
    return sqlite3.connect(path)


def get_total_time(conn: sqlite3.Connection) -> float:
    """Get total profile time in nanoseconds."""
    query = """
    SELECT MAX(end) - MIN(start) AS total
    FROM CUPTI_ACTIVITY_KIND_KERNEL
    """
    result = pd.read_sql_query(query, conn)
    return result['total'].iloc[0] if len(result) > 0 else 0


def get_kernel_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get kernel summary statistics."""
    query = """
    SELECT 
        s.value AS kernel_name,
        COUNT(*) AS calls,
        SUM(k.end - k.start) AS total_time_ns,
        AVG(k.end - k.start) AS avg_time_ns,
        MIN(k.end - k.start) AS min_time_ns,
        MAX(k.end - k.start) AS max_time_ns
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName = s.id
    GROUP BY s.value
    ORDER BY total_time_ns DESC
    """
    try:
        return pd.read_sql_query(query, conn)
    except:
        return pd.DataFrame()


def get_memcpy_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get memory copy summary statistics."""
    query = """
    SELECT 
        copyKind,
        COUNT(*) AS count,
        SUM(bytes) AS total_bytes,
        SUM(end - start) AS total_time_ns,
        AVG(bytes / NULLIF(end - start, 0) * 1e9) AS throughput_bytes_per_sec
    FROM CUPTI_ACTIVITY_KIND_MEMCPY
    GROUP BY copyKind
    """
    try:
        return pd.read_sql_query(query, conn)
    except:
        return pd.DataFrame()


def compare_profiles(before_path: str, after_path: str) -> Dict:
    """Compare two profiles and return comparison results."""
    before = connect_db(before_path)
    after = connect_db(after_path)
    
    results = {}
    
    # Total time comparison
    before_total = get_total_time(before)
    after_total = get_total_time(after)
    
    results['total_time'] = {
        'before_ms': before_total / 1e6,
        'after_ms': after_total / 1e6,
        'speedup': before_total / after_total if after_total > 0 else 0,
        'reduction_pct': (1 - after_total / before_total) * 100 if before_total > 0 else 0
    }
    
    # Kernel comparison
    before_kernels = get_kernel_summary(before)
    after_kernels = get_kernel_summary(after)
    
    if not before_kernels.empty and not after_kernels.empty:
        merged = pd.merge(
            before_kernels[['kernel_name', 'calls', 'total_time_ns', 'avg_time_ns']],
            after_kernels[['kernel_name', 'calls', 'total_time_ns', 'avg_time_ns']],
            on='kernel_name',
            how='outer',
            suffixes=('_before', '_after')
        ).fillna(0)
        
        merged['speedup'] = merged['avg_time_ns_before'] / merged['avg_time_ns_after'].replace(0, 1)
        merged['improvement_pct'] = (1 - merged['avg_time_ns_after'] / merged['avg_time_ns_before'].replace(0, 1)) * 100
        
        results['kernels'] = merged.to_dict('records')
        
        # Top improvements
        improved = merged[merged['speedup'] > 1.1].nlargest(5, 'speedup')
        results['top_improvements'] = improved[['kernel_name', 'speedup', 'improvement_pct']].to_dict('records')
        
        # Regressions
        regressed = merged[merged['speedup'] < 0.9].nsmallest(5, 'speedup')
        results['regressions'] = regressed[['kernel_name', 'speedup', 'improvement_pct']].to_dict('records')
    
    # Memory comparison
    before_mem = get_memcpy_summary(before)
    after_mem = get_memcpy_summary(after)
    
    if not before_mem.empty and not after_mem.empty:
        results['memory'] = {
            'before': before_mem.to_dict('records'),
            'after': after_mem.to_dict('records')
        }
    
    before.close()
    after.close()
    
    return results


def generate_markdown_report(results: Dict, before_path: str, after_path: str) -> str:
    """Generate a markdown report from comparison results."""
    report = []
    report.append("# Profile Comparison Report")
    report.append(f"\n**Generated:** {datetime.now().isoformat()}")
    report.append(f"\n**Before:** `{before_path}`")
    report.append(f"\n**After:** `{after_path}`")
    
    # Overall summary
    report.append("\n## Overall Summary")
    report.append("")
    total = results.get('total_time', {})
    report.append(f"| Metric | Before | After | Change |")
    report.append(f"|--------|--------|-------|--------|")
    report.append(f"| Total Time | {total.get('before_ms', 0):.2f} ms | {total.get('after_ms', 0):.2f} ms | **{total.get('speedup', 0):.2f}x** speedup |")
    
    # Verdict
    speedup = total.get('speedup', 1)
    if speedup > 1.1:
        report.append(f"\n✅ **IMPROVED** - {total.get('reduction_pct', 0):.1f}% reduction in execution time")
    elif speedup < 0.9:
        report.append(f"\n❌ **REGRESSED** - {abs(total.get('reduction_pct', 0)):.1f}% increase in execution time")
    else:
        report.append(f"\n➖ **NEUTRAL** - No significant change")
    
    # Top improvements
    if results.get('top_improvements'):
        report.append("\n## Top Kernel Improvements")
        report.append("")
        report.append("| Kernel | Speedup | Improvement |")
        report.append("|--------|---------|-------------|")
        for k in results['top_improvements']:
            name = k['kernel_name'][:50] + "..." if len(k['kernel_name']) > 50 else k['kernel_name']
            report.append(f"| `{name}` | {k['speedup']:.2f}x | {k['improvement_pct']:.1f}% |")
    
    # Regressions
    if results.get('regressions'):
        report.append("\n## ⚠️ Regressions")
        report.append("")
        report.append("| Kernel | Slowdown | Regression |")
        report.append("|--------|----------|------------|")
        for k in results['regressions']:
            name = k['kernel_name'][:50] + "..." if len(k['kernel_name']) > 50 else k['kernel_name']
            report.append(f"| `{name}` | {1/k['speedup']:.2f}x | {abs(k['improvement_pct']):.1f}% |")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Compare Nsight Systems profiles")
    parser.add_argument("before", help="SQLite database from before optimization")
    parser.add_argument("after", help="SQLite database from after optimization")
    parser.add_argument("--output", "-o", help="Output markdown file")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = parser.parse_args()
    
    print(f"Comparing profiles...")
    print(f"  Before: {args.before}")
    print(f"  After:  {args.after}")
    
    results = compare_profiles(args.before, args.after)
    
    if args.json:
        import json
        print(json.dumps(results, indent=2, default=str))
    else:
        report = generate_markdown_report(results, args.before, args.after)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {args.output}")
        else:
            print("\n" + report)


if __name__ == "__main__":
    main()
