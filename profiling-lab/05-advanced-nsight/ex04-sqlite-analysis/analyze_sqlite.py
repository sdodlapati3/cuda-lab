#!/usr/bin/env python3
"""
Custom Nsight Systems analysis using SQLite export.

Usage:
    # First export to SQLite
    nsys export --type=sqlite -o profile.sqlite profile.nsys-rep
    
    # Then run analysis
    python analyze_sqlite.py profile.sqlite
    python analyze_sqlite.py profile.sqlite -o results
"""

import sqlite3
import pandas as pd
from pathlib import Path
import argparse
import json


def load_profile_db(db_path: str) -> sqlite3.Connection:
    """Load SQLite profile database."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    return conn


def list_tables(conn: sqlite3.Connection) -> list:
    """List all tables in the database."""
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [row[0] for row in cursor.fetchall()]


def get_kernel_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get GPU kernel summary."""
    query = """
    SELECT 
        s.value AS kernel_name,
        COUNT(*) AS invocations,
        SUM(k.end - k.start) AS total_time_ns,
        AVG(k.end - k.start) AS avg_time_ns,
        MIN(k.end - k.start) AS min_time_ns,
        MAX(k.end - k.start) AS max_time_ns,
        AVG(k.gridX * k.gridY * k.gridZ) AS avg_grid_size,
        AVG(k.blockX * k.blockY * k.blockZ) AS avg_block_size,
        SUM(k.staticSharedMemory + k.dynamicSharedMemory) AS total_shared_mem
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName = s.id
    GROUP BY s.value
    ORDER BY total_time_ns DESC
    """
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Warning: Could not query kernels: {e}")
        return pd.DataFrame()


def get_memory_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get memory transfer summary."""
    query = """
    SELECT 
        CASE copyKind
            WHEN 1 THEN 'HtoD'
            WHEN 2 THEN 'DtoH'
            WHEN 8 THEN 'DtoD'
            WHEN 10 THEN 'HtoA'
            WHEN 11 THEN 'AtoH'
            ELSE 'Other'
        END AS transfer_type,
        COUNT(*) AS count,
        SUM(bytes) AS total_bytes,
        SUM(end - start) AS total_time_ns,
        AVG(bytes) AS avg_bytes,
        MIN(bytes) AS min_bytes,
        MAX(bytes) AS max_bytes
    FROM CUPTI_ACTIVITY_KIND_MEMCPY
    GROUP BY copyKind
    ORDER BY total_time_ns DESC
    """
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Warning: Could not query memory: {e}")
        return pd.DataFrame()


def get_api_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get CUDA API call summary."""
    query = """
    SELECT 
        s.value AS api_name,
        COUNT(*) AS calls,
        SUM(r.end - r.start) AS total_time_ns,
        AVG(r.end - r.start) AS avg_time_ns
    FROM CUPTI_ACTIVITY_KIND_RUNTIME r
    JOIN StringIds s ON r.nameId = s.id
    GROUP BY s.value
    ORDER BY total_time_ns DESC
    """
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Warning: Could not query API: {e}")
        return pd.DataFrame()


def get_gpu_utilization(conn: sqlite3.Connection) -> dict:
    """Calculate GPU utilization metrics."""
    query = """
    WITH bounds AS (
        SELECT MIN(start) AS first_start, MAX(end) AS last_end
        FROM CUPTI_ACTIVITY_KIND_KERNEL
    ),
    kernel_time AS (
        SELECT SUM(end - start) AS busy_time
        FROM CUPTI_ACTIVITY_KIND_KERNEL
    )
    SELECT 
        kt.busy_time,
        b.last_end - b.first_start AS total_time,
        CAST(kt.busy_time AS FLOAT) / NULLIF(b.last_end - b.first_start, 0) * 100 AS utilization_pct
    FROM kernel_time kt, bounds b
    """
    try:
        result = pd.read_sql_query(query, conn)
        if len(result) > 0 and result['total_time'].iloc[0]:
            return {
                'busy_time_ns': int(result['busy_time'].iloc[0]),
                'total_time_ns': int(result['total_time'].iloc[0]),
                'utilization_pct': float(result['utilization_pct'].iloc[0])
            }
    except Exception as e:
        print(f"Warning: Could not calculate utilization: {e}")
    return {}


def get_nvtx_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get NVTX range summary if available."""
    query = """
    SELECT 
        s.value AS range_name,
        COUNT(*) AS count,
        SUM(n.endTime - n.startTime) AS total_time_ns,
        AVG(n.endTime - n.startTime) AS avg_time_ns
    FROM NVTX_EVENTS n
    JOIN StringIds s ON n.textId = s.id
    WHERE n.eventType = 59
    GROUP BY s.value
    ORDER BY total_time_ns DESC
    """
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        return pd.DataFrame()


def get_gpu_info(conn: sqlite3.Connection) -> dict:
    """Get GPU information."""
    query = "SELECT * FROM TARGET_INFO_GPU LIMIT 1"
    try:
        result = pd.read_sql_query(query, conn)
        if len(result) > 0:
            return result.iloc[0].to_dict()
    except:
        pass
    return {}


def compute_derived_metrics(kernels: pd.DataFrame, memory: pd.DataFrame, util: dict) -> dict:
    """Compute derived performance metrics."""
    metrics = {}
    
    if len(kernels) > 0:
        metrics['total_kernel_time_ms'] = kernels['total_time_ns'].sum() / 1e6
        metrics['num_kernel_types'] = len(kernels)
        metrics['total_kernel_invocations'] = int(kernels['invocations'].sum())
        metrics['top_kernel'] = kernels.iloc[0]['kernel_name'] if len(kernels) > 0 else None
        metrics['top_kernel_pct'] = (kernels.iloc[0]['total_time_ns'] / 
                                      kernels['total_time_ns'].sum() * 100) if len(kernels) > 0 else 0
    
    if len(memory) > 0:
        metrics['total_memory_time_ms'] = memory['total_time_ns'].sum() / 1e6
        metrics['total_data_transferred_GB'] = memory['total_bytes'].sum() / 1e9
        
        # Calculate effective bandwidth
        if metrics['total_memory_time_ms'] > 0:
            metrics['avg_bandwidth_GBps'] = (metrics['total_data_transferred_GB'] / 
                                              (metrics['total_memory_time_ms'] / 1e3))
    
    if util:
        metrics['gpu_utilization_pct'] = util.get('utilization_pct', 0)
    
    return metrics


def generate_report(db_path: str, output_path: str = None, format: str = 'text'):
    """Generate comprehensive analysis report."""
    conn = load_profile_db(db_path)
    
    # Gather data
    tables = list_tables(conn)
    gpu_info = get_gpu_info(conn)
    kernels = get_kernel_summary(conn)
    memory = get_memory_summary(conn)
    api = get_api_summary(conn)
    util = get_gpu_utilization(conn)
    nvtx = get_nvtx_summary(conn)
    metrics = compute_derived_metrics(kernels, memory, util)
    
    if format == 'json':
        # JSON output
        report = {
            'database': db_path,
            'gpu_info': gpu_info,
            'metrics': metrics,
            'kernels': kernels.to_dict('records') if len(kernels) > 0 else [],
            'memory': memory.to_dict('records') if len(memory) > 0 else [],
            'utilization': util
        }
        if output_path:
            with open(f"{output_path}.json", 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Report saved to {output_path}.json")
        else:
            print(json.dumps(report, indent=2, default=str))
        return
    
    # Text output
    print("=" * 70)
    print("NSIGHT SYSTEMS CUSTOM ANALYSIS")
    print(f"Database: {db_path}")
    print(f"Tables found: {len(tables)}")
    print("=" * 70)
    
    # GPU Info
    if gpu_info:
        print(f"\n>>> GPU: {gpu_info.get('name', 'Unknown')}")
    
    # Summary Metrics
    print("\n>>> SUMMARY METRICS")
    print("-" * 70)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Kernel Summary
    if len(kernels) > 0:
        print("\n>>> TOP 10 GPU KERNELS BY TIME")
        print("-" * 70)
        display = kernels.copy()
        display['total_ms'] = display['total_time_ns'] / 1e6
        display['avg_ms'] = display['avg_time_ns'] / 1e6
        display['pct'] = display['total_time_ns'] / display['total_time_ns'].sum() * 100
        print(display[['kernel_name', 'invocations', 'total_ms', 'avg_ms', 'pct']].head(10).to_string(index=False))
    
    # Memory Summary
    if len(memory) > 0:
        print("\n>>> MEMORY TRANSFERS")
        print("-" * 70)
        display = memory.copy()
        display['total_GB'] = display['total_bytes'] / 1e9
        display['time_ms'] = display['total_time_ns'] / 1e6
        display['bandwidth_GBps'] = display['total_GB'] / (display['time_ms'] / 1e3)
        print(display[['transfer_type', 'count', 'total_GB', 'time_ms', 'bandwidth_GBps']].to_string(index=False))
    
    # API Summary
    if len(api) > 0:
        print("\n>>> TOP 10 CUDA API CALLS")
        print("-" * 70)
        display = api.copy()
        display['total_ms'] = display['total_time_ns'] / 1e6
        print(display[['api_name', 'calls', 'total_ms']].head(10).to_string(index=False))
    
    # NVTX
    if len(nvtx) > 0:
        print("\n>>> NVTX RANGES")
        print("-" * 70)
        display = nvtx.copy()
        display['total_ms'] = display['total_time_ns'] / 1e6
        print(display[['range_name', 'count', 'total_ms']].head(10).to_string(index=False))
    
    # GPU Utilization
    if util:
        print("\n>>> GPU UTILIZATION")
        print("-" * 70)
        print(f"  Busy Time:   {util['busy_time_ns']/1e6:.2f} ms")
        print(f"  Total Time:  {util['total_time_ns']/1e6:.2f} ms")
        print(f"  Utilization: {util['utilization_pct']:.1f}%")
    
    # Export CSVs
    if output_path:
        print("\n>>> EXPORTING DATA")
        print("-" * 70)
        if len(kernels) > 0:
            kernels.to_csv(f"{output_path}_kernels.csv", index=False)
            print(f"  Kernels: {output_path}_kernels.csv")
        if len(memory) > 0:
            memory.to_csv(f"{output_path}_memory.csv", index=False)
            print(f"  Memory: {output_path}_memory.csv")
        if len(api) > 0:
            api.to_csv(f"{output_path}_api.csv", index=False)
            print(f"  API: {output_path}_api.csv")
    
    print("\n" + "=" * 70)
    conn.close()


def compare_profiles(baseline_db: str, new_db: str, threshold: float = 0.1) -> dict:
    """
    Compare two profiles and detect regressions.
    
    Args:
        baseline_db: Path to baseline SQLite database
        new_db: Path to new SQLite database
        threshold: Regression threshold (0.1 = 10%)
    
    Returns:
        Dictionary with comparison results
    """
    baseline_conn = load_profile_db(baseline_db)
    new_conn = load_profile_db(new_db)
    
    baseline_kernels = get_kernel_summary(baseline_conn)
    new_kernels = get_kernel_summary(new_conn)
    
    baseline_conn.close()
    new_conn.close()
    
    # Merge on kernel name
    merged = pd.merge(
        baseline_kernels[['kernel_name', 'invocations', 'avg_time_ns']],
        new_kernels[['kernel_name', 'invocations', 'avg_time_ns']],
        on='kernel_name',
        suffixes=('_baseline', '_new')
    )
    
    # Calculate regression
    merged['time_change_pct'] = ((merged['avg_time_ns_new'] - merged['avg_time_ns_baseline']) / 
                                  merged['avg_time_ns_baseline'] * 100)
    
    # Find regressions
    regressions = merged[merged['time_change_pct'] > threshold * 100].copy()
    improvements = merged[merged['time_change_pct'] < -threshold * 100].copy()
    
    result = {
        'total_kernels_compared': len(merged),
        'regressions': len(regressions),
        'improvements': len(improvements),
        'regression_details': regressions.to_dict('records'),
        'improvement_details': improvements.to_dict('records')
    }
    
    print("\n>>> PROFILE COMPARISON")
    print("=" * 70)
    print(f"Kernels compared: {result['total_kernels_compared']}")
    print(f"Regressions (>{threshold*100:.0f}%): {result['regressions']}")
    print(f"Improvements (<{-threshold*100:.0f}%): {result['improvements']}")
    
    if len(regressions) > 0:
        print("\n>>> REGRESSIONS")
        print(regressions[['kernel_name', 'avg_time_ns_baseline', 'avg_time_ns_new', 'time_change_pct']].to_string())
    
    if len(improvements) > 0:
        print("\n>>> IMPROVEMENTS")
        print(improvements[['kernel_name', 'avg_time_ns_baseline', 'avg_time_ns_new', 'time_change_pct']].to_string())
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Nsight Systems SQLite export")
    parser.add_argument("database", help="Path to SQLite database (.sqlite)")
    parser.add_argument("-o", "--output", help="Output prefix for exported files")
    parser.add_argument("-f", "--format", choices=['text', 'json'], default='text',
                        help="Output format")
    parser.add_argument("--compare", help="Compare with another profile (baseline)")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Regression threshold (default: 0.1 = 10%%)")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_profiles(args.compare, args.database, args.threshold)
    else:
        generate_report(args.database, args.output, args.format)
