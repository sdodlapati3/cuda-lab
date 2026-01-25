# Exercise 04: SQLite Export & Custom Analysis

## Learning Objectives
- Export Nsight Systems reports to SQLite
- Query profiling data with SQL
- Build custom analysis tools in Python
- Create reusable analysis scripts

## Background

Nsight Systems can export to SQLite, enabling powerful custom analysis with SQL queries or Python. This is essential for:
- Automated regression testing
- Custom metrics not in standard reports
- Correlating with external data
- Building dashboards

---

## Part 1: Export to SQLite

```bash
# Export report to SQLite database
nsys export --type=sqlite -o profile.sqlite profile.nsys-rep

# The database contains multiple tables with all profiling data
```

---

## Part 2: Explore the Schema

```bash
# Open with sqlite3
sqlite3 profile.sqlite

# List all tables
.tables

# Common tables:
# - CUPTI_ACTIVITY_KIND_KERNEL      (GPU kernels)
# - CUPTI_ACTIVITY_KIND_MEMCPY      (Memory transfers)
# - CUPTI_ACTIVITY_KIND_RUNTIME     (CUDA API calls)
# - NVTX_EVENTS                     (NVTX ranges)
# - TARGET_INFO_GPU                 (GPU info)
# - StringIds                       (String lookup)
```

### Key Tables Schema:

```sql
-- GPU Kernels
SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 1;
-- Columns: start, end, deviceId, streamId, correlationId, 
--          gridX, gridY, gridZ, blockX, blockY, blockZ,
--          staticSharedMemory, dynamicSharedMemory, ...

-- Memory Operations  
SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY LIMIT 1;
-- Columns: start, end, bytes, copyKind, srcKind, dstKind, ...

-- String lookup (kernel names, etc)
SELECT * FROM StringIds LIMIT 5;
-- Columns: id, value
```

---

## Part 3: SQL Queries for Analysis

### Top Kernels by Time
```sql
SELECT 
    s.value AS kernel_name,
    COUNT(*) AS invocations,
    SUM(k.end - k.start) / 1e6 AS total_time_ms,
    AVG(k.end - k.start) / 1e6 AS avg_time_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
GROUP BY s.value
ORDER BY total_time_ms DESC
LIMIT 10;
```

### Memory Transfer Analysis
```sql
SELECT 
    CASE copyKind
        WHEN 1 THEN 'HtoD'
        WHEN 2 THEN 'DtoH'
        WHEN 8 THEN 'DtoD'
        ELSE 'Other'
    END AS transfer_type,
    COUNT(*) AS count,
    SUM(bytes) / 1e9 AS total_GB,
    SUM(end - start) / 1e6 AS total_time_ms,
    (SUM(bytes) / 1e9) / (SUM(end - start) / 1e9) AS bandwidth_GBps
FROM CUPTI_ACTIVITY_KIND_MEMCPY
GROUP BY copyKind;
```

### GPU Utilization Over Time
```sql
-- Find gaps between kernels (GPU idle time)
WITH kernel_times AS (
    SELECT start, end,
           LAG(end) OVER (ORDER BY start) AS prev_end
    FROM CUPTI_ACTIVITY_KIND_KERNEL
)
SELECT 
    SUM(CASE WHEN start > prev_end THEN start - prev_end ELSE 0 END) / 1e6 AS idle_time_ms,
    SUM(end - start) / 1e6 AS busy_time_ms
FROM kernel_times;
```

---

## Part 4: Python Analysis Script

```python
#!/usr/bin/env python3
"""
Custom Nsight Systems analysis using SQLite export.
"""

import sqlite3
import pandas as pd
from pathlib import Path
import argparse


def load_profile_db(db_path: str) -> sqlite3.Connection:
    """Load SQLite profile database."""
    conn = sqlite3.connect(db_path)
    return conn


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
        AVG(k.blockX * k.blockY * k.blockZ) AS avg_block_size
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName = s.id
    GROUP BY s.value
    ORDER BY total_time_ns DESC
    """
    return pd.read_sql_query(query, conn)


def get_memory_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get memory transfer summary."""
    query = """
    SELECT 
        CASE copyKind
            WHEN 1 THEN 'HtoD'
            WHEN 2 THEN 'DtoH'
            WHEN 8 THEN 'DtoD'
            ELSE 'Other'
        END AS transfer_type,
        COUNT(*) AS count,
        SUM(bytes) AS total_bytes,
        SUM(end - start) AS total_time_ns,
        AVG(bytes) AS avg_bytes
    FROM CUPTI_ACTIVITY_KIND_MEMCPY
    GROUP BY copyKind
    """
    return pd.read_sql_query(query, conn)


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
        CAST(kt.busy_time AS FLOAT) / (b.last_end - b.first_start) * 100 AS utilization_pct
    FROM kernel_time kt, bounds b
    """
    result = pd.read_sql_query(query, conn)
    if len(result) > 0:
        return {
            'busy_time_ms': result['busy_time'].iloc[0] / 1e6,
            'total_time_ms': result['total_time'].iloc[0] / 1e6,
            'utilization_pct': result['utilization_pct'].iloc[0]
        }
    return {}


def get_nvtx_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get NVTX range summary if available."""
    try:
        query = """
        SELECT 
            s.value AS range_name,
            COUNT(*) AS count,
            SUM(n.endTime - n.startTime) AS total_time_ns,
            AVG(n.endTime - n.startTime) AS avg_time_ns
        FROM NVTX_EVENTS n
        JOIN StringIds s ON n.textId = s.id
        WHERE n.eventType = 59  -- Range events
        GROUP BY s.value
        ORDER BY total_time_ns DESC
        """
        return pd.read_sql_query(query, conn)
    except:
        return pd.DataFrame()


def generate_report(db_path: str, output_path: str = None):
    """Generate comprehensive analysis report."""
    conn = load_profile_db(db_path)
    
    print("=" * 60)
    print("NSIGHT SYSTEMS CUSTOM ANALYSIS")
    print(f"Database: {db_path}")
    print("=" * 60)
    
    # Kernel Summary
    print("\n>>> GPU KERNEL SUMMARY")
    print("-" * 60)
    kernels = get_kernel_summary(conn)
    kernels['total_time_ms'] = kernels['total_time_ns'] / 1e6
    kernels['avg_time_ms'] = kernels['avg_time_ns'] / 1e6
    print(kernels[['kernel_name', 'invocations', 'total_time_ms', 'avg_time_ms']].head(10).to_string())
    
    # Memory Summary
    print("\n>>> MEMORY TRANSFER SUMMARY")
    print("-" * 60)
    memory = get_memory_summary(conn)
    if len(memory) > 0:
        memory['total_GB'] = memory['total_bytes'] / 1e9
        memory['total_time_ms'] = memory['total_time_ns'] / 1e6
        memory['bandwidth_GBps'] = memory['total_GB'] / (memory['total_time_ms'] / 1e3)
        print(memory[['transfer_type', 'count', 'total_GB', 'bandwidth_GBps']].to_string())
    else:
        print("No memory transfers found")
    
    # GPU Utilization
    print("\n>>> GPU UTILIZATION")
    print("-" * 60)
    util = get_gpu_utilization(conn)
    if util:
        print(f"GPU Busy Time:  {util['busy_time_ms']:.2f} ms")
        print(f"Total Time:     {util['total_time_ms']:.2f} ms")
        print(f"Utilization:    {util['utilization_pct']:.1f}%")
    
    # NVTX Summary
    nvtx = get_nvtx_summary(conn)
    if len(nvtx) > 0:
        print("\n>>> NVTX RANGES")
        print("-" * 60)
        nvtx['total_time_ms'] = nvtx['total_time_ns'] / 1e6
        print(nvtx[['range_name', 'count', 'total_time_ms']].head(10).to_string())
    
    # Export to CSV if requested
    if output_path:
        kernels.to_csv(f"{output_path}_kernels.csv", index=False)
        memory.to_csv(f"{output_path}_memory.csv", index=False)
        print(f"\nExported to {output_path}_*.csv")
    
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Nsight Systems SQLite export")
    parser.add_argument("database", help="Path to SQLite database (.sqlite)")
    parser.add_argument("-o", "--output", help="Output prefix for CSV files")
    args = parser.parse_args()
    
    generate_report(args.database, args.output)
```

---

## Part 5: Advanced Queries

### Kernel Launch Patterns
```sql
-- Find kernels with high variance (potential optimization target)
SELECT 
    s.value AS kernel_name,
    COUNT(*) AS n,
    AVG(k.end - k.start) AS mean,
    (AVG((k.end - k.start) * (k.end - k.start)) - 
     AVG(k.end - k.start) * AVG(k.end - k.start)) AS variance
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
GROUP BY s.value
HAVING COUNT(*) > 10
ORDER BY variance DESC
LIMIT 10;
```

### Memory-Compute Overlap
```sql
-- Find memory operations that overlap with kernels
SELECT 
    m.copyKind,
    COUNT(*) AS overlapping_transfers
FROM CUPTI_ACTIVITY_KIND_MEMCPY m
WHERE EXISTS (
    SELECT 1 FROM CUPTI_ACTIVITY_KIND_KERNEL k
    WHERE k.start <= m.end AND k.end >= m.start
)
GROUP BY m.copyKind;
```

---

## Part 6: Hands-On Exercise

### Task: Build a Regression Detector

Create a script that:
1. Loads two SQLite profiles (baseline vs new)
2. Compares kernel times
3. Reports regressions >10%

```python
def compare_profiles(baseline_db: str, new_db: str, threshold: float = 0.1):
    """Compare two profiles and detect regressions."""
    # Your implementation here
    pass
```

---

## Success Criteria

- [ ] Can export to SQLite
- [ ] Can query with SQL
- [ ] Built custom Python analysis
- [ ] Created regression detection tool
