# Exercise 08: Expert Systems & Auto-Analysis

## Learning Objectives
- Use Nsight Systems expert analysis features
- Understand automatic bottleneck detection
- Apply rule-based recommendations
- Create custom analysis rules

## Background

Nsight Systems includes "expert systems" - automatic analysis rules that detect common performance issues. This is especially useful when you don't know what to look for.

---

## Part 1: Enable Expert Analysis

```bash
# Profile with expert analysis (newer nsys versions)
nsys profile \
    --trace=cuda,nvtx,osrt \
    --stats=true \
    --force-overwrite=true \
    -o expert_profile \
    python train.py

# Run expert analysis
nsys analyze expert_profile.nsys-rep
```

### Expert Analysis Output:
```
=== EXPERT ANALYSIS REPORT ===

[WARNING] Low GPU Utilization Detected
  - GPU utilization: 45%
  - Recommendation: Check for CPU bottlenecks or sync points

[WARNING] Excessive Small Memory Transfers
  - 1000+ transfers < 1KB detected
  - Recommendation: Batch transfers or use pinned memory

[INFO] Kernel Launch Overhead
  - Average launch: 5.2 μs
  - Consider kernel fusion for short kernels
```

---

## Part 2: Built-in Rules

### Common Rules Detected:

| Rule | Trigger | Recommendation |
|------|---------|----------------|
| Low GPU Utilization | <70% busy | Find CPU bottlenecks |
| Small Memory Transfers | Many <1KB transfers | Batch or pin memory |
| High Sync Overhead | Frequent cudaDeviceSync | Use async operations |
| Kernel Launch Overhead | Short kernels | Fuse kernels |
| Memory Bandwidth Limited | Low achieved bandwidth | Optimize access patterns |
| Thread Divergence | High divergent branches | Restructure code |

---

## Part 3: Interpreting Recommendations

### Example: Low GPU Utilization

**Report:**
```
GPU Utilization: 52%
GPU Idle Time: 480 ms (48%)
Longest Idle Period: 45 ms at 1.234s
```

**Investigation Steps:**
1. Find the idle period in timeline
2. Check what CPU was doing
3. Look for sync calls
4. Check data loading

**Common Causes:**
- CPU preprocessing
- DataLoader stalls
- Explicit synchronization
- Python GIL contention

---

## Part 4: Custom Analysis Rules

### Create Your Own Rules:

```python
#!/usr/bin/env python3
"""custom_expert_rules.py - Custom performance analysis rules"""

import sqlite3
import pandas as pd
from typing import List, Dict, Any


class Rule:
    """Base class for analysis rules."""
    name: str = "BaseRule"
    severity: str = "INFO"  # INFO, WARNING, ERROR
    
    def check(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Check rule and return findings."""
        raise NotImplementedError


class LowGPUUtilizationRule(Rule):
    """Check for low GPU utilization."""
    name = "Low GPU Utilization"
    severity = "WARNING"
    threshold = 0.70  # 70%
    
    def check(self, conn):
        query = """
        WITH bounds AS (
            SELECT MIN(start) AS first, MAX(end) AS last
            FROM CUPTI_ACTIVITY_KIND_KERNEL
        ),
        busy AS (
            SELECT SUM(end - start) AS time
            FROM CUPTI_ACTIVITY_KIND_KERNEL
        )
        SELECT 
            busy.time AS busy_ns,
            bounds.last - bounds.first AS total_ns,
            CAST(busy.time AS FLOAT) / (bounds.last - bounds.first) AS utilization
        FROM busy, bounds
        """
        result = pd.read_sql_query(query, conn)
        
        if len(result) == 0:
            return {"triggered": False}
        
        util = result['utilization'].iloc[0]
        
        return {
            "triggered": util < self.threshold,
            "utilization": util,
            "recommendation": "Check for CPU bottlenecks, sync points, or data loading issues"
        }


class SmallTransfersRule(Rule):
    """Detect excessive small memory transfers."""
    name = "Excessive Small Transfers"
    severity = "WARNING"
    size_threshold = 4096  # 4KB
    count_threshold = 100
    
    def check(self, conn):
        query = f"""
        SELECT COUNT(*) AS small_transfer_count
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE bytes < {self.size_threshold}
        """
        result = pd.read_sql_query(query, conn)
        count = result['small_transfer_count'].iloc[0]
        
        return {
            "triggered": count > self.count_threshold,
            "count": count,
            "recommendation": "Batch transfers together or use pinned memory"
        }


class ShortKernelsRule(Rule):
    """Detect many short-running kernels."""
    name = "Short Kernel Launches"
    severity = "INFO"
    time_threshold = 10000  # 10 μs
    
    def check(self, conn):
        query = f"""
        SELECT 
            COUNT(*) AS short_kernels,
            (SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL) AS total_kernels
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE (end - start) < {self.time_threshold}
        """
        result = pd.read_sql_query(query, conn)
        
        short = result['short_kernels'].iloc[0]
        total = result['total_kernels'].iloc[0]
        ratio = short / total if total > 0 else 0
        
        return {
            "triggered": ratio > 0.3,  # >30% short kernels
            "short_kernels": short,
            "total_kernels": total,
            "ratio": ratio,
            "recommendation": "Consider kernel fusion to reduce launch overhead"
        }


class HighSyncOverheadRule(Rule):
    """Detect excessive synchronization calls."""
    name = "High Synchronization Overhead"
    severity = "WARNING"
    
    def check(self, conn):
        query = """
        SELECT 
            s.value AS api_name,
            COUNT(*) AS calls,
            SUM(r.end - r.start) AS total_time_ns
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        WHERE s.value LIKE '%Synchronize%' OR s.value LIKE '%DeviceSync%'
        GROUP BY s.value
        """
        try:
            result = pd.read_sql_query(query, conn)
            total_sync_time = result['total_time_ns'].sum() if len(result) > 0 else 0
            
            # Get total profile time
            total_query = """
            SELECT MAX(end) - MIN(start) AS total_time
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            """
            total_result = pd.read_sql_query(total_query, conn)
            total_time = total_result['total_time'].iloc[0] if len(total_result) > 0 else 1
            
            sync_ratio = total_sync_time / total_time if total_time > 0 else 0
            
            return {
                "triggered": sync_ratio > 0.05,  # >5% in sync
                "sync_time_ms": total_sync_time / 1e6,
                "sync_ratio": sync_ratio,
                "calls": result.to_dict('records') if len(result) > 0 else [],
                "recommendation": "Use async operations, reduce explicit syncs"
            }
        except:
            return {"triggered": False}


def run_expert_analysis(db_path: str) -> List[Dict]:
    """Run all expert rules on a profile."""
    conn = sqlite3.connect(db_path)
    
    rules = [
        LowGPUUtilizationRule(),
        SmallTransfersRule(),
        ShortKernelsRule(),
        HighSyncOverheadRule(),
    ]
    
    findings = []
    
    print("=" * 70)
    print("CUSTOM EXPERT ANALYSIS")
    print("=" * 70)
    
    for rule in rules:
        try:
            result = rule.check(conn)
            result['rule'] = rule.name
            result['severity'] = rule.severity
            findings.append(result)
            
            if result.get('triggered', False):
                icon = "⚠️" if rule.severity == "WARNING" else "ℹ️"
                print(f"\n{icon} [{rule.severity}] {rule.name}")
                for key, value in result.items():
                    if key not in ['triggered', 'rule', 'severity']:
                        print(f"   {key}: {value}")
        except Exception as e:
            print(f"Error running {rule.name}: {e}")
    
    conn.close()
    
    triggered = [f for f in findings if f.get('triggered', False)]
    print(f"\n{'=' * 70}")
    print(f"Analysis complete: {len(triggered)}/{len(findings)} rules triggered")
    
    return findings


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python custom_expert_rules.py <profile.sqlite>")
        sys.exit(1)
    
    run_expert_analysis(sys.argv[1])
```

---

## Part 5: Hands-On Exercise

### Task: Create a Custom Rule

Add a rule to detect memory-bound kernels:

```python
class MemoryBoundKernelsRule(Rule):
    """Detect kernels that may be memory-bound."""
    name = "Potential Memory-Bound Kernels"
    severity = "INFO"
    
    def check(self, conn):
        # Your implementation:
        # 1. Find kernels with high memory ops / compute ratio
        # 2. Compare achieved vs peak bandwidth
        # 3. Return findings with recommendations
        pass
```

---

## Part 6: Using Nsight's Built-in Expert System

Newer versions of Nsight Systems (2023.3+) have built-in expert analysis:

```bash
# Generate expert report
nsys analyze --output expert_report.txt profile.nsys-rep

# View in GUI
nsys-ui profile.nsys-rep
# Look for "Expert Systems" panel
```

---

## Part 7: Best Practices

### 1. Start with Automatic Analysis
Run expert analysis first to identify obvious issues.

### 2. Prioritize by Severity
Address ERRORs before WARNINGs before INFOs.

### 3. Validate Recommendations
Not all recommendations apply - understand your workload.

### 4. Build Domain-Specific Rules
Create rules for your specific use case (ML training, inference, etc.).

---

## Success Criteria

- [ ] Can run expert analysis on profiles
- [ ] Understand built-in rules and recommendations
- [ ] Created custom analysis rule
- [ ] Can prioritize findings by severity
