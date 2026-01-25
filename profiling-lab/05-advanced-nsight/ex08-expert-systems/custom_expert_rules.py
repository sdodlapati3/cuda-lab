#!/usr/bin/env python3
"""
Custom expert analysis rules for Nsight Systems profiles.

Usage:
    # First export to SQLite
    nsys export --type=sqlite -o profile.sqlite profile.nsys-rep
    
    # Run expert analysis
    python custom_expert_rules.py profile.sqlite
"""

import sqlite3
import pandas as pd
from typing import List, Dict, Any
import argparse
import json


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
            CAST(busy.time AS FLOAT) / NULLIF(bounds.last - bounds.first, 0) AS utilization
        FROM busy, bounds
        """
        try:
            result = pd.read_sql_query(query, conn)
            if len(result) == 0 or result['utilization'].iloc[0] is None:
                return {"triggered": False, "reason": "No kernel data"}
            
            util = result['utilization'].iloc[0]
            
            return {
                "triggered": util < self.threshold,
                "utilization": f"{util*100:.1f}%",
                "busy_time_ms": result['busy_ns'].iloc[0] / 1e6,
                "total_time_ms": result['total_ns'].iloc[0] / 1e6,
                "recommendation": "Check for CPU bottlenecks, sync points, or data loading issues"
            }
        except Exception as e:
            return {"triggered": False, "error": str(e)}


class SmallTransfersRule(Rule):
    """Detect excessive small memory transfers."""
    name = "Excessive Small Transfers"
    severity = "WARNING"
    size_threshold = 4096  # 4KB
    count_threshold = 100
    
    def check(self, conn):
        query = f"""
        SELECT 
            COUNT(*) AS small_count,
            (SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY) AS total_count
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE bytes < {self.size_threshold}
        """
        try:
            result = pd.read_sql_query(query, conn)
            small = result['small_count'].iloc[0]
            total = result['total_count'].iloc[0]
            
            return {
                "triggered": small > self.count_threshold,
                "small_transfers": small,
                "total_transfers": total,
                "ratio": f"{small/total*100:.1f}%" if total > 0 else "N/A",
                "recommendation": "Batch transfers together or use pinned memory"
            }
        except Exception as e:
            return {"triggered": False, "error": str(e)}


class ShortKernelsRule(Rule):
    """Detect many short-running kernels."""
    name = "Short Kernel Launches"
    severity = "INFO"
    time_threshold_ns = 10000  # 10 μs
    ratio_threshold = 0.3  # 30%
    
    def check(self, conn):
        query = f"""
        SELECT 
            COUNT(*) AS short_kernels,
            (SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL) AS total_kernels
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE (end - start) < {self.time_threshold_ns}
        """
        try:
            result = pd.read_sql_query(query, conn)
            short = result['short_kernels'].iloc[0]
            total = result['total_kernels'].iloc[0]
            ratio = short / total if total > 0 else 0
            
            return {
                "triggered": ratio > self.ratio_threshold,
                "short_kernels": short,
                "total_kernels": total,
                "ratio": f"{ratio*100:.1f}%",
                "threshold": f"<{self.time_threshold_ns/1000:.0f}μs",
                "recommendation": "Consider kernel fusion to reduce launch overhead"
            }
        except Exception as e:
            return {"triggered": False, "error": str(e)}


class HighSyncOverheadRule(Rule):
    """Detect excessive synchronization calls."""
    name = "High Synchronization Overhead"
    severity = "WARNING"
    threshold = 0.05  # 5%
    
    def check(self, conn):
        query = """
        SELECT 
            s.value AS api_name,
            COUNT(*) AS calls,
            SUM(r.end - r.start) AS total_time_ns
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        WHERE s.value LIKE '%ynchronize%'
        GROUP BY s.value
        """
        try:
            result = pd.read_sql_query(query, conn)
            total_sync_time = result['total_time_ns'].sum() if len(result) > 0 else 0
            
            total_query = """
            SELECT MAX(end) - MIN(start) AS total_time
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            """
            total_result = pd.read_sql_query(total_query, conn)
            total_time = total_result['total_time'].iloc[0] if len(total_result) > 0 else 1
            
            sync_ratio = total_sync_time / total_time if total_time > 0 else 0
            
            return {
                "triggered": sync_ratio > self.threshold,
                "sync_time_ms": total_sync_time / 1e6,
                "sync_ratio": f"{sync_ratio*100:.1f}%",
                "sync_calls": result.to_dict('records') if len(result) > 0 else [],
                "recommendation": "Use async operations, reduce explicit syncs"
            }
        except Exception as e:
            return {"triggered": False, "error": str(e)}


class UnbalancedKernelsRule(Rule):
    """Detect kernels with high execution time variance."""
    name = "High Kernel Time Variance"
    severity = "INFO"
    cv_threshold = 0.5  # Coefficient of variation > 50%
    
    def check(self, conn):
        query = """
        SELECT 
            s.value AS kernel_name,
            COUNT(*) AS n,
            AVG(k.end - k.start) AS mean_ns,
            AVG((k.end - k.start) * (k.end - k.start)) - 
                AVG(k.end - k.start) * AVG(k.end - k.start) AS variance
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        GROUP BY s.value
        HAVING COUNT(*) >= 10
        """
        try:
            result = pd.read_sql_query(query, conn)
            if len(result) == 0:
                return {"triggered": False, "reason": "Not enough kernel data"}
            
            # Calculate coefficient of variation
            result['std'] = result['variance'].apply(lambda x: x**0.5 if x > 0 else 0)
            result['cv'] = result['std'] / result['mean_ns'].replace(0, 1)
            
            high_variance = result[result['cv'] > self.cv_threshold]
            
            return {
                "triggered": len(high_variance) > 0,
                "high_variance_kernels": len(high_variance),
                "examples": high_variance[['kernel_name', 'n', 'cv']].head(5).to_dict('records'),
                "recommendation": "Investigate workload imbalance or input-dependent behavior"
            }
        except Exception as e:
            return {"triggered": False, "error": str(e)}


class MemoryBoundHintRule(Rule):
    """Hint at potentially memory-bound workloads."""
    name = "Memory-Bound Workload Indicators"
    severity = "INFO"
    
    def check(self, conn):
        # Compare memory time vs kernel time
        kernel_query = """
        SELECT SUM(end - start) AS total_kernel_time
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        """
        mem_query = """
        SELECT SUM(end - start) AS total_mem_time
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        """
        try:
            kernel_result = pd.read_sql_query(kernel_query, conn)
            mem_result = pd.read_sql_query(mem_query, conn)
            
            kernel_time = kernel_result['total_kernel_time'].iloc[0] or 0
            mem_time = mem_result['total_mem_time'].iloc[0] or 0
            
            mem_ratio = mem_time / (kernel_time + mem_time) if (kernel_time + mem_time) > 0 else 0
            
            return {
                "triggered": mem_ratio > 0.3,  # >30% time in memory ops
                "kernel_time_ms": kernel_time / 1e6,
                "memory_time_ms": mem_time / 1e6,
                "memory_ratio": f"{mem_ratio*100:.1f}%",
                "recommendation": "High memory operation time - consider memory access optimization"
            }
        except Exception as e:
            return {"triggered": False, "error": str(e)}


def run_expert_analysis(db_path: str, output_format: str = 'text') -> List[Dict]:
    """Run all expert rules on a profile."""
    conn = sqlite3.connect(db_path)
    
    rules = [
        LowGPUUtilizationRule(),
        SmallTransfersRule(),
        ShortKernelsRule(),
        HighSyncOverheadRule(),
        UnbalancedKernelsRule(),
        MemoryBoundHintRule(),
    ]
    
    findings = []
    
    for rule in rules:
        try:
            result = rule.check(conn)
            result['rule'] = rule.name
            result['severity'] = rule.severity
            findings.append(result)
        except Exception as e:
            findings.append({
                'rule': rule.name,
                'severity': rule.severity,
                'triggered': False,
                'error': str(e)
            })
    
    conn.close()
    
    if output_format == 'json':
        print(json.dumps(findings, indent=2, default=str))
    else:
        print("=" * 70)
        print("CUSTOM EXPERT ANALYSIS")
        print(f"Profile: {db_path}")
        print("=" * 70)
        
        triggered = [f for f in findings if f.get('triggered', False)]
        
        for finding in triggered:
            icon = {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}.get(finding['severity'], "•")
            print(f"\n{icon} [{finding['severity']}] {finding['rule']}")
            for key, value in finding.items():
                if key not in ['triggered', 'rule', 'severity', 'error']:
                    if isinstance(value, list):
                        print(f"   {key}:")
                        for item in value[:3]:
                            print(f"     - {item}")
                    else:
                        print(f"   {key}: {value}")
        
        if not triggered:
            print("\n✅ No issues detected!")
        
        print(f"\n{'=' * 70}")
        print(f"Rules checked: {len(findings)}")
        print(f"Issues found: {len(triggered)}")
    
    return findings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom expert analysis for Nsight Systems")
    parser.add_argument("database", help="SQLite database from nsys export")
    parser.add_argument("--format", choices=['text', 'json'], default='text')
    args = parser.parse_args()
    
    run_expert_analysis(args.database, args.format)
