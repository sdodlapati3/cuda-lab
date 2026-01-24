"""
extract_ncu_metrics.py - Extract roofline data from Nsight Compute reports

Parses NCU (Nsight Compute) output to extract:
- FLOPs executed
- Bytes transferred
- Arithmetic intensity
- Achieved throughput

Usage:
    # From NCU CSV export
    python extract_ncu_metrics.py --csv ncu_report.csv
    
    # Run NCU and extract (requires CUDA executable)
    python extract_ncu_metrics.py --run ./my_kernel
    
    # Output for roofline plotting
    python extract_ncu_metrics.py --csv report.csv --output kernels.json

Author: CUDA Lab
"""

import subprocess
import csv
import json
import re
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class KernelMetrics:
    """Metrics extracted from NCU for roofline analysis."""
    name: str
    
    # Compute metrics
    flops_fp32: int = 0
    flops_fp16: int = 0
    flops_fp64: int = 0
    
    # Memory metrics
    dram_bytes_read: int = 0
    dram_bytes_written: int = 0
    l2_bytes_read: int = 0
    l2_bytes_written: int = 0
    
    # Duration
    duration_ns: float = 0
    
    @property
    def total_flops(self) -> int:
        """Total FLOPs executed."""
        return self.flops_fp32 + self.flops_fp16 + self.flops_fp64
    
    @property
    def total_dram_bytes(self) -> int:
        """Total DRAM bytes transferred."""
        return self.dram_bytes_read + self.dram_bytes_written
    
    @property
    def arithmetic_intensity_dram(self) -> float:
        """AI based on DRAM traffic."""
        if self.total_dram_bytes == 0:
            return 0
        return self.total_flops / self.total_dram_bytes
    
    @property
    def arithmetic_intensity_l2(self) -> float:
        """AI based on L2 traffic."""
        l2_bytes = self.l2_bytes_read + self.l2_bytes_written
        if l2_bytes == 0:
            return 0
        return self.total_flops / l2_bytes
    
    @property
    def performance_tflops(self) -> float:
        """Achieved performance in TFLOPS."""
        if self.duration_ns == 0:
            return 0
        return self.total_flops / self.duration_ns / 1000  # ns to TFLOPS
    
    def to_roofline_point(self) -> dict:
        """Convert to roofline plot point format."""
        return {
            'name': self.name,
            'arithmetic_intensity': self.arithmetic_intensity_dram,
            'performance': self.performance_tflops,
        }


# NCU metric name mappings
METRIC_MAPPING = {
    # FLOPs metrics
    'smsp__sass_thread_inst_executed_op_fp32_pred_on.sum': 'flops_fp32',
    'smsp__sass_thread_inst_executed_op_fp16_pred_on.sum': 'flops_fp16',
    'smsp__sass_thread_inst_executed_op_fp64_pred_on.sum': 'flops_fp64',
    
    # DRAM metrics
    'dram__bytes_read.sum': 'dram_bytes_read',
    'dram__bytes_write.sum': 'dram_bytes_written',
    
    # L2 metrics
    'lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum': 'l2_bytes_read',
    'lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum': 'l2_bytes_written',
    
    # Duration
    'gpu__time_duration.avg': 'duration_ns',
}

# Simpler metric names (NCU output format varies)
SIMPLE_METRIC_MAPPING = {
    'FLOP': 'flops_fp32',
    'DRAM Read': 'dram_bytes_read',
    'DRAM Write': 'dram_bytes_written',
    'Duration': 'duration_ns',
    'L2 Read': 'l2_bytes_read',
    'L2 Write': 'l2_bytes_written',
}


def run_ncu(executable: str, args: List[str] = None, 
            output_csv: str = 'ncu_report.csv') -> str:
    """
    Run Nsight Compute on an executable.
    
    Returns path to CSV report.
    """
    ncu_cmd = [
        'ncu',
        '--csv',
        '--metrics', ','.join(METRIC_MAPPING.keys()),
        '-o', output_csv.replace('.csv', ''),  # NCU adds extension
        executable
    ]
    
    if args:
        ncu_cmd.extend(args)
    
    print(f"Running: {' '.join(ncu_cmd)}")
    
    try:
        result = subprocess.run(ncu_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"NCU Error: {result.stderr}")
            return None
        return output_csv
    except FileNotFoundError:
        print("Error: 'ncu' not found. Make sure CUDA toolkit is in PATH.")
        return None


def parse_ncu_csv(csv_path: str) -> List[KernelMetrics]:
    """
    Parse NCU CSV output to extract kernel metrics.
    """
    kernels = []
    
    with open(csv_path, 'r') as f:
        # NCU CSV format has header rows
        reader = csv.DictReader(f)
        
        current_kernel = None
        kernel_data = {}
        
        for row in reader:
            # Get kernel name
            kernel_name = row.get('Kernel Name', row.get('kernel', ''))
            
            if kernel_name and kernel_name != current_kernel:
                # Save previous kernel
                if current_kernel and kernel_data:
                    metrics = KernelMetrics(name=current_kernel)
                    for key, value in kernel_data.items():
                        if hasattr(metrics, key):
                            try:
                                setattr(metrics, key, int(float(value)))
                            except (ValueError, TypeError):
                                pass
                    kernels.append(metrics)
                
                current_kernel = kernel_name
                kernel_data = {}
            
            # Extract metrics
            metric_name = row.get('Metric Name', row.get('metric', ''))
            metric_value = row.get('Metric Value', row.get('value', 0))
            
            # Try to map metric name
            if metric_name in METRIC_MAPPING:
                attr = METRIC_MAPPING[metric_name]
                kernel_data[attr] = metric_value
            else:
                # Try simple mapping
                for simple, attr in SIMPLE_METRIC_MAPPING.items():
                    if simple.lower() in metric_name.lower():
                        kernel_data[attr] = metric_value
                        break
        
        # Don't forget last kernel
        if current_kernel and kernel_data:
            metrics = KernelMetrics(name=current_kernel)
            for key, value in kernel_data.items():
                if hasattr(metrics, key):
                    try:
                        setattr(metrics, key, int(float(value)))
                    except (ValueError, TypeError):
                        pass
            kernels.append(metrics)
    
    return kernels


def parse_ncu_report(report_path: str) -> List[KernelMetrics]:
    """
    Parse NCU .ncu-rep file (requires ncu-cli to export).
    """
    # Export to CSV first
    csv_path = report_path.replace('.ncu-rep', '.csv')
    
    cmd = ['ncu', '--import', report_path, '--csv']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        with open(csv_path, 'w') as f:
            f.write(result.stdout)
        return parse_ncu_csv(csv_path)
    except FileNotFoundError:
        print("Error: 'ncu' not found for report parsing")
        return []


def export_for_roofline(kernels: List[KernelMetrics], output_path: str):
    """Export kernel metrics for roofline plotting."""
    data = {
        'kernels': [k.to_roofline_point() for k in kernels]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {len(kernels)} kernels to {output_path}")


def print_summary(kernels: List[KernelMetrics]):
    """Print summary of extracted metrics."""
    print("\n" + "="*80)
    print("Roofline Metrics Summary")
    print("="*80)
    
    for k in kernels:
        print(f"\nKernel: {k.name}")
        print(f"  Total FLOPs:       {k.total_flops:,}")
        print(f"  DRAM Bytes:        {k.total_dram_bytes:,}")
        print(f"  Arithmetic Int.:   {k.arithmetic_intensity_dram:.2f} FLOPs/Byte")
        print(f"  Duration:          {k.duration_ns:.2f} ns")
        print(f"  Performance:       {k.performance_tflops:.4f} TFLOPS")
        
        # Classify kernel
        if k.arithmetic_intensity_dram < 10:
            bottleneck = "Memory-bound"
        elif k.arithmetic_intensity_dram > 100:
            bottleneck = "Compute-bound"
        else:
            bottleneck = "Transitional"
        print(f"  Classification:    {bottleneck}")


def estimate_metrics_from_code(
    kernel_name: str,
    problem_size: int,
    flops_per_element: int,
    bytes_read_per_element: int,
    bytes_written_per_element: int,
    duration_ms: float
) -> KernelMetrics:
    """
    Estimate metrics from analytical analysis.
    
    Useful when NCU is not available.
    """
    metrics = KernelMetrics(
        name=kernel_name,
        flops_fp32=problem_size * flops_per_element,
        dram_bytes_read=problem_size * bytes_read_per_element,
        dram_bytes_written=problem_size * bytes_written_per_element,
        duration_ns=duration_ms * 1e6
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Extract metrics for roofline analysis from NCU'
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--csv', type=str, help='NCU CSV report path')
    input_group.add_argument('--report', type=str, help='NCU .ncu-rep file path')
    input_group.add_argument('--run', type=str, help='CUDA executable to profile')
    
    parser.add_argument('--output', type=str, default='kernels.json',
                       help='Output JSON for roofline plotting')
    parser.add_argument('--summary', action='store_true',
                       help='Print metrics summary')
    args = parser.parse_args()
    
    kernels = []
    
    if args.csv:
        kernels = parse_ncu_csv(args.csv)
    elif args.report:
        kernels = parse_ncu_report(args.report)
    elif args.run:
        csv_path = run_ncu(args.run)
        if csv_path:
            kernels = parse_ncu_csv(csv_path)
    
    if kernels:
        if args.summary:
            print_summary(kernels)
        
        export_for_roofline(kernels, args.output)
        print(f"\nTo generate roofline plot:")
        print(f"  python plot_roofline.py --kernels {args.output}")
    else:
        print("No kernel metrics extracted")


if __name__ == "__main__":
    main()
