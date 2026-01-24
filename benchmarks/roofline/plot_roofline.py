"""
plot_roofline.py - Generate roofline plots for GPU kernels

The roofline model visualizes:
- X-axis: Arithmetic Intensity (FLOPs / Byte)
- Y-axis: Performance (GFLOPS or TFLOPS)
- Roofline: min(peak_compute, peak_bandwidth * AI)

Usage:
    # Plot with default hardware specs
    python plot_roofline.py
    
    # Add kernel points
    python plot_roofline.py --kernels kernels.json
    
    # Custom hardware
    python plot_roofline.py --peak-tflops 19.5 --peak-bw 2039

Author: CUDA Lab
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class KernelPoint:
    """A kernel's position on the roofline."""
    name: str
    arithmetic_intensity: float  # FLOPs / Byte
    performance: float  # TFLOPS
    color: str = 'blue'
    marker: str = 'o'


@dataclass 
class HardwareSpec:
    """GPU hardware specifications."""
    name: str
    peak_tflops_fp32: float
    peak_bandwidth_GB_s: float
    peak_tflops_fp16: Optional[float] = None
    peak_tflops_tf32: Optional[float] = None
    
    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity where compute becomes bottleneck."""
        return self.peak_tflops_fp32 * 1000 / self.peak_bandwidth_GB_s
    
    def roofline(self, ai: np.ndarray, precision: str = 'fp32') -> np.ndarray:
        """Calculate roofline performance at given arithmetic intensities."""
        if precision == 'fp32':
            peak = self.peak_tflops_fp32
        elif precision == 'fp16' and self.peak_tflops_fp16:
            peak = self.peak_tflops_fp16
        elif precision == 'tf32' and self.peak_tflops_tf32:
            peak = self.peak_tflops_tf32
        else:
            peak = self.peak_tflops_fp32
        
        # Performance = min(peak_compute, peak_bandwidth * AI)
        memory_bound = self.peak_bandwidth_GB_s * ai / 1000  # Convert to TFLOPS
        return np.minimum(peak, memory_bound)


# Predefined hardware specs
HARDWARE_SPECS = {
    'A100-80GB': HardwareSpec(
        name='NVIDIA A100-80GB',
        peak_tflops_fp32=19.5,
        peak_bandwidth_GB_s=2039,
        peak_tflops_fp16=312,
        peak_tflops_tf32=156
    ),
    'H100-SXM': HardwareSpec(
        name='NVIDIA H100-SXM',
        peak_tflops_fp32=67,
        peak_bandwidth_GB_s=3350,
        peak_tflops_fp16=1979,
        peak_tflops_tf32=989
    ),
    'V100': HardwareSpec(
        name='NVIDIA V100',
        peak_tflops_fp32=15.7,
        peak_bandwidth_GB_s=900,
        peak_tflops_fp16=125
    ),
    'T4': HardwareSpec(
        name='NVIDIA T4',
        peak_tflops_fp32=8.1,
        peak_bandwidth_GB_s=320,
        peak_tflops_fp16=65
    ),
    'RTX-4090': HardwareSpec(
        name='NVIDIA RTX 4090',
        peak_tflops_fp32=82.6,
        peak_bandwidth_GB_s=1008,
        peak_tflops_fp16=330
    ),
}


def plot_roofline(
    hardware: HardwareSpec,
    kernels: List[KernelPoint] = None,
    title: str = None,
    save_path: str = None,
    show_fp16: bool = True,
    show_tf32: bool = True,
    ai_range: tuple = (0.1, 1000),
):
    """
    Generate a roofline plot.
    
    Args:
        hardware: Hardware specifications
        kernels: List of kernel points to plot
        title: Plot title
        save_path: Path to save figure
        show_fp16: Show FP16 roofline
        show_tf32: Show TF32 roofline
        ai_range: Range of arithmetic intensity (min, max)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Arithmetic intensity range (log scale)
    ai = np.logspace(np.log10(ai_range[0]), np.log10(ai_range[1]), 1000)
    
    # Plot FP32 roofline
    perf_fp32 = hardware.roofline(ai, 'fp32')
    ax.loglog(ai, perf_fp32, 'b-', linewidth=2, label=f'FP32 ({hardware.peak_tflops_fp32:.1f} TFLOPS)')
    
    # Plot FP16 roofline
    if show_fp16 and hardware.peak_tflops_fp16:
        perf_fp16 = hardware.roofline(ai, 'fp16')
        ax.loglog(ai, perf_fp16, 'g--', linewidth=2, label=f'FP16 ({hardware.peak_tflops_fp16:.0f} TFLOPS)')
    
    # Plot TF32 roofline
    if show_tf32 and hardware.peak_tflops_tf32:
        perf_tf32 = hardware.roofline(ai, 'tf32')
        ax.loglog(ai, perf_tf32, 'r:', linewidth=2, label=f'TF32 ({hardware.peak_tflops_tf32:.0f} TFLOPS)')
    
    # Mark ridge point
    ridge = hardware.ridge_point
    ax.axvline(x=ridge, color='gray', linestyle='--', alpha=0.5)
    ax.text(ridge * 1.1, hardware.peak_tflops_fp32 * 0.5, 
            f'Ridge Point\n(AI={ridge:.1f})', fontsize=9, alpha=0.7)
    
    # Plot kernel points
    if kernels:
        for kernel in kernels:
            ax.plot(kernel.arithmetic_intensity, kernel.performance,
                   kernel.marker, markersize=12, color=kernel.color,
                   label=kernel.name, markeredgecolor='black', markeredgewidth=1)
            
            # Calculate efficiency
            max_perf = hardware.roofline(np.array([kernel.arithmetic_intensity]))[0]
            efficiency = kernel.performance / max_perf * 100
            
            # Annotate
            ax.annotate(
                f'{kernel.name}\n({efficiency:.0f}% eff.)',
                xy=(kernel.arithmetic_intensity, kernel.performance),
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, alpha=0.8
            )
    
    # Labels and formatting
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=12)
    ax.set_title(title or f'Roofline Model - {hardware.name}', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    
    # Set axis limits
    ax.set_xlim(ai_range)
    ax.set_ylim(0.1, hardware.peak_tflops_fp16 or hardware.peak_tflops_fp32 * 1.5)
    
    # Add bandwidth annotation
    ax.text(0.15, hardware.peak_bandwidth_GB_s / 1000 * 0.2,
            f'Memory BW\n{hardware.peak_bandwidth_GB_s:.0f} GB/s',
            fontsize=9, alpha=0.7, rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return fig, ax


def load_kernels_from_json(path: str) -> List[KernelPoint]:
    """Load kernel points from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    kernels = []
    for k in data.get('kernels', []):
        kernels.append(KernelPoint(
            name=k['name'],
            arithmetic_intensity=k['arithmetic_intensity'],
            performance=k['performance'],
            color=k.get('color', 'blue'),
            marker=k.get('marker', 'o')
        ))
    
    return kernels


def calculate_arithmetic_intensity(
    flops: int,
    bytes_read: int,
    bytes_written: int
) -> float:
    """
    Calculate arithmetic intensity.
    
    AI = FLOPs / (Bytes Read + Bytes Written)
    """
    total_bytes = bytes_read + bytes_written
    if total_bytes == 0:
        return float('inf')
    return flops / total_bytes


# Example kernel data for common operations
EXAMPLE_KERNELS = [
    # Memory-bound kernels (low AI)
    KernelPoint(
        name='Vector Add',
        arithmetic_intensity=0.083,  # 1 FLOP / 12 bytes (2 read + 1 write)
        performance=0.17,  # At 2000 GB/s
        color='red',
        marker='o'
    ),
    KernelPoint(
        name='Reduction',
        arithmetic_intensity=0.25,  # 1 FLOP / 4 bytes
        performance=0.5,
        color='orange',
        marker='s'
    ),
    KernelPoint(
        name='Softmax',
        arithmetic_intensity=2.5,  # ~10 FLOPs / 4 bytes
        performance=5.0,
        color='green',
        marker='^'
    ),
    # Compute-bound kernels (high AI)
    KernelPoint(
        name='MatMul (naive)',
        arithmetic_intensity=10,
        performance=8.0,
        color='blue',
        marker='D'
    ),
    KernelPoint(
        name='MatMul (tiled)',
        arithmetic_intensity=64,
        performance=18.0,
        color='purple',
        marker='*'
    ),
]


def main():
    parser = argparse.ArgumentParser(description='Generate Roofline Plot')
    parser.add_argument('--hardware', type=str, default='A100-80GB',
                       choices=list(HARDWARE_SPECS.keys()),
                       help='Hardware profile')
    parser.add_argument('--peak-tflops', type=float, default=None,
                       help='Custom peak FP32 TFLOPS')
    parser.add_argument('--peak-bw', type=float, default=None,
                       help='Custom peak bandwidth (GB/s)')
    parser.add_argument('--kernels', type=str, default=None,
                       help='JSON file with kernel data')
    parser.add_argument('--output', type=str, default='roofline.png',
                       help='Output file path')
    parser.add_argument('--examples', action='store_true',
                       help='Plot example kernels')
    args = parser.parse_args()
    
    # Get hardware spec
    if args.peak_tflops and args.peak_bw:
        hardware = HardwareSpec(
            name='Custom',
            peak_tflops_fp32=args.peak_tflops,
            peak_bandwidth_GB_s=args.peak_bw
        )
    else:
        hardware = HARDWARE_SPECS[args.hardware]
    
    # Load kernels
    kernels = None
    if args.kernels:
        kernels = load_kernels_from_json(args.kernels)
    elif args.examples:
        kernels = EXAMPLE_KERNELS
    
    # Generate plot
    plot_roofline(
        hardware=hardware,
        kernels=kernels,
        save_path=args.output
    )


if __name__ == "__main__":
    main()
