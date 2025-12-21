#!/usr/bin/env python3
"""
Roofline Plot Generator

Generates a roofline chart for your GPU with measured kernel positions.
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# GPU specs (update these with your measurements!)
# Run ./roofline_measure to get actual values
PEAK_BANDWIDTH_GB_S = 1555.0  # A100 HBM2e
PEAK_GFLOPS_FP32 = 19500.0    # A100 FP32

def plot_roofline(peak_bw=PEAK_BANDWIDTH_GB_S, peak_gflops=PEAK_GFLOPS_FP32, 
                  kernel_points=None):
    """Generate roofline plot."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Arithmetic intensity range (log scale)
    ai = np.logspace(-2, 3, 1000)  # 0.01 to 1000 FLOP/Byte
    
    # Ridge point
    ridge = peak_gflops / peak_bw
    
    # Roofline: min(compute, bandwidth * AI)
    performance = np.minimum(peak_gflops, peak_bw * ai)
    
    # Plot roofline
    ax.loglog(ai, performance, 'b-', linewidth=2.5, label='Roofline')
    
    # Mark ridge point
    ax.axvline(x=ridge, color='gray', linestyle='--', alpha=0.5)
    ax.annotate(f'Ridge Point\n({ridge:.2f} FLOP/Byte)', 
                xy=(ridge, peak_gflops), 
                xytext=(ridge * 3, peak_gflops * 0.7),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    # Add labels for regions
    ax.text(0.02, peak_gflops * 0.1, 'MEMORY\nBOUND', fontsize=14, 
            color='blue', alpha=0.7, ha='center')
    ax.text(ridge * 10, peak_gflops * 0.8, 'COMPUTE\nBOUND', fontsize=14,
            color='red', alpha=0.7, ha='center')
    
    # Plot kernel points
    if kernel_points:
        for name, ai_val, achieved, theoretical in kernel_points:
            color = 'green' if achieved / theoretical > 0.7 else 'orange'
            ax.scatter(ai_val, achieved, s=150, c=color, marker='o', 
                      edgecolors='black', linewidths=1.5, zorder=5)
            ax.annotate(name, (ai_val, achieved), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
    ax.set_title(f'Roofline Model\nPeak: {peak_gflops:.0f} GFLOPS, {peak_bw:.0f} GB/s', 
                fontsize=14)
    ax.set_xlim(0.01, 1000)
    ax.set_ylim(1, peak_gflops * 2)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    return fig

def load_kernel_data(csv_path):
    """Load kernel measurements from CSV."""
    points = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                points.append((
                    row['name'],
                    float(row['ai']),
                    float(row['achieved_gflops']),
                    float(row['theoretical_gflops'])
                ))
    except FileNotFoundError:
        print(f"No data file found at {csv_path}")
        print("Run ./roofline_measure first, then save CSV data.")
    return points

if __name__ == '__main__':
    # Try to load measured data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'my_gpu_data.csv')
    
    kernel_points = load_kernel_data(csv_path)
    
    if not kernel_points:
        # Use example data if no measurements
        print("Using example data (run roofline_measure for real data)")
        kernel_points = [
            ('vector_add', 0.083, 75.0, 129.6),
            ('saxpy', 0.167, 150.0, 259.2),
            ('poly_eval', 1.25, 1200.0, 1944.0),
        ]
    
    fig = plot_roofline(kernel_points=kernel_points)
    
    # Save plot
    output_path = os.path.join(script_dir, 'roofline.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Also show if running interactively
    plt.show()
