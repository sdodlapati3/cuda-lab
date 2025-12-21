/**
 * Day 5: Roofline - Main driver
 */

#include <cstdio>
#include <vector>

// External declarations
struct BandwidthResults {
    float copy_bandwidth_gb_s;
    float read_bandwidth_gb_s;
    float theoretical_peak_gb_s;
};

struct ComputeResults {
    float fp32_gflops;
    float fp16_gflops;
    float theoretical_fp32_gflops;
};

struct KernelPoint {
    const char* name;
    float ai;
    float achieved_gflops;
    float theoretical_gflops;
};

extern BandwidthResults measure_bandwidth();
extern ComputeResults measure_compute();
extern std::vector<KernelPoint> measure_example_kernels(float peak_bw, float peak_gflops);

int main() {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    
    printf("\n");
    printf("════════════════════════════════════════════════════════════════════\n");
    printf("  ROOFLINE MODEL MEASUREMENT - Day 5\n");
    printf("════════════════════════════════════════════════════════════════════\n");
    printf("  Device: %s\n", props.name);
    printf("  Compute Capability: %d.%d\n", props.major, props.minor);
    printf("  SM Count: %d\n", props.multiProcessorCount);
    printf("════════════════════════════════════════════════════════════════════\n");
    
    // Measure peak bandwidth
    BandwidthResults bw = measure_bandwidth();
    
    // Measure peak compute
    ComputeResults compute = measure_compute();
    
    // Calculate ridge point
    float ridge_point = compute.fp32_gflops / bw.copy_bandwidth_gb_s;
    printf("\n=== Roofline Summary ===\n");
    printf("Peak Bandwidth:    %.1f GB/s\n", bw.copy_bandwidth_gb_s);
    printf("Peak FP32:         %.1f GFLOPS\n", compute.fp32_gflops);
    printf("Ridge Point:       %.2f FLOP/Byte\n", ridge_point);
    printf("\n");
    printf("Kernels with AI < %.2f are MEMORY-BOUND\n", ridge_point);
    printf("Kernels with AI > %.2f are COMPUTE-BOUND\n", ridge_point);
    
    // Measure example kernels
    std::vector<KernelPoint> points = measure_example_kernels(
        bw.copy_bandwidth_gb_s, compute.fp32_gflops);
    
    // Export for plotting
    printf("\n=== CSV Data for Plotting ===\n");
    printf("# Copy to analysis/my_gpu_data.csv\n");
    printf("name,ai,achieved_gflops,theoretical_gflops\n");
    for (const auto& p : points) {
        printf("%s,%.4f,%.2f,%.2f\n", p.name, p.ai, p.achieved_gflops, p.theoretical_gflops);
    }
    
    printf("\n=== Run the plotter ===\n");
    printf("python3 analysis/plot_roofline.py\n\n");
    
    return 0;
}
