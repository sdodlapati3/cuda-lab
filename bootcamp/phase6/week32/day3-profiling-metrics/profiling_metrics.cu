/**
 * Week 32, Day 3: Profiling & Metrics
 * Performance analysis tools.
 */
#include <cstdio>

int main() {
    printf("Week 32 Day 3: Profiling & Metrics\n\n");
    
    printf("Triton Performance Analyzer (perf_analyzer):\n");
    printf("  perf_analyzer -m resnet50 -u localhost:8001 \\\n");
    printf("    --concurrency-range 1:16 \\\n");
    printf("    --measurement-interval 5000\n\n");
    
    printf("Key Metrics:\n");
    printf("┌─────────────────────┬────────────────────────────────────┐\n");
    printf("│ Metric              │ Description                        │\n");
    printf("├─────────────────────┼────────────────────────────────────┤\n");
    printf("│ Throughput          │ Inferences/second                  │\n");
    printf("│ Latency (p50/p99)   │ Response time percentiles          │\n");
    printf("│ Queue time          │ Time waiting for execution         │\n");
    printf("│ Compute time        │ Actual inference time              │\n");
    printf("│ GPU utilization     │ GPU busy percentage                │\n");
    printf("│ Memory usage        │ GPU memory consumption             │\n");
    printf("└─────────────────────┴────────────────────────────────────┘\n\n");
    
    printf("Prometheus Metrics (port 8002):\n");
    printf("  nv_inference_request_success\n");
    printf("  nv_inference_request_failure\n");
    printf("  nv_inference_count\n");
    printf("  nv_inference_exec_count\n");
    printf("  nv_inference_request_duration_us\n");
    printf("  nv_inference_queue_duration_us\n");
    printf("  nv_inference_compute_input_duration_us\n");
    printf("  nv_inference_compute_infer_duration_us\n");
    printf("  nv_inference_compute_output_duration_us\n\n");
    
    printf("Nsight Systems for Deep Analysis:\n");
    printf("  nsys profile -o inference_trace \\\n");
    printf("    --trace=cuda,nvtx python client.py\n\n");
    
    printf("Optimization Workflow:\n");
    printf("  1. Baseline with perf_analyzer\n");
    printf("  2. Identify bottleneck (queue, compute, network)\n");
    printf("  3. Apply optimization (batch size, instances, precision)\n");
    printf("  4. Measure improvement\n");
    printf("  5. Repeat\n");
    
    return 0;
}
