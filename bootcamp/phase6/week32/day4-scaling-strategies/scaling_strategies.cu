/**
 * Week 32, Day 4: Scaling Strategies
 * Multi-GPU and multi-instance deployment.
 */
#include <cstdio>

int main() {
    printf("Week 32 Day 4: Scaling Strategies\n\n");
    
    printf("1. Instance Groups (Multiple Model Copies):\n");
    printf("  instance_group [\n");
    printf("    { count: 2 kind: KIND_GPU gpus: [0] }  # 2 instances on GPU 0\n");
    printf("    { count: 2 kind: KIND_GPU gpus: [1] }  # 2 instances on GPU 1\n");
    printf("  ]\n\n");
    
    printf("2. Dynamic Batching Configuration:\n");
    printf("  dynamic_batching {\n");
    printf("    preferred_batch_size: [8, 16, 32]\n");
    printf("    max_queue_delay_microseconds: 100  # Wait up to 100μs\n");
    printf("  }\n\n");
    
    printf("3. Rate Limiter (Prevent Overload):\n");
    printf("  rate_limiter {\n");
    printf("    resources [\n");
    printf("      { name: \"GPU\" count: 1 }\n");
    printf("    ]\n");
    printf("  }\n\n");
    
    printf("Scaling Comparison:\n");
    printf("┌─────────────────────┬────────────────────────────────────┐\n");
    printf("│ Strategy            │ When to Use                        │\n");
    printf("├─────────────────────┼────────────────────────────────────┤\n");
    printf("│ More instances      │ Small models, latency-sensitive    │\n");
    printf("│ Larger batches      │ Large models, throughput-focused   │\n");
    printf("│ Multi-GPU           │ High throughput requirements       │\n");
    printf("│ Model parallelism   │ Very large models (>GPU memory)    │\n");
    printf("│ Horizontal scale    │ Kubernetes, multiple servers       │\n");
    printf("└─────────────────────┴────────────────────────────────────┘\n\n");
    
    printf("Kubernetes Deployment:\n");
    printf("  # HPA based on GPU utilization or custom metrics\n");
    printf("  apiVersion: autoscaling/v2\n");
    printf("  kind: HorizontalPodAutoscaler\n");
    printf("  spec:\n");
    printf("    minReplicas: 2\n");
    printf("    maxReplicas: 10\n");
    printf("    metrics:\n");
    printf("    - type: Pods\n");
    printf("      pods:\n");
    printf("        metric:\n");
    printf("          name: triton_queue_time\n");
    printf("        target:\n");
    printf("          type: AverageValue\n");
    printf("          averageValue: 10ms\n");
    
    return 0;
}
