/**
 * Week 32, Day 5: Best Practices
 * Production deployment checklist.
 */
#include <cstdio>

int main() {
    printf("Week 32 Day 5: Production Best Practices\n\n");
    
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║              Production Deployment Checklist                 ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("□ Model Optimization:\n");
    printf("  □ Convert to TensorRT/ONNX\n");
    printf("  □ Enable FP16/INT8 precision\n");
    printf("  □ Benchmark vs baseline\n");
    printf("  □ Validate accuracy within tolerance\n\n");
    
    printf("□ Server Configuration:\n");
    printf("  □ Tune batch size for workload\n");
    printf("  □ Configure instance groups\n");
    printf("  □ Set appropriate queue delays\n");
    printf("  □ Enable health checks\n\n");
    
    printf("□ Monitoring:\n");
    printf("  □ Prometheus metrics integration\n");
    printf("  □ Grafana dashboards\n");
    printf("  □ Alerting on latency/errors\n");
    printf("  □ Log aggregation\n\n");
    
    printf("□ Reliability:\n");
    printf("  □ Model versioning\n");
    printf("  □ Canary deployments\n");
    printf("  □ Rollback procedures\n");
    printf("  □ Load testing (10× expected traffic)\n\n");
    
    printf("□ Security:\n");
    printf("  □ TLS encryption\n");
    printf("  □ Authentication/authorization\n");
    printf("  □ Input validation\n");
    printf("  □ Rate limiting\n\n");
    
    printf("□ Resource Management:\n");
    printf("  □ GPU memory limits\n");
    printf("  □ CPU/memory requests\n");
    printf("  □ Horizontal pod autoscaling\n");
    printf("  □ Resource quotas\n\n");
    
    printf("Common Pitfalls:\n");
    printf("  ✗ No warmup before production traffic\n");
    printf("  ✗ Ignoring p99 latency (only looking at average)\n");
    printf("  ✗ Batch size too large for latency SLA\n");
    printf("  ✗ Missing health/readiness probes\n");
    printf("  ✗ No GPU memory headroom\n");
    
    return 0;
}
