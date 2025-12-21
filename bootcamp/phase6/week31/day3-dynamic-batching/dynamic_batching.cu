/**
 * Week 31, Day 3: Dynamic Batching
 * Variable batch size handling.
 */
#include <cstdio>

int main() {
    printf("Week 31 Day 3: Dynamic Batching\n\n");
    
    printf("TensorRT Dynamic Shapes:\n\n");
    
    printf("1. Define optimization profiles:\n");
    printf("   IOptimizationProfile* profile = builder->createOptProfile();\n");
    printf("   profile->setDimensions(\"input\",\n");
    printf("       OptProfileSelector::kMIN, Dims4{1, 3, 224, 224});\n");
    printf("   profile->setDimensions(\"input\",\n");
    printf("       OptProfileSelector::kOPT, Dims4{8, 3, 224, 224});\n");
    printf("   profile->setDimensions(\"input\",\n");
    printf("       OptProfileSelector::kMAX, Dims4{32, 3, 224, 224});\n\n");
    
    printf("2. At runtime, set actual dimensions:\n");
    printf("   context->setInputShape(\"input\", Dims4{batch, 3, 224, 224});\n\n");
    
    printf("Batching Strategies:\n");
    printf("┌─────────────────┬────────────────────────────────────┐\n");
    printf("│ Strategy        │ Use Case                           │\n");
    printf("├─────────────────┼────────────────────────────────────┤\n");
    printf("│ Static batch    │ Fixed workload, max throughput     │\n");
    printf("│ Dynamic batch   │ Variable requests, flexibility     │\n");
    printf("│ Request batching│ Combine requests (Triton Server)   │\n");
    printf("│ Continuous batch│ LLM serving (in-flight batching)   │\n");
    printf("└─────────────────┴────────────────────────────────────┘\n\n");
    
    printf("Performance vs Batch Size:\n");
    printf("  Batch 1:  Low latency, low throughput\n");
    printf("  Batch 8:  Good balance\n");
    printf("  Batch 32: High throughput, higher latency\n\n");
    
    printf("Dynamic batching trade-off:\n");
    printf("  - Pros: Flexibility, memory efficient\n");
    printf("  - Cons: Suboptimal kernel selection possible\n");
    
    return 0;
}
