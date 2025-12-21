/**
 * Week 31, Day 4: Precision Modes
 * TensorRT precision optimization.
 */
#include <cstdio>

int main() {
    printf("Week 31 Day 4: Precision Modes\n\n");
    
    printf("TensorRT Precision Flags:\n");
    printf("  config->setFlag(BuilderFlag::kFP16);   // Enable FP16\n");
    printf("  config->setFlag(BuilderFlag::kINT8);   // Enable INT8\n");
    printf("  config->setFlag(BuilderFlag::kTF32);   // Enable TF32\n\n");
    
    printf("INT8 Calibration:\n");
    printf("  1. Provide calibration dataset\n");
    printf("  2. TensorRT runs inference, collects statistics\n");
    printf("  3. Computes optimal scale factors per tensor\n");
    printf("  4. Stores calibration cache for reuse\n\n");
    
    printf("Precision Comparison (ResNet-50, A100):\n");
    printf("┌──────────┬──────────┬──────────┬──────────┐\n");
    printf("│ Precision│ Latency  │ Through. │ Accuracy │\n");
    printf("├──────────┼──────────┼──────────┼──────────┤\n");
    printf("│ FP32     │ 2.5 ms   │ 400 img/s│ Baseline │\n");
    printf("│ TF32     │ 1.8 ms   │ 550 img/s│ ~Same    │\n");
    printf("│ FP16     │ 1.2 ms   │ 800 img/s│ ~Same    │\n");
    printf("│ INT8     │ 0.8 ms   │ 1200 img/s│ <1%% drop │\n");
    printf("└──────────┴──────────┴──────────┴──────────┘\n\n");
    
    printf("Calibration Code Pattern:\n");
    printf("  class MyCalibrator : public IInt8EntropyCalibrator2 {\n");
    printf("      bool getBatch(void* bindings[], ...) override {\n");
    printf("          // Load calibration batch\n");
    printf("          return loadNextBatch(bindings);\n");
    printf("      }\n");
    printf("      // writeCalibrationCache / readCalibrationCache\n");
    printf("  };\n\n");
    
    printf("Best Practices:\n");
    printf("  - Use representative calibration data (1000+ samples)\n");
    printf("  - Try entropy vs minmax calibrators\n");
    printf("  - Cache calibration for faster rebuilds\n");
    
    return 0;
}
