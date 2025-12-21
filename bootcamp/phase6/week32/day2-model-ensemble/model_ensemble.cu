/**
 * Week 32, Day 2: Model Ensemble
 * Chaining models in Triton Inference Server.
 */
#include <cstdio>

int main() {
    printf("Week 32 Day 2: Model Ensemble\n\n");
    
    printf("Ensemble Pipeline Example:\n");
    printf("  Input Image → Preprocessing → ResNet → Postprocessing → Output\n\n");
    
    printf("Ensemble config.pbtxt:\n");
    printf("  name: \"image_classification_pipeline\"\n");
    printf("  platform: \"ensemble\"\n");
    printf("  max_batch_size: 32\n");
    printf("  \n");
    printf("  input [ { name: \"raw_image\" dims: [-1] data_type: TYPE_UINT8 } ]\n");
    printf("  output [ { name: \"class_labels\" dims: [5] data_type: TYPE_STRING } ]\n");
    printf("  \n");
    printf("  ensemble_scheduling {\n");
    printf("    step [\n");
    printf("      {\n");
    printf("        model_name: \"preprocess\"\n");
    printf("        model_version: -1\n");
    printf("        input_map { key: \"raw\" value: \"raw_image\" }\n");
    printf("        output_map { key: \"processed\" value: \"preprocessed\" }\n");
    printf("      },\n");
    printf("      {\n");
    printf("        model_name: \"resnet50\"\n");
    printf("        model_version: -1\n");
    printf("        input_map { key: \"input\" value: \"preprocessed\" }\n");
    printf("        output_map { key: \"output\" value: \"logits\" }\n");
    printf("      },\n");
    printf("      {\n");
    printf("        model_name: \"postprocess\"\n");
    printf("        model_version: -1\n");
    printf("        input_map { key: \"logits\" value: \"logits\" }\n");
    printf("        output_map { key: \"labels\" value: \"class_labels\" }\n");
    printf("      }\n");
    printf("    ]\n");
    printf("  }\n\n");
    
    printf("Ensemble Benefits:\n");
    printf("  - Clean separation of concerns\n");
    printf("  - Mix frameworks (Python preproc, TensorRT model)\n");
    printf("  - Independent versioning per component\n");
    printf("  - Parallel execution where possible\n");
    
    return 0;
}
