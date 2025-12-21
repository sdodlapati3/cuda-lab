/**
 * Week 32, Day 1: Triton Server Basics
 * NVIDIA Triton Inference Server overview.
 */
#include <cstdio>

int main() {
    printf("Week 32 Day 1: Triton Server Basics\n\n");
    
    printf("Triton Inference Server Features:\n");
    printf("  - Multi-framework support (TensorRT, PyTorch, ONNX, TF)\n");
    printf("  - Dynamic batching\n");
    printf("  - Model versioning\n");
    printf("  - Model ensemble\n");
    printf("  - HTTP/gRPC APIs\n");
    printf("  - Metrics (Prometheus)\n\n");
    
    printf("Model Repository Structure:\n");
    printf("  model_repository/\n");
    printf("  ├── resnet50/\n");
    printf("  │   ├── config.pbtxt\n");
    printf("  │   └── 1/\n");
    printf("  │       └── model.plan   # TensorRT engine\n");
    printf("  └── bert/\n");
    printf("      ├── config.pbtxt\n");
    printf("      └── 1/\n");
    printf("          └── model.onnx\n\n");
    
    printf("config.pbtxt Example:\n");
    printf("  name: \"resnet50\"\n");
    printf("  platform: \"tensorrt_plan\"\n");
    printf("  max_batch_size: 32\n");
    printf("  input [\n");
    printf("    { name: \"input\" dims: [3, 224, 224] data_type: TYPE_FP32 }\n");
    printf("  ]\n");
    printf("  output [\n");
    printf("    { name: \"output\" dims: [1000] data_type: TYPE_FP32 }\n");
    printf("  ]\n");
    printf("  dynamic_batching { preferred_batch_size: [8, 16, 32] }\n\n");
    
    printf("Start Server:\n");
    printf("  docker run --gpus all -p 8000:8000 -p 8001:8001 \\\n");
    printf("    -v /path/to/models:/models \\\n");
    printf("    nvcr.io/nvidia/tritonserver:23.10-py3 \\\n");
    printf("    tritonserver --model-repository=/models\n");
    
    return 0;
}
