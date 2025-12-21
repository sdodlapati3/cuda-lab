/**
 * Week 30, Day 5: TensorFlow Custom Op
 * Structure for TF CUDA extensions.
 */
#include <cstdio>

/*
 * TensorFlow Custom Op Structure:
 *
 * 1. Register the op (my_op.cc):
 *    REGISTER_OP("MyOp")
 *        .Input("input: float")
 *        .Output("output: float")
 *        .SetShapeFn([](InferenceContext* c) {
 *            c->set_output(0, c->input(0));
 *            return Status::OK();
 *        });
 *
 * 2. Implement kernel (my_op.cu.cc):
 *    template <typename T>
 *    __global__ void MyOpKernel(const T* in, T* out, int n) {
 *        int idx = blockIdx.x * blockDim.x + threadIdx.x;
 *        if (idx < n) out[idx] = process(in[idx]);
 *    }
 *
 * 3. Register kernel:
 *    REGISTER_KERNEL_BUILDER(
 *        Name("MyOp").Device(DEVICE_GPU).TypeConstraint<float>("T"),
 *        MyOpOp<GPUDevice, float>);
 *
 * 4. Build with bazel or cmake
 *
 * 5. Load in Python:
 *    my_op_module = tf.load_op_library('./my_op.so')
 *    result = my_op_module.my_op(input_tensor)
 */

int main() {
    printf("Week 30 Day 5: TensorFlow Custom Op\n\n");
    
    printf("TensorFlow vs PyTorch Extension Comparison:\n");
    printf("┌───────────────────┬─────────────────┬─────────────────┐\n");
    printf("│ Feature           │ PyTorch         │ TensorFlow      │\n");
    printf("├───────────────────┼─────────────────┼─────────────────┤\n");
    printf("│ Build system      │ setuptools      │ Bazel/CMake     │\n");
    printf("│ Binding           │ pybind11        │ TF C++ API      │\n");
    printf("│ JIT compilation   │ Yes (load())    │ Limited         │\n");
    printf("│ Gradient reg      │ autograd.Func   │ RegisterGrad    │\n");
    printf("│ Shape inference   │ Automatic       │ SetShapeFn      │\n");
    printf("└───────────────────┴─────────────────┴─────────────────┘\n\n");
    
    printf("TensorFlow Op Components:\n");
    printf("  1. Op Registration (REGISTER_OP)\n");
    printf("  2. Kernel Implementation (OpKernel)\n");
    printf("  3. Kernel Registration (REGISTER_KERNEL_BUILDER)\n");
    printf("  4. Gradient Registration (REGISTER_GRADIENT)\n\n");
    
    printf("Build Command:\n");
    printf("  nvcc -shared -o my_op.so my_op.cu.cc \\\n");
    printf("       -I$TF_INC -L$TF_LIB -ltensorflow_framework\n");
    
    return 0;
}
