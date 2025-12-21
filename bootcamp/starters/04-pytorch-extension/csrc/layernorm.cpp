/**
 * PyTorch C++ bindings for Fused LayerNorm
 */

#include <torch/extension.h>
#include <vector>

// ============================================================================
// Input checking macros
// ============================================================================
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ============================================================================
// Forward declarations of CUDA kernels
// ============================================================================
void layernorm_forward_cuda(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    float* mean,
    float* rstd,
    int batch_size,
    int hidden_size,
    float eps
);

void layernorm_backward_cuda(
    const float* grad_output,
    const float* input,
    const float* weight,
    const float* mean,
    const float* rstd,
    float* grad_input,
    float* grad_weight,
    float* grad_bias,
    int batch_size,
    int hidden_size
);

// ============================================================================
// PyTorch interface
// ============================================================================
std::vector<torch::Tensor> layernorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    
    TORCH_CHECK(input.dim() == 2, "Input must be 2D (batch, hidden)");
    
    int batch_size = input.size(0);
    int hidden_size = input.size(1);
    
    TORCH_CHECK(weight.size(0) == hidden_size, "Weight size mismatch");
    TORCH_CHECK(bias.size(0) == hidden_size, "Bias size mismatch");
    
    // Allocate outputs
    auto output = torch::empty_like(input);
    auto mean = torch::empty({batch_size}, input.options());
    auto rstd = torch::empty({batch_size}, input.options());
    
    layernorm_forward_cuda(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        batch_size,
        hidden_size,
        eps
    );
    
    return {output, mean, rstd};
}

std::vector<torch::Tensor> layernorm_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor mean,
    torch::Tensor rstd
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(mean);
    CHECK_INPUT(rstd);
    
    int batch_size = input.size(0);
    int hidden_size = input.size(1);
    
    // Allocate gradients
    auto grad_input = torch::empty_like(input);
    auto grad_weight = torch::zeros_like(weight);  // zeros for atomic add
    auto grad_bias = torch::zeros_like(weight);
    
    layernorm_backward_cuda(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        grad_weight.data_ptr<float>(),
        grad_bias.data_ptr<float>(),
        batch_size,
        hidden_size
    );
    
    return {grad_input, grad_weight, grad_bias};
}

// ============================================================================
// Python bindings
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layernorm_forward, "Fused LayerNorm forward (CUDA)");
    m.def("backward", &layernorm_backward, "Fused LayerNorm backward (CUDA)");
}
