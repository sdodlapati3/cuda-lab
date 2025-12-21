/**
 * pybind11 Python bindings for CUDA operations
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace py = pybind11;

// Declare external CUDA functions
extern "C" {
    void cuda_vector_add(const float* a, const float* b, float* c, int n);
    void cuda_vector_mul(const float* a, const float* b, float* c, int n);
    float cuda_sum(const float* input, int n);
    int cuda_get_device_count();
    const char* cuda_get_device_name(int device);
}

// Python wrapper for vector addition
py::array_t<float> vector_add(py::array_t<float> a, py::array_t<float> b) {
    // Get buffer info
    auto buf_a = a.request();
    auto buf_b = b.request();
    
    // Validate inputs
    if (buf_a.ndim != 1 || buf_b.ndim != 1) {
        throw std::runtime_error("Inputs must be 1-dimensional");
    }
    if (buf_a.size != buf_b.size) {
        throw std::runtime_error("Input arrays must have same size");
    }
    
    // Create output array
    auto result = py::array_t<float>(buf_a.size);
    auto buf_result = result.request();
    
    // Call CUDA function
    cuda_vector_add(
        static_cast<float*>(buf_a.ptr),
        static_cast<float*>(buf_b.ptr),
        static_cast<float*>(buf_result.ptr),
        buf_a.size
    );
    
    return result;
}

// Python wrapper for element-wise multiplication
py::array_t<float> vector_mul(py::array_t<float> a, py::array_t<float> b) {
    auto buf_a = a.request();
    auto buf_b = b.request();
    
    if (buf_a.ndim != 1 || buf_b.ndim != 1) {
        throw std::runtime_error("Inputs must be 1-dimensional");
    }
    if (buf_a.size != buf_b.size) {
        throw std::runtime_error("Input arrays must have same size");
    }
    
    auto result = py::array_t<float>(buf_a.size);
    auto buf_result = result.request();
    
    cuda_vector_mul(
        static_cast<float*>(buf_a.ptr),
        static_cast<float*>(buf_b.ptr),
        static_cast<float*>(buf_result.ptr),
        buf_a.size
    );
    
    return result;
}

// Python wrapper for sum reduction
float array_sum(py::array_t<float> arr) {
    auto buf = arr.request();
    
    if (buf.ndim != 1) {
        throw std::runtime_error("Input must be 1-dimensional");
    }
    
    return cuda_sum(static_cast<float*>(buf.ptr), buf.size);
}

// Get device information
int get_device_count() {
    return cuda_get_device_count();
}

std::string get_device_name(int device) {
    return std::string(cuda_get_device_name(device));
}

// Module definition
PYBIND11_MODULE(cuda_ops, m) {
    m.doc() = "CUDA operations with Python bindings";
    
    m.def("vector_add", &vector_add, 
          "Add two vectors element-wise on GPU",
          py::arg("a"), py::arg("b"));
    
    m.def("vector_mul", &vector_mul,
          "Multiply two vectors element-wise on GPU",
          py::arg("a"), py::arg("b"));
    
    m.def("array_sum", &array_sum,
          "Sum all elements of an array on GPU",
          py::arg("arr"));
    
    m.def("get_device_count", &get_device_count,
          "Get number of CUDA devices");
    
    m.def("get_device_name", &get_device_name,
          "Get name of CUDA device",
          py::arg("device") = 0);
}
