cmake_minimum_required(VERSION 3.18)
project(python_bindings CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Try to find pybind11 (optional - will skip if not found)
find_package(pybind11 QUIET)

# Build the CUDA library
add_library(cuda_vector_ops STATIC vector_ops.cu)
target_compile_options(cuda_vector_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Build simple test executable
add_executable(test_vector_ops test_vector_ops.cpp)
target_link_libraries(test_vector_ops cuda_vector_ops)

# Build Python module if pybind11 is available
if(pybind11_FOUND)
    pybind11_add_module(cuda_ops python_bindings.cpp)
    target_link_libraries(cuda_ops PRIVATE cuda_vector_ops)
    message(STATUS "pybind11 found - Python module will be built")
else()
    message(STATUS "pybind11 not found - Python module will not be built")
    message(STATUS "Install with: pip install pybind11")
endif()
