cmake_minimum_required(VERSION 3.18)
project(CudaTests LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -lineinfo")

add_executable(run_tests
    src/main.cpp
    src/test_framework.cpp
    src/cuda_test.cu
    tests/test_vector_ops.cpp
    tests/test_reduction.cpp
)

target_include_directories(run_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(run_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add CTest support
enable_testing()
add_test(NAME CudaTests COMMAND run_tests)
