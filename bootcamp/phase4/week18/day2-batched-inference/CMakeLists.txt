cmake_minimum_required(VERSION 3.18)
project(batched_inference CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)

add_executable(batched_inference batched_inference.cu)

target_link_libraries(batched_inference PRIVATE CUDA::cublas)

target_compile_options(batched_inference PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
)
