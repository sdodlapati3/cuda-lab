cmake_minimum_required(VERSION 3.18)
project(baseline_comparison CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)

add_executable(baseline baseline.cu)
target_link_libraries(baseline CUDA::cublas)
target_compile_options(baseline PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ptxas-options=-v>)
