cmake_minimum_required(VERSION 3.18)
project(basic_tiled CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)

add_executable(basic_tiled basic_tiled.cu)
target_link_libraries(basic_tiled CUDA::cublas)
target_compile_options(basic_tiled PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ptxas-options=-v>)
