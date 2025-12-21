cmake_minimum_required(VERSION 3.18)
project(tile_size CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)

add_executable(tile_size tile_size.cu)
target_link_libraries(tile_size CUDA::cublas)
target_compile_options(tile_size PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ptxas-options=-v>)
