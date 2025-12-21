cmake_minimum_required(VERSION 3.18)
project(week25_day1 CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --ptxas-options=-v")

find_package(CUDAToolkit REQUIRED)
add_executable(warp_tiling warp_tiling.cu)
target_link_libraries(warp_tiling CUDA::cublas)
