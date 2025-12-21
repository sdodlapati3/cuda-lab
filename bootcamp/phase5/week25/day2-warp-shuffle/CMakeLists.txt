cmake_minimum_required(VERSION 3.18)
project(week25_day2 CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --ptxas-options=-v")

find_package(CUDAToolkit REQUIRED)
add_executable(warp_shuffle warp_shuffle.cu)
target_link_libraries(warp_shuffle CUDA::cublas)
