cmake_minimum_required(VERSION 3.18)
project(week23_day4 CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --ptxas-options=-v")

find_package(CUDAToolkit REQUIRED)
add_executable(thread_tile_8x8 thread_tile_8x8.cu)
target_link_libraries(thread_tile_8x8 CUDA::cublas)
