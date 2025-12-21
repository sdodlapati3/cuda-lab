cmake_minimum_required(VERSION 3.18)
project(week24_day5 CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --ptxas-options=-v")

find_package(CUDAToolkit REQUIRED)
add_executable(combined_vectorized combined_vectorized.cu)
target_link_libraries(combined_vectorized CUDA::cublas)
