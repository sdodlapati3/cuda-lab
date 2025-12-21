cmake_minimum_required(VERSION 3.18)
project(week24_day6 CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --ptxas-options=-v")

find_package(CUDAToolkit REQUIRED)
add_executable(vectorization_summary vectorization_summary.cu)
target_link_libraries(vectorization_summary CUDA::cublas)
