cmake_minimum_required(VERSION 3.18)
project(week24_day1 CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --ptxas-options=-v")

find_package(CUDAToolkit REQUIRED)
add_executable(float4_intro float4_intro.cu)
target_link_libraries(float4_intro CUDA::cublas)
