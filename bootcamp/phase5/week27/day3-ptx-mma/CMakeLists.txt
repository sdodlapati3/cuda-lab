cmake_minimum_required(VERSION 3.18)
project(week27_day3 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(ptx_mma ptx_mma.cu)
