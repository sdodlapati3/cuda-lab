cmake_minimum_required(VERSION 3.18)
project(week30_day4 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(fused_ops fused_ops.cu)
