cmake_minimum_required(VERSION 3.18)
project(week31_day1 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(tensorrt_basics tensorrt_basics.cu)
