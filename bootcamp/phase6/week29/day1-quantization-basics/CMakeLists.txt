cmake_minimum_required(VERSION 3.18)
project(week29_day1 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(quantization_basics quantization_basics.cu)
