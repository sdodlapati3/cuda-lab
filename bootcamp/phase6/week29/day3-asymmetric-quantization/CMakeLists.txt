cmake_minimum_required(VERSION 3.18)
project(week29_day3 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(asymmetric_quant asymmetric_quant.cu)
