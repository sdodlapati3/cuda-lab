cmake_minimum_required(VERSION 3.18)
project(week29_day2 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(symmetric_quant symmetric_quant.cu)
