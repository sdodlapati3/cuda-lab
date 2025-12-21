cmake_minimum_required(VERSION 3.18)
project(week31_day2 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(layer_fusion layer_fusion.cu)
