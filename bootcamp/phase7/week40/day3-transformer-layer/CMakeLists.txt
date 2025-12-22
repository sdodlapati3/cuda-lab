cmake_minimum_required(VERSION 3.18)
project(week40_day3 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(transformer_layer transformer_layer.cu)
