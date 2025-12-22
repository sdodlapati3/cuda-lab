cmake_minimum_required(VERSION 3.18)
project(week34_day3 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(layernorm_forward layernorm_forward.cu)
