cmake_minimum_required(VERSION 3.18)
project(week34_day1 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(layernorm_basics layernorm_basics.cu)
