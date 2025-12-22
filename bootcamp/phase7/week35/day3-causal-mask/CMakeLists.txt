cmake_minimum_required(VERSION 3.18)
project(week35_day3 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(causal_mask causal_mask.cu)
