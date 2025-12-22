cmake_minimum_required(VERSION 3.18)
project(week35_day4 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(padding_mask padding_mask.cu)
