cmake_minimum_required(VERSION 3.18)
project(week31_day4 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(precision_modes precision_modes.cu)
