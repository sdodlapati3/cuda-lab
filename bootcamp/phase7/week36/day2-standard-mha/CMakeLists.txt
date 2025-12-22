cmake_minimum_required(VERSION 3.18)
project(week36_day2 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(standard_mha standard_mha.cu)
