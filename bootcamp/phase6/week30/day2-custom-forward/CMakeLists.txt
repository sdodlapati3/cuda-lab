cmake_minimum_required(VERSION 3.18)
project(week30_day2 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(custom_forward custom_forward.cu)
