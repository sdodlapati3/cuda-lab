cmake_minimum_required(VERSION 3.18)
project(week39_day2 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(bias_activation bias_activation.cu)
