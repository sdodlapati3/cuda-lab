cmake_minimum_required(VERSION 3.18)
project(week27_day2 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(multi_stage multi_stage.cu)
