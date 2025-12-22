cmake_minimum_required(VERSION 3.18)
project(week38_day2 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(flashv2 flashv2.cu)
