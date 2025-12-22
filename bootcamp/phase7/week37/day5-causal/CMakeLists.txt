cmake_minimum_required(VERSION 3.18)
project(week37_day5 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(causal_flash causal_flash.cu)
