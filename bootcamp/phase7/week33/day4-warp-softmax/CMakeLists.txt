cmake_minimum_required(VERSION 3.18)
project(week33_day4 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(warp_softmax warp_softmax.cu)
