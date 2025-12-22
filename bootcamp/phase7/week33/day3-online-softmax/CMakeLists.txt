cmake_minimum_required(VERSION 3.18)
project(week33_day3 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(online_softmax online_softmax.cu)
