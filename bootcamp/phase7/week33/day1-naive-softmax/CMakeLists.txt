cmake_minimum_required(VERSION 3.18)
project(week33_day1 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(naive_softmax naive_softmax.cu)
