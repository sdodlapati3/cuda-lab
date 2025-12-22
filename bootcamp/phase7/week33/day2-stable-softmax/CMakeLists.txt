cmake_minimum_required(VERSION 3.18)
project(week33_day2 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(stable_softmax stable_softmax.cu)
