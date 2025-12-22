cmake_minimum_required(VERSION 3.18)
project(week37_day3 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(online_softmax_attention online_softmax_attention.cu)
