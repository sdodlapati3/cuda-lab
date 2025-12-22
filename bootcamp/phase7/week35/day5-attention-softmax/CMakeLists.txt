cmake_minimum_required(VERSION 3.18)
project(week35_day5 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(attention_softmax attention_softmax.cu)
