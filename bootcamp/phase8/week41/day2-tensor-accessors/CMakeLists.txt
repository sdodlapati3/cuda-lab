cmake_minimum_required(VERSION 3.18)
project(week41_day2 CUDA CXX)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(tensor_accessors tensor_accessors.cu)
