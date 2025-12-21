cmake_minimum_required(VERSION 3.18)
project(week30_day1 CUDA CXX)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(pytorch_ext_demo pytorch_ext_demo.cu)
