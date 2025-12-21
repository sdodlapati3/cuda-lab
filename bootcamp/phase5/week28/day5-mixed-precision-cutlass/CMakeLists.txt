cmake_minimum_required(VERSION 3.18)
project(week28_day5 CUDA CXX)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(mixed_precision_cutlass mixed_precision_cutlass.cu)
