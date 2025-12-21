cmake_minimum_required(VERSION 3.18)
project(week28_day3 CUDA CXX)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(custom_epilogues custom_epilogues.cu)
