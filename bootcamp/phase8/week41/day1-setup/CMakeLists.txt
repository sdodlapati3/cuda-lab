cmake_minimum_required(VERSION 3.18)
project(week41_day1 CUDA CXX)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(setup_demo setup_demo.cu)
