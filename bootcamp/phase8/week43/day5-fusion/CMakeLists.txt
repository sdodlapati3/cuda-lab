cmake_minimum_required(VERSION 3.18)
project(week43 CUDA CXX)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(main main.cu)
