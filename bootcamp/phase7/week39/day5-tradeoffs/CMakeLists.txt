cmake_minimum_required(VERSION 3.18)
project(week39_day5 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(tradeoffs tradeoffs.cu)
