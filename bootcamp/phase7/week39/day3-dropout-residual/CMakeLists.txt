cmake_minimum_required(VERSION 3.18)
project(week39_day3 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(dropout_residual dropout_residual.cu)
