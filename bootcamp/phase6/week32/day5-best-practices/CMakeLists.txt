cmake_minimum_required(VERSION 3.18)
project(week32_day5 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(best_practices best_practices.cu)
