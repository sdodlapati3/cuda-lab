cmake_minimum_required(VERSION 3.18)
project(week31_day5 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(plugin_development plugin_development.cu)
