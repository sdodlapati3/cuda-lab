cmake_minimum_required(VERSION 3.18)
project(week40_day6 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(phase_summary phase_summary.cu)
