cmake_minimum_required(VERSION 3.18)
project(week27_day6 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(advanced_tc_summary advanced_tc_summary.cu)
