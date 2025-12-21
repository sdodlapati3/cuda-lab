cmake_minimum_required(VERSION 3.18)
project(week27_day5 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(tc_occupancy tc_occupancy.cu)
