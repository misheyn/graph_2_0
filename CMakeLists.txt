cmake_minimum_required(VERSION 3.21)
project(graph_2_0 CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(graph_2_0 main.cu)

set_target_properties(graph_2_0 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
