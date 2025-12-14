cmake_minimum_required(VERSION 3.23)

project(pmpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)

add_subdirectory(kernels)
add_subdirectory(test)