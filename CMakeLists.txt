cmake_minimum_required(VERSION 3.23)

project(pmpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(Python_ROOT_DIR "C:/Users/maksy/AppData/Local/Python/pythoncore-3.14-64")

find_package(Python 3.14 EXACT COMPONENTS Interpreter Development.Module REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(kernels)
add_subdirectory(pybind11)
add_subdirectory(bindings)