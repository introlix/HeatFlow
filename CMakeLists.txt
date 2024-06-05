cmake_minimum_required(VERSION 3.0.0)
project(heatflow VERSION 0.1.0)

add_library(tensor STATIC src/tensor/tensor.cpp)

add_subdirectory(pybind11)
include_directories(${CMAKE_SOURCE_DIR}/eigen-3.4.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

pybind11_add_module(heatflow src/main.cpp)
target_link_libraries(heatflow PRIVATE tensor)