cmake_minimum_required(VERSION 2.8)
project(CommonAlg C CXX)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fPIC -std=c++11")
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-fPIC")

add_subdirectory(src)
