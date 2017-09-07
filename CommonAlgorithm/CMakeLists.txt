cmake_minimum_required(VERSION 2.8)
project(CommonAlg C CXX)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fPIC -std=c++11")
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-fPIC")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-g -Xcompiler -fPIC -std=c++11
    -gencode=arch=compute_20,code=\"sm_20\"
    -gencode=arch=compute_20,code=\"sm_21\"
    -gencode=arch=compute_30,code=\"sm_30\"
    -gencode=arch=compute_35,code=\"sm_35\"
    -gencode=arch=compute_50,code=\"sm_50\"
    -gencode=arch=compute_50,code=\"compute_50\"
    ")

set(lib_deps "")
include(${PROJECT_SOURCE_DIR}/deps/deps.cmake)

add_subdirectory(src)
