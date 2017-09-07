find_package(CUDA 8.0 REQUIRED)
include_directories(system ${CUDA_INCLUDE_DIRS})
list(APPEND lib_deps ${CUDA_LIBRARIES})

find_package(OpenCV REQUIRED COMPONENTS core highgui)
include_directories(system ${OpenCV_INCLUDE_DIRS})
list(APPEND lib_deps ${OpenCV_LIBS})
