cmake_minimum_required(VERSION 3.16)
project(pyramid_texture_filter LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Source files
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "src/*.cu"
)

file(GLOB_RECURSE HEADERS
    "include/*.h"
    "include/*.cuh"
)

# Create library
add_library(pyramid_texture_filter ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(pyramid_texture_filter ${OpenCV_LIBS})
set_target_properties(pyramid_texture_filter PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(pyramid_texture_filter PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
)


# Create demo executable
add_executable(demo examples/main.cpp)
target_link_libraries(demo pyramid_texture_filter)
set_target_properties(demo PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
)