cmake_minimum_required(VERSION 3.1)
project(NBody LANGUAGES CUDA CXX)


set(CMAKE-CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(DMAKE_CXX_FLAGS "-Wall -02 -Wextra -Wno-missing-field-initializers")

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_executable(nbody src/nbody.cu)

set(SOURCE_FILES
src/nbody.cu
)
