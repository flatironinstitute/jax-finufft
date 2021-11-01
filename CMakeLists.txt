cmake_minimum_required(VERSION 3.12)
project(jax_finufft LANGUAGES CXX)

message(STATUS "Using CMake version: " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenMP)

pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
target_link_libraries(PkgConfig::FFTW)

if(OpenMP_CXX_FOUND)
    link_libraries(OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP was not found")
endif()

add_compile_options(-Wall -O3 -funroll-loops)
include_directories(${CMAKE_CURRENT_LIST_DIR}/finufft/include)
pybind11_add_module(jax_finufft
    ${CMAKE_CURRENT_LIST_DIR}/lib/jax_finufft.cc
    ${CMAKE_CURRENT_LIST_DIR}/finufft/src/spreadinterp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/finufft/src/utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/finufft/src/finufft.cpp
    ${CMAKE_CURRENT_LIST_DIR}/finufft/src/simpleinterfaces.cpp)
install(TARGETS jax_finufft DESTINATION jax_finufft)
