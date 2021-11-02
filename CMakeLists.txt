cmake_minimum_required(VERSION 3.12)
project(jax_finufft LANGUAGES C CXX)

message(STATUS "Using CMake version: " ${CMAKE_VERSION})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(FFTW REQUIRED COMPONENTS FLOAT_LIB DOUBLE_LIB)
link_libraries(${FFTW_FLOAT_LIB} ${FFTW_DOUBLE_LIB})

include_directories(${FFTW_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

add_compile_options(-Wall -O3 -funroll-loops)
include_directories(${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/include)

add_library(finufft STATIC 
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/spreadinterp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/finufft.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/simpleinterfaces.cpp)

add_library(finufft_32 STATIC 
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/spreadinterp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/finufft.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/simpleinterfaces.cpp)
target_compile_definitions(finufft_32 PUBLIC SINGLE)

pybind11_add_module(jax_finufft
    ${CMAKE_CURRENT_LIST_DIR}/lib/jax_finufft.cc
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/utils_precindep.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/contrib/legendre_rule_fast.c)
target_link_libraries(jax_finufft PRIVATE finufft finufft_32)
install(TARGETS jax_finufft DESTINATION jax_finufft)
