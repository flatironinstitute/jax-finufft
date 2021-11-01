cmake_minimum_required(VERSION 3.12)
project(jax_finufft LANGUAGES C CXX)

message(STATUS "Using CMake version: " ${CMAKE_VERSION})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(FFTW COMPONENTS FLOAT_LIB DOUBLE_LIB)
# find_package(OpenMP)

# if(OpenMP_CXX_FOUND)
#     link_libraries(OpenMP::OpenMP_CXX)
# else()
#     message(STATUS "OpenMP was not found")
# endif()

include_directories(${FFTW_INCLUDE_DIRS})

add_compile_options(-Wall -O3 -funroll-loops)
include_directories(${CMAKE_CURRENT_LIST_DIR}/finufft/include)

set(FINUFFT_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/lib/jax_finufft.cc
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/spreadinterp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/utils_precindep.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/contrib/legendre_rule_fast.c
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/finufft.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/simpleinterfaces.cpp)

pybind11_add_module(jax_finufft ${FINUFFT_SOURCES})
install(TARGETS jax_finufft DESTINATION jax_finufft)
# if(FFTW_DOUBLE_OPENMP_LIB_FOUND)
#     target_link_libraries(jax_finufft PRIVATE ${FFTW_DOUBLE_OPENMP_LIB})
# else
if(FFTW_DOUBLE_LIB_FOUND)
    target_link_libraries(jax_finufft PRIVATE ${FFTW_DOUBLE_LIB})
else()
    message(FATAL_ERROR "FFTW library with double support not found")
endif()

pybind11_add_module(jax_finufft_single ${FINUFFT_SOURCES})
target_compile_definitions(jax_finufft_single PUBLIC SINGLE)
install(TARGETS jax_finufft_single DESTINATION jax_finufft)
# if(FFTW_FLOAT_OPENMP_LIB_FOUND)
#     target_link_libraries(jax_finufft_single PRIVATE ${FFTW_FLOAT_OPENMP_LIB})
# else
if(FFTW_FLOAT_LIB_FOUND)
    target_link_libraries(jax_finufft_single PRIVATE ${FFTW_FLOAT_LIB})
else()
    message(FATAL_ERROR "FFTW library with float support not found")
endif()
