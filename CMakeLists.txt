cmake_minimum_required(VERSION 3.12)
project(jax_finufft LANGUAGES C CXX)

# Add the /cmake directory to the module path so that we can find FFTW
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

# Handle Python settings passed from scikit-build
if(SKBUILD)
    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
    set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
    set(Python_LIBRARY "${PYTHON_LIBRARY}")
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _tmp_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
      list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
else()
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _tmp_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
      list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

# Find and link pybind11 and fftw
find_package(pybind11 CONFIG REQUIRED)
find_package(FFTW REQUIRED COMPONENTS FLOAT_LIB DOUBLE_LIB)
link_libraries(${FFTW_FLOAT_LIB} ${FFTW_DOUBLE_LIB})

# Work out compiler flags
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-Wall -Wno-unknown-pragmas -O3 -funroll-loops)
set(FINUFFT_INCLUDE_DIRS
    ${CMAKE_CURRENT_LIST_DIR}/lib
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/include
    ${FFTW_INCLUDE_DIRS})

# Build single and double point versions of the FINUFFT library
add_library(finufft STATIC
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/spreadinterp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/finufft.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/simpleinterfaces.cpp)
target_include_directories(finufft PRIVATE ${FINUFFT_INCLUDE_DIRS})

add_library(finufft_32 STATIC
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/spreadinterp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/finufft.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/simpleinterfaces.cpp)
target_compile_definitions(finufft_32 PUBLIC SINGLE)
target_include_directories(finufft_32 PRIVATE ${FINUFFT_INCLUDE_DIRS})

# Build the XLA bindings to those libraries
pybind11_add_module(jax_finufft
    ${CMAKE_CURRENT_LIST_DIR}/lib/jax_finufft.cc
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/utils_precindep.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/contrib/legendre_rule_fast.c)
target_link_libraries(jax_finufft PRIVATE finufft finufft_32)
target_include_directories(jax_finufft PRIVATE ${FINUFFT_INCLUDE_DIRS})
install(TARGETS jax_finufft DESTINATION .)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75")
    endif()

    set(CUFINUFFT_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/lib
        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/include
        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/contrib/cuda_samples
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    set(CUFINUFFT_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/src/2d/spreadinterp2d.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/src/2d/spread2d_wrapper.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/src/2d/spread2d_wrapper_paul.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/src/2d/interp2d_wrapper.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/src/2d/cufinufft2d.cu

        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/src/3d/spreadinterp3d.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/src/3d/spread3d_wrapper.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/src/3d/interp3d_wrapper.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/src/3d/cufinufft3d.cu

        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/src/memtransfer_wrapper.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/src/deconvolve_wrapper.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/src/cufinufft.cu

        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/contrib/dirft2d.cpp
        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/contrib/common.cpp
        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/contrib/spreadinterp.cpp
        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/contrib/utils_fp.cpp)

    add_library(cufinufft STATIC
        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/src/precision_independent.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/src/profile.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/contrib/legendre_rule_fast.c
        ${CMAKE_CURRENT_LIST_DIR}/vendor/cufinufft/contrib/utils.cpp)
    target_include_directories(cufinufft PRIVATE ${CUFINUFFT_INCLUDE_DIRS})

    add_library(cufinufft_64 STATIC ${CUFINUFFT_SOURCES})
    target_include_directories(cufinufft_64 PRIVATE ${CUFINUFFT_INCLUDE_DIRS})

    add_library(cufinufft_32 STATIC ${CUFINUFFT_SOURCES})
    target_compile_definitions(cufinufft_32 PUBLIC SINGLE)
    target_include_directories(cufinufft_32 PRIVATE ${CUFINUFFT_INCLUDE_DIRS})
    
else()
    message(STATUS "No CUDA compiler found; GPU support will be disabled")
endif()
