cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)
message(STATUS "Using CMake version: " ${CMAKE_VERSION})

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

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(FFTW REQUIRED COMPONENTS FLOAT_LIB DOUBLE_LIB)
link_libraries(${FFTW_FLOAT_LIB} ${FFTW_DOUBLE_LIB})

# Work out compiler flags
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-Wall -Wno-unknown-pragmas -O3 -funroll-loops -fdiagnostics-color)
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
pybind11_add_module(jax_finufft_cpu
    ${CMAKE_CURRENT_LIST_DIR}/lib/jax_finufft_cpu.cc
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/utils_precindep.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/contrib/legendre_rule_fast.cpp)

target_link_libraries(jax_finufft_cpu PRIVATE finufft finufft_32)
target_include_directories(jax_finufft_cpu PRIVATE ${FINUFFT_INCLUDE_DIRS})
install(TARGETS jax_finufft_cpu DESTINATION .)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;90")
    endif()
    
    # Find cufft
    find_package(CUDAToolkit)

    set(CUFINUFFT_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/lib
        # ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/include/cufinufft/contrib/
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/include/cufinufft/contrib/cuda_samples
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/include
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/contrib
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    set(CUFINUFFT_SOURCES
        # TODO: 1D not supported via JAX, but needed for compilation
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/cuda/1d/spread1d_wrapper.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/cuda/1d/interp1d_wrapper.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/cuda/1d/cufinufft1d.cu

        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/cuda/2d/spread2d_wrapper.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/cuda/2d/interp2d_wrapper.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/cuda/2d/cufinufft2d.cu

        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/cuda/3d/spread3d_wrapper.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/cuda/3d/interp3d_wrapper.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/cuda/3d/cufinufft3d.cu

        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/cuda/deconvolve_wrapper.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/cuda/memtransfer_wrapper.cu
    
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/cuda/spreadinterp.cpp
        
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/cuda/precision_independent.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/contrib/legendre_rule_fast.cpp
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/cuda/utils.cpp
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/cuda/cufinufft.cu
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/cuda/common.cu)

    add_library(cufinufft STATIC ${CUFINUFFT_SOURCES})
    target_include_directories(cufinufft PRIVATE ${CUFINUFFT_INCLUDE_DIRS})
    
    pybind11_add_module(jax_finufft_gpu
        ${CMAKE_CURRENT_LIST_DIR}/lib/jax_finufft_gpu.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/kernels.cc.cu)
    target_link_libraries(jax_finufft_gpu PRIVATE cufinufft)
    target_link_libraries(jax_finufft_gpu PRIVATE ${CUDA_cufft_LIBRARY} ${CUDA_nvToolsExt_LIBRARY})
    target_include_directories(jax_finufft_gpu PRIVATE ${CUFINUFFT_INCLUDE_DIRS})
    install(TARGETS jax_finufft_gpu DESTINATION .)

else()
    message(STATUS "No CUDA compiler found; GPU support will be disabled")
endif()
