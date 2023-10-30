cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)
message(STATUS "Using CMake version: " ${CMAKE_VERSION})

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(FINUFFT_USE_CUDA ON)
else()
    message(STATUS "No CUDA compiler found; GPU support will be disabled")
    set(FINUFFT_USE_CUDA OFF)
endif()

# TODO(dfm): OpenMP segfaults on my system - can we enable this somehow?
set(FINUFFT_USE_OPENMP OFF)

# Add the FINUFFT project using the vendored version
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/vendor/finufft")

# Find pybind11
set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Build the CPU XLA bindings
pybind11_add_module(jax_finufft_cpu ${CMAKE_CURRENT_LIST_DIR}/lib/jax_finufft_cpu.cc)
target_link_libraries(jax_finufft_cpu PRIVATE finufft_static)
install(TARGETS jax_finufft_cpu LIBRARY DESTINATION .)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    pybind11_add_module(jax_finufft_gpu
        ${CMAKE_CURRENT_LIST_DIR}/lib/jax_finufft_gpu.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/kernels.cc.cu)
    target_link_libraries(jax_finufft_gpu PRIVATE cufinufft_static)
    install(TARGETS jax_finufft_gpu LIBRARY DESTINATION .)
endif()
