cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)
message(STATUS "Using CMake version: " ${CMAKE_VERSION})

# Workaround for LTO applied incorrectly to CUDA fatbin
# https://github.com/pybind/pybind11/issues/4825
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# Enable CUDA if requested and available
option(JAX_FINUFFT_USE_CUDA "Enable CUDA build if available" OFF)
if(JAX_FINUFFT_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        message(STATUS "CUDA compiler found; compiling with GPU support")
        enable_language(CUDA)
        set(FINUFFT_USE_CUDA ON)
    else()
        message(STATUS "No CUDA compiler found; compiling without GPU support")
        set(FINUFFT_USE_CUDA OFF)
    endif()
else()
    message(STATUS "GPU support was not requested")
    set(FINUFFT_USE_CUDA OFF)
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # TODO(dfm): OpenMP segfaults on my system - can we enable this somehow?
    set(FINUFFT_USE_OPENMP OFF)
else()
    set(FINUFFT_USE_OPENMP ON)
endif()

# Add the FINUFFT project using the vendored version
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/vendor/finufft")

# Find pybind11
set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Build the CPU XLA bindings
pybind11_add_module(jax_finufft_cpu ${CMAKE_CURRENT_LIST_DIR}/lib/jax_finufft_cpu.cc)
target_link_libraries(jax_finufft_cpu PRIVATE finufft_static)
install(TARGETS jax_finufft_cpu LIBRARY DESTINATION .)

# Include the CUDA extensions if possible - see above for where this is set
if(FINUFFT_USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    # TODO(dfm): The ${CUFINUFFT_INCLUDE_DIRS} variable doesn't seem to get set
    # properly when FINUFFT is included as a submodule (maybe because of the use
    # of ${PROJECT_SOURCE_DIR}). This is just copied from there, linking to the
    # appropriate vendored directories.
    set(CUFINUFFT_VENDORED_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/include
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/contrib
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/include/cufinufft/contrib/cuda_samples
    )
    pybind11_add_module(jax_finufft_gpu
        ${CMAKE_CURRENT_LIST_DIR}/lib/jax_finufft_gpu.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/kernels.cc.cu)
    target_include_directories(jax_finufft_gpu PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
    target_include_directories(jax_finufft_gpu PUBLIC ${CUFINUFFT_VENDORED_INCLUDE_DIRS})
    target_link_libraries(jax_finufft_gpu PRIVATE cufinufft_static)
    install(TARGETS jax_finufft_gpu LIBRARY DESTINATION .)
endif()
