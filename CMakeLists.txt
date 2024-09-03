cmake_minimum_required(VERSION 3.19...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)
message(STATUS "Using CMake version: " ${CMAKE_VERSION})

# for cuda-gdb and verbose PTXAS output
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G -Xptxas -v")

# Enable OpenMP if requested and available
option(JAX_FINUFFT_USE_OPENMP "Enable OpenMP" ON)

if(JAX_FINUFFT_USE_OPENMP)
    find_package(OpenMP)

    if(OpenMP_CXX_FOUND)
        message(STATUS "jax_finufft: OpenMP found")
        set(FINUFFT_USE_OPENMP ON)
    else()
        message(STATUS "jax_finufft: OpenMP not found")
        set(FINUFFT_USE_OPENMP OFF)
    endif()
else()
    message(STATUS "jax_finufft: OpenMP support was not requested")
    set(FINUFFT_USE_OPENMP OFF)
endif()

# Enable CUDA if requested and available
option(JAX_FINUFFT_USE_CUDA "Enable CUDA build" OFF)

if(JAX_FINUFFT_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        message(STATUS "jax_finufft: CUDA compiler found; compiling with GPU support")
        enable_language(CUDA)
        set(FINUFFT_USE_CUDA ON)
    else()
        message(FATAL_ERROR "jax_finufft: No CUDA compiler found! Please ensure the "
            "CUDA Toolkit is installed, or set JAX_FINUFFT_USE_CUDA=OFF to disable "
            "GPU support.")
        set(FINUFFT_USE_CUDA OFF)
    endif()
else()
    message(STATUS "jax_finufft: GPU support was not requested")
    set(FINUFFT_USE_CUDA OFF)
endif()

# Add the FINUFFT project using the vendored version
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/vendor/finufft")
set_property(TARGET finufft PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET finufft_f32 PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET finufft_f64 PROPERTY POSITION_INDEPENDENT_CODE ON)

# Find Python and nanobind
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Build the CPU XLA bindings
nanobind_add_module(jax_finufft_cpu ${CMAKE_CURRENT_LIST_DIR}/lib/jax_finufft_cpu.cc)
target_link_libraries(jax_finufft_cpu PRIVATE finufft)
install(TARGETS jax_finufft_cpu LIBRARY DESTINATION .)

if(FINUFFT_USE_OPENMP)
    target_compile_definitions(jax_finufft_cpu PRIVATE FINUFFT_USE_OPENMP)
endif()

# Include the CUDA extensions if possible - see above for where this is set
if(FINUFFT_USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET cufinufft_common_objects PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET cufinufft_objects PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET cufinufft PROPERTY POSITION_INDEPENDENT_CODE ON)

    # TODO(dfm): The ${CUFINUFFT_INCLUDE_DIRS} variable doesn't seem to get set
    # properly when FINUFFT is included as a submodule (maybe because of the use
    # of ${PROJECT_SOURCE_DIR}). This is just copied from there, linking to the
    # appropriate vendored directories.
    set(CUFINUFFT_VENDORED_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/include
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/contrib
        ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/include/cufinufft/contrib/cuda_samples
    )
    nanobind_add_module(jax_finufft_gpu
        ${CMAKE_CURRENT_LIST_DIR}/lib/jax_finufft_gpu.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/cufinufft_wrapper.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/kernels.cc.cu)
    target_include_directories(jax_finufft_gpu PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
    target_include_directories(jax_finufft_gpu PUBLIC ${CUFINUFFT_VENDORED_INCLUDE_DIRS})
    target_link_libraries(jax_finufft_gpu PRIVATE cufinufft)
    install(TARGETS jax_finufft_gpu LIBRARY DESTINATION .)
endif()
