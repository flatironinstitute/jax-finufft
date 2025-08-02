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

        set(FINUFFT_USE_CUDA ON)

        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "native")
        endif()

        message(STATUS "jax_finufft: CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

        # This needs to be run after the CMAKE_CUDA_ARCHITECTURES check, otherwise
        # it will set it to the compiler default
        enable_language(CUDA)
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

set(FINUFFT_POSITION_INDEPENDENT_CODE ON)

# Add the FINUFFT project using the vendored version
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/vendor/finufft")

# Workaround for CUDA files using the cpp extension
if (FINUFFT_USE_CUDA)
    set_source_files_properties(
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/cuda/utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vendor/finufft/src/cuda/spreadinterp.cpp
    TARGET_DIRECTORY cufinufft
    PROPERTIES LANGUAGE CUDA
    )
endif()

# Find Python and nanobind
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Build the CPU XLA bindings
nanobind_add_module(jax_finufft_cpu FREE_THREADED ${CMAKE_CURRENT_LIST_DIR}/lib/jax_finufft_cpu.cc)
target_link_libraries(jax_finufft_cpu PRIVATE finufft)

if (NOT FFTW_INCLUDE_DIRS)
    get_target_property(FFTW_INCLUDE_DIRS fftw3 INTERFACE_INCLUDE_DIRECTORIES)
endif()

target_include_directories(jax_finufft_cpu PRIVATE ${FFTW_INCLUDE_DIRS})
install(TARGETS jax_finufft_cpu LIBRARY DESTINATION .)

if(FINUFFT_USE_OPENMP)
    target_compile_definitions(jax_finufft_cpu PRIVATE FINUFFT_USE_OPENMP)
endif()

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
    nanobind_add_module(jax_finufft_gpu
        FREE_THREADED
        ${CMAKE_CURRENT_LIST_DIR}/lib/jax_finufft_gpu.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/cufinufft_wrapper.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/kernels.cc.cu)
    target_include_directories(jax_finufft_gpu PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
    target_include_directories(jax_finufft_gpu PUBLIC ${CUFINUFFT_VENDORED_INCLUDE_DIRS})
    target_link_libraries(jax_finufft_gpu PRIVATE cufinufft)
    install(TARGETS jax_finufft_gpu LIBRARY DESTINATION .)
endif()
