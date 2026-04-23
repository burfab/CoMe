cmake_minimum_required(VERSION 4.2.0)
project(fast_traversal LANGUAGES CXX CUDA)
enable_language(CUDA)

include(FetchContent)

# Add paths to our CMake code to the module path, so they can be found automatically by CMake.
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_BUILD_TYPE "Release")

# --- cnpy (build from source, since it's not available in your env) ---
FetchContent_Declare(
  cnpy
  GIT_REPOSITORY https://github.com/rogersce/cnpy.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(cnpy)

find_package(ZLIB REQUIRED)
include_directories(${CONDA_PREFIX}/include)

# --- Torch / CGAL ---
find_package(Torch REQUIRED)
find_package(CGAL REQUIRED)

# --- Python dev (needed for pybind11_add_module / python_add_library) ---
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# --- pybind11: use ONE method (FetchContent). Remove find_package(pybind11 REQUIRED). ---
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.9.2
)
FetchContent_MakeAvailable(pybind11)

# --- torch_python: needed for at::Tensor pybind caster (fixes your undefined symbol) ---
# Torch_DIR is .../torch/share/cmake/Torch, so prefix is ../../..
get_filename_component(TORCH_INSTALL_PREFIX "${Torch_DIR}/../../.." ABSOLUTE)
find_library(TORCH_PYTHON_LIBRARY
  NAMES torch_python
  PATHS "${TORCH_INSTALL_PREFIX}/lib" "${TORCH_INSTALL_PREFIX}/lib64"
)
if(NOT TORCH_PYTHON_LIBRARY)
  message(FATAL_ERROR "torch_python not found under ${TORCH_INSTALL_PREFIX}/lib. Needed for pybind11 <-> at::Tensor caster.")
endif()

# Put all the runtime stuff in the same directory.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(BUILD_DIRECTORY "${CMAKE_BINARY_DIR}/build")
set(CUDA_GENERATED_OUTPUT_DIR "${BUILD_DIRECTORY}")

#------------------------------------------------------------------------------
# main build
#------------------------------------------------------------------------------

add_link_options(-flto=auto)  # disables some warning

pybind11_add_module(tetranerf_cpp_extension
  src/triangulation.cpp
  src/py_binding.cpp
)

# Torch compile flags are important for ABI/settings
target_compile_options(tetranerf_cpp_extension PRIVATE ${TORCH_CXX_FLAGS})
target_include_directories(tetranerf_cpp_extension PRIVATE ${TORCH_INCLUDE_DIRS})

target_link_libraries(tetranerf_cpp_extension
  PRIVATE
    Python::Module
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    CGAL::CGAL
    cnpy
    ZLIB::ZLIB
)

target_compile_features(tetranerf_cpp_extension PRIVATE cxx_std_17)

set_target_properties(tetranerf_cpp_extension
  PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tetranerf/utils/extension"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
