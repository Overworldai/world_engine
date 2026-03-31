cmake_minimum_required(VERSION 3.27)
project(we_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

# ----------------------------- Dependencies -----------------------------
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

# ----------------------------- Library -----------------------------
add_library(we_kernels_lib)
target_sources(we_kernels_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/kernels/w8a8_gemm.cpp)
target_include_directories(we_kernels_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(we_kernels_lib PUBLIC mlx)

# ----------------------------- Metal -----------------------------
if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET we_kernels_metallib
    TITLE we_kernels
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/kernels/w8a8_gemm.metal
    INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR} ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  )
  add_dependencies(we_kernels_lib we_kernels_metallib)
endif()

# ----------------------------- Python Bindings -----------------------------
nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE we_kernels_lib)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
