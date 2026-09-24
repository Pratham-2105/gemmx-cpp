cmake_minimum_required(VERSION 3.24)
project(gemmx VERSION 0.1.0 LANGUAGES CXX)

# ---- Language standard -------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)          # -std=c++20, not -std=gnu++20
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # compile_commands.json for clangd

# Default to Release for single-config generators (Makefiles/Ninja).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Options -----------------------------------------------------------------
# OFF by default: portable x86-64 baseline code. Whether the compiler may use
# AVX2 on the "scalar" kernels is an experimental variable, so it must be an
# explicit, recorded choice rather than an accident of the build.
option(GEMMX_NATIVE      "Compile with -march=native (/arch:AVX2 on MSVC)" OFF)
option(GEMMX_BUILD_TESTS "Build unit tests"                                ON)

# ---- Core library ------------------------------------------------------------
add_library(gemmx_core STATIC
  src/gemm_reference.cpp
)
target_include_directories(gemmx_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(gemmx_core PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(gemmx_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(gemmx_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(GEMMX_NATIVE)
  if(MSVC)
    target_compile_options(gemmx_core PRIVATE /arch:AVX2)
  else()
    target_compile_options(gemmx_core PRIVATE -march=native)
  endif()
endif()

# ---- Tests -------------------------------------------------------------------
if(GEMMX_BUILD_TESTS)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(gtest_force_shared_crt ON  CACHE BOOL "" FORCE)  # MSVC runtime match
  set(INSTALL_GTEST          OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(gemmx_tests
    tests/test_matrix.cpp
    tests/test_gemm_correctness.cpp
  )
  target_link_libraries(gemmx_tests PRIVATE gemmx_core GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(gemmx_tests)
endif()
