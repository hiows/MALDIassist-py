cmake_minimum_required(VERSION 3.15...3.30)
project(maldiassist LANGUAGES CXX)

if(NOT SKBUILD)
  message(WARNING
    "This CMake project is meant to be built through scikit-build-core "
    "(e.g. `pip install .`). Building it directly is unsupported.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate the Python interpreter/headers provided by scikit-build-core.
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

# Discover nanobind's CMake config shipped with the build-time dependency.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# STABLE_ABI produces a limited-API (abi3) module on Python >= 3.12 and falls
# back to the version-specific ABI on older interpreters.
nanobind_add_module(_spectrum_math_cpp
  STABLE_ABI
  NB_STATIC
  src/spectrum_math_cpp.cpp)

# Install the compiled extension inside the maldiassist package directory so it
# is importable as `maldiassist._spectrum_math_cpp`.
install(TARGETS _spectrum_math_cpp LIBRARY DESTINATION maldiassist)
