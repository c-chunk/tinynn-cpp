cmake_minimum_required(VERSION 3.16)

project(tinynn-cpp
  VERSION 0.1.0
  DESCRIPTION "Tiny neural network framework in C++20"
  LANGUAGES CXX
)

# =========================
# Options
# =========================
option(TINYNN_BUILD_EXAMPLES "Build examples" ON)
option(TINYNN_BUILD_TESTS "Build tests" ON)

# =========================
# C++ Standard
# =========================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =========================
# Library (header-only style)
# =========================
add_library(tinynn INTERFACE)

target_include_directories(tinynn
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# =========================
# Examples
# =========================
if (TINYNN_BUILD_EXAMPLES)

  # ---- minimal ----
  add_executable(example_minimal
    examples/minimal/main.cc
  )
  target_link_libraries(example_minimal PRIVATE tinynn)

  # ---- qmnist ----
  add_executable(example_qmnist_mlp
    examples/qmnist/fit_mlp.cc
  )
  target_link_libraries(example_qmnist_mlp PRIVATE tinynn)

  add_executable(example_qmnist_cnn
    examples/qmnist/qmnist_cnn.cc
  )
  target_link_libraries(example_qmnist_cnn PRIVATE tinynn)

  add_executable(example_trainer_mlp
    examples/qmnist/trainer_mlp.cc
  )
  target_link_libraries(example_trainer_mlp PRIVATE tinynn)

endif()

# =========================
# Tests
# =========================
if (TINYNN_BUILD_TESTS)

  enable_testing()

  file(GLOB_RECURSE TINYNN_TEST_SOURCES
    tests/*.cc
  )

  foreach(test_src ${TINYNN_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)

    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE tinynn)

    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()

endif()
