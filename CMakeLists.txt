cmake_minimum_required(VERSION 3.20)

project(ml-project LANGUAGES CXX)

option(ML_PREPARE_MNIST "Download and prepare MNIST dataset for the example model" ON)
option(ML_BUILD_MNIST_EXAMPLE "Build the MNIST example" ON)
option(ML_BUILD_BENCHMARKS "Build the benchmarks" OFF)
option(ML_ENABLE_SANITIZERS, "Enabled sanitizers" OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Generate compile_commands.json for clangd (neovim)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(BLAS REQUIRED)

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
  "${SRC_DIR}/*.cpp"
)

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS
  "${INCLUDE_DIR}/*.hpp"
  "${INCLUDE_DIR}/*.inl"
)

add_library(ml ${SOURCES} ${HEADERS})
target_include_directories(ml PUBLIC ${INCLUDE_DIR} ${SRC_DIR})
add_library(ml::ml ALIAS ml)

target_link_libraries(ml PRIVATE BLAS::BLAS)

if (MSVC)
  target_compile_options(ml PRIVATE
        /W4
        /permissive-
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<CONFIG:Release>:/O2>
    )
else()
  target_compile_options(ml PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        $<$<CONFIG:Debug>:-g -O0>
        $<$<CONFIG:Release>:-O3>
    )

    if(ML_ENABLE_SANITIZERS)
      target_compile_options(ml PUBLIC -fsanitize=address, undefined -fno-omit-frame-pointer)
      target_link_options(ml PUBLIC -fsanitize=address,undefined)
      target_link_options(ml PUBLIC -fsanitize=address,undefined)
    endif()
endif()

if(ML_PREPARE_MNIST AND ML_BUILD_MNIST_EXAMPLE)
  find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)

  set(MNIST_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/download_mnist.py")
  set(MNIST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
  set(MNIST_VENV "${CMAKE_CURRENT_BINARY_DIR}/.venv")
  set(MNIST_DEPENDENCIES_STAMP "${MNIST_VENV}/.dependencies-installed")

  if (WIN32)
    set(MNIST_PYTHON "${MNIST_VENV}/Scripts/python.exe")
  else()
    set(MNIST_PYTHON "${MNIST_VENV}/bin/python")
  endif()

  set(MNIST_DATA_FILES
    "${MNIST_DATA_DIR}/train_images.bin"
    "${MNIST_DATA_DIR}/train_labels.bin"
    "${MNIST_DATA_DIR}/test_images.bin"
    "${MNIST_DATA_DIR}/test_labels.bin"
  )

  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/help2.py" "${MNIST_SCRIPT}" COPYONLY)

  add_custom_command(
    OUTPUT "${MNIST_DEPENDENCIES_STAMP}"
    COMMAND "${Python3_EXECUTABLE}" -m venv "${MNIST_VENV}"
    COMMAND "${MNIST_PYTHON}" -m pip install tensorflow tensorflow-datasets importlib-resources
    COMMAND "${CMAKE_COMMAND}" -E touch "${MNIST_DEPENDENCIES_STAMP}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
    COMMENT "Installing TensorFlow and TensorFlow Datasets"
    VERBATIM
    USES_TERMINAL
  )

  add_custom_command(
    OUTPUT ${MNIST_DATA_FILES}
    COMMAND "${MNIST_PYTHON}" "${MNIST_SCRIPT}"
    DEPENDS "${MNIST_DEPENDENCIES_STAMP}" "${CMAKE_CURRENT_SOURCE_DIR}/help2.py" "${MNIST_SCRIPT}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Downloading and preparing the MNIST dataset"
    VERBATIM
    USES_TERMINAL
  )
  add_executable(ml-mnist-example examples/main.cpp)
  target_link_libraries(ml-mnist-example PRIVATE ml::ml)

  add_custom_target(mnist-data DEPENDS ${MNIST_DATA_FILES})
  add_dependencies(ml-mnist-example mnist-data)
  target_compile_definitions(ml-mnist-example PRIVATE ML_MNIST_DATA_DIR="${MNIST_DATA_DIR}")

endif()



