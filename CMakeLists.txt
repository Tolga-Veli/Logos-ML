cmake_minimum_required(VERSION 3.20)

project(ml-project LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate compile_commands.json for clangd (neovim)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(BLAS REQUIRED)

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
  "${SRC_DIR}/*.inl"
  "${SRC_DIR}/*.cpp"
)

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS
  "${SRC_DIR}/*.hpp"
)

add_executable(ml-project ${SOURCES} ${HEADERS})
target_include_directories(ml-project PRIVATE  "${SRC_DIR}")

target_link_libraries(ml-project PRIVATE BLAS::BLAS)

if (BUILD_TESTING)
  add_executable(ml-tests tests/tests.cpp)
  target_include_directories(ml-tests PRIVATE "${SRC_DIR}")
  target_link_libraries(ml-tests PRIVATE BLAS::BLAS)
  add_test(NAME ml-tests COMMAND ml-tests)
endif()

if (MSVC)
  target_compile_options(ml-project PRIVATE
        /W4
        /permissive-
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<CONFIG:Release>:/O2>
    )
  if (BUILD_TESTING)
    target_compile_options(ml-tests PRIVATE /W4 /permissive-)
  endif()
else()
  target_compile_options(ml-project PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        $<$<CONFIG:Debug>:-g -O0>
        $<$<CONFIG:Release>:-O3>
    )
  if (BUILD_TESTING)
    target_compile_options(ml-tests PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()
