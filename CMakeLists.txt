cmake_minimum_required(VERSION 3.20)

project(ml-project LANGUAGES CXX)

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

if (MSVC)
  target_compile_options(ml-project PRIVATE
        /W4
        /permissive-
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<CONFIG:Release>:/O2>
    )
else()
  target_compile_options(ml-project PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        $<$<CONFIG:Debug>:-g -O0>
        $<$<CONFIG:Release>:-O3>
    )
endif()
