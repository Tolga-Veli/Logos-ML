cmake_minimum_required(VERSION 3.20)

project(Logos VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# neovim / clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(ROOT    "${CMAKE_CURRENT_SOURCE_DIR}")
set(SRC_DIR "${ROOT}/src")

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
  "${SRC_DIR}/*.inl"
  "${SRC_DIR}/*.cpp"
)

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS
  "${SRC_DIR}/*.hpp"
)

add_executable(Logos ${SOURCES} ${HEADERS})
target_include_directories(Logos PRIVATE  "${SRC_DIR}")

set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY VS_STARTUP_PROJECT Logos)

source_group(TREE "${SRC_DIR}" PREFIX "src" FILES
  ${SOURCES} ${HEADERS}
)

if (MSVC)
  target_compile_options(Logos PRIVATE /W4 /permissive-)
  target_compile_options(Logos PRIVATE 
    $<$<CONFIG:Debug>:/Od /Zi>
    $<$<CONFIG:Release>:/O2>
  )
else()
  target_compile_options(Logos PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(Logos PRIVATE 
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3>
  )
endif()