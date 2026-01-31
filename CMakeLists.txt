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

file(GLOB_RECURSE SRC_SUBDIRS LIST_DIRECTORIES true
  "${SRC_DIR}/*"
)

set(ALL_INCLUDE_DIRS
  "${SRC_DIR}"
)

foreach(p ${SRC_SUBDIRS})
  if(IS_DIRECTORY "${p}")
    list(APPEND ALL_INCLUDE_DIRS "${p}")
  endif()
endforeach()

list(REMOVE_DUPLICATES ALL_INCLUDE_DIRS)


add_executable(Logos ${SOURCES} ${HEADERS})

target_include_directories(Logos PRIVATE
  ${ALL_INCLUDE_DIRS}
)

source_group(TREE "${SRC_DIR}" PREFIX "src" FILES
  ${SOURCES} ${HEADERS}
)

if (MSVC)
  target_compile_options(Logos PRIVATE /W4 /permissive-)
  target_compile_options(Logos PRIVATE /O2)
else()
  target_compile_options(Logos PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(Logos PRIVATE -O3)
endif()

