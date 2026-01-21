cmake_minimum_required(VERSION 3.20)

project(Logos
    VERSION 1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/libs)

file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "src/*.hpp"
)

set(CMAKE_CXX_FLAGS "-O3 -march=native")

add_executable(Logos ${SOURCES})

