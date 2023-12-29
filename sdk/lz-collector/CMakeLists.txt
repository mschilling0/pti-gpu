if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                        "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules/")
endif()
cmake_minimum_required(VERSION 3.14)

project(lz-collector VERSION 0.0.1 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

find_package(Threads REQUIRED)
find_package(LevelZero REQUIRED)

add_library(lz-collector SHARED "${PROJECT_SOURCE_DIR}/src/lz-collector.cc")
add_library(Pti::lz-collector ALIAS lz-collector)
target_include_directories(
  lz-collector
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  PRIVATE ${PROJECT_SOURCE_DIR}/src
)
set_target_properties(
  lz-collector
  PROPERTIES VERSION "${PROJECT_VERSION}"
             SOVERSION "${PROJECT_VERSION_MAJOR}"
             POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(
  lz-collector
  PUBLIC
    $<$<TARGET_EXISTS:LevelZero::level-zero>:PTI_LEVEL_ZERO=1>
)
target_link_libraries(
  lz-collector
  PUBLIC
    Threads::Threads LevelZero::level-zero
)
target_compile_features(lz-collector PRIVATE cxx_std_17)

add_library(lz-view-collector SHARED
  "${PROJECT_SOURCE_DIR}/src/lz-collector-new.cc")
add_library(Pti::lz-view-collector ALIAS lz-view-collector)
target_include_directories(
  lz-view-collector
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  PRIVATE ${PROJECT_SOURCE_DIR}/src
)
set_target_properties(
  lz-view-collector
  PROPERTIES VERSION "${PROJECT_VERSION}"
             SOVERSION "${PROJECT_VERSION_MAJOR}"
             POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(
  lz-view-collector
  PUBLIC
    $<$<TARGET_EXISTS:LevelZero::level-zero>:PTI_LEVEL_ZERO=1>
)
target_link_libraries(
  lz-view-collector
  PUBLIC
    Threads::Threads LevelZero::level-zero
)
target_compile_features(lz-view-collector PRIVATE cxx_std_17)

add_subdirectory(dpc_gemm)
add_subdirectory(graph)
