cmake_minimum_required(VERSION 3.14)

project(
  ptitrace
  VERSION 0.0.1
  LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_VISIBILITY_PRESET "hidden")

set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)

FetchContent_MakeAvailable(json)


add_executable(pti-collector "${PROJECT_SOURCE_DIR}/pti-collector.cc")

find_package(Pti QUIET)

if(NOT Pti::pti_view)
  FetchContent_Declare(
    pti-sdk
    GIT_REPOSITORY https://github.com/intel/pti-gpu.git
    GIT_TAG master
    SOURCE_SUBDIR sdk
  )
  set(PTI_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(PTI_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(pti-sdk)
endif()

target_link_libraries(pti-collector PUBLIC Pti::pti_view)

add_library(pti-adapter SHARED "${PROJECT_SOURCE_DIR}/pti-adapter.cc")

target_link_libraries(pti-adapter PUBLIC Pti::pti_view PRIVATE nlohmann_json::nlohmann_json)

target_link_libraries(pti-collector PUBLIC pti-adapter)
