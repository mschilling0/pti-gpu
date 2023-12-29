cmake_minimum_required(VERSION 3.14)

project(PTI_Samples_sycl_graph CXX)

option(NEW_COLLECTOR "Build with new collector" OFF)

# TODO(matthew.schilling@intel.com): find_package(IntelSYCL)?
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fsycl" has_sycl)


if (has_sycl)
add_executable(sycl_graph main.cc)

target_link_options(sycl_graph PUBLIC -fsycl -gline-tables-only)
target_compile_options(sycl_graph PUBLIC -fsycl -gline-tables-only)

target_include_directories(sycl_graph PRIVATE
)

# Allow building in-source and out-of-source
##if (NOT TARGET Pti::pti_view)
#  find_package(Pti REQUIRED)
#endif()
#
if(NOT NEW_COLLECTOR)
  target_link_libraries(sycl_graph PUBLIC Pti::lz-collector)
else()
  target_link_libraries(sycl_graph PUBLIC Pti::lz-view-collector)
endif()

endif()
