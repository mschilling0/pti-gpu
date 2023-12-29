cmake_minimum_required(VERSION 3.14)

project(PTI_Samples_DPC_GEMM CXX)

option(NEW_COLLECTOR "Build with new collector" OFF)

# TODO(matthew.schilling@intel.com): find_package(IntelSYCL)?
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fsycl" has_sycl)


if (has_sycl)
add_executable(dpc_gemm main.cc)

target_link_options(dpc_gemm PUBLIC -fsycl -gline-tables-only)
target_compile_options(dpc_gemm PUBLIC -fsycl -gline-tables-only)

target_include_directories(dpc_gemm PRIVATE
)

# Allow building in-source and out-of-source
##if (NOT TARGET Pti::pti_view)
#  find_package(Pti REQUIRED)
#endif()
#
if(NOT NEW_COLLECTOR)
  target_link_libraries(dpc_gemm PUBLIC Pti::lz-collector)
else()
  target_link_libraries(dpc_gemm PUBLIC Pti::lz-view-collector)
endif()

endif()
