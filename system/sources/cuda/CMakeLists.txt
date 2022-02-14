# ------------------------------------------------------------------------------
# Set NVCC flags
# ------------------------------------------------------------------------------
# set(DF_CUDA_ARCH Auto CACHE STRING "A list of CUDA architectures to compile for. Specifying 'Auto' will attempt to autodetect available GPU devices")
# CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_NVCC_ARCH_FLAGS ${DF_CUDA_ARCH})
# set(CMAKE_CUDA_FLAGS ${CUDA_NVCC_ARCH_FLAGS};--expt-relaxed-constexpr;--expt-extended-lambda;--use_fast_math)
# message("Compiling for CUDA architectures: ${CUDA_NVCC_ARCH_FLAGS}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
#     set(CMAKE_CUDA_ARCHITECTURES 52 61 75)
# endif()
# message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")


set(cuda_sources
  geometric_factor_kernels.cpp
  reprojection_factor_kernels.cpp
  match_geometry_factor_kernels.cpp
  photometric_factor_kernels.cpp
)

set(cuda_headers
  geometric_factor_kernels.h
  reprojection_factor_kernels.h
  match_geometry_factor_kernels.h
  photometric_factor_kernels.h
)

set_source_files_properties(${cuda_sources} PROPERTIES LANGUAGE CUDA)

# ------------------------------------------------------------------------------
# Build target
# ------------------------------------------------------------------------------
add_library(df_cuda STATIC ${cuda_sources} ${cuda_headers})
target_include_directories(df_cuda PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(df_cuda 
    PUBLIC
    df_common
    ${TORCH_LIBRARIES}
)

set_target_properties(df_cuda PROPERTIES CUDA_STANDARD 14)
set_target_properties(df_cuda PROPERTIES RELOCATABLE_DEVICE_CODE ON)
set_target_properties(df_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
