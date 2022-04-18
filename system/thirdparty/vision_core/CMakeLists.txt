# Copyright (c) 2016, Robert Lukierski.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# 
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 
# Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ------------------------------------------------------------------------------

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
# TODO: set it here instead of the default one above when a different cuda compiler is needed
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.1/bin/nvcc")


cmake_minimum_required(VERSION 3.9)
if(POLICY_CMP0058)
    cmake_policy(SET CMP0058 OLD) # Workaround for Ninja + FindCUDA
endif()
if(POLICY_CMP0007)
    cmake_policy(SET CMP0007 OLD) # Workaround FindCUDA
endif()

# ------------------------------------------------------------------------------
# Default build configuration
# ------------------------------------------------------------------------------
if(DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Build type.")
else()
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type.")
endif()

# ------------------------------------------------------------------------------
# Project name & version
# ------------------------------------------------------------------------------
project(VisionCore LANGUAGES CXX CUDA)
set(${PROJECT_NAME}_VERSION 1.0.0)


# ------------------------------------------------------------------------------
# Enable C++11 and high level warnings
# ------------------------------------------------------------------------------
#add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
#add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
#add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-pedantic>)
#add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>)
#add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-variable>)

#include(CheckCXXCompilerFlag)
#check_cxx_compiler_flag("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
#if(COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
#    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
#    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mtune=native>)
#endif()

# ------------------------------------------------------------------------------
# Ready for testing
# ------------------------------------------------------------------------------
option(BUILD_TESTS "Enable to build unit tests" ON)
find_package(GTest QUIET)
if(BUILD_TESTS AND GTest_FOUND)
    enable_testing()
      
    # Code Coverage Build Type
    set(CMAKE_CXX_FLAGS_COVERAGE "-g -O0 --coverage -fprofile-arcs -ftest-coverage" CACHE STRING "Flags used by the C++ compiler during coverage builds." FORCE)
    set(CMAKE_C_FLAGS_COVERAGE "-g -O0 --coverage -fprofile-arcs -ftest-coverage" CACHE STRING "Flags used by the C compiler during coverage builds." FORCE )
    set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "" CACHE STRING "Flags used for linking binaries during coverage builds." FORCE )
    set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "" CACHE STRING "Flags used by the shared libraries linker during coverage builds." FORCE)
    mark_as_advanced(CMAKE_CXX_FLAGS_COVERAGE CMAKE_C_FLAGS_COVERAGE CMAKE_EXE_LINKER_FLAGS_COVERAGE CMAKE_SHARED_LINKER_FLAGS_COVERAGE)
    message("Testing enabled")
endif()

include(GenerateExportHeader)

# ---------------------------------------------
# Where to look for Find*.cmake
# ---------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts)

option(USE_GLBINDING "Use glbinding & globjects instead" OFF)
option(USE_OPENCL "Use OpenCL" ON)

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
find_package(Eigen3 REQUIRED QUIET)
find_package(Sophus REQUIRED QUIET)
find_package(TBB REQUIRED QUIET)
find_package(Boost COMPONENTS system REQUIRED QUIET)

if(COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
    set(CMAKE_REQUIRED_DEFINITIONS -march=native -mtune=native)
endif()
set(CMAKE_REQUIRED_INCLUDES ${EIGEN3_INCLUDE_DIR})
try_run(EIGEN_IDEAL_MAX_ALIGN_BYTES EIGEN_GET_ALIGNMENT_OK 
            ${CMAKE_BINARY_DIR} 
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/EigenAlignmentCheck.cpp
            COMPILE_DEFINITIONS ${CMAKE_REQUIRED_DEFINITIONS}
            CMAKE_FLAGS -DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES})
if(EIGEN_GET_ALIGNMENT_OK)
    message("Eigen Host Alignment: ${EIGEN_IDEAL_MAX_ALIGN_BYTES}")
endif()

# TODO FIXME not supported by the mainline Eigen
set(EIGEN_GET_ALIGNMENT_OK FALSE)

if(USE_GLBINDING)
    find_package(glbinding REQUIRED QUIET)
    find_package(globjects REQUIRED QUIET)
else()
    find_package(OpenGL REQUIRED QUIET)
    find_package(GLEW REQUIRED QUIET)
endif()

function(JOIN VALUES GLUE OUTPUT)
  string (REPLACE ";" "${GLUE}" _TMP_STR "${VALUES}")
  set (${OUTPUT} "${_TMP_STR}" PARENT_SCOPE)
endfunction()

find_package(CUDA QUIET)
if(CUDA_FOUND)
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_NVCC_ARCH_FLAGS Auto)
    set(CMAKE_CUDA_FLAGS ${CUDA_NVCC_ARCH_FLAGS};--expt-relaxed-constexpr;--expt-extended-lambda;-use_fast_math)
    # suppress NVCC warning triggered in Sophus:
    # "__device__ annotation is ignored on a function that is explicitly defaulted on its first declaration"
    # see: https://github.com/kokkos/kokkos/issues/1473
    list(APPEND CMAKE_CUDA_FLAGS -Xcudafe;--diag_suppress=esa_on_defaulted_function_ignored)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        list(APPEND CMAKE_CUDA_FLAGS -G;-g;-lineinfo;-Xcompiler;-rdynamic)
    elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
        list(APPEND CMAKE_CUDA_FLAGS -g;-lineinfo)
    endif()
    join("${CMAKE_CUDA_FLAGS}" " " CMAKE_CUDA_FLAGS)
    # Fix for CUDA9/GCC 7.2.0
    #
    # need also fix for __builtin_addressof in /usr/lib/gcc/x86_64-pc-linux-gnu/7.2.0/include/g++-v7/bits/move.h
    # #ifndef __CUDACC__
    #   template<typename _Tp>
    #   inline _GLIBCXX_CONSTEXPR _Tp*
    #   __addressof(_Tp& __r) _GLIBCXX_NOEXCEPT 
    #   { return __builtin_addressof(__r); }
    # #else // __CUDACC__
    #   template<typename _Tp>
    #   inline _Tp*
    #   __addressof(_Tp& __r) _GLIBCXX_NOEXCEPT
    #   { return reinterpret_cast<_Tp*> (&const_cast<char&>(reinterpret_cast<const volatile char&>(__r))); }
    # #endif // __CUDACC__
    #
    # and /opt/cuda/include/crt/host_config.h
    #
    # and possibly, if using glibc 2.26, define #define _BITS_FLOATN_H in /opt/cuda/include/host_defines.h
    #
    list(APPEND CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}")
    message("CUDA Flags: ${CMAKE_CUDA_FLAGS}")
endif()

find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    message("OpenCL Version: ${OpenCL_VERSION_MAJOR}.${OpenCL_VERSION_MINOR}")
endif()

# ------------------------------------------------------------------------------
# Print Project Info
# ------------------------------------------------------------------------------
message("Project: ${PROJECT_NAME} / ${${PROJECT_NAME}_VERSION}, build type: ${CMAKE_BUILD_TYPE}, compiled on: ${CMAKE_SYSTEM}, flags: ${CMAKE_CXX_FLAGS}, GLBinding: ${USE_GLBINDING} CUDA: ${CUDA_FOUND} OpenCL: ${OpenCL_FOUND}")

find_package(OpenCV 4 QUIET)
find_package(Ceres QUIET)
find_package(cereal QUIET)
find_package(Vulkan QUIET)
find_package(FREEGLUT QUIET)
find_package(X11 QUIET)
find_package(FFTW QUIET)

# ---------------------------------------------
# List of sources
# ---------------------------------------------
set(HEADERS
include/VisionCore/CUDAException.hpp
include/VisionCore/CUDAGenerics.hpp
include/VisionCore/HelpersAutoDiff.hpp
include/VisionCore/HelpersCUDA.hpp
include/VisionCore/HelpersEigen.hpp
include/VisionCore/HelpersMisc.hpp
include/VisionCore/HelpersSophus.hpp
include/VisionCore/LaunchUtils.hpp
include/VisionCore/MemoryPolicyCUDA.hpp
include/VisionCore/MemoryPolicyHost.hpp
include/VisionCore/MemoryPolicy.hpp
include/VisionCore/MemoryPolicyOpenCL.hpp
include/VisionCore/Platform.hpp
include/VisionCore/TypeTraits.hpp
include/VisionCore/Buffers/Buffer1D.hpp
include/VisionCore/Buffers/Buffer2D.hpp
include/VisionCore/Buffers/Buffer3D.hpp
include/VisionCore/Buffers/BufferPyramid.hpp
include/VisionCore/Buffers/CUDATexture.hpp
include/VisionCore/Buffers/GPUVariable.hpp
include/VisionCore/Buffers/Image2D.hpp
include/VisionCore/Buffers/ImagePyramid.hpp
include/VisionCore/Buffers/PyramidBase.hpp
include/VisionCore/Buffers/Reductions.hpp
include/VisionCore/Buffers/Volume.hpp
include/VisionCore/Control/PID.hpp
include/VisionCore/Control/VelocityProfile.hpp
include/VisionCore/Image/BufferOps.hpp
include/VisionCore/Image/ColorMap.hpp
include/VisionCore/Image/ConnectedComponents.hpp
include/VisionCore/Image/Filters.hpp
include/VisionCore/Image/ImagePatch.hpp
include/VisionCore/Image/PixelConvert.hpp
include/VisionCore/IO/File.hpp
include/VisionCore/IO/ImageIO.hpp
include/VisionCore/IO/PLYModel.hpp
include/VisionCore/Math/Angles.hpp
include/VisionCore/Math/Convolution.hpp
include/VisionCore/Math/DenavitHartenberg.hpp
include/VisionCore/Math/Divergence.hpp
include/VisionCore/Math/Fitting.hpp
include/VisionCore/Math/Fourier.hpp
include/VisionCore/Math/HammingDistance.hpp
include/VisionCore/Math/Kalman.hpp
include/VisionCore/Math/LeastSquares.hpp
include/VisionCore/Math/LiangBarsky.hpp
include/VisionCore/Math/LocalParamSE3.hpp
include/VisionCore/Math/LossFunctions.hpp
include/VisionCore/Math/PolarSpherical.hpp
include/VisionCore/Math/Random.hpp
include/VisionCore/Math/RANSAC.hpp
include/VisionCore/Math/Statistics.hpp
include/VisionCore/Types/AxisAlignedBoundingBox.hpp
include/VisionCore/Types/CostVolumeElement.hpp
include/VisionCore/Types/Gaussian.hpp
include/VisionCore/Types/Hypersphere.hpp
include/VisionCore/Types/Polynomial.hpp
include/VisionCore/Types/Rectangle.hpp
include/VisionCore/Types/SDF.hpp
include/VisionCore/Types/SquareUpperTriangularMatrix.hpp
include/VisionCore/WrapGL/WrapGLBuffer.hpp
include/VisionCore/WrapGL/WrapGLCommon.hpp
include/VisionCore/WrapGL/WrapGLContext.hpp
include/VisionCore/WrapGL/WrapGLFramebuffer.hpp
include/VisionCore/WrapGL/WrapGL.hpp
include/VisionCore/WrapGL/WrapGLQuery.hpp
include/VisionCore/WrapGL/WrapGLSampler.hpp
include/VisionCore/WrapGL/WrapGLProgram.hpp
include/VisionCore/WrapGL/WrapGLTexture.hpp
include/VisionCore/WrapGL/WrapGLTransformFeedback.hpp
include/VisionCore/WrapGL/WrapGLVertexArrayObject.hpp
)

set(SOURCES
sources/Image/BufferOpsCPU.cpp
sources/Image/ColorMapCPU.cpp
sources/Image/ConnectedComponents.cpp
sources/Image/FiltersCPU.cpp
sources/Image/PixelConvertCPU.cpp
sources/Image/ColorMapDefs.hpp
sources/Image/JoinSplitHelpers.hpp
sources/IO/ImageIO.cpp
sources/IO/ImageUtilsCPU.cpp
sources/IO/PLYModel.cpp
sources/IO/SaveBuffer.cpp
sources/Math/ConvolutionCPU.cpp
sources/VisionCore.cpp
sources/WrapGL/WrapGLBuffer.cpp
sources/WrapGL/WrapGLCommon.cpp
sources/WrapGL/WrapGLFramebuffer.cpp
)

set(KERNEL_SOURCES
sources/Image/BufferOpsGPU.cpp
sources/Image/ColorMapGPU.cpp
sources/Image/FiltersGPU.cpp
sources/Image/PixelConvertGPU.cpp
sources/Math/ConvolutionGPU.cpp
sources/Math/FourierGPU.cpp
)

if(CUDA_FOUND)
    list(APPEND SOURCES sources/Math/Random.cpp)
endif()

if(USE_GLBINDING AND glbinding_FOUND)
    if(FREEGLUT_FOUND)
        list(APPEND SOURCES sources/WrapGL/context_glut.cpp)
    endif()

    if(X11_FOUND)
        list(APPEND SOURCES sources/WrapGL/context_windowless.cpp sources/WrapGL/context_x11.cpp)
    endif()
endif()

if(FFTW_FOUND)
    list(APPEND SOURCES sources/Math/FourierCPU.cpp)
endif()

set_source_files_properties(${KERNEL_SOURCES} PROPERTIES LANGUAGE CUDA)

# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------
if(CUDA_FOUND)
    add_library(${PROJECT_NAME} SHARED ${HEADERS} ${SOURCES} ${KERNEL_SOURCES})
else()
    add_library(${PROJECT_NAME} SHARED ${HEADERS} ${SOURCES})
endif()

check_cxx_compiler_flag(-std=c++17 HAVE_FLAG_STD_CXX17)
check_cxx_compiler_flag(-std=c++1z HAVE_FLAG_STD_CXX1Z)
check_cxx_compiler_flag(-std=c++14 HAVE_FLAG_STD_CXX14)
check_cxx_compiler_flag(-std=c++1y HAVE_FLAG_STD_CXX1Y)

if(HAVE_FLAG_STD_CXX17 OR HAVE_FLAG_STD_CXX1Z)
    target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14) # TODO CUDA not ready
elseif(HAVE_FLAG_STD_CXX14 OR HAVE_FLAG_STD_CXX1Y)
    target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)
else()
    target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:>
    PRIVATE sources)
    
target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sophus::Sophus)

if(CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_curand_LIBRARY} ${CUDA_CUFFT_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME} PUBLIC VISIONCORE_HAVE_CUDA)
    
    if(EIGEN_GET_ALIGNMENT_OK)
        target_compile_definitions(${PROJECT_NAME} PUBLIC EIGEN_CUDA_MAX_ALIGN_BYTES=${EIGEN_IDEAL_MAX_ALIGN_BYTES})
    endif()
endif()

if(OpenCL_FOUND AND USE_OPENCL)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenCL::OpenCL)
    target_compile_definitions(${PROJECT_NAME} PUBLIC VISIONCORE_HAVE_OPENCL CL_HPP_MINIMUM_OPENCL_VERSION=120 CL_HPP_TARGET_OPENCL_VERSION=${OpenCL_VERSION_MAJOR}${OpenCL_VERSION_MINOR}0 CL_HPP_ENABLE_EXCEPTIONS)
endif()

if(TBB_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${TBB_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PUBLIC ${TBB_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME} PUBLIC VISIONCORE_HAVE_TBB ${TBB_DEFINITIONS})
endif()

if(OpenCV_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME} PUBLIC VISIONCORE_HAVE_OPENCV)
endif()

if(cereal_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC cereal)
    target_compile_definitions(${PROJECT_NAME} PUBLIC VISIONCORE_HAVE_CEREAL)
endif()

if(CERES_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${CERES_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PUBLIC ${CERES_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME} PUBLIC ${CERES_DEFINITIONS} VISIONCORE_HAVE_CERES)
endif()

if(USE_GLBINDING AND glbinding_FOUND)
    target_compile_definitions(${PROJECT_NAME} PUBLIC VISIONCORE_HAVE_GLBINDING)
    target_link_libraries(${PROJECT_NAME} PUBLIC glbinding)
    
    if(FREEGLUT_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${FREEGLUT_LIBRARIES})
    endif()
    if(X11_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${X11_LIBRARIES})
    endif()
    
    if(globjects_FOUND)
        target_link_libraries(${PROJECT_NAME} PUBLIC globjects)
    endif()
else()
    # Just basic OpenGL functionality
    target_include_directories(${PROJECT_NAME} PUBLIC ${OPENGL_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OPENGL_LIBRARIES})
    target_link_libraries(${PROJECT_NAME} PUBLIC GLEW::GLEW)
endif()

if(Vulkan_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC Vulkan::Vulkan)
endif()

if(FFTW_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${FFTW_INCLUDES})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${FFTW_LIBRARIES})
endif()

# ------------------------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------------------------
if(BUILD_TESTS AND GTest_FOUND)
    add_subdirectory(tests)
endif()

# ------------------------------------------------------------------------------
# Installation - library
# ------------------------------------------------------------------------------
generate_export_header(${PROJECT_NAME} EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_export.h)
set_property(TARGET ${PROJECT_NAME} PROPERTY VERSION ${${PROJECT_NAME}_VERSION})
set_property(TARGET ${PROJECT_NAME} PROPERTY SOVERSION 1)
set_property(TARGET ${PROJECT_NAME} PROPERTY INTERFACE_${PROJECT_NAME}_MAJOR_VERSION 1)
set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY COMPATIBLE_INTERFACE_STRING ${PROJECT_NAME}_MAJOR_VERSION)

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_export.h" DESTINATION include/VisionCore)
install(DIRECTORY include/VisionCore DESTINATION include)


# ------------------------------------------------------------------------------
# Installation - generate version file
# ------------------------------------------------------------------------------
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${${PROJECT_NAME}_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# ------------------------------------------------------------------------------
# Installation - Export Targets
# ------------------------------------------------------------------------------
export(EXPORT ${PROJECT_NAME}Targets FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake") 
configure_file(${PROJECT_NAME}Config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" @ONLY)
set(ConfigPackageLocation lib/cmake/${PROJECT_NAME})
install(EXPORT ${PROJECT_NAME}Targets FILE ${PROJECT_NAME}Targets.cmake DESTINATION ${ConfigPackageLocation})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake" DESTINATION ${ConfigPackageLocation})
export(PACKAGE ${PROJECT_NAME})
