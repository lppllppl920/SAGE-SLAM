cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

set(GLIBCXX_USE_CXX11_ABI 1)


# ------------------------------------------------------------------------------
# Project name
# ------------------------------------------------------------------------------
project(mydeepfactors LANGUAGES CUDA CXX C)
set(${PROJECT_NAME}_VERSION 0.1)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CheckLanguage)
check_language(CUDA)
# # ------------------------------------------------------------------------------
# # Default build configuration
# # ------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif() 

# output all executables to a single dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# ------------------------------------------------------------------------------
# CMake modules directory
# ------------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/thirparty/install_${CMAKE_BUILD_TYPE}")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# ------------------------------------------------------------------------------
# Build Options
# ------------------------------------------------------------------------------
option(DF_BUILD_TESTS "Build unit tests" OFF)
option(DF_ENABLE_TIMING "Enable timing of crucial parts of the system" OFF)
option(DF_WITH_FLYCAP "Compile support for PoinGrey FlyCapture SDK cameras" OFF)

## Allow selecting the network code_size at compile-time
set(DF_CODE_SIZE 16)
set(DF_FEAT_SIZE 16)

add_definitions(-DDF_CODE_SIZE=${DF_CODE_SIZE})
add_definitions(-DDF_FEAT_SIZE=${DF_FEAT_SIZE})

# # disable eigen vectorization and alignment
# # we were getting nasty segfaults heisenbugs with them
# add_definitions(-DEIGEN_DONT_VECTORIZE 
#                 -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)

message(STATUS "Code Size: ${DF_CODE_SIZE}")
message(STATUS "Feat Size: ${DF_FEAT_SIZE}")

if(DF_ENABLE_TIMING)
  message(STATUS "adding time enabling")
  add_definitions(-DENABLE_TIMING)
endif()

# ------------------------------------------------------------------------------
# Compile Options
# ------------------------------------------------------------------------------
# enable c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_COMPILE_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_CREATE_SHARED_LIBRARY ON)

set(CMAKE_VERBOSE_MAKEFILE ON)
# enable warnings
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-dev>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)

# ------------------------------------------------------------------------------
# Print Project Info
# ------------------------------------------------------------------------------
message("Project: ${PROJECT_NAME}")
message("Version: ${${PROJECT_NAME}_VERSION}")
message("Build type: ${CMAKE_BUILD_TYPE}")
message("Compiled on: ${CMAKE_SYSTEM}")

# ------------------------------------------------------------------------------
# Dependencies
# For the downloaded deps, we rely here on XXX_DIR cache variables
# that were set by thirdparty/CMakeLists.txt file parsed earlier
# ------------------------------------------------------------------------------
add_subdirectory(thirdparty)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
find_package(CUDA         REQUIRED)

find_package(PythonInterp REQUIRED)
set(Torch_PATH "/opt/conda/lib/python3.8/site-packages/torch/share/cmake/Torch" CACHE STRING "Directory of Torch")
set(Torch_DIR ${Torch_PATH})
find_package(Torch        REQUIRED)

find_package(Boost        REQUIRED COMPONENTS filesystem)
find_package(OpenCV       REQUIRED)
find_package(HDF5         REQUIRED COMPONENTS C CXX)
find_package(PkgConfig    REQUIRED)
pkg_check_modules(glog REQUIRED libglog)
find_package(gflags       REQUIRED)

pkg_check_modules(Curses REQUIRED ncurses)

set(Eigen3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/install_${CMAKE_BUILD_TYPE}/share/eigen3/cmake")
find_package(Eigen3       REQUIRED)

set(Sophus_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/install_${CMAKE_BUILD_TYPE}/share/sophus/cmake")
message("Sophus ${Sophus_DIR}")
find_package(Sophus       REQUIRED)

set(opengv_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/install_${CMAKE_BUILD_TYPE}/lib/cmake/opengv-1.0")
find_package(opengv       REQUIRED)

set(GTSAM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/install_${CMAKE_BUILD_TYPE}/lib/cmake/GTSAM")
find_package(GTSAM        REQUIRED)

set(teaserpp_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/install_${CMAKE_BUILD_TYPE}/lib/cmake/teaserpp")
find_package(teaserpp     REQUIRED)

set(CMAKE_DISABLE_FIND_PACKAGE_Eigen3 TRUE)

set(Pangolin_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/install_${CMAKE_BUILD_TYPE}/lib/cmake/Pangolin")
find_package(Pangolin     REQUIRED)

set(VisionCore_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/build_${CMAKE_BUILD_TYPE}/vision_core")
find_package(VisionCore     REQUIRED)

set(CameraDrivers_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/build_${CMAKE_BUILD_TYPE}/camera_drivers")
find_package(CameraDrivers  REQUIRED)

set(DBoW2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/build_${CMAKE_BUILD_TYPE}/DBoW2")
find_package(DBoW2          REQUIRED)

add_definitions(-DC10_USE_GLOG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${HDF5_INCLUDE_DIRS})
include_directories(${INTERFACE_INCLUDE_DIRECTORIES})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${glog_INCLUDE_DIRS})

# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------
add_subdirectory(sources)

# ------------------------------------------------------------------------------
# Installation - generate version file
# ------------------------------------------------------------------------------
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${${PROJECT_NAME}_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# export package to cmake cache
export(PACKAGE ${PROJECT_NAME})


