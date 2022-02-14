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

cmake_minimum_required(VERSION 3.1)

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
project(CameraDrivers)
set(${PROJECT_NAME}_VERSION 1.0.0)

# ------------------------------------------------------------------------------
# Enable C++11 and high level warnings
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 11) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -pedantic -Wno-unused-parameter -Wno-unused-variable)

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
endif()

include(GenerateExportHeader)

# ---------------------------------------------
# Where to look for Find*.cmake
# ---------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts)

# ------------------------------------------------------------------------------
# Print Project Info
# ------------------------------------------------------------------------------
message("Project: ${PROJECT_NAME} / ${${PROJECT_NAME}_VERSION}, build type: ${CMAKE_BUILD_TYPE}, compiled on: ${CMAKE_SYSTEM}, flags: ${CMAKE_CXX_FLAGS}")

# --------------------------------------------------------------------------------
# External dependencies
# --------------------------------------------------------------------------------
find_package(cereal QUIET)
find_package(CameraModels QUIET)

# Hardware
find_package(FlyCapture2 QUIET)
find_package(DC1394 QUIET)
find_package(VRMagicSDK QUIET)
if(VRMAGICSDK_FOUND)
    find_package(Boost COMPONENTS thread system QUIET)
    if(NOT Boost_FOUND)
        unset(VRMAGICSDK_FOUND)
    endif()
endif()
find_package(OpenNI2 QUIET)
find_package(realsense QUIET)
find_package(realsense2 QUIET)
find_package(freenect2 QUIET)
find_package(IDSuEye QUIET)
find_package(V4L2 QUIET)

# --------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------
set(HEADERS
include/CameraDrivers.hpp
include/FireWireDriver.hpp
include/KinectOneDriver.hpp
include/OpenNIDriver.hpp
include/PointGreyDriver.hpp
include/RealSenseDriver.hpp
include/V4LDriver.hpp
include/VRMagicDriver.hpp
#include/IDSuEyeDriver.hpp
)

set(SOURCES
sources/CameraDriver.cpp 
)

if(FLYCAPTURE2_FOUND)
    message("FOUND: PointGrey FlyCapture2")
    set(CAMERADRIVERS_SUPPORTS_POINTGREY ON)
    list(APPEND SOURCES sources/PointGreyDriver.cpp)
endif()

if(DC1394_FOUND)
    message("FOUND: FireWire")
    set(CAMERADRIVERS_SUPPORTS_FIREWIRE ON)
    list(APPEND SOURCES sources/FireWireDriver.cpp)
endif()

if(VRMAGICSDK_FOUND)
    message("FOUND: VRMagic")
    set(CAMERADRIVERS_SUPPORTS_VRMAGIC ON)
    list(APPEND SOURCES sources/VRMagicDriver.cpp ${VRMAGICSDK_CPP_WRAPPER})
    set_source_files_properties(${VRMAGICSDK_CPP_WRAPPER} PROPERTIES COMPILE_FLAGS "-Wno-overflow -Wno-pedantic")
endif()

if(OPENNI2_FOUND)
    message("FOUND: OpenNI2")
    set(CAMERADRIVERS_SUPPORTS_OPENNI2 ON)
    list(APPEND SOURCES sources/OpenNIDriver.cpp)
endif()

if(realsense_FOUND)
    message("FOUND: librealsense")
    set(CAMERADRIVERS_SUPPORTS_REALSENSE ON)
    list(APPEND SOURCES sources/RealSenseDriver.cpp)
endif()

if(realsense2_FOUND)
    message("FOUND: librealsense2")
    set(CAMERADRIVERS_SUPPORTS_REALSENSE2 ON)
    #list(APPEND SOURCES sources/RealSense2Driver.cpp) TODO FIXME
endif()

if(freenect2_FOUND)
    message("FOUND: freenect2")
    set(CAMERADRIVERS_SUPPORTS_KINECTONE ON)
    list(APPEND SOURCES sources/KinectOneDriver.cpp)
endif()

if(IDSUEYE_FOUND)
    message("FOUND: IDS uEye")
    set(CAMERADRIVERS_SUPPORTS_IDSUEYE ON)
    #list(APPEND SOURCES sources/IDSuEyeDriver.cpp) TODO FIXME
endif()

if(V4L2_FOUND)
    message("FOUND: V4L2")
    set(CAMERADRIVERS_SUPPORTS_V4L2 ON)
    #list(APPEND SOURCES sources/V4LDriver.cpp) TODO FIXME
endif()

# --------------------------------------------------------------------------------
# Build library
# --------------------------------------------------------------------------------
add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE sources)
    
if(cereal_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC cereal)
    target_compile_definitions(${PROJECT_NAME} PUBLIC CAMERA_DRIVERS_HAVE_CEREAL)
endif()
if(CameraModels_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC CameraModels)
    target_compile_definitions(${PROJECT_NAME} PUBLIC CAMERA_DRIVERS_HAVE_CAMERA_MODELS)
endif()

if(FLYCAPTURE2_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${FLYCAPTURE2_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${FLYCAPTURE2_LIBRARY})
endif()
if(DC1394_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${DC1394_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${DC1394_LIBRARIES})
endif()
if(VRMAGICSDK_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${VRMAGICSDK_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${VRMAGICSDK_LIBRARIES} ${Boost_LIBRARIES})
endif()
if(freenect2_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${freenect2_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${freenect2_LIBRARIES})
endif()
if(OPENNI2_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENNI2_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OPENNI2_LIBRARIES})
endif()
if(realsense_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${realsense_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${realsense_LIBRARY})
endif()
if(realsense2_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${realsense2_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${realsense2_LIBRARY})
endif()
if(IDSUEYE_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${IDSUEYE_LIBRARY})
    target_include_directories(${PROJECT_NAME} PRIVATE ${IDSUEYE_INCLUDE_DIR})
endif()
if(V4L2_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${V4L2_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PRIVATE ${V4L2_INCLUDE_DIRS})
endif()

# ---------------------------------------------
# Unit Tests
# ---------------------------------------------
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
install(FILES ${HEADERS} "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_export.h" DESTINATION include)

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
