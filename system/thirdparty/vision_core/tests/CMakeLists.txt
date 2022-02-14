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
# Dependencies
# ------------------------------------------------------------------------------
# Google Test
find_package(GTest REQUIRED QUIET)
include_directories(${GTEST_INCLUDE_DIR})

# Sophus (Optional)
find_package(Sophus QUIET)
if(Sophus_FOUND)
    include_directories(${Sophus_INCLUDE_DIRS})
    add_definitions(-DCORE_TESTS_HAVE_SOPHUS)
endif()

include_directories(.)

# ------------------------------------------------------------------------------
# Build subtests
# ------------------------------------------------------------------------------
add_subdirectory(Buffers)
add_subdirectory(Control)
add_subdirectory(IO)
add_subdirectory(Image)
add_subdirectory(Math)
add_subdirectory(Types)
add_subdirectory(WrapGL)

# =========================================================================
# Build top level tests
# =========================================================================
set(TEST_SOURCES
tests_main.cpp
UT_Platform.cpp
EigenConfigCPU.cpp
UT_EigenConfig.cpp
)

if(OpenCL_FOUND)
    list(APPEND TEST_SOURCES UT_OpenCL.cpp)
endif()

if(CUDA_FOUND)
set(TEST_KERNEL_SOURCES
EigenConfigCUDA.cu
)
endif()

add_executable(UT_VisionCore ${TEST_SOURCES} ${TEST_KERNEL_SOURCES})
target_link_libraries(UT_VisionCore PUBLIC ${GTEST_LIBRARY} ${PROJECT_NAME})
if(CUDA_FOUND)
    target_link_libraries(UT_VisionCore PUBLIC ${CUDA_LIBRARIES})
endif()
if(OpenCL_FOUND)
    target_link_libraries(UT_VisionCore PUBLIC ${OpenCL_LIBRARIES})
endif()
add_test(UT_VisionCore UT_VisionCore --gtest_output=xml:UT_VisionCore.xml)
