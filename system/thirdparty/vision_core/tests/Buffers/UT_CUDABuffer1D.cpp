/**
 * ****************************************************************************
 * Copyright (c) 2015, Robert Lukierski.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * ****************************************************************************
 * Basic CUDA Buffer1D Tests.
 * ****************************************************************************
 */

// testing framework & libraries
#include <gtest/gtest.h>

// google logger
#include <glog/logging.h>

#include <VisionCore/Buffers/Buffer1D.hpp>

#include <VisionCore/LaunchUtils.hpp>
#include <BufferTestHelpers.hpp>

static constexpr std::size_t BufferSize = 4097;

template<typename T>
class Test_CUDABuffer1D : public ::testing::Test
{
public:   
    Test_CUDABuffer1D()
    {
        
    }
    
    virtual ~Test_CUDABuffer1D()
    {
        
    }
};

typedef ::testing::Types<float,Eigen::Vector3f,Sophus::SE3f> TypesToTest;
TYPED_TEST_CASE(Test_CUDABuffer1D, TypesToTest);

TYPED_TEST(Test_CUDABuffer1D, TestHostDevice) 
{
    typedef TypeParam BufferElementT;
  
    
    
    // CPU buffer 1
    vc::Buffer1DManaged<BufferElementT, vc::TargetHost> buffer_cpu1(BufferSize);
    
    // Fill H
    for(std::size_t i = 0 ; i < BufferSize ; ++i) 
    { 
        BufferElementOps<BufferElementT>::assign(buffer_cpu1(i),i,BufferSize);
    }
    
    // GPU Buffer
    vc::Buffer1DManaged<BufferElementT, vc::TargetDeviceCUDA> buffer_gpu(BufferSize);
    
    LOG(INFO) << "Buffer1D Sizes, CPU: " << buffer_cpu1.bytes() << ", GPU: " << buffer_gpu.bytes() << " [bytes]";
    
    // H->D
    buffer_gpu.copyFrom(buffer_cpu1);
    
    // CPU buffer 2
    vc::Buffer1DManaged<BufferElementT, vc::TargetHost> buffer_cpu2(BufferSize);
    
    // D->H
    buffer_cpu2.copyFrom(buffer_gpu);
    
    // Check
    for(std::size_t i = 0 ; i < BufferSize ; ++i) 
    {
        BufferElementOps<BufferElementT>::check(buffer_cpu2(i),buffer_cpu1(i),i);
    }
    
    // Now write from kernel
    LaunchKernel_WriteBuffer1D(buffer_gpu,BufferSize);
    
    // D->H
    buffer_cpu1.copyFrom(buffer_gpu);
    
    // Check
    for(std::size_t i = 0 ; i < BufferSize ; ++i) 
    {
        BufferElementOps<BufferElementT>::check(buffer_cpu1(i),buffer_cpu2(i),i);
    }
}
