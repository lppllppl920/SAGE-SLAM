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
 * Basic OpenCL Buffer1D Tests.
 * ****************************************************************************
 */

// system
#include <stdint.h>
#include <stddef.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <exception>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <valarray>

// testing framework & libraries
#include <gtest/gtest.h>

// google logger
#include <glog/logging.h>

#include <VisionCore/Buffers/Buffer1D.hpp>

class Test_OpenCLBuffer1D : public ::testing::Test
{
public:   
    Test_OpenCLBuffer1D()
    {
        
    }
    
    virtual ~Test_OpenCLBuffer1D()
    {
        
    }
};

TEST_F(Test_OpenCLBuffer1D, OpenCL) 
{
    cl_int error;
    cl::CommandQueue queue = cl::CommandQueue::getDefault(&error);
    
    vc::Buffer1DManaged<float, vc::TargetHost> cpu_src(10), cpu_dst(10);
    
    for(std::size_t i = 0 ; i < 10 ; ++i)
    {
        cpu_src(i) = i + 1;
    }
    
    vc::Buffer1DManaged<float, vc::TargetDeviceOpenCL> bufcl(10, cl::Context::getDefault(), CL_MEM_READ_ONLY);
    
    bufcl.copyFrom(queue, cpu_src);
    
    ASSERT_TRUE(bufcl.isValid());
    
    vc::Buffer1DView<float, vc::TargetDeviceOpenCL> viewcl(bufcl);
    
    ASSERT_TRUE(viewcl.isValid());
    
    cpu_dst.copyFrom(queue, viewcl);
    
    for(std::size_t i = 0 ; i < 10 ; ++i)
    {
        ASSERT_TRUE(cpu_dst(i) == cpu_src(i)) << "Error at " << i << " and " << cpu_dst(i) << " where it should be " << cpu_src(i);
    }
}
