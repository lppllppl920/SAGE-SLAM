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
 * Basic Buffer1D Tests.
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
#include <BufferTestHelpers.hpp>

static constexpr std::size_t BufferSize = 4097;

template<typename T>
class Test_Buffer1D : public ::testing::Test
{
public:   
    Test_Buffer1D()
    {
        
    }
    
    virtual ~Test_Buffer1D()
    {
        
    }
};

typedef ::testing::Types<float,Eigen::Vector3f,Sophus::SE3f> TypesToTest;
TYPED_TEST_CASE(Test_Buffer1D, TypesToTest);

TYPED_TEST(Test_Buffer1D, CPU) 
{
    typedef TypeParam BufferElementT;
    
    vc::Buffer1DManaged<BufferElementT, vc::TargetHost> bufcpu(BufferSize);
    
    ASSERT_TRUE(bufcpu.isValid()) << "Wrong managed state";
    ASSERT_EQ(bufcpu.size(), BufferSize) << "Wrong managed size";
    ASSERT_EQ(bufcpu.bytes(), BufferSize * sizeof(BufferElementT)) << "Wrong managed size bytes";
    
    for(std::size_t i = 0 ; i < bufcpu.size() ; ++i)
    {
        BufferElementOps<BufferElementT>::assign(bufcpu(i),i,BufferSize);
    }
    
    vc::Buffer1DView<BufferElementT, vc::TargetHost> viewcpu(bufcpu);
    
    ASSERT_TRUE(viewcpu.isValid()) << "Wrong view state";
    ASSERT_EQ(viewcpu.size(), BufferSize) << "Wrong size for view";
    
    for(std::size_t i = 0 ; i < viewcpu.size() ; ++i)
    {
        BufferElementOps<BufferElementT>::check(viewcpu(i), viewcpu.ptr()[i],i);
    }
}
