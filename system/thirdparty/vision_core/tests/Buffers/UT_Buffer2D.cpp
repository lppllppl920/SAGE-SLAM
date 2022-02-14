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
 * Basic Buffer2D Tests.
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

#include <VisionCore/Buffers/Buffer2D.hpp>
#include <BufferTestHelpers.hpp>

static constexpr std::size_t BufferSizeX = 1025;
static constexpr std::size_t BufferSizeY = 769;

template<typename T>
class Test_Buffer2D : public ::testing::Test
{
public:   
    Test_Buffer2D()
    {
        
    }
    
    virtual ~Test_Buffer2D()
    {
        
    }
};

typedef ::testing::Types<float,Eigen::Vector3f,Sophus::SE3f> TypesToTest;
TYPED_TEST_CASE(Test_Buffer2D, TypesToTest);

TYPED_TEST(Test_Buffer2D, CPU) 
{
    typedef TypeParam BufferElementT;
    
    vc::Buffer2DManaged<BufferElementT, vc::TargetHost> bufcpu(BufferSizeX,BufferSizeY);
    
    ASSERT_TRUE(bufcpu.isValid()) << "Wrong managed state";
    ASSERT_EQ(bufcpu.width(), BufferSizeX) << "Wrong managed width size";
    ASSERT_EQ(bufcpu.height(), BufferSizeY) << "Wrong managed height size";
    ASSERT_EQ(bufcpu.bytes(), BufferSizeX * BufferSizeY * sizeof(BufferElementT)) << "Wrong managed size bytes";
    
    for(std::size_t y = 0 ; y < bufcpu.height() ; ++y)
    {
        for(std::size_t x = 0 ; x < bufcpu.width() ; ++x)
        {
            const std::size_t LinIndex = y * BufferSizeX + x;
            BufferElementOps<BufferElementT>::assign(bufcpu(x,y),LinIndex,BufferSizeX*BufferSizeY);
        }
    }
    
    vc::Buffer2DView<BufferElementT, vc::TargetHost> viewcpu(bufcpu);
    
    ASSERT_TRUE(viewcpu.isValid()) << "Wrong view state";
    ASSERT_EQ(viewcpu.width(), BufferSizeX) << "Wrong view width size";
    ASSERT_EQ(viewcpu.height(), BufferSizeY) << "Wrong view height size";
    
    for(std::size_t y = 0 ; y < viewcpu.height() ; ++y)
    {
        for(std::size_t x = 0 ; x < viewcpu.width() ; ++x)
        {
            BufferElementT gt;
            const std::size_t LinIndex = y * BufferSizeX + x;
            BufferElementOps<BufferElementT>::assign(gt,LinIndex,BufferSizeX*BufferSizeY);
            BufferElementOps<BufferElementT>::check(viewcpu(x,y), gt, LinIndex);
        }
    }
}
