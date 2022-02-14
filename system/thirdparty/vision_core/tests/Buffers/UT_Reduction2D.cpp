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
 * Basic Reduction2D Tests.
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
#include <VisionCore/LaunchUtils.hpp>

class Test_Reduction2D : public ::testing::Test
{
public:   
    Test_Reduction2D()
    {
        
    }
    
    virtual ~Test_Reduction2D()
    {
        
    }
};

TEST_F(Test_Reduction2D, CPU) 
{
    vc::Buffer2DManaged<float, vc::TargetHost> bufcpu(128,128);
    
    float ground_truth = 0.0f;
    
    // fill
    for(std::size_t y = 0 ; y< bufcpu.height() ; ++y)
    {
        for(std::size_t x = 0 ; x < bufcpu.width() ; ++x)
        {
            const float val = 1.0f;
            bufcpu(x,y) = val;
            ground_truth += bufcpu(x,y);
        }
    }
    
    float our_result = vc::launchParallelReduce(bufcpu.width(), bufcpu.height(), 0.0f,
    [&](const std::size_t x, const std::size_t y, float& ret)
    {
        ret += bufcpu(x,y);
    },
    [&](const float& v1, const float& v2)
    {
        return v1 + v2;
    });
    
    ASSERT_FLOAT_EQ(ground_truth, our_result);
}
