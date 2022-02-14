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
#include <cstdlib>
#include <valarray>

// testing framework & libraries
#include <gtest/gtest.h>

#include <FireWireDriver.hpp>
#include <KinectOneDriver.hpp>
#include <OpenNIDriver.hpp>
#include <PointGreyDriver.hpp>
#include <RealSenseDriver.hpp>
#include <V4LDriver.hpp>
#include <VRMagicDriver.hpp>

class Test_FrameBuffer : public ::testing::Test
{
public:   
    Test_FrameBuffer()
    {
        
    }
    
    virtual ~Test_FrameBuffer()
    {
        
    }
};

struct SomeExternalBufferSource
{
    SomeExternalBufferSource(bool& f, const std::size_t bytes) : ptr(malloc(bytes)), flag(f)
    {
        flag = true;
    }
    
    ~SomeExternalBufferSource()
    {
        free(ptr);
        flag = false;
    }
    
    void* ptr;
    bool& flag;
};

static void sebs_deleter(const void* ptr)
{
    delete static_cast<const SomeExternalBufferSource*>(ptr);
}

TEST_F(Test_FrameBuffer, TestHeapAllocation) 
{
    drivers::camera::FrameBuffer fb;
    
    EXPECT_FALSE(fb.isValid()) << "Should be invalid";
    ASSERT_EQ(fb.getData(), nullptr) << "Should be null";
    
    fb.create(640, 480, drivers::camera::EPixelFormat::PIXEL_FORMAT_RGBA8);
    
    EXPECT_TRUE(fb.isValid()) << "Should be valid";
    ASSERT_NE(fb.getData(), nullptr) << "Invalid pointer";
    
    ASSERT_EQ(fb.getWidth(), 640) << "Wrong width";
    ASSERT_EQ(fb.getHeight(), 480) << "Wrong height";
    ASSERT_EQ(fb.getDataSize(), 640*480*4) << "Wrong buffer size";
    ASSERT_EQ(fb.getPixelFormat(), drivers::camera::EPixelFormat::PIXEL_FORMAT_RGBA8) << "Wrong pixel format";
    
    fb.release();
    
    EXPECT_FALSE(fb.isValid()) << "Should be invalid";
    ASSERT_EQ(fb.getData(), nullptr) << "Should be null";
}

TEST_F(Test_FrameBuffer, TestZeroCopyMode) 
{
    drivers::camera::FrameBuffer fb;
    
    EXPECT_FALSE(fb.isValid()) << "Should be invalid";
    ASSERT_EQ(fb.getData(), nullptr) << "Should be null";
    
    bool allocation_flag = false;
    SomeExternalBufferSource* sebs = new SomeExternalBufferSource(allocation_flag,640*480*4);
    EXPECT_TRUE(allocation_flag) << "Should be allocated";
    
    fb.create(sebs, std::bind(sebs_deleter, std::placeholders::_1), static_cast<uint8_t*>(sebs->ptr), 640, 480, drivers::camera::EPixelFormat::PIXEL_FORMAT_RGBA8);
    
    EXPECT_TRUE(fb.isValid()) << "Should be valid";
    ASSERT_NE(fb.getData(), nullptr) << "Invalid pointer";
    ASSERT_EQ(fb.getData(), sebs->ptr) << "Wrong memory"; // should use external memory
    
    ASSERT_EQ(fb.getWidth(), 640) << "Wrong width";
    ASSERT_EQ(fb.getHeight(), 480) << "Wrong height";
    ASSERT_EQ(fb.getDataSize(), 640*480*4) << "Wrong buffer size";
    ASSERT_EQ(fb.getPixelFormat(), drivers::camera::EPixelFormat::PIXEL_FORMAT_RGBA8) << "Wrong pixel format";
    
    // let's move
    drivers::camera::FrameBuffer fb2(std::move(fb));
    EXPECT_TRUE(allocation_flag) << "Should be still allocated";
    EXPECT_FALSE(fb.isValid()) << "Should be invalid";
    ASSERT_EQ(fb.getData(), nullptr) << "Should be null";
    
    EXPECT_TRUE(fb2.isValid()) << "Should be valid";
    ASSERT_NE(fb2.getData(), nullptr) << "Invalid pointer";
    ASSERT_EQ(fb2.getData(), sebs->ptr) << "Wrong memory"; // should use external memory
    
    ASSERT_EQ(fb2.getWidth(), 640) << "Wrong width";
    ASSERT_EQ(fb2.getHeight(), 480) << "Wrong height";
    ASSERT_EQ(fb2.getDataSize(), 640*480*4) << "Wrong buffer size";
    ASSERT_EQ(fb2.getPixelFormat(), drivers::camera::EPixelFormat::PIXEL_FORMAT_RGBA8) << "Wrong pixel format";
    
    // do a copy
    drivers::camera::FrameBuffer fb3(fb2);
    
    EXPECT_TRUE(allocation_flag) << "Should be still allocated";
    
    EXPECT_TRUE(fb2.isValid()) << "Should be valid";
    ASSERT_NE(fb2.getData(), nullptr) << "Invalid pointer";
    ASSERT_EQ(fb2.getData(), sebs->ptr) << "Wrong memory"; // should use external memory
    
    ASSERT_EQ(fb2.getWidth(), 640) << "Wrong width";
    ASSERT_EQ(fb2.getHeight(), 480) << "Wrong height";
    ASSERT_EQ(fb2.getDataSize(), 640*480*4) << "Wrong buffer size";
    ASSERT_EQ(fb2.getPixelFormat(), drivers::camera::EPixelFormat::PIXEL_FORMAT_RGBA8) << "Wrong pixel format";
    
    EXPECT_TRUE(fb3.isValid()) << "Should be valid";
    ASSERT_NE(fb3.getData(), nullptr) << "Invalid pointer";
    ASSERT_NE(fb3.getData(), sebs->ptr) << "Wrong memory"; // should use heap memory
    
    ASSERT_EQ(fb3.getWidth(), 640) << "Wrong width";
    ASSERT_EQ(fb3.getHeight(), 480) << "Wrong height";
    ASSERT_EQ(fb3.getDataSize(), 640*480*4) << "Wrong buffer size";
    ASSERT_EQ(fb3.getPixelFormat(), drivers::camera::EPixelFormat::PIXEL_FORMAT_RGBA8) << "Wrong pixel format";
    
    fb2.release();
    
    EXPECT_FALSE(fb2.isValid()) << "Should be invalid";
    ASSERT_EQ(fb2.getData(), nullptr) << "Should be null";
    
    EXPECT_FALSE(allocation_flag) << "Should be deallocated";
}
