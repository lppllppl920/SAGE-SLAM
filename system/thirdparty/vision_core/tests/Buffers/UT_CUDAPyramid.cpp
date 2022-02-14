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
 * Basic CUDA Pyramid Tests.
 * ****************************************************************************
 */

// testing framework & libraries
#include <gtest/gtest.h>

// google logger
#include <glog/logging.h>

#include <VisionCore/Buffers/BufferPyramid.hpp>
#include <VisionCore/Buffers/ImagePyramid.hpp>

#include <VisionCore/LaunchUtils.hpp>

static constexpr std::size_t BufferSizeX = 640;
static constexpr std::size_t BufferSizeY = 480;
static constexpr std::size_t Levels = 2;
typedef float BufferElementT;

class Test_CUDAPyramid : public ::testing::Test
{
public:   
    Test_CUDAPyramid() : buffer_pyramid(BufferSizeX,BufferSizeY), image_pyramid(BufferSizeX, BufferSizeY)
    {
        
    }
    
    virtual ~Test_CUDAPyramid()
    {
        
    }
    
    vc::BufferPyramidManaged<BufferElementT,Levels,vc::TargetDeviceCUDA> buffer_pyramid;
    vc::ImagePyramidManaged<BufferElementT,Levels,vc::TargetDeviceCUDA> image_pyramid;
};

__global__ void Kernel_PyramidReadBuffer2D(const vc::Buffer2DView<BufferElementT,vc::TargetDeviceCUDA> buf)
{
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf.inBounds(x,y)) // is valid
    {
        BufferElementT elem = buf(x,y);
    }
}

__global__ void Kernel_PyramidReadImage2D(const vc::Image2DView<BufferElementT,vc::TargetDeviceCUDA> buf)
{
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf.inBounds(x,y)) // is valid
    {
        BufferElementT elem = buf(x,y);
    }
}


TEST_F(Test_CUDAPyramid, TestReadBuffer) 
{
    dim3 gridDim, blockDim;
    
    for(std::size_t lvl = 0 ; lvl < Levels ; ++lvl)
    {
        vc::InitDimFromOutputImage(blockDim, gridDim, buffer_pyramid[lvl]);
        
        Kernel_PyramidReadBuffer2D<<<gridDim,blockDim>>>(buffer_pyramid[lvl]);
        
        // wait for it
        const cudaError err = cudaDeviceSynchronize();
        if(err != cudaSuccess)
        {
            throw vc::CUDAException(err, "Error launching the kernel");
        }
    }
}

TEST_F(Test_CUDAPyramid, TestReadImage) 
{
    dim3 gridDim, blockDim;
    
    for(std::size_t lvl = 0 ; lvl < Levels ; ++lvl)
    {
        vc::InitDimFromOutputImage(blockDim, gridDim, image_pyramid[lvl]);
        
        Kernel_PyramidReadImage2D<<<gridDim,blockDim>>>(image_pyramid[lvl]);
        
        // wait for it
        const cudaError err = cudaDeviceSynchronize();
        if(err != cudaSuccess)
        {
            throw vc::CUDAException(err, "Error launching the kernel");
        }
    }
}

__global__ void Kernel_PyramidWriteBuffer2D(vc::Buffer2DView<BufferElementT,vc::TargetDeviceCUDA> buf)
{
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf.inBounds(x,y)) // is valid
    {
        BufferElementT& elem = buf(x,y);
        elem = x * y;
    }
}

__global__ void Kernel_PyramidWriteImage2D(vc::Image2DView<BufferElementT,vc::TargetDeviceCUDA> buf)
{
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf.inBounds(x,y)) // is valid
    {
        BufferElementT& elem = buf(x,y);
        elem = x * y;
    }
}

TEST_F(Test_CUDAPyramid, TestWriteBuffer) 
{
    dim3 gridDim, blockDim;
    
    for(std::size_t lvl = 0 ; lvl < Levels ; ++lvl)
    {
        vc::InitDimFromOutputImage(blockDim, gridDim, buffer_pyramid[lvl]);
        
        Kernel_PyramidWriteBuffer2D<<<gridDim,blockDim>>>(buffer_pyramid[lvl]);
        
        // wait for it
        const cudaError err = cudaDeviceSynchronize();
        if(err != cudaSuccess)
        {
            throw vc::CUDAException(err, "Error launching the kernel");
        }
    }
}

TEST_F(Test_CUDAPyramid, TestWriteImage) 
{
    dim3 gridDim, blockDim;
    
    for(std::size_t lvl = 0 ; lvl < Levels ; ++lvl)
    {
        vc::InitDimFromOutputImage(blockDim, gridDim, image_pyramid[lvl]);
        
        Kernel_PyramidWriteImage2D<<<gridDim,blockDim>>>(image_pyramid[lvl]);
        
        // wait for it
        const cudaError err = cudaDeviceSynchronize();
        if(err != cudaSuccess)
        {
            throw vc::CUDAException(err, "Error launching the kernel");
        }
    }
}
