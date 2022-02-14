/**
 * ****************************************************************************
 * Copyright (c) 2017, Robert Lukierski.
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
 * Templated kernels for tests.
 * ****************************************************************************
 */

#include <VisionCore/LaunchUtils.hpp>
#include <BufferTestHelpers.hpp>


template<typename BufferElementT>
__global__ void Kernel_WriteBuffer1D(vc::Buffer1DView<BufferElementT,vc::TargetDeviceCUDA> buf, std::size_t BufferSize)
{
    const std::size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(buf.size() != BufferSize)
    {
        asm("trap;");
    }
    
    if(buf.inBounds(i)) // is valid
    {
        BufferElementOps<BufferElementT>::assign(buf(i),i,BufferSize);
    }
}

template<typename BufferElementT>
void LaunchKernel_WriteBuffer1D(const vc::Buffer1DView<BufferElementT, vc::TargetDeviceCUDA>& buffer_gpu, std::size_t BufferSize)
{
    dim3 gridDim, blockDim;
    
    vc::InitDimFromBuffer(blockDim, gridDim, buffer_gpu);
    
    Kernel_WriteBuffer1D<BufferElementT><<<gridDim,blockDim>>>(buffer_gpu, BufferSize);
    
    // Wait for it
    cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename BufferElementT>
__global__ void Kernel_WriteBuffer2D(vc::Buffer2DView<BufferElementT,vc::TargetDeviceCUDA> buf, 
                                     std::size_t BufferSizeX, std::size_t BufferSizeY)
{
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf.width() != BufferSizeX)
    {
        asm("trap;");
    }
    
    if(buf.height() != BufferSizeY)
    {
        asm("trap;");
    }
    
    if(buf.inBounds(x,y)) // is valid
    {
        const std::size_t LinIndex = y * BufferSizeX + x;
        BufferElementOps<BufferElementT>::assign(buf(x,y),LinIndex,BufferSizeX*BufferSizeY);
    }
}

template<typename BufferElementT>
void LaunchKernel_WriteBuffer2D(const vc::Buffer2DView<BufferElementT, vc::TargetDeviceCUDA>& buffer_gpu, 
                                std::size_t BufferSizeX, std::size_t BufferSizeY)
{
    dim3 gridDim, blockDim;
    
    vc::InitDimFromBuffer(blockDim, gridDim, buffer_gpu);
    
    Kernel_WriteBuffer2D<BufferElementT><<<gridDim,blockDim>>>(buffer_gpu, BufferSizeX, BufferSizeY);
    
    // Wait for it
    cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template void LaunchKernel_WriteBuffer1D<float>(const vc::Buffer1DView<float, vc::TargetDeviceCUDA>& buffer_gpu, std::size_t BufferSize);
template void LaunchKernel_WriteBuffer1D<Eigen::Vector3f>(const vc::Buffer1DView<Eigen::Vector3f, vc::TargetDeviceCUDA>& buffer_gpu, std::size_t BufferSize);
template void LaunchKernel_WriteBuffer1D<Sophus::SE3f>(const vc::Buffer1DView<Sophus::SE3f, vc::TargetDeviceCUDA>& buffer_gpu, std::size_t BufferSize);

template void LaunchKernel_WriteBuffer2D<float>(const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buffer_gpu, std::size_t BufferSizeX, std::size_t BufferSizeY);
template void LaunchKernel_WriteBuffer2D<Eigen::Vector3f>(const vc::Buffer2DView<Eigen::Vector3f, vc::TargetDeviceCUDA>& buffer_gpu, std::size_t BufferSizeX, std::size_t BufferSizeY);
template void LaunchKernel_WriteBuffer2D<Sophus::SE3f>(const vc::Buffer2DView<Sophus::SE3f, vc::TargetDeviceCUDA>& buffer_gpu, std::size_t BufferSizeX, std::size_t BufferSizeY);
