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

#include <GetEigenConfig.hpp>
#include <VisionCore/Buffers/Buffer1D.hpp>

void GetEigenConfigCUDAHost(float* data, const Sophus::SE3f& v)
{
    getEigenConfiguration(data,v);
}

__global__ void Kernel_GetEigenConfig(vc::Buffer1DView<float, vc::TargetDeviceCUDA> buf, const Sophus::SE3f v)
{
    getEigenConfiguration(buf.ptr(),v);
}

void GetEigenConfigCUDADevice(float* data, const Sophus::SE3f& v)
{
    vc::Buffer1DManaged<float,vc::TargetHost> cpu_buf(MaxEigenConfigurationCount);
    vc::Buffer1DManaged<float,vc::TargetDeviceCUDA> gpu_buf(MaxEigenConfigurationCount);
    
    Kernel_GetEigenConfig<<<1,1>>>(gpu_buf, v);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
    
    cpu_buf.copyFrom(gpu_buf);
    for(std::size_t i = 0 ; i < MaxEigenConfigurationCount ; ++i)
    {
        data[i] = cpu_buf[i];
    }
}
