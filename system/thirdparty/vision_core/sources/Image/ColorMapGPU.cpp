/**
 * ****************************************************************************
 * Copyright (c) 2016, Robert Lukierski.
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
 * Color Maps.
 * ****************************************************************************
 */

#define COLOR_MAP_THIS_IS_GPU
#include <Image/ColorMapDefs.hpp>
#include <VisionCore/LaunchUtils.hpp>

__device__ __constant__ float3 CurrentColorMap[256];

template<typename T, typename TOUT, typename Target>
__global__ void Kernel_createColorMap1D(std::size_t cms, const vc::Buffer1DView<T,Target> img_in, const T vmin, const T vmax, vc::Buffer1DView<TOUT,Target> img_out)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(img_in.inBounds((int)x)) // is valid
    {
        const T& val_in = img_in(x); 
        TOUT& val_out = img_out(x);
        
        float3 result = getColorMapValuePreload(cms, CurrentColorMap, vmin, vmax, val_in);
        val_out = ConvertToTarget<TOUT>::run(result);
    }
}

template<typename T, typename TOUT, typename Target>
void vc::image::createColorMap(ColorMap cm, const vc::Buffer1DView<T,Target>& img_in, const T& vmin, const T& vmax, vc::Buffer1DView<TOUT,Target>& img_out)
{
    const std::size_t cms = getColorMapSizeForwarder(cm);
    const float3* data = getColorMapData(cm);
    
    // copy a lot to constant memory
    cudaError err = cudaMemcpyToSymbol(CurrentColorMap, data, cms * sizeof(float3));
    if( err != cudaSuccess ) 
    {
        throw vc::CUDAException(err, "Unable to cudaMemcpyToSymbol");
    }
    
    dim3 gridDim, blockDim;
    vc::InitDimFromBuffer(blockDim, gridDim, img_out);
    
    // run kernel
    Kernel_createColorMap1D<T, TOUT, Target><<<gridDim,blockDim>>>(cms, img_in, vmin, vmax, img_out);
    
    // wait for it
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }
}

template<typename T, typename TOUT, typename Target>
__global__ void Kernel_createColorMap2D(std::size_t cms, const vc::Buffer2DView<T,Target> img_in, const T vmin, const T vmax, vc::Buffer2DView<TOUT,Target> img_out)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(img_in.inBounds((int)x,(int)y)) // is valid
    {
        const T& val_in = img_in(x,y); 
        TOUT& val_out = img_out(x,y);
        
        float3 result = getColorMapValuePreload(cms, CurrentColorMap, vmin, vmax, val_in);
        val_out = ConvertToTarget<TOUT>::run(result);
    }
}

template<typename T, typename TOUT, typename Target>
void vc::image::createColorMap(ColorMap cm, const vc::Buffer2DView<T,Target>& img_in, const T& vmin, const T& vmax, vc::Buffer2DView<TOUT,Target>& img_out)
{
    const std::size_t cms = getColorMapSizeForwarder(cm);
    const float3* data = getColorMapData(cm);
    
    // copy a lot to constant memory
    cudaError err = cudaMemcpyToSymbol(CurrentColorMap, data, cms * sizeof(float3));
    if( err != cudaSuccess ) 
    {
        throw vc::CUDAException(err, "Unable to cudaMemcpyToSymbol");
    }
    
    dim3 gridDim, blockDim;
    vc::InitDimFromBufferOver(blockDim, gridDim, img_in);
    
    // run kernel
    Kernel_createColorMap2D<T, TOUT, Target><<<gridDim,blockDim>>>(cms, img_in, vmin, vmax, img_out);
    
    // wait for it
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }
}

#define GENERATE_IMPL(TIN,TOUT)\
template void vc::image::createColorMap<TIN,TOUT,vc::TargetDeviceCUDA>(ColorMap cm, const vc::Buffer1DView<TIN,vc::TargetDeviceCUDA>& img_in, const TIN& vmin, const TIN& vmax, vc::Buffer1DView<TOUT,vc::TargetDeviceCUDA>& img_out); \
template void vc::image::createColorMap<TIN,TOUT,vc::TargetDeviceCUDA>(ColorMap cm, const vc::Buffer2DView<TIN,vc::TargetDeviceCUDA>& img_in, const TIN& vmin, const TIN& vmax, vc::Buffer2DView<TOUT,vc::TargetDeviceCUDA>& img_out);

GENERATE_IMPL(float,float3)
GENERATE_IMPL(float,float4)
GENERATE_IMPL(float,Eigen::Vector3f)
GENERATE_IMPL(float,Eigen::Vector4f)
