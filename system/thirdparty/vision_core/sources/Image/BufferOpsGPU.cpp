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
 * Various operations on buffers.
 * ****************************************************************************
 */

#include <VisionCore/Image/BufferOps.hpp>

#include <VisionCore/LaunchUtils.hpp>
#include <VisionCore/Buffers/Reductions.hpp>
#include <VisionCore/CUDAException.hpp>
#include <VisionCore/Image/PixelConvert.hpp>
#include <VisionCore/Math/LossFunctions.hpp>

#include <Image/JoinSplitHelpers.hpp>

#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>

template<typename T>
struct rescale_functor : public thrust::unary_function<T, T>
{    
    rescale_functor(T al, T be, T cmin, T cmax) : alpha(al), beta(be), clamp_min(cmin), clamp_max(cmax) { }
    
    EIGEN_DEVICE_FUNC T operator()(T val)
    {
        return clamp(val * alpha + beta, clamp_min, clamp_max); 
    }
    
    T alpha, beta, clamp_min, clamp_max;
};

template<typename T, typename Target>
void vc::image::rescaleBufferInplace(vc::Buffer1DView< T, Target>& buf_in, T alpha, T beta, T clamp_min, T clamp_max)
{
    thrust::transform(buf_in.begin(), buf_in.end(), buf_in.begin(), rescale_functor<T>(alpha, beta, clamp_min, clamp_max) );
}

template<typename T, typename Target>
void vc::image::rescaleBufferInplace(vc::Buffer2DView<T, Target>& buf_in, T alpha, T beta, T clamp_min, T clamp_max)
{
    rescaleBuffer(buf_in, buf_in, alpha, beta, clamp_min, clamp_max);
}

template<typename T, typename Target>
void vc::image::rescaleBufferInplaceMinMax(vc::Buffer2DView<T, Target>& buf_in, T vmin, T vmax, T clamp_min, T clamp_max)
{
    rescaleBuffer(buf_in, buf_in, T(1.0f) / (vmax - vmin), -vmin * (T(1.0)/(vmax - vmin)), clamp_min, clamp_max);
}

template<typename T1, typename T2, typename Target>
__global__ void Kernel_rescaleBuffer(const vc::Buffer2DView<T1, Target> buf_in, 
                                     vc::Buffer2DView<T2, Target> buf_out, float alpha, float beta, float clamp_min, float clamp_max)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_in.inBounds((int)x,(int)y)) // is valid
    {
        const T2 val = vc::image::convertPixel<T1,T2>(buf_in(x,y));
        buf_out(x,y) = clamp(val * alpha + beta, clamp_min, clamp_max); 
    }
}

template<typename T1, typename T2, typename Target>
void vc::image::rescaleBuffer(const vc::Buffer2DView<T1, Target>& buf_in, vc::Buffer2DView<T2, Target>& buf_out, float alpha, float beta, float clamp_min, float clamp_max)
{
    dim3 gridDim, blockDim;
    
    if(!( (buf_in.width() == buf_out.width()) && (buf_in.height() == buf_out.height())))
    {
        throw std::runtime_error("In/Out dimensions don't match");
    }
    
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_in);
    
    // run kernel
    Kernel_rescaleBuffer<T1,T2><<<gridDim,blockDim>>>(buf_in, buf_out, alpha, beta, clamp_min, clamp_max);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T, typename Target>
void vc::image::normalizeBufferInplace(vc::Buffer2DView< T, Target >& buf_in)
{
    const T min_val = calcBufferMin(buf_in);
    const T max_val = calcBufferMax(buf_in);

    rescaleBufferInplace(buf_in, T(1.0f) / (max_val - min_val), -min_val * (T(1.0)/(max_val - min_val)));
}

template<typename T>
struct clamp_functor : public thrust::unary_function<T, T>
{    
    clamp_functor(T al, T be) : alpha(al), beta(be) { }
    
    EIGEN_DEVICE_FUNC T operator()(T val)
    {
        return clamp(val, alpha, beta); 
    }
    
    T alpha, beta;
};

template<typename T, typename Target>
void vc::image::clampBuffer(vc::Buffer1DView<T, Target>& buf_io, T a, T b)
{
    thrust::transform(buf_io.begin(), buf_io.end(), buf_io.begin(), clamp_functor<T>(a, b) );
}

template<typename T, typename Target>
__global__ void Kernel_clampBuffer(vc::Buffer2DView<T, Target> buf_io, T a, T b)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_io.inBounds((int)x,(int)y)) // is valid
    {
        buf_io(x,y) = clamp(buf_io(x,y),a,b);
    }
}

template<typename T, typename Target>
void vc::image::clampBuffer(vc::Buffer2DView<T, Target>& buf_io, T a, T b)
{
    dim3 gridDim, blockDim;
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_io);
    
    // run kernel
    Kernel_clampBuffer<T,Target><<<gridDim,blockDim>>>(buf_io, a, b);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }
}

template<typename T, typename Target>
T vc::image::calcBufferMin(const vc::Buffer1DView< T, Target >& buf_in)
{
    thrust::device_ptr<T> iter = thrust::min_element(buf_in.begin(), buf_in.end());
    return iter[0];
}

template<typename T, typename Target>
T vc::image::calcBufferMax(const vc::Buffer1DView< T, Target >& buf_in)
{
    thrust::device_ptr<T> iter = thrust::max_element(buf_in.begin(), buf_in.end());
    return iter[0];
}

template<typename T, typename Target>
T vc::image::calcBufferMean(const vc::Buffer1DView< T, Target >& buf_in)
{
    T sum = thrust::reduce(buf_in.begin(), buf_in.end());
    return sum / buf_in.size();
}

template<typename T, typename Target>
T vc::image::calcBufferMin(const vc::Buffer2DView< T, Target >& buf_in)
{
    thrust::device_ptr<T> iter = thrust::min_element(buf_in.begin(), buf_in.end());
    return iter[0];
}

template<typename T, typename Target>
T vc::image::calcBufferMax(const vc::Buffer2DView< T, Target >& buf_in)
{
    thrust::device_ptr<T> iter = thrust::max_element(buf_in.begin(), buf_in.end());
    return iter[0];
}

template<typename T, typename Target>
T vc::image::calcBufferMean(const vc::Buffer2DView< T, Target >& buf_in)
{
    T sum = thrust::reduce(buf_in.begin(), buf_in.end());
    return sum / buf_in.area();
}

template<typename T, typename Target>
__global__ void Kernel_leaveQuarter(const vc::Buffer2DView<T, Target> buf_in, vc::Buffer2DView<T, Target> buf_out)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds((int)x,(int)y)) // is valid
    {
        buf_out(x,y) = buf_in(2*x + 1,2*y + 1);
    }
}

template<typename T, typename Target>
void vc::image::leaveQuarter(const vc::Buffer2DView<T, Target>& buf_in, vc::Buffer2DView<T, Target>& buf_out)
{
    dim3 gridDim, blockDim;
    
    if(!( (buf_in.width()/2 == buf_out.width()) && (buf_in.height()/2 == buf_out.height())))
    {
        throw std::runtime_error("In/Out dimensions don't match");
    }
    
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_leaveQuarter<T,Target><<<gridDim,blockDim>>>(buf_in, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T, typename Target>
__global__ void Kernel_downsampleHalf(const vc::Buffer2DView<T, Target> buf_in, vc::Buffer2DView<T, Target> buf_out)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds((int)x,(int)y)) // is valid
    {
        const T* tl = buf_in.ptr(2*x,2*y);
        const T* bl = buf_in.ptr(2*x,2*y+1);
        
        buf_out(x,y) = (T)(*tl + *(tl+1) + *bl + *(bl+1)) / 4;
    }
}

template<typename T, typename Target>
void vc::image::downsampleHalf(const vc::Buffer2DView<T, Target>& buf_in, vc::Buffer2DView<T, Target>& buf_out)
{
    dim3 gridDim, blockDim;
    
    if(!( (buf_in.width()/2 == buf_out.width()) && (buf_in.height()/2 == buf_out.height())))
    {
        throw std::runtime_error("In/Out dimensions don't match");
    }
    
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_downsampleHalf<T,Target><<<gridDim,blockDim>>>(buf_in, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T, typename Target>
__global__ void Kernel_downsampleHalfNoInvalid(const vc::Buffer2DView<T, Target> buf_in, vc::Buffer2DView<T, Target> buf_out)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds((int)x,(int)y)) // is valid
    {
        const T* tl = buf_in.ptr(2*x,2*y);
        const T* bl = buf_in.ptr(2*x,2*y+1);
        const T v1 = *tl;
        const T v2 = *(tl+1);
        const T v3 = *bl;
        const T v4 = *(bl+1);
        
        int n = 0;
        T sum = vc::zero<T>();
        
        if(vc::isvalid(v1)) { sum += v1; n++; }
        if(vc::isvalid(v2)) { sum += v2; n++; }
        if(vc::isvalid(v3)) { sum += v3; n++; }
        if(vc::isvalid(v4)) { sum += v4; n++; }     
        
        buf_out(x,y) = n > 0 ? (T)(sum / n) : vc::getInvalid<T>();
    }
}

template<typename T, typename Target>
void vc::image::downsampleHalfNoInvalid(const vc::Buffer2DView<T, Target>& buf_in, vc::Buffer2DView<T, Target>& buf_out)
{
    dim3 gridDim, blockDim;
    
    if(!( (buf_in.width()/2 == buf_out.width()) && (buf_in.height()/2 == buf_out.height())))
    {
        throw std::runtime_error("In/Out dimensions don't match");
    }
    
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_downsampleHalfNoInvalid<T,Target><<<gridDim,blockDim>>>(buf_in, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename TCOMP, typename Target>
__global__ void Kernel_join2(const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target> buf_in1, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target> buf_in2, vc::Buffer2DView<TCOMP, Target> buf_out)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds((int)x,(int)y)) // is valid
    {
        ::internal::JoinSplitHelper<TCOMP>::join(buf_in1(x,y), buf_in2(x,y), buf_out(x,y));
    }
}

template<typename TCOMP, typename Target>
__global__ void Kernel_join3(const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target> buf_in1, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target> buf_in2, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target> buf_in3, vc::Buffer2DView<TCOMP, Target> buf_out)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds((int)x,(int)y)) // is valid
    {
        ::internal::JoinSplitHelper<TCOMP>::join(buf_in1(x,y), buf_in2(x,y), buf_in3(x,y), buf_out(x,y));
    }
}

template<typename TCOMP, typename Target>
__global__ void Kernel_join4(const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target> buf_in1, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target> buf_in2, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target> buf_in3, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target> buf_in4, vc::Buffer2DView<TCOMP, Target> buf_out)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds((int)x,(int)y)) // is valid
    {
        ::internal::JoinSplitHelper<TCOMP>::join(buf_in1(x,y), buf_in2(x,y), buf_in3(x,y), buf_in4(x,y), buf_out(x,y));
    }
}

template<typename TCOMP, typename Target>
void vc::image::join(const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_in1, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_in2, vc::Buffer2DView<TCOMP, Target>& buf_out)
{
    assert((buf_out.width() == buf_in1.width()) && (buf_out.height() == buf_in1.height()));
    assert((buf_in1.width() == buf_in2.width()) && (buf_in1.height() == buf_in2.height()));
    
    dim3 gridDim, blockDim;
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_join2<TCOMP,Target><<<gridDim,blockDim>>>(buf_in1, buf_in2, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }
}

template<typename TCOMP, typename Target>
void vc::image::join(const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_in1, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_in2, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_in3, vc::Buffer2DView<TCOMP, Target>& buf_out)
{
    assert((buf_out.width() == buf_in1.width()) && (buf_out.height() == buf_in1.height()));
    assert((buf_in1.width() == buf_in2.width()) && (buf_in1.height() == buf_in2.height()));
    assert((buf_in2.width() == buf_in3.width()) && (buf_in2.height() == buf_in3.height()));
    
    dim3 gridDim, blockDim;
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_join3<TCOMP,Target><<<gridDim,blockDim>>>(buf_in1, buf_in2, buf_in3, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }
}

template<typename TCOMP, typename Target>
void vc::image::join(const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_in1, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_in2, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_in3, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_in4, vc::Buffer2DView<TCOMP, Target>& buf_out)
{
    assert((buf_out.width() == buf_in1.width()) && (buf_out.height() == buf_in1.height()));
    assert((buf_in1.width() == buf_in2.width()) && (buf_in1.height() == buf_in2.height()));
    assert((buf_in2.width() == buf_in3.width()) && (buf_in2.height() == buf_in3.height()));
    assert((buf_in3.width() == buf_in4.width()) && (buf_in3.height() == buf_in4.height()));
    
    dim3 gridDim, blockDim;
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_join4<TCOMP,Target><<<gridDim,blockDim>>>(buf_in1, buf_in2, buf_in3, buf_in4, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }
}

template<typename TCOMP, typename Target>
__global__ void Kernel_split2(const vc::Buffer2DView<TCOMP, Target> buf_in, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target> buf_out1, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target> buf_out2)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_in.inBounds((int)x,(int)y)) // is valid
    {
        ::internal::JoinSplitHelper<TCOMP>::split(buf_in(x,y), buf_out1(x,y), buf_out2(x,y));
    }
}

template<typename TCOMP, typename Target>
__global__ void Kernel_split3(const vc::Buffer2DView<TCOMP, Target> buf_in, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target> buf_out1, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target> buf_out2, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target> buf_out3)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_in.inBounds((int)x,(int)y)) // is valid
    {
        ::internal::JoinSplitHelper<TCOMP>::split(buf_in(x,y), buf_out1(x,y), buf_out2(x,y), buf_out3(x,y));
    }
}

template<typename TCOMP, typename Target>
__global__ void Kernel_split4(const vc::Buffer2DView<TCOMP, Target> buf_in, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target> buf_out1, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target> buf_out2, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target> buf_out3, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target> buf_out4)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_in.inBounds((int)x,(int)y)) // is valid
    {
        ::internal::JoinSplitHelper<TCOMP>::split(buf_in(x,y), buf_out1(x,y), buf_out2(x,y), buf_out3(x,y), buf_out4(x,y));
    }
}

template<typename TCOMP, typename Target>
void vc::image::split(const vc::Buffer2DView<TCOMP, Target>& buf_in, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_out1, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_out2)
{
    assert((buf_in.width() == buf_out1.width()) && (buf_in.height() == buf_out1.height()));
    assert((buf_out1.width() == buf_out2.width()) && (buf_out1.height() == buf_out2.height()));
    
    dim3 gridDim, blockDim;
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_in);
    
    // run kernel
    Kernel_split2<TCOMP,Target><<<gridDim,blockDim>>>(buf_in, buf_out1, buf_out2);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }
}

template<typename TCOMP, typename Target>
void vc::image::split(const vc::Buffer2DView<TCOMP, Target>& buf_in, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_out1, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_out2, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_out3)
{
    assert((buf_in.width() == buf_out1.width()) && (buf_in.height() == buf_out1.height()));
    assert((buf_out1.width() == buf_out2.width()) && (buf_out1.height() == buf_out2.height()));
    assert((buf_out2.width() == buf_out3.width()) && (buf_out2.height() == buf_out3.height()));
    
    dim3 gridDim, blockDim;
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_in);
    
    // run kernel
    Kernel_split3<TCOMP,Target><<<gridDim,blockDim>>>(buf_in, buf_out1, buf_out2, buf_out3);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }
}

template<typename TCOMP, typename Target>
void vc::image::split(const vc::Buffer2DView<TCOMP, Target>& buf_in, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_out1, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_out2, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_out3, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_out4)
{
    assert((buf_in.width() == buf_out1.width()) && (buf_in.height() == buf_out1.height()));
    assert((buf_out1.width() == buf_out2.width()) && (buf_out1.height() == buf_out2.height()));
    assert((buf_out2.width() == buf_out3.width()) && (buf_out2.height() == buf_out3.height()));
    assert((buf_out3.width() == buf_out4.width()) && (buf_out3.height() == buf_out4.height()));
    
    dim3 gridDim, blockDim;
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_in);
    
    // run kernel
    Kernel_split4<TCOMP,Target><<<gridDim,blockDim>>>(buf_in, buf_out1, buf_out2, buf_out3, buf_out4);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }
}

template<typename T, typename Target>
__global__ void Kernel_fillBuffer1D(vc::Buffer1DView<T, Target> buf_in, const typename vc::type_traits<T>::ChannelType v)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(buf_in.inBounds(x)) // is valid
    {
        buf_in(x) = vc::internal::type_dispatcher_helper<T>::fill(v);
    }
}

template<typename T, typename Target>
__global__ void Kernel_fillBuffer2D(vc::Buffer2DView<T, Target> buf_in, const typename vc::type_traits<T>::ChannelType v)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_in.inBounds(x,y)) // is valid
    {
        buf_in(x,y) = vc::internal::type_dispatcher_helper<T>::fill(v);
    }
}

/**
 * fillBuffer
 */
template<typename T, typename Target>
void vc::image::fillBuffer(vc::Buffer1DView<T, Target>& buf_in, const typename vc::type_traits<T>::ChannelType& v)
{
    dim3 gridDim, blockDim;

    vc::InitDimFromBuffer(blockDim, gridDim, buf_in);
    
    // run kernel
    Kernel_fillBuffer1D<T,Target><<<gridDim,blockDim>>>(buf_in, v);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

/**
 * fillBuffer
 */
template<typename T, typename Target>
void vc::image::fillBuffer(vc::Buffer2DView<T, Target>& buf_in, const typename vc::type_traits<T>::ChannelType& v)
{
    dim3 gridDim, blockDim;
    
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_in);
    
    // run kernel
    Kernel_fillBuffer2D<T,Target><<<gridDim,blockDim>>>(buf_in, v);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T, typename Target>
__global__ void Kernel_invertBuffer2D(vc::Buffer2DView<T, Target> buf_io)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_io.inBounds(x,y)) // is valid
    {
        buf_io(x,y) = ::internal::JoinSplitHelper<T>::invertedValue(buf_io(x,y));
    }
}

/**
 * Invert Buffer
 */
template<typename T, typename Target>
void vc::image::invertBuffer(vc::Buffer2DView<T, Target>& buf_io)
{
    dim3 gridDim, blockDim;
    
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_io);
    
    // run kernel
    Kernel_invertBuffer2D<T,Target><<<gridDim,blockDim>>>(buf_io);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T, typename Target>
__global__ void Kernel_thresholdBufferSimple(vc::Buffer2DView<T, Target> buf_in, vc::Buffer2DView<T, Target> buf_out, T thr, T val_below, T val_above )
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_in.inBounds(x,y)) // is valid
    {
        const T& val = buf_in(x,y);
        if(val < thr)
        {
            buf_out(x,y) = val_below;
        }
        else
        {
            buf_out(x,y) = val_above;
        }
    }
}

/**
 * Threshold Buffer
 */
template<typename T, typename Target>
void vc::image::thresholdBuffer(const vc::Buffer2DView<T, Target>& buf_in, vc::Buffer2DView<T, Target>& buf_out, T thr, T val_below, T val_above)
{
    dim3 gridDim, blockDim;
    
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_in);
    
    // run kernel
    Kernel_thresholdBufferSimple<T,Target><<<gridDim,blockDim>>>(buf_in, buf_out, thr, val_below, val_above);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T, typename Target>
__global__ void Kernel_thresholdBufferAdvanced(vc::Buffer2DView<T, Target> buf_in, vc::Buffer2DView<T, Target> buf_out, T thr, T val_below, T val_above, T minval, T maxval, bool saturation )
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_in.inBounds(x,y)) // is valid
    {
        T val = buf_in(x,y);
        
        if(saturation)
        {
            val = clamp(val, minval, maxval);
        }
        
        const T relative_val = (val - minval) / (maxval - minval);
        
        if(relative_val < thr)
        {
            buf_out(x,y) = val_below;
        }
        else
        {
            buf_out(x,y) = val_above;
        }
    }
}

/**
 * Threshold Buffer
 */
template<typename T, typename Target>
void vc::image::thresholdBuffer(const vc::Buffer2DView<T, Target>& buf_in, vc::Buffer2DView<T, Target>& buf_out, T thr, T val_below, T val_above, T minval, T maxval, bool saturation)
{
    dim3 gridDim, blockDim;
    
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_in);
    
    // run kernel
    Kernel_thresholdBufferAdvanced<T,Target><<<gridDim,blockDim>>>(buf_in, buf_out, thr, val_below, val_above, minval, maxval, saturation);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T, typename Target>
__global__ void Kernel_flipXBuffer(const vc::Buffer2DView<T, Target> buf_in, vc::Buffer2DView<T, Target> buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds(x,y)) // is valid
    {
        const std::size_t nx = ((buf_in.width() - 1) - x);
        if(buf_in.inBounds(nx,y))
        {
            buf_out(x,y) = buf_in(nx,y);
        }
    }
}

/**
 * Flip X.
 */
template<typename T, typename Target>
void vc::image::flipXBuffer(const vc::Buffer2DView<T, Target>& buf_in, vc::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in.width() == buf_out.width()) && (buf_in.height() == buf_out.height()));
    
    dim3 gridDim, blockDim;
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_flipXBuffer<T,Target><<<gridDim,blockDim>>>(buf_in, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }
}

template<typename T, typename Target>
__global__ void Kernel_flipYBuffer(const vc::Buffer2DView<T, Target> buf_in, vc::Buffer2DView<T, Target> buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds(x,y)) // is valid
    {
        const std::size_t ny = ((buf_in.height() - 1) - y);
        if(buf_in.inBounds(x,ny))
        {
            buf_out(x,y) = buf_in(x,ny);
        }
    }
}

/**
 * Flip Y.
 */
template<typename T, typename Target>
void vc::image::flipYBuffer(const vc::Buffer2DView<T, Target>& buf_in, vc::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in.width() == buf_out.width()) && (buf_in.height() == buf_out.height()));
    
    dim3 gridDim, blockDim;
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_flipYBuffer<T,Target><<<gridDim,blockDim>>>(buf_in, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }
}

template<typename T, typename Target>
__global__ void Kernel_bufferSubstract(const vc::Buffer2DView<T, Target> buf_in1, const vc::Buffer2DView<T, Target> buf_in2, vc::Buffer2DView<T, Target> buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds(x,y)) // is valid
    {
        buf_out(x,y) = buf_in1(x,y) - buf_in2(x,y);
    }
}

template<typename T, typename Target>
void vc::image::bufferSubstract(const vc::Buffer2DView<T, Target>& buf_in1, const vc::Buffer2DView<T, Target>& buf_in2, vc::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in1.width() == buf_out.width()) && (buf_in1.height() == buf_out.height()));
    assert((buf_in2.width() == buf_out.width()) && (buf_in2.height() == buf_out.height()));
    
    dim3 gridDim, blockDim;
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_bufferSubstract<T,Target><<<gridDim,blockDim>>>(buf_in1, buf_in2, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }
}

template<typename T, typename Target>
__global__ void Kernel_bufferSubstractL1(const vc::Buffer2DView<T, Target> buf_in1, const vc::Buffer2DView<T, Target> buf_in2, vc::Buffer2DView<T, Target> buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds(x,y)) // is valid
    {
        buf_out(x,y) = vc::math::lossL1(buf_in1(x,y) - buf_in2(x,y));
    }
}

template<typename T, typename Target>
void vc::image::bufferSubstractL1(const vc::Buffer2DView<T, Target>& buf_in1, const vc::Buffer2DView<T, Target>& buf_in2, vc::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in1.width() == buf_out.width()) && (buf_in1.height() == buf_out.height()));
    assert((buf_in2.width() == buf_out.width()) && (buf_in2.height() == buf_out.height()));
    
    dim3 gridDim, blockDim;
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_bufferSubstractL1<T,Target><<<gridDim,blockDim>>>(buf_in1, buf_in2, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }
}

template<typename T, typename Target>
__global__ void Kernel_bufferSubstractL2(const vc::Buffer2DView<T, Target> buf_in1, const vc::Buffer2DView<T, Target> buf_in2, vc::Buffer2DView<T, Target> buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds(x,y)) // is valid
    {
        buf_out(x,y) = vc::math::lossL2(buf_in1(x,y) - buf_in2(x,y));
    }
}

template<typename T, typename Target>
void vc::image::bufferSubstractL2(const vc::Buffer2DView<T, Target>& buf_in1, const vc::Buffer2DView<T, Target>& buf_in2, vc::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in1.width() == buf_out.width()) && (buf_in1.height() == buf_out.height()));
    assert((buf_in2.width() == buf_out.width()) && (buf_in2.height() == buf_out.height()));
    
    dim3 gridDim, blockDim;
    vc::InitDimFromBufferOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_bufferSubstractL2<T,Target><<<gridDim,blockDim>>>(buf_in1, buf_in2, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }
}

template<typename T, typename Target>
__global__ void Kernel_bufferSum1D(const vc::Buffer1DView<T, Target> bin, 
                                   vc::Buffer1DView<T, Target> bout,
                                   unsigned int Nblocks)
{
    T sum = vc::zero<T>();
    
    vc::runReductions(Nblocks, [&] __device__ (unsigned int i) 
    { 
        sum = bin(i);
    });
    
    vc::finalizeReduction(bout.ptr(), &sum, vc::internal::warpReduceSum<T>, vc::zero<T>());
}

template<typename T, typename Target>
T vc::image::bufferSum(const vc::Buffer1DView<T, Target>& buffer, const T& initial, unsigned int tpb)
{
    const std::size_t threads = tpb;
    const std::size_t blocks = std::min((buffer.size() + threads - 1) / threads, (std::size_t)1024);
    
    vc::Buffer1DManaged<T, Target> scratch_buffer(blocks);
    
    // run kernel
    Kernel_bufferSum1D<T,Target><<<blocks, threads, 32 * sizeof(T)>>>(buffer, scratch_buffer, buffer.size());
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }
    
    // second pass
    Kernel_bufferSum1D<T,Target><<<1, 1024, 32 * sizeof(T)>>>(scratch_buffer, scratch_buffer, blocks);
    
    T retval = vc::zero<T>();
    vc::Buffer1DView<T,vc::TargetHost> varproxy(&retval, 1);
    
    varproxy.copyFrom(scratch_buffer);
    
    return initial + retval;
}

template<typename T, typename Target>
__global__ void Kernel_bufferSum2D(vc::Buffer2DView<T, Target> bin, 
                                   vc::Buffer1DView<T, Target> bout,
                                   unsigned int Nblocks)
{
    T sum = vc::zero<T>();
    
    vc::runReductions(Nblocks, [&] __device__ (unsigned int i) 
    { 
        const unsigned int y = i / bin.width();
        const unsigned int x = i - (y * bin.width());
      
        sum = bin(x,y);
    });
    
    vc::finalizeReduction(bout.ptr(), &sum, vc::internal::warpReduceSum<T>, vc::zero<T>());
}

template<typename T, typename Target>
T vc::image::bufferSum(const vc::Buffer2DView<T, Target>& buffer, const T& initial, unsigned int tpb)
{
    const std::size_t threads = tpb;
    const std::size_t blocks = std::min((buffer.area() + threads - 1) / threads, (std::size_t)1024);
    
    vc::Buffer1DManaged<T, Target> scratch_buffer(blocks);
    
    // run kernel
    Kernel_bufferSum2D<T,Target><<<blocks, threads, 32 * sizeof(T)>>>(buffer, scratch_buffer, buffer.area());
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }
    
    // second pass
    Kernel_bufferSum1D<T,Target><<<1, 1024, 32 * sizeof(T)>>>(scratch_buffer, scratch_buffer, blocks);
    
    T retval = vc::zero<T>();
    vc::Buffer1DView<T,vc::TargetHost> varproxy(&retval, 1);
    
    varproxy.copyFrom(scratch_buffer);
    
    return initial + retval;
}

#define JOIN_SPLIT_FUNCTIONS2(TCOMP) \
template void vc::image::join(const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetDeviceCUDA>& buf_in1, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetDeviceCUDA>& buf_in2, vc::Buffer2DView<TCOMP, vc::TargetDeviceCUDA>& buf_out); \
template void vc::image::split(const vc::Buffer2DView<TCOMP, vc::TargetDeviceCUDA>& buf_in, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetDeviceCUDA>& buf_out1, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetDeviceCUDA>& buf_out2); 

#define JOIN_SPLIT_FUNCTIONS3(TCOMP) \
template void vc::image::join(const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetDeviceCUDA>& buf_in1, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetDeviceCUDA>& buf_in2, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetDeviceCUDA>& buf_in3, vc::Buffer2DView<TCOMP, vc::TargetDeviceCUDA>& buf_out); \
template void vc::image::split(const vc::Buffer2DView<TCOMP, vc::TargetDeviceCUDA>& buf_in, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetDeviceCUDA>& buf_out1, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetDeviceCUDA>& buf_out2, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetDeviceCUDA>& buf_out3); 

#define JOIN_SPLIT_FUNCTIONS4(TCOMP) \
template void vc::image::join(const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetDeviceCUDA>& buf_in1, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetDeviceCUDA>& buf_in2, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetDeviceCUDA>& buf_in3, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetDeviceCUDA>& buf_in4, vc::Buffer2DView<TCOMP, vc::TargetDeviceCUDA>& buf_out); \
template void vc::image::split(const vc::Buffer2DView<TCOMP, vc::TargetDeviceCUDA>& buf_in, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetDeviceCUDA>& buf_out1, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetDeviceCUDA>& buf_out2, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetDeviceCUDA>& buf_out3, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetDeviceCUDA>& buf_out4); \

// instantiations

JOIN_SPLIT_FUNCTIONS2(Eigen::Vector2f)
JOIN_SPLIT_FUNCTIONS3(Eigen::Vector3f)
JOIN_SPLIT_FUNCTIONS4(Eigen::Vector4f)
JOIN_SPLIT_FUNCTIONS2(Eigen::Vector2d)
JOIN_SPLIT_FUNCTIONS3(Eigen::Vector3d)
JOIN_SPLIT_FUNCTIONS4(Eigen::Vector4d)

JOIN_SPLIT_FUNCTIONS2(float2)
JOIN_SPLIT_FUNCTIONS3(float3)
JOIN_SPLIT_FUNCTIONS4(float4)

// instantiations
template void vc::image::rescaleBufferInplace<float, vc::TargetDeviceCUDA>(vc::Buffer1DView< float, vc::TargetDeviceCUDA >& buf_in, float alpha, float beta, float clamp_min, float clamp_max);
template void vc::image::rescaleBufferInplace<float, vc::TargetDeviceCUDA>(vc::Buffer2DView< float, vc::TargetDeviceCUDA >& buf_in, float alpha, float beta, float clamp_min, float clamp_max);
template void vc::image::rescaleBufferInplaceMinMax<float, vc::TargetDeviceCUDA>(vc::Buffer2DView< float, vc::TargetDeviceCUDA >& buf_in, float alpha, float beta, float clamp_min, float clamp_max);
template void vc::image::normalizeBufferInplace<float, vc::TargetDeviceCUDA>(vc::Buffer2DView< float, vc::TargetDeviceCUDA >& buf_in);

template void vc::image::rescaleBuffer<uint8_t, float, vc::TargetDeviceCUDA>(const vc::Buffer2DView<uint8_t, vc::TargetDeviceCUDA>& buf_in, vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void vc::image::rescaleBuffer<uint16_t, float, vc::TargetDeviceCUDA>(const vc::Buffer2DView<uint16_t, vc::TargetDeviceCUDA>& buf_in, vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void vc::image::rescaleBuffer<uint32_t, float, vc::TargetDeviceCUDA>(const vc::Buffer2DView<uint32_t, vc::TargetDeviceCUDA>& buf_in, vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void vc::image::rescaleBuffer<float, float, vc::TargetDeviceCUDA>(const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_in, vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);

template void vc::image::rescaleBuffer<float, uint8_t, vc::TargetDeviceCUDA>(const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_in, vc::Buffer2DView<uint8_t, vc::TargetDeviceCUDA>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void vc::image::rescaleBuffer<float, uint16_t, vc::TargetDeviceCUDA>(const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_in, vc::Buffer2DView<uint16_t, vc::TargetDeviceCUDA>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void vc::image::rescaleBuffer<float, uint32_t, vc::TargetDeviceCUDA>(const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_in, vc::Buffer2DView<uint32_t, vc::TargetDeviceCUDA>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);

// statistics

#define MIN_MAX_MEAN_THR_FUNCS(BUF_TYPE) \
template BUF_TYPE vc::image::calcBufferMin<BUF_TYPE, vc::TargetDeviceCUDA>(const vc::Buffer1DView< BUF_TYPE, vc::TargetDeviceCUDA >& buf_in); \
template BUF_TYPE vc::image::calcBufferMax<BUF_TYPE, vc::TargetDeviceCUDA>(const vc::Buffer1DView< BUF_TYPE, vc::TargetDeviceCUDA >& buf_in); \
template BUF_TYPE vc::image::calcBufferMean<BUF_TYPE, vc::TargetDeviceCUDA>(const vc::Buffer1DView< BUF_TYPE, vc::TargetDeviceCUDA >& buf_in); \
template BUF_TYPE vc::image::calcBufferMin<BUF_TYPE, vc::TargetDeviceCUDA>(const vc::Buffer2DView< BUF_TYPE, vc::TargetDeviceCUDA >& buf_in); \
template BUF_TYPE vc::image::calcBufferMax<BUF_TYPE, vc::TargetDeviceCUDA>(const vc::Buffer2DView< BUF_TYPE, vc::TargetDeviceCUDA >& buf_in); \
template BUF_TYPE vc::image::calcBufferMean<BUF_TYPE, vc::TargetDeviceCUDA>(const vc::Buffer2DView< BUF_TYPE, vc::TargetDeviceCUDA >& buf_in); \
template void vc::image::thresholdBuffer<BUF_TYPE, vc::TargetDeviceCUDA>(const vc::Buffer2DView< BUF_TYPE, vc::TargetDeviceCUDA>& buf_in, vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_out, BUF_TYPE thr, BUF_TYPE val_below, BUF_TYPE val_above); \
template void vc::image::thresholdBuffer<BUF_TYPE, vc::TargetDeviceCUDA>(const vc::Buffer2DView< BUF_TYPE, vc::TargetDeviceCUDA>& buf_in, vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_out, BUF_TYPE thr, BUF_TYPE val_below, BUF_TYPE val_above, BUF_TYPE minval, BUF_TYPE maxval, bool saturation);

MIN_MAX_MEAN_THR_FUNCS(float)
MIN_MAX_MEAN_THR_FUNCS(uint8_t)
MIN_MAX_MEAN_THR_FUNCS(uint16_t)

// various

#define SIMPLE_TYPE_FUNCS(BUF_TYPE) \
template void vc::image::leaveQuarter<BUF_TYPE, vc::TargetDeviceCUDA>(const vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_in, vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_out); \
template void vc::image::downsampleHalf<BUF_TYPE, vc::TargetDeviceCUDA>(const vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_in, vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_out); \
template void vc::image::fillBuffer<BUF_TYPE, vc::TargetDeviceCUDA>(vc::Buffer1DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_in, const typename vc::type_traits<BUF_TYPE>::ChannelType& v); \
template void vc::image::fillBuffer<BUF_TYPE, vc::TargetDeviceCUDA>(vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_in, const typename vc::type_traits<BUF_TYPE>::ChannelType& v); \
template void vc::image::invertBuffer<BUF_TYPE, vc::TargetDeviceCUDA>(vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_io); \
template void vc::image::flipXBuffer(const vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_in, vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_out); \
template void vc::image::flipYBuffer(const vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_in, vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_out); 

#define BUFFER_SUMATION_FUNCS(BUF_TYPE)\
template BUF_TYPE vc::image::bufferSum(const vc::Buffer1DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_in, const BUF_TYPE& initial, unsigned int tpb);\
template BUF_TYPE vc::image::bufferSum(const vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_in, const BUF_TYPE& initial, unsigned int tpb);

SIMPLE_TYPE_FUNCS(uint8_t)
SIMPLE_TYPE_FUNCS(uint16_t)
SIMPLE_TYPE_FUNCS(uchar3)
SIMPLE_TYPE_FUNCS(uchar4)
SIMPLE_TYPE_FUNCS(float)
SIMPLE_TYPE_FUNCS(float2)
SIMPLE_TYPE_FUNCS(float3)
SIMPLE_TYPE_FUNCS(float4)
SIMPLE_TYPE_FUNCS(Eigen::Vector2f)
SIMPLE_TYPE_FUNCS(Eigen::Vector3f)
SIMPLE_TYPE_FUNCS(Eigen::Vector4f)

BUFFER_SUMATION_FUNCS(float)
BUFFER_SUMATION_FUNCS(float2)
BUFFER_SUMATION_FUNCS(float3)
BUFFER_SUMATION_FUNCS(float4)
BUFFER_SUMATION_FUNCS(Eigen::Vector2f)
BUFFER_SUMATION_FUNCS(Eigen::Vector3f)
BUFFER_SUMATION_FUNCS(Eigen::Vector4f)

#define CLAMP_FUNC_TYPES(BUF_TYPE) \
template void vc::image::clampBuffer(vc::Buffer1DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_io, BUF_TYPE a, BUF_TYPE b); \
template void vc::image::clampBuffer(vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_io, BUF_TYPE a, BUF_TYPE b);

CLAMP_FUNC_TYPES(uint8_t)
CLAMP_FUNC_TYPES(uint16_t)
CLAMP_FUNC_TYPES(float)
CLAMP_FUNC_TYPES(float2)
CLAMP_FUNC_TYPES(float3)
CLAMP_FUNC_TYPES(float4)

#define STUPID_FUNC_TYPES(BUF_TYPE)\
template void vc::image::bufferSubstract(const vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_in1, const vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_in2, vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_out);\
template void vc::image::bufferSubstractL1(const vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_in1, const vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_in2, vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_out);\
template void vc::image::bufferSubstractL2(const vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_in1, const vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_in2, vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_out);
STUPID_FUNC_TYPES(float)

//template void vc::image::downsampleHalfNoInvalid<BUF_TYPE, vc::TargetDeviceCUDA>(const vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_in, vc::Buffer2DView<BUF_TYPE, vc::TargetDeviceCUDA>& buf_out); 
