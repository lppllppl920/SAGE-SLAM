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
 * Memory Policies for CUDA.
 * ****************************************************************************
 */

#ifndef VISIONCORE_MEMORY_POLICY_CUDA_HPP
#define VISIONCORE_MEMORY_POLICY_CUDA_HPP

#include <cstdlib>

#include <VisionCore/Platform.hpp>

#include <thrust/device_vector.h>

namespace vc
{

struct TargetDeviceCUDA;
    
namespace internal
{
    template<typename TargetTo, typename TargetFrom>
    cudaMemcpyKind TargetCopyKind();
    
    template<> inline cudaMemcpyKind TargetCopyKind<TargetHost,TargetHost>() { return cudaMemcpyHostToHost;}
    template<> inline cudaMemcpyKind TargetCopyKind<TargetDeviceCUDA,TargetHost>() { return cudaMemcpyHostToDevice;}
    template<> inline cudaMemcpyKind TargetCopyKind<TargetHost,TargetDeviceCUDA>() { return cudaMemcpyDeviceToHost;}
    template<> inline cudaMemcpyKind TargetCopyKind<TargetDeviceCUDA,TargetDeviceCUDA>() { return cudaMemcpyDeviceToDevice;}
    
    template<typename T, typename Target> struct ThrustType;
    template<typename T> struct ThrustType<T,TargetHost> { typedef T* Ptr; };
    template<typename T> struct ThrustType<T,TargetDeviceCUDA> { typedef thrust::device_ptr<T> Ptr; };
}
    
struct TargetDeviceCUDA
{
    template<typename T> using PointerType = void*;
    template<typename T> using TextureHandleType = cudaTextureObject_t;
    
    template<typename T>
    inline static void AllocateMem(PointerType<T>* devPtr, std::size_t s)
    {
        const cudaError err = cudaMalloc(devPtr, sizeof(T) * s);
        if( err != cudaSuccess ) { throw CUDAException(err, "Unable to cudaMalloc"); }
    }
    
    template<typename T>
    inline static void AllocatePitchedMem(PointerType<T>* devPtr, std::size_t* pitch, std::size_t w, std::size_t h)
    {
        const cudaError err = cudaMallocPitch(devPtr, pitch, w * sizeof(T), h);
        if( err != cudaSuccess ) { throw CUDAException(err, "Unable to cudaMallocPitch"); }
    }

    template<typename T>
    inline static void AllocatePitchedMem(PointerType<T>* devPtr, std::size_t* pitch, std::size_t* img_pitch, std::size_t w, std::size_t h, std::size_t d)
    {
        const cudaError err = cudaMallocPitch(devPtr, pitch, w * sizeof(T), h * d);
        if( err != cudaSuccess ) { throw CUDAException(err, "Unable to cudaMallocPitch"); }
        
        *img_pitch = *pitch * h;
    }

    template<typename T>
    inline static bool DeallocatePitchedMem(PointerType<T> devPtr) throw()
    {
#ifndef VISIONCORE_CUDA_KERNEL_SPACE
        const cudaError err = cudaFree(devPtr);
        return err == cudaSuccess;
#endif // VISIONCORE_CUDA_KERNEL_SPACE
        return false;
    }
    
    template<typename T> 
    inline static void memset(PointerType<T> devPtr, int value, std::size_t count) 
    {
        cudaMemset(devPtr, value, count);
    }
    
    template<typename T>
    inline static void memset2D(PointerType<T> ptr, std::size_t pitch, int value, std::size_t width, std::size_t height)
    {
        cudaMemset2D(ptr, pitch, value, width, height);
    }
    
    template<typename T>
    inline static void memset3D(PointerType<T> ptr, std::size_t pitch, int value, std::size_t width, std::size_t height, std::size_t depth)
    {
        cudaMemset3D(make_cudaPitchedPtr(ptr, pitch, width, height), value, make_cudaExtent(width,height,depth));
    }
};

template<> struct TargetTransfer<TargetHost,TargetDeviceCUDA>
{
    typedef TargetHost TargetFrom;
    typedef TargetDeviceCUDA TargetTo;
    static constexpr cudaMemcpyKind CopyKind = cudaMemcpyHostToDevice;
    
    template<typename T>
    inline static void memcpy(typename TargetTo::template PointerType<T> dst,
                              const typename TargetFrom::template PointerType<T> src, std::size_t count)
    {
        const cudaError err = cudaMemcpy(dst, src, count, CopyKind );
        if( err != cudaSuccess ) { throw CUDAException(err, "Unable to cudaMemcpy"); }
    }
    
    template<typename T>
    inline static void memcpy2D(typename TargetTo::template PointerType<T> dst, std::size_t  dpitch, 
                                const typename TargetFrom::template PointerType<T> src, 
                                std::size_t  spitch, std::size_t  width, std::size_t  height)
    {
        const cudaError err = cudaMemcpy2D(dst,dpitch,src,spitch, width, height, CopyKind );
        if(err != cudaSuccess) { throw CUDAException(err, "Unable to cudaMemcpy2D"); }
    }
};

template<> struct TargetTransfer<TargetDeviceCUDA,TargetHost>
{
    typedef TargetDeviceCUDA TargetFrom;
    typedef TargetHost TargetTo;
    static constexpr cudaMemcpyKind CopyKind = cudaMemcpyDeviceToHost;
    
    template<typename T>
    inline static void memcpy(typename TargetTo::template PointerType<T> dst, 
                              const typename TargetFrom::template PointerType<T> src, 
                              std::size_t count)
    {
        const cudaError err = cudaMemcpy(dst, src, count, CopyKind );
        if( err != cudaSuccess ) { throw CUDAException(err, "Unable to cudaMemcpy"); }
    }
    
    template<typename T>
    inline static void memcpy2D(typename TargetTo::template PointerType<T> dst, std::size_t dpitch, 
                                const typename TargetFrom::template PointerType<T> src, 
                                std::size_t spitch, std::size_t  width, std::size_t  height)
    {
        const cudaError err = cudaMemcpy2D(dst,dpitch,src,spitch, width, height, CopyKind );
        if(err != cudaSuccess) { throw CUDAException(err, "Unable to cudaMemcpy2D"); }
    }
};

template<> struct TargetTransfer<TargetDeviceCUDA,TargetDeviceCUDA>
{
    typedef TargetDeviceCUDA TargetFrom;
    typedef TargetDeviceCUDA TargetTo;
    static constexpr cudaMemcpyKind CopyKind = cudaMemcpyDeviceToDevice;
    
    template<typename T>
    inline static void memcpy(typename TargetTo::template PointerType<T> dst, 
                              const typename TargetFrom::template PointerType<T> src, 
                              std::size_t count)
    {
        const cudaError err = cudaMemcpy(dst, src, count, CopyKind );
        if( err != cudaSuccess ) { throw CUDAException(err, "Unable to cudaMemcpy"); }
    }
    
    template<typename T>
    inline static void memcpy2D(typename TargetTo::template PointerType<T> dst, std::size_t dpitch, 
                                const typename TargetFrom::template PointerType<T> src, 
                                std::size_t  spitch, std::size_t  width, std::size_t  height)
    {
        const cudaError err = cudaMemcpy2D(dst,dpitch,src,spitch, width, height, CopyKind );
        if(err != cudaSuccess) { throw CUDAException(err, "Unable to cudaMemcpy2D"); }
    }
};

}

#endif // VISIONCORE_MEMORY_POLICY_CUDA_HPP
