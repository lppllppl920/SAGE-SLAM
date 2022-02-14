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
 * 2D Host/Device Buffer Pyramids.
 * ****************************************************************************
 */

#ifndef VISIONCORE_BUFFER_PYRAMID_HPP
#define VISIONCORE_BUFFER_PYRAMID_HPP

#include <VisionCore/Buffers/PyramidBase.hpp>
#include <VisionCore/Buffers/Buffer2D.hpp>

namespace vc
{

template<typename T, std::size_t Levels, typename Target = TargetHost>
class BufferPyramidView { };

template<typename T, typename Target = TargetHost>
class RuntimeBufferPyramidView { };

/**
 * Buffer Pyramid View - Host.
 */
template<typename T, std::size_t Levels>
class BufferPyramidView<T,Levels,TargetHost> : public PyramidViewBase<Buffer2DView, T,Levels,TargetHost>
{
public:
    typedef PyramidViewBase<Buffer2DView, T,Levels,TargetHost> BaseT;
    typedef typename BaseT::ViewType LevelT;
    static const std::size_t LevelCount = BaseT::LevelCount;
    typedef typename BaseT::ValueType ValueType;
    typedef typename BaseT::TargetType TargetType;
    
    EIGEN_DEVICE_FUNC inline BufferPyramidView() { }
    
    EIGEN_DEVICE_FUNC inline ~BufferPyramidView() { }
    
    EIGEN_DEVICE_FUNC inline BufferPyramidView(const BufferPyramidView<T,LevelCount,TargetType>& pyramid) : BaseT(pyramid) { }
    
    EIGEN_DEVICE_FUNC inline BufferPyramidView(BufferPyramidView<T,LevelCount,TargetType>&& pyramid) : BaseT(std::move(pyramid)) { }
    
    EIGEN_DEVICE_FUNC inline BufferPyramidView<T,LevelCount,TargetType>& operator=(const BufferPyramidView<T,LevelCount,TargetType>& pyramid)
    {
        BaseT::operator=(pyramid);
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline BufferPyramidView<T,LevelCount,TargetType>& operator=(BufferPyramidView<T,LevelCount,TargetType>&& pyramid)
    {
        BaseT::operator=(std::move(pyramid));
        return *this;
    }
    
    inline void memset(unsigned char v = 0)
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            BaseT::imgs[l].memset(v);
        }
    }
    
    inline void copyFrom(const BufferPyramidView<T,Levels,TargetHost>& pyramid)
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            BaseT::imgs[l].copyFrom(pyramid.imgs[l]);
        }
    }
    
#ifdef VISIONCORE_HAVE_CUDA
    inline void copyFrom(const BufferPyramidView<T,Levels,TargetDeviceCUDA>& pyramid)
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            BaseT::imgs[l].copyFrom(pyramid.imgs[l]);
        }
    }
#endif // VISIONCORE_HAVE_CUDA

#ifdef VISIONCORE_HAVE_OPENCL
    inline void copyFrom(const cl::CommandQueue& queue, const BufferPyramidView<T,Levels,TargetDeviceOpenCL>& pyramid)
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            BaseT::imgs[l].copyFrom(queue, pyramid.imgs[l]);
        }
    }
#endif // VISIONCORE_HAVE_OPENCL
};

/**
 * Runtime Buffer Pyramid View - Host.
 */
template<typename T>
class RuntimeBufferPyramidView<T,TargetHost> : public RuntimePyramidViewBase<Buffer2DView,T,TargetHost>
{
public:
    typedef RuntimePyramidViewBase<Buffer2DView,T,TargetHost> BaseT;
    typedef typename BaseT::ViewType LevelT;
    typedef typename BaseT::ValueType ValueType;
    typedef typename BaseT::TargetType TargetType;
    
    inline RuntimeBufferPyramidView() = delete;
    
    inline RuntimeBufferPyramidView(std::size_t Levels) : BaseT(Levels) { }
    
    inline ~RuntimeBufferPyramidView() { }
    
    inline RuntimeBufferPyramidView(const RuntimeBufferPyramidView<T,TargetType>& pyramid) : BaseT(pyramid) { }
    
    inline RuntimeBufferPyramidView(RuntimeBufferPyramidView<T,TargetType>&& pyramid) : BaseT(std::move(pyramid)) { }
    
    inline RuntimeBufferPyramidView<T,TargetType>& operator=(const RuntimeBufferPyramidView<T,TargetType>& pyramid)
    {
        BaseT::operator=(pyramid);
        return *this;
    }
    
    inline RuntimeBufferPyramidView<T,TargetType>& operator=(RuntimeBufferPyramidView<T,TargetType>&& pyramid)
    {
        BaseT::operator=(std::move(pyramid));
        return *this;
    }
    
    inline void memset(unsigned char v = 0)
    {
        for(std::size_t l = 0 ; l < BaseT::getLevelCount() ; ++l) 
        {
            BaseT::imgs[l].memset(v);
        }
    }
    
    inline void copyFrom(const RuntimeBufferPyramidView<T,TargetHost>& pyramid)
    {
        for(std::size_t l = 0 ; l < BaseT::getLevelCount() ; ++l) 
        {
            BaseT::imgs[l].copyFrom(pyramid[l]);
        }
    }
    
#ifdef VISIONCORE_HAVE_CUDA
    inline void copyFrom(const RuntimeBufferPyramidView<T,TargetDeviceCUDA>& pyramid)
    {
        for(std::size_t l = 0 ; l < BaseT::getLevelCount() ; ++l) 
        {
            BaseT::imgs[l].copyFrom(pyramid[l]);
        }
    }
#endif // VISIONCORE_HAVE_CUDA
    
#ifdef VISIONCORE_HAVE_OPENCL
    inline void copyFrom(const cl::CommandQueue& queue, const RuntimeBufferPyramidView<T,TargetDeviceOpenCL>& pyramid)
    {
        for(std::size_t l = 0 ; l < BaseT::getLevelCount() ; ++l) 
        {
            BaseT::imgs[l].copyFrom(queue, pyramid[l]);
        }
    }
#endif // VISIONCORE_HAVE_OPENCL
};

#ifdef VISIONCORE_HAVE_CUDA
/**
 * Buffer Pyramid View - CUDA.
 */
template<typename T, std::size_t Levels>
class BufferPyramidView<T,Levels,TargetDeviceCUDA> : public PyramidViewBase<Buffer2DView, T,Levels,TargetDeviceCUDA>
{
public:
    typedef PyramidViewBase<Buffer2DView, T,Levels,TargetDeviceCUDA> BaseT;
    typedef typename BaseT::ViewType LevelT;
    static const std::size_t LevelCount = BaseT::LevelCount;
    typedef typename BaseT::ValueType ValueType;
    typedef typename BaseT::TargetType TargetType;
    
    EIGEN_DEVICE_FUNC inline BufferPyramidView() { }
    
    EIGEN_DEVICE_FUNC inline ~BufferPyramidView() { }
    
    EIGEN_DEVICE_FUNC inline BufferPyramidView(const BufferPyramidView<T,LevelCount,TargetType>& pyramid) : BaseT(pyramid) { }
    
    EIGEN_DEVICE_FUNC inline BufferPyramidView(BufferPyramidView<T,LevelCount,TargetType>&& pyramid) : BaseT(std::move(pyramid)) { }
    
    EIGEN_DEVICE_FUNC inline BufferPyramidView<T,LevelCount,TargetType>& operator=(const BufferPyramidView<T,LevelCount,TargetType>& pyramid)
    {
        BaseT::operator=(pyramid);
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline BufferPyramidView<T,LevelCount,TargetType>& operator=(BufferPyramidView<T,LevelCount,TargetType>&& pyramid)
    {
        BaseT::operator=(std::move(pyramid));
        return *this;
    }
    
    inline void memset(unsigned char v = 0)
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            BaseT::imgs[l].memset(v);
        }
    }
    
    template<typename TargetFrom>
    inline void copyFrom(const BufferPyramidView<T,Levels,TargetFrom>& pyramid)
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            BaseT::imgs[l].copyFrom(pyramid.imgs[l]);
        }
    }
};

/**
 * Runtime Buffer Pyramid View - CUDA.
 */
template<typename T>
class RuntimeBufferPyramidView<T,TargetDeviceCUDA> : public RuntimePyramidViewBase<Buffer2DView,T,TargetDeviceCUDA>
{
public:
  typedef RuntimePyramidViewBase<Buffer2DView,T,TargetDeviceCUDA> BaseT;
  typedef typename BaseT::ViewType LevelT;
  typedef typename BaseT::ValueType ValueType;
  typedef typename BaseT::TargetType TargetType;
  
  inline RuntimeBufferPyramidView() = delete;
  
  inline RuntimeBufferPyramidView(std::size_t Levels) : BaseT(Levels) { }
  
  inline ~RuntimeBufferPyramidView() { }
  
  inline RuntimeBufferPyramidView(const RuntimeBufferPyramidView<T,TargetType>& pyramid) : BaseT(pyramid) { }
  
  inline RuntimeBufferPyramidView(RuntimeBufferPyramidView<T,TargetType>&& pyramid) : BaseT(std::move(pyramid)) { }
  
  inline RuntimeBufferPyramidView<T,TargetType>& operator=(const RuntimeBufferPyramidView<T,TargetType>& pyramid)
  {
      BaseT::operator=(pyramid);
      return *this;
  }
  
  inline RuntimeBufferPyramidView<T,TargetType>& operator=(RuntimeBufferPyramidView<T,TargetType>&& pyramid)
  {
      BaseT::operator=(std::move(pyramid));
      return *this;
  }
  
  inline void memset(unsigned char v = 0)
  {
      for(std::size_t l = 0 ; l < BaseT::getLevelCount() ; ++l) 
      {
          BaseT::imgs[l].memset(v);
      }
  }
  
  template<typename TargetFrom>
  inline void copyFrom(const RuntimeBufferPyramidView<T,TargetFrom>& pyramid)
  {
      for(std::size_t l = 0 ; l < BaseT::getLevelCount() ; ++l) 
      {
          BaseT::imgs[l].copyFrom(pyramid[l]);
      }
  }
};

#endif // VISIONCORE_HAVE_CUDA

#ifdef VISIONCORE_HAVE_OPENCL
/**
 * Buffer Pyramid View - OpenCL.
 */
template<typename T, std::size_t Levels>
class BufferPyramidView<T,Levels,TargetDeviceOpenCL> : public PyramidViewBase<Buffer2DView, T,Levels,TargetDeviceOpenCL>
{
public:
    typedef PyramidViewBase<Buffer2DView, T,Levels,TargetDeviceOpenCL> BaseT;
    typedef typename BaseT::ViewType LevelT;
    static const std::size_t LevelCount = BaseT::LevelCount;
    typedef typename BaseT::ValueType ValueType;
    typedef typename BaseT::TargetType TargetType;
    
    inline BufferPyramidView() { }
    
    inline ~BufferPyramidView() { }
    
    inline BufferPyramidView(const BufferPyramidView<T,LevelCount,TargetType>& pyramid) : BaseT(pyramid) { }
    
    inline BufferPyramidView(BufferPyramidView<T,LevelCount,TargetType>&& pyramid) : BaseT(std::move(pyramid)) { }
    
    inline BufferPyramidView<T,LevelCount,TargetType>& operator=(const BufferPyramidView<T,LevelCount,TargetType>& pyramid)
    {
        BaseT::operator=(pyramid);
        return *this;
    }
    
    inline BufferPyramidView<T,LevelCount,TargetType>& operator=(BufferPyramidView<T,LevelCount,TargetType>&& pyramid)
    {
        BaseT::operator=(std::move(pyramid));
        return *this;
    }
    
    inline void memset(unsigned char v = 0)
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            BaseT::imgs[l].memset(v);
        }
    }
    
    inline void copyFrom(const cl::CommandQueue& queue, const BufferPyramidView<T,Levels,TargetHost>& pyramid)
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            BaseT::imgs[l].copyFrom(queue, pyramid.imgs[l]);
        }
    }
    
    inline void copyFrom(const cl::CommandQueue& queue, const BufferPyramidView<T,Levels,TargetDeviceOpenCL>& pyramid)
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            BaseT::imgs[l].copyFrom(queue, pyramid.imgs[l]);
        }
    }

    inline void memset(const cl::CommandQueue& queue, cl_float4 v)
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            BaseT::imgs[l].memset(queue, v);
        }
    }
};

/**
 * Runtime Buffer Pyramid View - OpenCL.
 */
template<typename T>
class RuntimeBufferPyramidView<T,TargetDeviceOpenCL> : public RuntimePyramidViewBase<Buffer2DView,T,TargetDeviceOpenCL>
{
public:
    typedef RuntimePyramidViewBase<Buffer2DView,T,TargetDeviceOpenCL> BaseT;
    typedef typename BaseT::ViewType LevelT;
    typedef typename BaseT::ValueType ValueType;
    typedef typename BaseT::TargetType TargetType;
    
    inline RuntimeBufferPyramidView() = delete;
    
    inline RuntimeBufferPyramidView(std::size_t Levels) : BaseT(Levels) { }
    
    inline ~RuntimeBufferPyramidView() { }
    
    inline RuntimeBufferPyramidView(const RuntimeBufferPyramidView<T,TargetType>& pyramid) : BaseT(pyramid) { }
    
    inline RuntimeBufferPyramidView(RuntimeBufferPyramidView<T,TargetType>&& pyramid) : BaseT(std::move(pyramid)) { }
    
    inline RuntimeBufferPyramidView<T,TargetType>& operator=(const RuntimeBufferPyramidView<T,TargetType>& pyramid)
    {
        BaseT::operator=(pyramid);
        return *this;
    }
    
    inline RuntimeBufferPyramidView<T,TargetType>& operator=(RuntimeBufferPyramidView<T,TargetType>&& pyramid)
    {
        BaseT::operator=(std::move(pyramid));
        return *this;
    }
    
    inline void memset(unsigned char v = 0)
    {
        for(std::size_t l = 0 ; l < BaseT::getLevelCount() ; ++l) 
        {
            BaseT::imgs[l].memset(v);
        }
    }
    
    inline void copyFrom(const cl::CommandQueue& queue, const RuntimeBufferPyramidView<T,TargetHost>& pyramid)
    {
        for(std::size_t l = 0 ; l < BaseT::getLevelCount() ; ++l) 
        {
            BaseT::imgs[l].copyFrom(queue, pyramid[l]);
        }
    }
    
    inline void copyFrom(const cl::CommandQueue& queue, const RuntimeBufferPyramidView<T,TargetDeviceOpenCL>& pyramid)
    {
        for(std::size_t l = 0 ; l < BaseT::getLevelCount() ; ++l) 
        {
            BaseT::imgs[l].copyFrom(queue, pyramid[l]);
        }
    }
    
    inline void memset(const cl::CommandQueue& queue, cl_float4 v)
    {
        for(std::size_t l = 0 ; l < BaseT::getLevelCount() ; ++l) 
        {
            BaseT::imgs[l].memset(queue, v);
        }
    }
};

#endif //  VISIONCORE_HAVE_OPENCL

/// ***********************************************************************

/**
 * CUDA/Host Buffer Pyramid Creation.
 */
template<typename T, std::size_t Levels, typename Target = TargetHost>
class BufferPyramidManaged : public BufferPyramidView<T, Levels, Target>
{
public:
    typedef BufferPyramidView<T, Levels, Target> ViewT;
    typedef typename ViewT::ViewType LevelT;
    static const std::size_t LevelCount = Levels;
    typedef T ValueType;
    typedef Target TargetType;
    
    BufferPyramidManaged() = delete;
    
    inline BufferPyramidManaged(std::size_t w, std::size_t h)
    {        
        // Build power of two structure
        for(std::size_t l = 0; l < LevelCount && (w>>l > 0) && (h>>l > 0); ++l ) 
        {
            typename Target::template PointerType<T> ptr = 0;
            std::size_t line_pitch = 0;
            
            Target::template AllocatePitchedMem<T>(&ptr, &line_pitch, w >> l, h >> l);
            
            ViewT::imgs[l] = LevelT(ptr, w >> l, h >> l, line_pitch);
        }
    }
    
    BufferPyramidManaged(const BufferPyramidManaged<T,LevelCount,Target>& img) = delete;
    
    inline BufferPyramidManaged(BufferPyramidManaged<T,LevelCount,Target>&& img) : ViewT(std::move(img))
    {
        
    }
    
    BufferPyramidManaged<T,LevelCount,Target>& operator=(const BufferPyramidManaged<T,LevelCount,Target>& img) = delete;
    
    inline BufferPyramidManaged<T,LevelCount,Target>& operator=(BufferPyramidManaged<T,LevelCount,Target>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline ~BufferPyramidManaged()
    {
        for(std::size_t l = 0; l < LevelCount ; ++l)
        {
            Target::template DeallocatePitchedMem<T>(ViewT::imgs[l].rawPtr());
        }
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

#ifdef VISIONCORE_HAVE_OPENCL

/**
 * OpenCL Buffer Pyramid Creation.
 */
template<typename T, std::size_t Levels>
class BufferPyramidManaged<T,Levels,TargetDeviceOpenCL> : public BufferPyramidView<T, Levels, TargetDeviceOpenCL>
{
public:
    typedef BufferPyramidView<T, Levels, TargetDeviceOpenCL> ViewT;
    typedef typename ViewT::ViewType LevelT;
    static const std::size_t LevelCount = Levels;
    typedef T ValueType;
    typedef TargetDeviceOpenCL TargetType;
    
    BufferPyramidManaged() = delete;

    inline BufferPyramidManaged(std::size_t w, std::size_t h, const cl::Context& context, cl_mem_flags flags)
    {        
        // Build power of two structure
        for(std::size_t l = 0; l < LevelCount && (w>>l > 0) && (h>>l > 0); ++l ) 
        {
            const std::size_t new_w = w >> l;
            const std::size_t new_h = h >> l;
            ViewT::imgs[l] = Buffer2DView<T,TargetType>(new cl::Buffer(context, flags, new_w*new_h*sizeof(T)), new_w, new_h, new_w * sizeof(T));
        }
    }
    
    BufferPyramidManaged(const BufferPyramidManaged<T,LevelCount,TargetType>& img) = delete;
    
    inline BufferPyramidManaged(BufferPyramidManaged<T,LevelCount,TargetType>&& img) : ViewT(std::move(img))
    {
        
    }
    
    BufferPyramidManaged<T,LevelCount,TargetType>& operator=(const BufferPyramidManaged<T,LevelCount,TargetType>& img) = delete;
    
    inline BufferPyramidManaged<T,LevelCount,TargetType>& operator=(BufferPyramidManaged<T,LevelCount,TargetType>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline ~BufferPyramidManaged()
    {
        for(std::size_t l = 0; l < LevelCount ; ++l)
        {
            if(ViewT::imgs[l].isValid())
            {
                cl::Buffer* clb = static_cast<cl::Buffer*>(ViewT::imgs[l].rawPtr());
                delete clb;
            }
        }
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

#endif // VISIONCORE_HAVE_OPENCL

/**
 * CUDA/Host Runtime Buffer Pyramid Creation.
 */
template<typename T, typename Target = TargetHost>
class RuntimeBufferPyramidManaged : public RuntimeBufferPyramidView<T, Target>
{
public:
    typedef RuntimeBufferPyramidView<T, Target> ViewT;
    typedef typename ViewT::ViewType LevelT;
    typedef T ValueType;
    typedef Target TargetType;
    
    RuntimeBufferPyramidManaged() = delete;
    
    inline RuntimeBufferPyramidManaged(std::size_t LevelCount, std::size_t w, std::size_t h) : ViewT(LevelCount)
    {        
        // Build power of two structure
        for(std::size_t l = 0; l < LevelCount && (w>>l > 0) && (h>>l > 0); ++l ) 
        {
            typename Target::template PointerType<T> ptr = 0;
            std::size_t line_pitch = 0;
            
            Target::template AllocatePitchedMem<T>(&ptr, &line_pitch, w >> l, h >> l);
            
            ViewT::imgs[l] = LevelT(ptr, w >> l, h >> l, line_pitch);
        }
    }
    
    RuntimeBufferPyramidManaged(const RuntimeBufferPyramidManaged<T,Target>& img) = delete;
    
    inline RuntimeBufferPyramidManaged(RuntimeBufferPyramidManaged<T,Target>&& img) : ViewT(std::move(img))
    {
      
    }
    
    RuntimeBufferPyramidManaged<T,Target>& operator=(const RuntimeBufferPyramidManaged<T,Target>& img) = delete;
    
    inline RuntimeBufferPyramidManaged<T,Target>& operator=(RuntimeBufferPyramidManaged<T,Target>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline ~RuntimeBufferPyramidManaged()
    {
        for(std::size_t l = 0; l < ViewT::getLevelCount() ; ++l)
        {
            Target::template DeallocatePitchedMem<T>(ViewT::imgs[l].rawPtr());
        }
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

}

#endif // VISIONCORE_BUFFER_PYRAMID_HPP
