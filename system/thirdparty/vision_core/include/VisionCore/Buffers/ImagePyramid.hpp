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
 * Image Host/Device Pyramid.
 * ****************************************************************************
 */

#ifndef VISIONCORE_IMAGE_PYRAMID_HPP
#define VISIONCORE_IMAGE_PYRAMID_HPP

#include <VisionCore/Buffers/PyramidBase.hpp>
#include <VisionCore/Buffers/Image2D.hpp>

namespace vc
{

template<typename T, std::size_t Levels, typename Target = TargetHost>
class ImagePyramidView { };

/**
 * Image Pyramid View - Host.
 */
template<typename T, std::size_t Levels>
class ImagePyramidView<T,Levels,TargetHost> : public PyramidViewBase<Image2DView, T,Levels,TargetHost>
{
public:
    typedef PyramidViewBase<Image2DView, T,Levels,TargetHost> BaseT;
    typedef typename BaseT::ViewType LevelT;
    static const std::size_t LevelCount = BaseT::LevelCount;
    typedef typename BaseT::ValueType ValueType;
    typedef typename BaseT::TargetType TargetType;
    
    EIGEN_DEVICE_FUNC inline ImagePyramidView() { }
    
    EIGEN_DEVICE_FUNC inline ~ImagePyramidView() { }
    
    EIGEN_DEVICE_FUNC inline ImagePyramidView(const ImagePyramidView<T,LevelCount,TargetType>& pyramid) : BaseT(pyramid) { }
    
    EIGEN_DEVICE_FUNC inline ImagePyramidView(ImagePyramidView<T,LevelCount,TargetType>&& pyramid) : BaseT(std::move(pyramid)) { }
    
    EIGEN_DEVICE_FUNC inline ImagePyramidView<T,LevelCount,TargetType>& operator=(const ImagePyramidView<T,LevelCount,TargetType>& pyramid)
    {
        BaseT::operator=(pyramid);
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline ImagePyramidView<T,LevelCount,TargetType>& operator=(ImagePyramidView<T,LevelCount,TargetType>&& pyramid)
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
    
    inline void copyFrom(const ImagePyramidView<T,Levels,TargetHost>& pyramid)
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            BaseT::imgs[l].copyFrom(pyramid.imgs[l]);
        }
    }
    
#ifdef VISIONCORE_HAVE_CUDA
    inline void copyFrom(const ImagePyramidView<T,Levels,TargetDeviceCUDA>& pyramid)
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            BaseT::imgs[l].copyFrom(pyramid.imgs[l]);
        }
    }
#endif // VISIONCORE_HAVE_CUDA

#ifdef VISIONCORE_HAVE_OPENCL
    inline void copyFrom(const cl::CommandQueue& queue, const ImagePyramidView<T,Levels,TargetDeviceOpenCL>& pyramid)
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            BaseT::imgs[l].copyFrom(queue, pyramid.imgs[l]);
        }
    }
#endif // VISIONCORE_HAVE_OPENCL
};

#ifdef VISIONCORE_HAVE_CUDA

/**
 * Image Pyramid View - CUDA.
 */
template<typename T, std::size_t Levels>
class ImagePyramidView<T,Levels,TargetDeviceCUDA> : public PyramidViewBase<Image2DView, T,Levels,TargetDeviceCUDA>
{
public:
    typedef PyramidViewBase<Image2DView, T,Levels,TargetDeviceCUDA> BaseT;
    typedef typename BaseT::ViewType LevelT;
    static const std::size_t LevelCount = BaseT::LevelCount;
    typedef typename BaseT::ValueType ValueType;
    typedef typename BaseT::TargetType TargetType;
    
    EIGEN_DEVICE_FUNC inline ImagePyramidView() { }
    
    EIGEN_DEVICE_FUNC inline ~ImagePyramidView() { }
    
    EIGEN_DEVICE_FUNC inline ImagePyramidView(const ImagePyramidView<T,LevelCount,TargetType>& pyramid) : BaseT(pyramid) { }
    
    EIGEN_DEVICE_FUNC inline ImagePyramidView(ImagePyramidView<T,LevelCount,TargetType>&& pyramid) : BaseT(std::move(pyramid)) { }
    
    EIGEN_DEVICE_FUNC inline ImagePyramidView<T,LevelCount,TargetType>& operator=(const ImagePyramidView<T,LevelCount,TargetType>& pyramid)
    {
        BaseT::operator=(pyramid);
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline ImagePyramidView<T,LevelCount,TargetType>& operator=(ImagePyramidView<T,LevelCount,TargetType>&& pyramid)
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
    inline void copyFrom(const ImagePyramidView<T,Levels,TargetFrom>& pyramid)
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            BaseT::imgs[l].copyFrom(pyramid.imgs[l]);
        }
    }
};

#endif // VISIONCORE_HAVE_CUDA

#ifdef VISIONCORE_HAVE_OPENCL
/**
 * Image Pyramid View - OpenCL.
 */
template<typename T, std::size_t Levels>
class ImagePyramidView<T,Levels,TargetDeviceOpenCL> : public PyramidViewBase<Image2DView, T,Levels,TargetDeviceOpenCL>
{
public:
    typedef PyramidViewBase<Image2DView, T,Levels,TargetDeviceOpenCL> BaseT;
    typedef typename BaseT::ViewType LevelT;
    static const std::size_t LevelCount = BaseT::LevelCount;
    typedef typename BaseT::ValueType ValueType;
    typedef typename BaseT::TargetType TargetType;
    
    inline ImagePyramidView() { }
    
    inline ~ImagePyramidView() { }
    
    inline ImagePyramidView(const ImagePyramidView<T,LevelCount,TargetType>& pyramid) : BaseT(pyramid) { }
    
    inline ImagePyramidView(ImagePyramidView<T,LevelCount,TargetType>&& pyramid) : BaseT(std::move(pyramid)) { }
    
    inline ImagePyramidView<T,LevelCount,TargetType>& operator=(const ImagePyramidView<T,LevelCount,TargetType>& pyramid)
    {
        BaseT::operator=(pyramid);
        return *this;
    }
    
    inline ImagePyramidView<T,LevelCount,TargetType>& operator=(ImagePyramidView<T,LevelCount,TargetType>&& pyramid)
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
    
    inline void copyFrom(const cl::CommandQueue& queue, const ImagePyramidView<T,Levels,TargetHost>& pyramid)
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            BaseT::imgs[l].copyFrom(queue, pyramid.imgs[l]);
        }
    }
    
    inline void copyFrom(const cl::CommandQueue& queue, const ImagePyramidView<T,Levels,TargetDeviceOpenCL>& pyramid)
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

#endif //  VISIONCORE_HAVE_OPENCL

/// ***********************************************************************

/**
 * CUDA/Host Image Pyramid Creation.
 */
template<typename T, std::size_t Levels, typename Target = TargetHost>
class ImagePyramidManaged : public ImagePyramidView<T, Levels, Target>
{
public:
    typedef ImagePyramidView<T, Levels, Target> ViewT;
    typedef typename ViewT::ViewType LevelT;
    static const std::size_t LevelCount = Levels;
    typedef T ValueType;
    typedef Target TargetType;
    
    ImagePyramidManaged() = delete;
    
    inline ImagePyramidManaged(std::size_t w, std::size_t h)
    {        
        // Build power of two structure
        for(std::size_t l = 0; l < LevelCount && (w>>l > 0) && (h>>l > 0); ++l ) 
        {
            typename Target::template PointerType<T> ptr = 0;
            std::size_t line_pitch = 0;
            
            Target::template AllocatePitchedMem<T>(&ptr, &line_pitch, w >> l, h >> l);
            
            ViewT::imgs[l] = Image2DView<T,Target>((T*)ptr, w >> l, h >> l, line_pitch);
        }
    }
    
    ImagePyramidManaged(const ImagePyramidManaged<T,LevelCount,Target>& img) = delete;
    
    inline ImagePyramidManaged(ImagePyramidManaged<T,LevelCount,Target>&& img) : ViewT(std::move(img))
    {
        
    }
    
    ImagePyramidManaged<T,LevelCount,Target>& operator=(const ImagePyramidManaged<T,LevelCount,Target>& img) = delete;
    
    inline ImagePyramidManaged<T,LevelCount,Target>& operator=(ImagePyramidManaged<T,LevelCount,Target>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline ~ImagePyramidManaged()
    {
        for(std::size_t l = 0; l < LevelCount ; ++l)
        {
            Target::template DeallocatePitchedMem<T>(ViewT::imgs[l].ptr());
        }
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

#ifdef VISIONCORE_HAVE_OPENCL

/**
 * OpenCL Image Pyramid Creation.
 */
template<typename T, std::size_t Levels>
class ImagePyramidManaged<T,Levels,TargetDeviceOpenCL> : public ImagePyramidView<T, Levels, TargetDeviceOpenCL>
{
public:
    typedef ImagePyramidView<T, Levels, TargetDeviceOpenCL> ViewT;
    typedef typename ViewT::ViewType LevelT;
    static const std::size_t LevelCount = Levels;
    typedef T ValueType;
    typedef TargetDeviceOpenCL TargetType;
    
    ImagePyramidManaged() = delete;
    
    inline ImagePyramidManaged(std::size_t w, std::size_t h, const cl::Context& context, cl_mem_flags flags, const cl::ImageFormat& fmt)
    {        
        // Build power of two structure
        for(std::size_t l = 0; l < LevelCount && (w>>l > 0) && (h>>l > 0); ++l ) 
        {
            ViewT::imgs[l] = Image2DView<T,TargetType>(new cl::Image2D(context, flags, fmt, w >> l, h >> l, 0, nullptr), w >> l, h >> l, w * sizeof(T));
        }
    }
    
    inline ImagePyramidManaged(std::size_t w, std::size_t h, const cl::Context& context, cl_mem_flags flags)
    {        
        // Build power of two structure
        for(std::size_t l = 0; l < LevelCount && (w>>l > 0) && (h>>l > 0); ++l ) 
        {
            const std::size_t new_w = w >> l;
            const std::size_t new_h = h >> l;
            ViewT::imgs[l] = Image2DView<T,TargetType>(new cl::Image2D(context, flags, cl::ImageFormat(internal::ChannelCountToOpenCLChannelOrder<LevelT::Channels>::ChannelOrder,
                                                                                                             internal::TypeToOpenCLChannelType<typename LevelT::ValueType>::ChannelType), new_w, new_h, 0, nullptr), new_w, new_h, new_w * sizeof(T));
        }
    }
    
    ImagePyramidManaged(const ImagePyramidManaged<T,LevelCount,TargetType>& img) = delete;
    
    inline ImagePyramidManaged(ImagePyramidManaged<T,LevelCount,TargetType>&& img) : ViewT(std::move(img))
    {
        
    }
    
    ImagePyramidManaged<T,LevelCount,TargetType>& operator=(const ImagePyramidManaged<T,LevelCount,TargetType>& img) = delete;
    
    inline ImagePyramidManaged<T,LevelCount,TargetType>& operator=(ImagePyramidManaged<T,LevelCount,TargetType>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline ~ImagePyramidManaged()
    {
        for(std::size_t l = 0; l < LevelCount ; ++l)
        {
            if(ViewT::imgs[l].isValid())
            {
                cl::Image2D* clb = static_cast<cl::Image2D*>(ViewT::imgs[l].rawPtr());
                delete clb;
            }
        }
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

#endif // VISIONCORE_HAVE_OPENCL

}

#endif // VISIONCORE_IMAGE_PYRAMID_HPP
