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
 * 3D Host/Device Buffer.
 * ****************************************************************************
 */
#ifndef VISIONCORE_BUFFER3D_HPP
#define VISIONCORE_BUFFER3D_HPP

#include <VisionCore/Platform.hpp>

// Core
#include <VisionCore/Buffers/Buffer2D.hpp>
#include <VisionCore/MemoryPolicy.hpp>

namespace vc
{

/**
 * Buffer 3D View - Basics.
 */
template<typename T, typename Target = TargetHost>
class Buffer3DViewBase
{
public:
    typedef T ValueType;
    typedef Target TargetType;
    
#ifndef VISIONCORE_CUDA_KERNEL_SPACE
    EIGEN_DEVICE_FUNC inline Buffer3DViewBase() : memptr(nullptr) { }
#else // VISIONCORE_CUDA_KERNEL_SPACE
    EIGEN_DEVICE_FUNC inline Buffer3DViewBase() { }
#endif // VISIONCORE_CUDA_KERNEL_SPACE
    
    EIGEN_DEVICE_FUNC inline ~Buffer3DViewBase()
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline Buffer3DViewBase(const Buffer3DViewBase<T,TargetType>& img)
        : memptr(img.memptr), xsize(img.xsize), ysize(img.ysize), zsize(img.zsize), line_pitch(img.line_pitch), plane_pitch(img.plane_pitch)
    {
        
    } 
    
    EIGEN_DEVICE_FUNC inline Buffer3DViewBase(Buffer3DViewBase<T,TargetType>&& img)
    : memptr(img.memptr), xsize(img.xsize), ysize(img.ysize), zsize(img.zsize), line_pitch(img.line_pitch), plane_pitch(img.plane_pitch)
    {
        img.memptr = 0;
    }
    
    EIGEN_DEVICE_FUNC inline Buffer3DViewBase<T,TargetType>& operator=(const Buffer3DViewBase<T,TargetType>& img)
    {
        memptr = (void*)img.rawPtr();
        line_pitch = img.pitch();
        xsize = img.width();
        ysize = img.height();
        zsize = img.depth();
        plane_pitch = img.planePitch();
        
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline Buffer3DViewBase<T,TargetType>& operator=(Buffer3DViewBase<T,TargetType>&& img)
    {
        memptr = img.memptr;
        line_pitch = img.line_pitch;
        xsize = img.xsize;
        ysize = img.ysize;
        zsize = img.zsize;
        plane_pitch = img.plane_pitch;
        img.memptr = 0;
        
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline Buffer3DViewBase(typename TargetType::template PointerType<T> optr) : memptr(optr), xsize(0), ysize(0), zsize(0), line_pitch(0), plane_pitch(0)
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline Buffer3DViewBase(typename TargetType::template PointerType<T> optr, std::size_t w, std::size_t h, std::size_t d)
    {
        memptr = optr;
        xsize = w;
        ysize = h;
        zsize = d;
        line_pitch = sizeof(T) * xsize;
        plane_pitch = line_pitch * ysize; 
    }
    
    EIGEN_DEVICE_FUNC inline Buffer3DViewBase(typename TargetType::template PointerType<T> optr, std::size_t w, std::size_t h, std::size_t d, std::size_t opitch)
    {   
        memptr = optr;
        xsize = w;
        ysize = h;
        zsize = d;
        line_pitch = opitch;
        plane_pitch = line_pitch * ysize; 
    }
    
    EIGEN_DEVICE_FUNC inline Buffer3DViewBase(typename TargetType::template PointerType<T> optr, std::size_t w, std::size_t h, std::size_t d, std::size_t opitch, std::size_t oimg_pitch)
    {
        memptr = optr;
        xsize = w;
        ysize = h;
        zsize = d;
        line_pitch = opitch;
        plane_pitch = oimg_pitch; 
    }
    
    EIGEN_DEVICE_FUNC inline std::size_t width() const { return xsize; }
    EIGEN_DEVICE_FUNC inline std::size_t height() const { return ysize; }
    EIGEN_DEVICE_FUNC inline std::size_t depth() const { return zsize; }
    EIGEN_DEVICE_FUNC inline std::size_t planePitch() const { return plane_pitch; }
    EIGEN_DEVICE_FUNC inline std::size_t elementPlanePitch() const { return plane_pitch / sizeof(T); }
    EIGEN_DEVICE_FUNC inline std::size_t pitch() const { return line_pitch; }
    EIGEN_DEVICE_FUNC inline std::size_t elementPitch() const { return line_pitch / sizeof(T); }
    EIGEN_DEVICE_FUNC inline std::size_t volume() const { return xsize * ysize * zsize; }
    EIGEN_DEVICE_FUNC inline std::size_t bytes() const { return plane_pitch * zsize; }
    EIGEN_DEVICE_FUNC inline std::size_t totalElements() const { return (elementPlanePitch() * zsize); }
    
    EIGEN_DEVICE_FUNC inline bool inBounds(std::size_t x, std::size_t y, std::size_t z) const
    {
        return (x < xsize) && (y < ysize) && (z < zsize);
    }
    
    EIGEN_DEVICE_FUNC inline bool isValid() const { return rawPtr() != nullptr; }
    EIGEN_DEVICE_FUNC inline const typename TargetType::template PointerType<T> rawPtr() const { return memptr; }
    EIGEN_DEVICE_FUNC inline typename TargetType::template PointerType<T> rawPtr() { return memptr; }
    
    EIGEN_DEVICE_FUNC inline uint3 voxels() const
    {
        return make_uint3(xsize,ysize,zsize);
    }
protected:
    typename Target::template PointerType<T> memptr;
    std::size_t xsize;    
    std::size_t ysize;    
    std::size_t zsize;
    std::size_t line_pitch;    
    std::size_t plane_pitch;
};

/**
 * Buffer 3D View - Add contents access methods.
 */
template<typename T, typename Target>
class Buffer3DViewAccessible : public Buffer3DViewBase<T,Target>
{
public:
    typedef Buffer3DViewBase<T,Target> BaseT;
    typedef typename BaseT::ValueType ValueType;
    typedef typename BaseT::TargetType TargetType;
    
    EIGEN_DEVICE_FUNC inline Buffer3DViewAccessible() { }
    EIGEN_DEVICE_FUNC inline ~Buffer3DViewAccessible() { }
    EIGEN_DEVICE_FUNC inline Buffer3DViewAccessible(const Buffer3DViewAccessible<T,TargetType>& img) : BaseT(img) { } 
    EIGEN_DEVICE_FUNC inline Buffer3DViewAccessible(Buffer3DViewAccessible<T,TargetType>&& img) : BaseT(std::move(img)) { }
    
    EIGEN_DEVICE_FUNC inline Buffer3DViewAccessible<T,TargetType>& operator=(const Buffer3DViewAccessible<T,TargetType>& img)
    {
        BaseT::operator=(img);
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline Buffer3DViewAccessible<T,TargetType>& operator=(Buffer3DViewAccessible<T,TargetType>&& img)
    {
        BaseT::operator=(std::move(img));
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline Buffer3DViewAccessible(typename TargetType::template PointerType<T> optr, std::size_t w, std::size_t h, std::size_t d) : BaseT(optr,w,h,d) { }
    EIGEN_DEVICE_FUNC inline Buffer3DViewAccessible(typename TargetType::template PointerType<T> optr, std::size_t w, std::size_t h, std::size_t d, std::size_t opitch) : BaseT(optr,w,h,d,opitch) { }
    EIGEN_DEVICE_FUNC inline Buffer3DViewAccessible(typename TargetType::template PointerType<T> optr, std::size_t w, std::size_t h, std::size_t d, std::size_t opitch, std::size_t oimg_pitch) : BaseT(optr,w,h,d,opitch, oimg_pitch) { }
    
    EIGEN_DEVICE_FUNC inline T* ptr() { return (T*)BaseT::rawPtr(); }
    EIGEN_DEVICE_FUNC inline const T* ptr() const { return (T*)BaseT::rawPtr(); }
    
    EIGEN_DEVICE_FUNC inline const T* planePtr(std::size_t z) const
    {
        return (const T*)( ((const unsigned char*)BaseT::rawPtr()) + (z * BaseT::planePitch() ));
    }
    
    EIGEN_DEVICE_FUNC inline T* planePtr(std::size_t z)
    {
        return (T*)( ((unsigned char*)BaseT::rawPtr()) + (z * BaseT::planePitch() ));
    }
    
    EIGEN_DEVICE_FUNC inline T* rowPtr(std::size_t y, std::size_t z)
    {
        return (T*)( ((unsigned char*)BaseT::rawPtr()) + (z * BaseT::planePitch()) + (y * BaseT::pitch()));
    }
    
    EIGEN_DEVICE_FUNC inline const T* rowPtr(std::size_t y, std::size_t z) const
    {
        return (const T*)( ((const unsigned char*)BaseT::rawPtr()) + (z * BaseT::planePitch()) + (y * BaseT::pitch()));
    }
    
    EIGEN_DEVICE_FUNC inline T& operator[](std::size_t ix)
    {
        return *planePtr(ix);
    }
    
    EIGEN_DEVICE_FUNC inline const T& operator[](std::size_t ix) const
    {
        return *planePtr(ix);
    }
    
    EIGEN_DEVICE_FUNC inline T* ptr(std::size_t x, std::size_t y, std::size_t z) 
    {
        return rowPtr(y,z) + x;
    }
    
    EIGEN_DEVICE_FUNC inline const T* ptr(std::size_t x, std::size_t y, std::size_t z) const
    {
        return rowPtr(y,z) + x;
    }
    
    EIGEN_DEVICE_FUNC inline T& operator()(std::size_t x, std::size_t y, std::size_t z)
    {
        return *ptr(x,y,z);
    }
    
    EIGEN_DEVICE_FUNC inline const T& operator()(std::size_t x, std::size_t y, std::size_t z) const
    {
        return *ptr(x,y,z);
    }
    
    EIGEN_DEVICE_FUNC inline T& get(std::size_t x, std::size_t y, std::size_t z)
    {
        return *ptr(x,y,z);
    }
    
    EIGEN_DEVICE_FUNC inline const T& get(std::size_t x, std::size_t y, std::size_t z) const
    {
        return *ptr(x,y,z);
    }
    
    EIGEN_DEVICE_FUNC inline T& get(uint3 p)
    {
        return *ptr(p.x,p.y,p.z);
    }
    
    EIGEN_DEVICE_FUNC inline const T& get(uint3 p) const
    {
        return *ptr(p.x,p.y,p.z);
    }
    
    EIGEN_DEVICE_FUNC inline Buffer3DViewAccessible<T,TargetType> subBuffer3D(uint3 start, uint3 size)
    {
        return Buffer3DViewAccessible<T,TargetType>(&get(start), size.x, size.y, size.z, BaseT::pitch(), BaseT::planePitch());
    }
    
    EIGEN_DEVICE_FUNC inline Buffer2DView<T,TargetType> buffer2DXY(std::size_t z)
    {
        return Buffer2DView<T,TargetType>( planePtr(z), BaseT::width(), BaseT::height(), BaseT::pitch());
    }
    
    EIGEN_DEVICE_FUNC inline Buffer2DView<T,TargetType> buffer2DXZ(std::size_t y)
    {
        return Buffer2DView<T,TargetType>( rowPtr(y,0), BaseT::width(), BaseT::depth(), BaseT::planePitch());
    }
};

/**
 * View on a 3D Buffer.
 */    
template<typename T, typename Target = TargetHost>
class Buffer3DView
{
};

/**
 * View on a 3D Buffer Specialization - Host.
 */ 
template<typename T>
class Buffer3DView<T,TargetHost> : public Buffer3DViewAccessible<T,TargetHost>
{
public:
    typedef Buffer3DViewAccessible<T,TargetHost> BaseT;
    typedef typename BaseT::ValueType ValueType;
    typedef typename BaseT::TargetType TargetType;
    
    inline Buffer3DView() { }
    inline ~Buffer3DView() { }
    inline Buffer3DView(const Buffer3DView<T,TargetType>& img) : BaseT(img) { } 
    inline Buffer3DView(Buffer3DView<T,TargetType>&& img) : BaseT(std::move(img)) { }
    
    inline Buffer3DView<T,TargetType>& operator=(const Buffer3DView<T,TargetType>& img)
    {
        BaseT::operator=(img);
        return *this;
    }
    
    inline Buffer3DView<T,TargetType>& operator=(Buffer3DView<T,TargetType>&& img)
    {
        BaseT::operator=(std::move(img));
        return *this;
    }
    
    inline Buffer3DView(typename TargetType::template PointerType<T> optr, std::size_t w, std::size_t h, std::size_t d) : BaseT(optr,w,h,d) { }
    inline Buffer3DView(typename TargetType::template PointerType<T> optr, std::size_t w, std::size_t h, std::size_t d, std::size_t opitch) : BaseT(optr,w,h,d,opitch) { }
    inline Buffer3DView(typename TargetType::template PointerType<T> optr, std::size_t w, std::size_t h, std::size_t d, std::size_t opitch, std::size_t oimg_pitch) : BaseT(optr,w,h,d,opitch, oimg_pitch) { }
    
    inline void memset(unsigned char v = 0)
    {
        TargetType::template memset3D<T>(BaseT::rawPtr(), BaseT::pitch(), v, BaseT::width(), BaseT::height(), BaseT::depth());
    }

    inline void copyFrom(const Buffer3DView<T,TargetHost>& img)
    {
        typedef TargetHost TargetFrom;
        TargetTransfer<TargetFrom,TargetType>::template memcpy2D<T>(BaseT::rawPtr(), BaseT::pitch(), (typename TargetFrom::template PointerType<T>)img.ptr(), img.pitch(), std::min(img.width(), BaseT::width()) * sizeof(T), BaseT::height()*std::min(img.depth(), BaseT::depth()));
    }
    
#ifdef VISIONCORE_HAVE_CUDA
    inline void copyFrom(const Buffer3DView<T,TargetDeviceCUDA>& img)
    {
        typedef TargetDeviceCUDA TargetFrom;
        TargetTransfer<TargetFrom,TargetType>::template memcpy2D<T>(BaseT::rawPtr(), BaseT::pitch(), (typename TargetFrom::template PointerType<T>)img.ptr(), img.pitch(), std::min(img.width(), BaseT::width()) * sizeof(T), BaseT::height()*std::min(img.depth(), BaseT::depth()));
    }
#endif // VISIONCORE_HAVE_CUDA

#ifdef VISIONCORE_HAVE_OPENCL
    inline void copyFrom(const cl::CommandQueue& queue, const Buffer3DView<T,TargetDeviceOpenCL>& img)
    {
        queue.enqueueReadBuffer(img.clType(), true, 0, std::min(BaseT::bytes(), img.bytes()), BaseT::rawPtr());
    }
#endif // VISIONCORE_HAVE_OPENCL
    
    inline T* begin() { return BaseT::ptr(); }
    inline const T* begin() const { return BaseT::ptr(); }
    inline T* end() { return (BaseT::ptr() + BaseT::totalElements()); }
    inline const T* end() const { return (BaseT::ptr() + BaseT::totalElements()); }
};

#ifdef VISIONCORE_HAVE_CUDA
/**
 * View on a 3D Buffer Specialization - CUDA.
 */ 
template<typename T>
class Buffer3DView<T,TargetDeviceCUDA> : public Buffer3DViewAccessible<T,TargetDeviceCUDA>
{
public:
    typedef Buffer3DViewAccessible<T,TargetDeviceCUDA> BaseT;
    typedef typename BaseT::ValueType ValueType;
    typedef typename BaseT::TargetType TargetType;
    
    EIGEN_DEVICE_FUNC inline Buffer3DView() { }
    EIGEN_DEVICE_FUNC inline ~Buffer3DView() { }
    EIGEN_DEVICE_FUNC inline Buffer3DView(const Buffer3DView<T,TargetType>& img) : BaseT(img) { } 
    EIGEN_DEVICE_FUNC inline Buffer3DView(Buffer3DView<T,TargetType>&& img) : BaseT(std::move(img)) { }
    
    EIGEN_DEVICE_FUNC inline Buffer3DView<T,TargetType>& operator=(const Buffer3DView<T,TargetType>& img)
    {
        BaseT::operator=(img);
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline Buffer3DView<T,TargetType>& operator=(Buffer3DView<T,TargetType>&& img)
    {
        BaseT::operator=(std::move(img));
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline Buffer3DView(typename TargetType::template PointerType<T> optr, std::size_t w, std::size_t h, std::size_t d) : BaseT(optr,w,h,d) { }
    EIGEN_DEVICE_FUNC inline Buffer3DView(typename TargetType::template PointerType<T> optr, std::size_t w, std::size_t h, std::size_t d, std::size_t opitch) : BaseT(optr,w,h,d,opitch) { }
    EIGEN_DEVICE_FUNC inline Buffer3DView(typename TargetType::template PointerType<T> optr, std::size_t w, std::size_t h, std::size_t d, std::size_t opitch, std::size_t oimg_pitch) : BaseT(optr,w,h,d,opitch, oimg_pitch) { }
    
    inline void memset(unsigned char v = 0)
    {
        TargetType::template memset3D<T>(BaseT::rawPtr(), BaseT::pitch(), v, BaseT::width(), BaseT::height(), BaseT::depth());
    }
    
    template<typename TargetFrom>
    inline void copyFrom(const Buffer3DView<T,TargetFrom>& img)
    {
#ifdef VISIONCORE_HAVE_OPENCL
        static_assert(std::is_same<TargetFrom,TargetDeviceOpenCL>::value != true, "Not possible to do OpenCL-CUDA copy");
#endif
        
        TargetTransfer<TargetFrom,TargetType>::template memcpy2D<T>(BaseT::rawPtr(), BaseT::pitch(), (typename TargetFrom::template PointerType<T>)img.ptr(), img.pitch(), std::min(img.width(), BaseT::width()) * sizeof(T), BaseT::height()*std::min(img.depth(), BaseT::depth()));
    }
    
    EIGEN_DEVICE_FUNC inline typename internal::ThrustType<T,TargetType>::Ptr begin() 
    {
        return (typename internal::ThrustType<T,TargetType>::Ptr)(BaseT::rawPtr());
    }

    EIGEN_DEVICE_FUNC inline typename internal::ThrustType<T,TargetType>::Ptr end() 
    {
        return (typename internal::ThrustType<T,TargetType>::Ptr)( BaseT::ptr(BaseT::width(), BaseT::height()-1,BaseT::depth()-1) );
    }

    inline void fill(T val) 
    {
        thrust::fill(begin(), end(), val);
    }
};

#endif // VISIONCORE_HAVE_CUDA

#ifdef VISIONCORE_HAVE_OPENCL

/**
 * View on a 3D Buffer Specialization - OpenCL.
 */ 
template<typename T>
class Buffer3DView<T,TargetDeviceOpenCL> : public Buffer3DViewBase<T,TargetDeviceOpenCL>
{
public:
    typedef Buffer3DViewBase<T,TargetDeviceOpenCL> BaseT;
    typedef typename BaseT::ValueType ValueType;
    typedef typename BaseT::TargetType TargetType;
    
    inline Buffer3DView() { }
    inline ~Buffer3DView() { }
    inline Buffer3DView(const Buffer3DView<T,TargetType>& img) : BaseT(img) { } 
    inline Buffer3DView(Buffer3DView<T,TargetType>&& img) : BaseT(std::move(img)) { }
    
    inline Buffer3DView<T,TargetType>& operator=(const Buffer3DView<T,TargetType>& img)
    {
        BaseT::operator=(img);
        return *this;
    }
    
    inline Buffer3DView<T,TargetType>& operator=(Buffer3DView<T,TargetType>&& img)
    {
        BaseT::operator=(std::move(img));
        return *this;
    }
    
    inline Buffer3DView(typename TargetType::template PointerType<T> optr, std::size_t w, std::size_t h, std::size_t d) : BaseT(optr,w,h,d) { }
    inline Buffer3DView(typename TargetType::template PointerType<T> optr, std::size_t w, std::size_t h, std::size_t d, std::size_t opitch) : BaseT(optr,w,h,d,opitch) { }
    inline Buffer3DView(typename TargetType::template PointerType<T> optr, std::size_t w, std::size_t h, std::size_t d, std::size_t opitch, std::size_t oimg_pitch) : BaseT(optr,w,h,d,opitch, oimg_pitch) { }
    
    inline const cl::Buffer& clType() const { return *static_cast<const cl::Buffer*>(BaseT::rawPtr()); }
    inline cl::Buffer& clType() { return *static_cast<cl::Buffer*>(BaseT::rawPtr()); }
    
    inline void copyFrom(const cl::CommandQueue& queue, const Buffer3DView<T,TargetHost>& img)
    {
        queue.enqueueWriteBuffer(clType(), true, 0, std::min(BaseT::bytes(), img.bytes()), img.rawPtr());
    }
    
    inline void copyFrom(const cl::CommandQueue& queue, const Buffer3DView<T,TargetDeviceOpenCL>& img)
    {
        queue.enqueueCopyBuffer(img.clType(), clType(), 0, 0, std::min(BaseT::bytes(), img.bytes()));
    }
    
    inline void memset(const cl::CommandQueue& queue, T v)
    {
        queue.enqueueFillBuffer(clType(), v, 0, BaseT::bytes());
    }
};

/**
 * OpenCL Buffer to Host Mapping.
 */
struct Buffer3DMapper
{
    template<typename T>
    static inline Buffer3DView<T,TargetHost> map(const cl::CommandQueue& queue, cl_map_flags flags, const Buffer3DView<T,TargetDeviceOpenCL>& buf, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        typename TargetHost::template PointerType<T> ptr = queue.enqueueMapBuffer(buf.clType(), true, flags, 0, buf.bytes(), events, event);
        return Buffer3DView<T,TargetHost>(ptr, buf.width(), buf.height(), buf.depth());
    }
    
    template<typename T>
    static inline void unmap(const cl::CommandQueue& queue, const Buffer3DView<T,TargetDeviceOpenCL>& buf, const Buffer3DView<T,TargetHost>& bufcpu,  const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        queue.enqueueUnmapMemObject(buf.clType(), bufcpu.rawPtr(), events, event);
    }
};

#endif // VISIONCORE_HAVE_OPENCL

/// ***********************************************************************

/**
 * CUDA/Host Buffer 3D Creation.
 */
template<typename T, typename Target = TargetHost>
class Buffer3DManaged : public Buffer3DView<T,Target>
{
public:
    typedef Buffer3DView<T,Target> ViewT;    
    
    Buffer3DManaged() = delete;
    
    inline Buffer3DManaged(std::size_t w, std::size_t h, std::size_t d) : ViewT() 
    {
        ViewT::xsize = w;
        ViewT::ysize = h;
        ViewT::zsize = d;
        
        std::size_t line_pitch = 0;
        std::size_t plane_pitch = 0;
        typename Target::template PointerType<T> ptr = 0;
        
        Target::template AllocatePitchedMem<T>(&ptr, &line_pitch, &plane_pitch, ViewT::xsize, ViewT::ysize, ViewT::zsize);
        
        ViewT::memptr = ptr;
        ViewT::line_pitch = line_pitch;
        ViewT::plane_pitch = plane_pitch;
    }
    
    inline ~Buffer3DManaged()
    {
        Target::template DeallocatePitchedMem<T>(ViewT::memptr);
    }
    
    Buffer3DManaged(const Buffer3DManaged<T,Target>& img) = delete;
    
    inline Buffer3DManaged(Buffer3DManaged<T,Target>&& img) : ViewT(std::move(img))
    {
        
    }
    
    Buffer3DManaged<T,Target>& operator=(const Buffer3DManaged<T,Target>& img) = delete;
    
    inline Buffer3DManaged<T,Target>& operator=(Buffer3DManaged<T,Target>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

#ifdef VISIONCORE_HAVE_OPENCL

/**
 * OpenCL Buffer 3D Creation.
 */
template<typename T>
class Buffer3DManaged<T,TargetDeviceOpenCL> : public Buffer3DView<T,TargetDeviceOpenCL>
{
public:
    typedef TargetDeviceOpenCL Target;
    typedef Buffer3DView<T,Target> ViewT;
    
    Buffer3DManaged() = delete;
    
    inline Buffer3DManaged(std::size_t w, std::size_t h, std::size_t d, const cl::Context& context, cl_mem_flags flags, typename TargetHost::template PointerType<T> hostptr = nullptr) : ViewT()
    {        
        ViewT::memptr = new cl::Buffer(context, flags, d*w*h*sizeof(T), hostptr);
        ViewT::xsize = w;
        ViewT::ysize = h;
        ViewT::zsize = d;
        ViewT::line_pitch = w * sizeof(T);
        ViewT::plane_pitch = h * ViewT::line_pitch;
    }
    
    inline ~Buffer3DManaged()
    {
        if(ViewT::isValid())
        {
            cl::Buffer* clb = static_cast<cl::Buffer*>(ViewT::memptr);
            delete clb;
            ViewT::memptr = nullptr;
            ViewT::xsize = 0;
            ViewT::ysize = 0;
            ViewT::zsize = 0;
            ViewT::line_pitch = 0;
            ViewT::plane_pitch = 0;
        }
    }
    
    Buffer3DManaged(const Buffer3DManaged<T,Target>& img) = delete;
    
    inline Buffer3DManaged(Buffer3DManaged<T,Target>&& img) : ViewT(std::move(img))
    {
        
    }
    
    Buffer3DManaged<T,Target>& operator=(const Buffer3DManaged<T,Target>& img) = delete;
    
    inline Buffer3DManaged<T,Target>& operator=(Buffer3DManaged<T,Target>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

#endif // VISIONCORE_HAVE_OPENCL

}

#endif // VISIONCORE_BUFFER3D_HPP
