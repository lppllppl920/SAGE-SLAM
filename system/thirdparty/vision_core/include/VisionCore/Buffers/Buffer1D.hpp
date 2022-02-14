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
 * Linear Host/Device Buffer.
 * ****************************************************************************
 */

#ifndef VISIONCORE_BUFFER1D_HPP
#define VISIONCORE_BUFFER1D_HPP

#include <VisionCore/Platform.hpp>
#include <VisionCore/MemoryPolicy.hpp>
#include <vector>

namespace vc
{

/**
 * View on a 1D Buffer - Base.
 */    
template<typename T, typename Target>
class Buffer1DViewBase
{
public:
    typedef T ValueType;
    typedef typename vc::TypeAlignmentTraits<T,Target>::ResultT RealValueType;
    typedef Target TargetType;
    typedef typename TargetType::template PointerType<RealValueType> PointerType;
    
#ifndef VISIONCORE_CUDA_KERNEL_SPACE
    EIGEN_DEVICE_FUNC inline Buffer1DViewBase() : memptr(0) { }
#else // VISIONCORE_CUDA_KERNEL_SPACE
    EIGEN_DEVICE_FUNC inline Buffer1DViewBase() { }
#endif // VISIONCORE_CUDA_KERNEL_SPACE
    
    EIGEN_DEVICE_FUNC inline ~Buffer1DViewBase()
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline Buffer1DViewBase( const Buffer1DViewBase<ValueType,TargetType>& img ) : 
        memptr(img.memptr), xsize(img.xsize)
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline Buffer1DViewBase( Buffer1DViewBase<ValueType,TargetType>&& img ) : 
        memptr(img.memptr), xsize(img.xsize)
    {
        // This object will take over managing data (if Management = Manage)
        img.memptr = 0;
    }
    
    EIGEN_DEVICE_FUNC inline Buffer1DViewBase(PointerType optr, std::size_t s) :
        memptr(optr), xsize(s)
    {  
        
    }
        
    EIGEN_DEVICE_FUNC inline Buffer1DViewBase<ValueType,TargetType>& operator=(const Buffer1DViewBase<ValueType,TargetType>& other)
    {
        memptr = (void*)other.rawPtr();
        xsize = other.size();
        
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline Buffer1DViewBase<ValueType,TargetType>& operator=(Buffer1DViewBase<ValueType,TargetType>&& img)
    {
        memptr = img.memptr;
        xsize = img.xsize;
        img.memptr = 0;
        
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline std::size_t size() const { return xsize; }
    EIGEN_DEVICE_FUNC inline std::size_t bytes() const { return xsize * sizeof(RealValueType); }
    
    inline void swap(Buffer1DViewBase<ValueType,TargetType>& img)
    {
        std::swap(img.memptr, memptr);
        std::swap(img.xsize, xsize);
    }
    
    EIGEN_DEVICE_FUNC inline bool isValid() const { return rawPtr() != nullptr; }
    
    EIGEN_DEVICE_FUNC inline const PointerType rawPtr() const { return memptr; }
    EIGEN_DEVICE_FUNC inline PointerType rawPtr() { return memptr; }
    
    EIGEN_DEVICE_FUNC inline bool inBounds(std::size_t x) const
    {
        return x < xsize;
    }
    
    EIGEN_DEVICE_FUNC inline bool inBounds(int x) const
    {
        return 0 <= x && x < xsize;
    }
    
    EIGEN_DEVICE_FUNC inline std::size_t indexClamped(int x) const { return clamp(x, 0, (int)xsize-1); }
    EIGEN_DEVICE_FUNC inline std::size_t indexCircular(int x) const 
    { 
        if(x < 0) 
        { 
            return x + xsize; 
        } 
        else if(x >= xsize) 
        { 
            return x - xsize;           
        } 
        else 
        { 
            return x; 
        }
    }
    
    EIGEN_DEVICE_FUNC inline std::size_t indexReflected(int x) const 
    { 
        if(x < 0) 
        { 
            return -x-1; 
        } 
        else if(x >= xsize) 
        {   
            return 2 * xsize - x - 1; 
        } 
        else 
        { 
            return x; 
        }       
    }
protected:
    PointerType memptr;
    std::size_t xsize;
};

/**
 * Buffer 1D View - Add contents access methods.
 */
template<typename T, typename Target>
class Buffer1DViewAccessible : public Buffer1DViewBase<T,Target>
{
public:
    typedef Buffer1DViewBase<T,Target> BaseT;
    typedef typename BaseT::RealValueType RealValueType;
    typedef typename BaseT::ValueType ValueType;
    typedef typename BaseT::PointerType PointerType;
    typedef typename BaseT::TargetType TargetType;
    
    EIGEN_DEVICE_FUNC inline Buffer1DViewAccessible() : BaseT() { }
    
    EIGEN_DEVICE_FUNC inline ~Buffer1DViewAccessible() { }
    
    EIGEN_DEVICE_FUNC inline Buffer1DViewAccessible(const Buffer1DViewAccessible<ValueType,TargetType>& img) : BaseT(img) { }
    
    EIGEN_DEVICE_FUNC inline Buffer1DViewAccessible(Buffer1DViewAccessible<ValueType,TargetType>&& img) : BaseT(std::move(img)) { }
    
    EIGEN_DEVICE_FUNC inline Buffer1DViewAccessible(PointerType optr, std::size_t s) : BaseT(optr, s) { }
    
    EIGEN_DEVICE_FUNC inline Buffer1DViewAccessible<ValueType,TargetType>& operator=(const Buffer1DViewAccessible<ValueType,TargetType>& img)
    {
        BaseT::operator=(img);
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline Buffer1DViewAccessible<ValueType,TargetType>& operator=(Buffer1DViewAccessible<ValueType,TargetType>&& img)
    {
        BaseT::operator=(std::move(img));
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline ValueType* ptr()
    {
        return (ValueType*)BaseT::rawPtr();
    }
    
    EIGEN_DEVICE_FUNC inline const ValueType* ptr() const
    {
        return (const ValueType*)BaseT::rawPtr();
    }
        
    EIGEN_DEVICE_FUNC inline ValueType* ptr(std::size_t idx)
    {
        return (ValueType*)(ptr() + idx);
    }
    
    EIGEN_DEVICE_FUNC inline const ValueType* ptr(std::size_t idx) const
    {
        return (const ValueType*)(ptr() + idx);
    }
    
    EIGEN_DEVICE_FUNC inline ValueType& operator()(std::size_t s)
    {
        return ptr()[s];
    }
    
    EIGEN_DEVICE_FUNC inline const ValueType& operator()(std::size_t s) const
    {
        return ptr()[s];
    }
    
    EIGEN_DEVICE_FUNC inline ValueType& operator[](std::size_t ix)
    {
        //return static_cast<ValueType*>(BaseT::rawPtr())[ix];
        return operator()(ix);
    }
    
    EIGEN_DEVICE_FUNC inline const ValueType& operator[](std::size_t ix) const
    {
        //return static_cast<ValueType*>(BaseT::rawPtr())[ix];
        return operator()(ix);
    }
    
    EIGEN_DEVICE_FUNC inline const ValueType& get(std::size_t ix) const
    {
        return operator()(ix);
    }
    
    EIGEN_DEVICE_FUNC inline const ValueType& getWithClampedRange(int x) const
    {
        return operator()(BaseT::indexClamped(x));
    }
    
    EIGEN_DEVICE_FUNC inline const ValueType& getWithCircularRange(int x) const
    {
        return operator()(BaseT::indexCircular(x));
    }
    
    EIGEN_DEVICE_FUNC inline const ValueType& getWithReflectedRange(int x) const
    {
        return operator()(BaseT::indexReflected(x));
    }
};

/**
 * View on a 1D Buffer.
 */    
template<typename T, typename Target = TargetHost>
class Buffer1DView
{
    
};

/**
 * View on a 1D Buffer Specialization - Host.
 */ 
template<typename T>
class Buffer1DView<T,TargetHost> : public Buffer1DViewAccessible<T,TargetHost>
{
public:
    typedef Buffer1DViewAccessible<T,TargetHost> BaseT;
    typedef typename BaseT::RealValueType RealValueType;
    typedef typename BaseT::ValueType ValueType;
    typedef typename BaseT::PointerType PointerType;
    typedef typename BaseT::TargetType TargetType;
    
    inline Buffer1DView() : BaseT() { }
    
    inline ~Buffer1DView() { }
    
    inline Buffer1DView(const Buffer1DView<ValueType,TargetType>& img) : BaseT(img) { }
    
    inline Buffer1DView(Buffer1DView<ValueType,TargetType>&& img) : BaseT(std::move(img)) { }
    
    inline Buffer1DView(PointerType optr, std::size_t s) : BaseT(optr, s) { }
    
    template<typename AllocT>
    inline Buffer1DView(std::vector<ValueType,AllocT>& vec) : BaseT(vec.data(), vec.size()) { }
    
    inline Buffer1DView<ValueType,TargetType>& operator=(const Buffer1DView<ValueType,TargetType>& img)
    {
        BaseT::operator=(img);
        return *this;
    }
    
    inline Buffer1DView<ValueType,TargetType>& operator=(Buffer1DView<ValueType,TargetType>&& img)
    {
        BaseT::operator=(std::move(img));
        return *this;
    }

    inline void copyFrom(const Buffer1DView<ValueType,TargetHost>& img)
    {
        typedef TargetHost TargetFrom;
        TargetTransfer<TargetFrom,TargetType>::template memcpy<ValueType>(BaseT::rawPtr(), 
                                                                          (PointerType)img.ptr(), 
                                                                          std::min(img.bytes(),BaseT::bytes()));
    }
    
#ifdef VISIONCORE_HAVE_CUDA
    inline void copyFrom(const Buffer1DView<ValueType,TargetDeviceCUDA>& img)
    {
        typedef TargetDeviceCUDA TargetFrom;
        TargetTransfer<TargetFrom,TargetType>::template memcpy<ValueType>(BaseT::rawPtr(), 
                                                                          (PointerType)img.ptr(), std::min(img.bytes(),BaseT::bytes()));
    }
#endif // VISIONCORE_HAVE_CUDA
    
#ifdef VISIONCORE_HAVE_OPENCL
    inline void copyFrom(const cl::CommandQueue& queue, 
                         const Buffer1DView<ValueType,TargetDeviceOpenCL>& img, 
                         const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        queue.enqueueReadBuffer(img.clType(), true, 0, std::min(BaseT::bytes(), img.bytes()), BaseT::rawPtr(), events, event);
    }
#endif // VISIONCORE_HAVE_OPENCL
    
    inline void memset(unsigned char v = 0)
    {
        TargetType::template memset<T>(BaseT::rawPtr(), v, BaseT::bytes());
    }
    
    inline ValueType* begin() { return BaseT::ptr(); }
    inline const ValueType* begin() const { return BaseT::ptr(); }
    inline ValueType* end() { return (BaseT::ptr() + BaseT::size()); }
    inline const ValueType* end() const { return (BaseT::ptr() + BaseT::size()); }
};

#ifdef VISIONCORE_HAVE_CUDA

/**
 * View on a 1D Buffer Specialization - CUDA.
 */ 
template<typename T>
class Buffer1DView<T,TargetDeviceCUDA> : public Buffer1DViewAccessible<T,TargetDeviceCUDA>
{
public:
    typedef Buffer1DViewAccessible<T,TargetDeviceCUDA> BaseT;
    typedef typename BaseT::RealValueType RealValueType;
    typedef typename BaseT::ValueType ValueType;
    typedef typename BaseT::PointerType PointerType;
    typedef typename BaseT::TargetType TargetType;
    
    EIGEN_DEVICE_FUNC inline Buffer1DView() : BaseT() { }
    
    EIGEN_DEVICE_FUNC inline ~Buffer1DView() { }
    
    EIGEN_DEVICE_FUNC inline Buffer1DView(const Buffer1DView<ValueType,TargetType>& img) : BaseT(img) { }
    
    EIGEN_DEVICE_FUNC inline Buffer1DView(Buffer1DView<ValueType,TargetType>&& img) : BaseT(std::move(img)) { }
    
    EIGEN_DEVICE_FUNC inline Buffer1DView(PointerType optr, std::size_t s) : BaseT(optr, s) { }
    
    EIGEN_DEVICE_FUNC inline Buffer1DView<ValueType,TargetType>& operator=(const Buffer1DView<ValueType,TargetType>& img)
    {
        BaseT::operator=(img);
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline Buffer1DView<ValueType,TargetType>& operator=(Buffer1DView<ValueType,TargetType>&& img)
    {
        BaseT::operator=(std::move(img));
        return *this;
    }
    
    template<typename TargetFrom>
    inline void copyFrom(const Buffer1DView<ValueType,TargetFrom>& img)
    {
#ifdef VISIONCORE_HAVE_OPENCL
        static_assert(std::is_same<TargetFrom,TargetDeviceOpenCL>::value != true, "Not possible to do OpenCL-CUDA copy");
#endif
        
        TargetTransfer<TargetFrom,TargetType>::template memcpy<ValueType>(BaseT::rawPtr(), 
                                                                  (PointerType)img.ptr(), std::min(img.bytes(),BaseT::bytes()));
    }
    
    inline void memset(unsigned char v = 0)
    {
        TargetType::template memset<ValueType>(BaseT::rawPtr(), v, BaseT::bytes());
    }

    EIGEN_DEVICE_FUNC inline typename internal::ThrustType<ValueType,TargetType>::Ptr begin() 
    {
        return (typename internal::ThrustType<ValueType,TargetType>::Ptr)(BaseT::ptr());
    }
    
    EIGEN_DEVICE_FUNC inline typename internal::ThrustType<ValueType,TargetType>::Ptr begin() const
    {
        return (typename internal::ThrustType<ValueType,TargetType>::Ptr)(const_cast<ValueType*>(BaseT::ptr()));
    }

    EIGEN_DEVICE_FUNC inline typename internal::ThrustType<ValueType,TargetType>::Ptr end() 
    {
        return (typename internal::ThrustType<ValueType,TargetType>::Ptr)( BaseT::ptr() + BaseT::size() );
    }
    
    EIGEN_DEVICE_FUNC inline typename internal::ThrustType<ValueType,TargetType>::Ptr end() const
    {
        return (typename internal::ThrustType<ValueType,TargetType>::Ptr)( const_cast<ValueType*>(BaseT::ptr() + BaseT::size()) );
    }

    inline void fill(ValueType val) 
    {
        thrust::fill(begin(), end(), val);
    }
};

#endif // VISIONCORE_HAVE_CUDA

#ifdef VISIONCORE_HAVE_OPENCL

/**
 * View on a 1D Buffer Specialization - OpenCL.
 */ 
template<typename T>
class Buffer1DView<T,TargetDeviceOpenCL> : public Buffer1DViewBase<T,TargetDeviceOpenCL>
{
public:
    typedef Buffer1DViewBase<T,TargetDeviceOpenCL> BaseT;
    typedef typename BaseT::RealValueType RealValueType;
    typedef typename BaseT::ValueType ValueType;
    typedef typename BaseT::PointerType PointerType;
    typedef typename BaseT::TargetType TargetType;
    
    inline Buffer1DView() : BaseT() { }
    
    inline ~Buffer1DView() { }
    
    inline Buffer1DView(const Buffer1DView<ValueType,TargetType>& img) : BaseT(img) { }
    
    inline Buffer1DView(Buffer1DView<ValueType,TargetType>&& img) : BaseT(std::move(img)) { }
    
    inline Buffer1DView(PointerType optr, std::size_t s) : BaseT(optr, s) { }
    
    inline Buffer1DView<ValueType,TargetType>& operator=(const Buffer1DView<ValueType,TargetType>& img)
    {
        BaseT::operator=(img);
        return *this;
    }
    
    inline Buffer1DView<ValueType,TargetType>& operator=(Buffer1DView<ValueType,TargetType>&& img)
    {
        BaseT::operator=(std::move(img));
        return *this;
    }
    
    inline const cl::Buffer& clType() const { return *static_cast<const cl::Buffer*>(BaseT::rawPtr()); }
    inline cl::Buffer& clType() { return *static_cast<cl::Buffer*>(BaseT::rawPtr()); }
    
    inline void copyFrom(const cl::CommandQueue& queue, const Buffer1DView<ValueType,TargetHost>& img, 
                         const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        queue.enqueueWriteBuffer(clType(), true, 0, std::min(BaseT::bytes(), img.bytes()), img.rawPtr(), events, event);
    }
    
    inline void copyFrom(const cl::CommandQueue& queue, const Buffer1DView<ValueType,TargetDeviceOpenCL>& img, 
                         const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        queue.enqueueCopyBuffer(img.clType(), clType(), 0, 0, std::min(BaseT::bytes(), img.bytes()), events, event);
    }
    
    inline void memset(const cl::CommandQueue& queue, T v, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        queue.enqueueFillBuffer(clType(), v, 0, BaseT::bytes(), events, event);
    }
};

/**
 * OpenCL Buffer to Host Mapping.
 */
struct Buffer1DMapper
{
    template<typename T>
    static inline Buffer1DView<T,TargetHost> map(const cl::CommandQueue& queue, cl_map_flags flags, 
                                                 const Buffer1DView<T,TargetDeviceOpenCL>& buf, 
                                                 const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        typedef typename Buffer1DView<T,TargetHost>::PointerType PointerType;
        PointerType ptr = queue.enqueueMapBuffer(buf.clType(), true, flags, 0, buf.bytes(), events, event);
        return Buffer1DView<T,TargetHost>(ptr, buf.size());
    }
    
    template<typename T>
    static inline void unmap(const cl::CommandQueue& queue, const Buffer1DView<T,TargetDeviceOpenCL>& buf, 
                             const Buffer1DView<T,TargetHost>& bufcpu,  
                             const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        queue.enqueueUnmapMemObject(buf.clType(), bufcpu.rawPtr(), events, event);
    }
};

#endif // VISIONCORE_HAVE_OPENCL

/// ***********************************************************************

/**
 * Buffer 1D Creation.
 */
template<typename T, typename Target = TargetHost>
class Buffer1DManaged { };

/**
 * Buffer 1D Creation - Host Specialization.
 */
template<typename T>
class Buffer1DManaged<T,TargetHost> : public Buffer1DView<T,TargetHost>
{
public:
    typedef TargetHost Target;
    typedef Buffer1DView<T,Target> ViewT;
    typedef typename ViewT::RealValueType RealValueType;
    typedef typename ViewT::ValueType ValueType;
    typedef typename ViewT::PointerType PointerType;
    typedef typename ViewT::TargetType TargetType;
    
    Buffer1DManaged() = delete;
    
    Buffer1DManaged(std::size_t s) : ViewT()
    {
        ViewT::memptr = 0;
        ViewT::xsize = s;
        PointerType ptr = 0;
        Target::template AllocateMem<RealValueType>(&ptr, ViewT::xsize);
        ViewT::memptr = ptr;
    }
    
    ~Buffer1DManaged()
    {
        if(ViewT::memptr != 0)
        {
            Target::template DeallocatePitchedMem<RealValueType>(ViewT::memptr);
        }
    }
    
    Buffer1DManaged(const Buffer1DManaged<ValueType,Target>& img) = delete;
    
    inline Buffer1DManaged(Buffer1DManaged<ValueType,Target>&& img) : ViewT(std::move(img))
    {
        
    }
    
    Buffer1DManaged<ValueType,Target>& operator=(const Buffer1DManaged<ValueType,Target>& img) = delete;
    
    inline Buffer1DManaged<ValueType,Target>& operator=(Buffer1DManaged<ValueType,Target>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    void resize(std::size_t new_s)
    {
        if(ViewT::memptr != 0)
        {
            Target::template DeallocatePitchedMem<RealValueType>(ViewT::memptr);
            ViewT::xsize = 0;
            ViewT::memptr = 0;
        }
        
        ViewT::memptr = 0;
        ViewT::xsize = new_s;
        PointerType ptr = 0;
        Target::template AllocateMem<RealValueType>(&ptr, ViewT::xsize);
        ViewT::memptr = ptr;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

#ifdef VISIONCORE_HAVE_CUDA

/**
 * Buffer 1D Creation - CUDA Specialization.
 */
template<typename T>
class Buffer1DManaged<T,TargetDeviceCUDA> : public Buffer1DView<T,TargetDeviceCUDA>
{
public:
    typedef TargetDeviceCUDA Target;
    typedef Buffer1DView<T,Target> ViewT;
    typedef typename ViewT::RealValueType RealValueType;
    typedef typename ViewT::ValueType ValueType;
    typedef typename ViewT::PointerType PointerType;
    typedef typename ViewT::TargetType TargetType;
    
    Buffer1DManaged() = delete;
    
    Buffer1DManaged(std::size_t s) : ViewT()
    {
        ViewT::memptr = 0;
        ViewT::xsize = s;
        PointerType ptr = 0;
        Target::template AllocateMem<RealValueType>(&ptr, ViewT::xsize);
        ViewT::memptr = ptr;
    }
    
    ~Buffer1DManaged()
    {
        if(ViewT::memptr != 0)
        {
            Target::template DeallocatePitchedMem<RealValueType>(ViewT::memptr);
        }
    }
    
    Buffer1DManaged(const Buffer1DManaged<ValueType,Target>& img) = delete;
    
    inline Buffer1DManaged(Buffer1DManaged<ValueType,Target>&& img) : ViewT(std::move(img))
    {
        
    }
    
    Buffer1DManaged<ValueType,Target>& operator=(const Buffer1DManaged<ValueType,Target>& img) = delete;
    
    inline Buffer1DManaged<ValueType,Target>& operator=(Buffer1DManaged<ValueType,Target>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    void resize(std::size_t new_s)
    {
        if(ViewT::memptr != 0)
        {
            Target::template DeallocatePitchedMem<RealValueType>(ViewT::memptr);
            ViewT::xsize = 0;
            ViewT::memptr = 0;
        }
        
        ViewT::memptr = 0;
        ViewT::xsize = new_s;
        PointerType ptr = 0;
        Target::template AllocateMem<RealValueType>(&ptr, ViewT::xsize);
        ViewT::memptr = ptr;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

#endif // VISIONCORE_HAVE_CUDA

#ifdef VISIONCORE_HAVE_OPENCL

/**
 * OpenCL Buffer 1D Creation.
 */
template<typename T>
class Buffer1DManaged<T,TargetDeviceOpenCL> : public Buffer1DView<T,TargetDeviceOpenCL>
{
public:
    typedef TargetDeviceOpenCL Target;
    typedef Buffer1DView<T,Target> ViewT;
    typedef typename ViewT::RealValueType RealValueType;
    typedef typename ViewT::ValueType ValueType;
    typedef typename ViewT::PointerType PointerType;
    typedef typename ViewT::TargetType TargetType;
    
    Buffer1DManaged() = delete;
    
    Buffer1DManaged(std::size_t s, const cl::Context& context, cl_mem_flags flags, PointerType hostptr = nullptr) : ViewT()
    {
        ViewT::memptr = new cl::Buffer(context, flags, s * sizeof(RealValueType), hostptr);
        ViewT::xsize = s;
    }
    
    ~Buffer1DManaged()
    {
        if(ViewT::isValid())
        {
            cl::Buffer* clb = static_cast<cl::Buffer*>(ViewT::memptr);
            delete clb;
            ViewT::memptr = nullptr;
            ViewT::xsize = 0;
        }
    }
    
    Buffer1DManaged(const Buffer1DManaged<ValueType,Target>& img) = delete;
    
    inline Buffer1DManaged(Buffer1DManaged<ValueType,Target>&& img) : ViewT(std::move(img))
    {
        
    }
    
    Buffer1DManaged<ValueType,Target>& operator=(const Buffer1DManaged<ValueType,Target>& img) = delete;
    
    inline Buffer1DManaged<ValueType,Target>& operator=(Buffer1DManaged<ValueType,Target>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

#endif // VISIONCORE_HAVE_OPENCL

}

#endif // VISIONCORE_BUFFER1D_HPP
