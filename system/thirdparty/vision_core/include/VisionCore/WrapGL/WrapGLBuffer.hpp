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
 * OpenGL Buffers.
 * ****************************************************************************
 */
#ifndef VISIONCORE_WRAPGL_BUFFER_HPP
#define VISIONCORE_WRAPGL_BUFFER_HPP

#include <array>
#include <vector>

#include <VisionCore/WrapGL/WrapGLCommon.hpp>

#include <VisionCore/Buffers/Buffer1D.hpp>
#include <VisionCore/Buffers/Buffer2D.hpp>
#include <VisionCore/Buffers/Buffer3D.hpp>

/**
 * TODO:
 * 
 * glTexBuffer / glTexBufferRange - texture as buffer?
 * 
 * 
 */

namespace vc
{

namespace wrapgl
{
       
class Buffer
{
public:
    typedef ScopeBinder<Buffer> Binder;
    
    enum class Type
    {
        eArray = GL_ARRAY_BUFFER, 
        eAtomicCounter = GL_ATOMIC_COUNTER_BUFFER, 
        eCopyRead = GL_COPY_READ_BUFFER, 
        eCopyWrite = GL_COPY_WRITE_BUFFER, 
        eDrawIndirect = GL_DRAW_INDIRECT_BUFFER, 
        eDispatchIndirect = GL_DISPATCH_INDIRECT_BUFFER, 
        eElementArray = GL_ELEMENT_ARRAY_BUFFER, 
        ePixelPack = GL_PIXEL_PACK_BUFFER, 
        ePixelUnpack = GL_PIXEL_UNPACK_BUFFER, 
        eQuery = GL_QUERY_BUFFER, 
        eShaderStorage = GL_SHADER_STORAGE_BUFFER, 
        eTexture = GL_TEXTURE_BUFFER, 
        eTransformFeedback = GL_TRANSFORM_FEEDBACK_BUFFER,
        eUniform = GL_UNIFORM_BUFFER,
        Invalid
    };
  
    inline Buffer();
    
    inline Buffer(Type bt, GLuint numel, GLenum dtype, GLuint cpe, 
                  GLenum gluse = GL_DYNAMIC_DRAW, const GLvoid* data = nullptr);
    
    inline ~Buffer();
    
    template<typename T, typename AllocatorT>
    inline void create(Type bt, const std::vector<T,AllocatorT>& vec, GLenum gluse = GL_DYNAMIC_DRAW);
    
    template<typename T>
    inline void create(Type bt, GLuint numel, GLenum gluse = GL_DYNAMIC_DRAW, const T* data = nullptr);
    
    template<typename T>
    inline void create(Type bt, const Buffer1DView<T,TargetHost>& buf, GLenum gluse = GL_DYNAMIC_DRAW);
    
    template<typename T>
    inline void create(Type bt, const Buffer2DView<T,TargetHost>& buf, GLenum gluse = GL_DYNAMIC_DRAW);
    
    template<typename T>
    inline void create(Type bt, const Buffer3DView<T,TargetHost>& buf, GLenum gluse = GL_DYNAMIC_DRAW);
    
    inline void create(Type bt, GLuint numel, GLenum dtype, GLuint cpe, 
                       GLenum gluse = GL_DYNAMIC_DRAW, 
                       const GLvoid* data = nullptr);
    
    inline void destroy();
    
    inline bool isValid() const;
    inline void resize(GLuint new_num_elements, GLenum gluse = GL_DYNAMIC_DRAW);
    
    inline void bind() const;
    inline void unbind() const;
    
    inline void upload(const GLvoid* data, GLsizeiptr size_bytes, GLintptr offset = 0);
    
    template<typename T, typename AllocatorT>
    inline void upload(const std::vector<T,AllocatorT>& vec, GLintptr offset = 0);
    
    template<typename T>
    inline void upload(const Buffer1DView<T,TargetHost>& buf, GLintptr offset = 0);
    
    template<typename T>
    inline void upload(const Buffer2DView<T,TargetHost>& buf, GLintptr offset = 0);
    
    template<typename T>
    inline void upload(const Buffer3DView<T,TargetHost>& buf, GLintptr offset = 0);
    
    inline void download(GLvoid* data, GLsizeiptr size_bytes, GLintptr offset = 0);
    
    template<typename T, typename AllocatorT>
    inline void download(std::vector<T,AllocatorT>& vec, GLintptr offset = 0);
    
    template<typename T>
    inline void download(Buffer1DView<T,TargetHost>& buf, GLintptr offset = 0);    
    
    template<typename T>
    inline void download(Buffer2DView<T,TargetHost>& buf, GLintptr offset = 0);
    
    template<typename T>
    inline void download(Buffer3DView<T,TargetHost>& buf, GLintptr offset = 0);
    
    inline GLuint id() const;
    inline Type   type() const;
    inline GLuint size() const;
    inline GLuint sizeBytes() const;
    inline GLenum dataType() const;
    inline GLuint countPerElement() const;
private:
    GLuint bufid;
    Type   buffer_type;
    GLuint num_elements;
    GLenum datatype;
    GLuint count_per_element;
    GLenum bufuse;
};
    
}

template<typename T, typename Target>
class Buffer1DFromOpenGL;
template<typename T, typename Target>
class Buffer2DFromOpenGL;
template<typename T, typename Target>
class Buffer3DFromOpenGL;

template<typename T>
class Buffer1DFromOpenGL<T,TargetHost> : public Buffer1DView<T,TargetHost>
{
public:
    typedef Buffer1DView<T,TargetHost> ViewT;
    
    inline Buffer1DFromOpenGL() = delete;
    
    inline Buffer1DFromOpenGL(wrapgl::Buffer& glbuf, GLenum acc = GL_READ_WRITE);
    inline ~Buffer1DFromOpenGL();
    
    inline Buffer1DFromOpenGL(const Buffer1DFromOpenGL<T,TargetHost>& img) = delete;
    inline Buffer1DFromOpenGL(Buffer1DFromOpenGL<T,TargetHost>&& img) : ViewT(std::move(img)), buf(std::move(img.buf)) { }
    inline Buffer1DFromOpenGL<T,TargetHost>& operator=(const Buffer1DFromOpenGL<T,TargetHost>& img) = delete;
    inline Buffer1DFromOpenGL<T,TargetHost>& operator=(Buffer1DFromOpenGL<T,TargetHost>&& img)
    {
        ViewT::operator=(std::move(img));
        buf = std::move(img.buf);
        return *this;
    }
    
    inline const ViewT& view() const { return static_cast<const ViewT&>(*this);  }
    inline ViewT& view() { return static_cast<ViewT&>(*this); }
  
private:
    wrapgl::Buffer& buf;
};

template<typename T>
class Buffer2DFromOpenGL<T,TargetHost> : public Buffer2DView<T,TargetHost>
{
public:
    typedef Buffer2DView<T,TargetHost> ViewT;
    
    inline Buffer2DFromOpenGL() = delete;
    
    inline Buffer2DFromOpenGL(wrapgl::Buffer& glbuf, std::size_t height, GLenum acc = GL_READ_WRITE);
    inline ~Buffer2DFromOpenGL();
    
    inline Buffer2DFromOpenGL(const Buffer2DFromOpenGL<T,TargetHost>& img) = delete;
    inline Buffer2DFromOpenGL(Buffer2DFromOpenGL<T,TargetHost>&& img) : ViewT(std::move(img)), buf(std::move(img.buf)) { }
    inline Buffer2DFromOpenGL<T,TargetHost>& operator=(const Buffer2DFromOpenGL<T,TargetHost>& img) = delete;
    inline Buffer2DFromOpenGL<T,TargetHost>& operator=(Buffer2DFromOpenGL<T,TargetHost>&& img)
    {
        ViewT::operator=(std::move(img));
        buf = std::move(img.buf);
        return *this;
    }
    
    inline const ViewT& view() const { return static_cast<const ViewT&>(*this);  }
    inline ViewT& view() { return static_cast<ViewT&>(*this); }
    
private:
    wrapgl::Buffer& buf;
};

template<typename T>
class Buffer3DFromOpenGL<T,TargetHost> : public Buffer3DView<T,TargetHost>
{
public:
    typedef Buffer3DView<T,TargetHost> ViewT;
    
    inline Buffer3DFromOpenGL() = delete;
    
    inline Buffer3DFromOpenGL(wrapgl::Buffer& glbuf, std::size_t height, std::size_t depth, GLenum acc = GL_READ_WRITE);
    inline ~Buffer3DFromOpenGL();
    
    inline Buffer3DFromOpenGL(const Buffer3DFromOpenGL<T,TargetHost>& img) = delete;
    inline Buffer3DFromOpenGL(Buffer3DFromOpenGL<T,TargetHost>&& img) : ViewT(std::move(img)), buf(std::move(img.buf)) { }
    inline Buffer3DFromOpenGL<T,TargetHost>& operator=(const Buffer3DFromOpenGL<T,TargetHost>& img) = delete;
    inline Buffer3DFromOpenGL<T,TargetHost>& operator=(Buffer3DFromOpenGL<T,TargetHost>&& img)
    {
        ViewT::operator=(std::move(img));
        buf = std::move(img.buf);
        return *this;
    }
    
    inline const ViewT& view() const { return static_cast<const ViewT&>(*this);  }
    inline ViewT& view() { return static_cast<ViewT&>(*this); }
    
private:
    wrapgl::Buffer& buf;
};

}

#ifdef VISIONCORE_HAVE_CUDA

#include <cuda_runtime_api.h>

#include <VisionCore/CUDAException.hpp>

namespace vc
{
 
// some functions wrapper here as cuda_gl_interop header leaks horrible stuff
namespace internal
{
cudaGraphicsResource* registerOpenGLBuffer(GLuint id, unsigned int flags = cudaGraphicsMapFlagsNone);
}
    
template<typename T>
class Buffer1DFromOpenGL<T,TargetDeviceCUDA> : public Buffer1DView<T,TargetDeviceCUDA>
{
public:
    typedef Buffer1DView<T,TargetDeviceCUDA> ViewT;
    
    inline Buffer1DFromOpenGL() = delete;
    
    inline Buffer1DFromOpenGL(wrapgl::Buffer& glbuf);
    inline ~Buffer1DFromOpenGL();
    
    inline Buffer1DFromOpenGL(const Buffer1DFromOpenGL<T,TargetDeviceCUDA>& img) = delete;
    inline Buffer1DFromOpenGL(Buffer1DFromOpenGL<T,TargetDeviceCUDA>&& img)
        : ViewT(std::move(img)), cuda_res(img.cuda_res)
    {
        img.cuda_res = 0;
    }
    inline Buffer1DFromOpenGL<T,TargetDeviceCUDA>& operator=(const Buffer1DFromOpenGL<T,TargetDeviceCUDA>& img) = delete;
    inline Buffer1DFromOpenGL<T,TargetDeviceCUDA>& operator=(Buffer1DFromOpenGL<T,TargetDeviceCUDA>&& img)
    {
        ViewT::operator=(std::move(img));
        cuda_res = img.cuda_res;
        img.cuda_res = 0;
        return *this;
    }
    
    inline const ViewT& view() const { return static_cast<const ViewT&>(*this);  }
    inline ViewT& view() { return static_cast<ViewT&>(*this); }
    
private:
    cudaGraphicsResource* cuda_res;
};

template<typename T>
class Buffer2DFromOpenGL<T,TargetDeviceCUDA> : public Buffer2DView<T,TargetDeviceCUDA>
{
public:
    typedef Buffer2DView<T,TargetDeviceCUDA> ViewT;
    
    inline Buffer2DFromOpenGL() = delete;
    
    inline Buffer2DFromOpenGL(wrapgl::Buffer& glbuf, std::size_t height);
    inline ~Buffer2DFromOpenGL();
    
    inline Buffer2DFromOpenGL(const Buffer2DFromOpenGL<T,TargetDeviceCUDA>& img) = delete;
    inline Buffer2DFromOpenGL(Buffer2DFromOpenGL<T,TargetDeviceCUDA>&& img)
        : ViewT(std::move(img)), cuda_res(img.cuda_res)
    {
        img.cuda_res = 0;
    }
    inline Buffer2DFromOpenGL<T,TargetDeviceCUDA>& operator=(const Buffer2DFromOpenGL<T,TargetDeviceCUDA>& img) = delete;
    inline Buffer2DFromOpenGL<T,TargetDeviceCUDA>& operator=(Buffer2DFromOpenGL<T,TargetDeviceCUDA>&& img)
    {
        ViewT::operator=(std::move(img));
        cuda_res = img.cuda_res;
        img.cuda_res = 0;
        return *this;
    }
    
    inline const ViewT& view() const { return static_cast<const ViewT&>(*this);  }
    inline ViewT& view() { return static_cast<ViewT&>(*this); }
    
private:
    cudaGraphicsResource* cuda_res;
};

template<typename T>
class Buffer3DFromOpenGL<T,TargetDeviceCUDA> : public Buffer3DView<T,TargetDeviceCUDA>
{
public:
    typedef Buffer3DView<T,TargetDeviceCUDA> ViewT;
    
    inline Buffer3DFromOpenGL() = delete;
    
    inline Buffer3DFromOpenGL(wrapgl::Buffer& glbuf, std::size_t height, std::size_t depth);
    inline ~Buffer3DFromOpenGL();
    
    inline Buffer3DFromOpenGL(const Buffer3DFromOpenGL<T,TargetDeviceCUDA>& img) = delete;
    inline Buffer3DFromOpenGL(Buffer3DFromOpenGL<T,TargetDeviceCUDA>&& img)
        : ViewT(std::move(img)), cuda_res(img.cuda_res)
    {
        img.cuda_res = 0;
    }
    inline Buffer3DFromOpenGL<T,TargetDeviceCUDA>& operator=(const Buffer3DFromOpenGL<T,TargetDeviceCUDA>& img) = delete;
    inline Buffer3DFromOpenGL<T,TargetDeviceCUDA>& operator=(Buffer3DFromOpenGL<T,TargetDeviceCUDA>&& img)
    {
        ViewT::operator=(std::move(img));
        cuda_res = img.cuda_res;
        img.cuda_res = 0;
        return *this;
    }
    
    inline const ViewT& view() const { return static_cast<const ViewT&>(*this);  }
    inline ViewT& view() { return static_cast<ViewT&>(*this); }
    
private:
    cudaGraphicsResource* cuda_res;
};
    
}
#endif // VISIONCORE_HAVE_CUDA

#ifdef VISIONCORE_HAVE_OPENCL
namespace vc
{
    
template<typename T>
class Buffer1DFromOpenGL<T,TargetDeviceOpenCL> : public Buffer1DView<T,TargetDeviceOpenCL>
{
public:
    typedef Buffer1DView<T,TargetDeviceOpenCL> ViewT;
    
    inline Buffer1DFromOpenGL() = delete;
    
    inline Buffer1DFromOpenGL(const cl::Context& context, cl_mem_flags flags, wrapgl::Buffer& glbuf);
    inline ~Buffer1DFromOpenGL();
    
    inline Buffer1DFromOpenGL(const Buffer1DFromOpenGL<T,TargetDeviceOpenCL>& img) = delete;
    inline Buffer1DFromOpenGL(Buffer1DFromOpenGL<T,TargetDeviceOpenCL>&& img) : ViewT(std::move(img)) { }
    inline Buffer1DFromOpenGL<T,TargetDeviceOpenCL>& operator=(const Buffer1DFromOpenGL<T,TargetDeviceOpenCL>& img) = delete;
    inline Buffer1DFromOpenGL<T,TargetDeviceOpenCL>& operator=(Buffer1DFromOpenGL<T,TargetDeviceOpenCL>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline const ViewT& view() const { return static_cast<const ViewT&>(*this);  }
    inline ViewT& view() { return static_cast<ViewT&>(*this); }
};

template<typename T>
class Buffer2DFromOpenGL<T,TargetDeviceOpenCL> : public Buffer2DView<T,TargetDeviceOpenCL>
{
public:
    typedef Buffer2DView<T,TargetDeviceOpenCL> ViewT;
    
    inline Buffer2DFromOpenGL() = delete;
    
    inline Buffer2DFromOpenGL(const cl::Context& context, cl_mem_flags flags, wrapgl::Buffer& glbuf, std::size_t height);
    inline ~Buffer2DFromOpenGL();
    
    inline Buffer2DFromOpenGL(const Buffer2DFromOpenGL<T,TargetDeviceOpenCL>& img) = delete;
    inline Buffer2DFromOpenGL(Buffer2DFromOpenGL<T,TargetDeviceOpenCL>&& img) : ViewT(std::move(img)) { }
    inline Buffer2DFromOpenGL<T,TargetDeviceOpenCL>& operator=(const Buffer2DFromOpenGL<T,TargetDeviceOpenCL>& img) = delete;
    inline Buffer2DFromOpenGL<T,TargetDeviceOpenCL>& operator=(Buffer2DFromOpenGL<T,TargetDeviceOpenCL>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline const ViewT& view() const { return static_cast<const ViewT&>(*this);  }
    inline ViewT& view() { return static_cast<ViewT&>(*this); }
};

template<typename T>
class Buffer3DFromOpenGL<T,TargetDeviceOpenCL> : public Buffer3DView<T,TargetDeviceOpenCL>
{
public:
    typedef Buffer3DView<T,TargetDeviceOpenCL> ViewT;
    
    inline Buffer3DFromOpenGL() = delete;
    
    inline Buffer3DFromOpenGL(const cl::Context& context, cl_mem_flags flags, 
                              wrapgl::Buffer& glbuf, std::size_t height, std::size_t depth);
    inline ~Buffer3DFromOpenGL();
    
    inline Buffer3DFromOpenGL(const Buffer3DFromOpenGL<T,TargetDeviceOpenCL>& img) = delete;
    inline Buffer3DFromOpenGL(Buffer3DFromOpenGL<T,TargetDeviceOpenCL>&& img) : ViewT(std::move(img)) { }
    inline Buffer3DFromOpenGL<T,TargetDeviceOpenCL>& operator=(const Buffer3DFromOpenGL<T,TargetDeviceOpenCL>& img) = delete;
    inline Buffer3DFromOpenGL<T,TargetDeviceOpenCL>& operator=(Buffer3DFromOpenGL<T,TargetDeviceOpenCL>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline const ViewT& view() const { return static_cast<const ViewT&>(*this);  }
    inline ViewT& view() { return static_cast<ViewT&>(*this); }
};  
    
}
#endif // VISIONCORE_HAVE_OPENCL

#include <VisionCore/WrapGL/impl/WrapGLBuffer_impl.hpp>

#endif // VISIONCORE_WRAPGL_BUFFER_HPP
