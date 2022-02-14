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
 * Texture.
 * ****************************************************************************
 */

#ifndef VISIONCORE_WRAPGL_TEXTURE_HPP
#define VISIONCORE_WRAPGL_TEXTURE_HPP

#include <VisionCore/WrapGL/WrapGLCommon.hpp>

#include <VisionCore/Buffers/Buffer1D.hpp>
#include <VisionCore/Buffers/Buffer2D.hpp>
#include <VisionCore/Buffers/Buffer3D.hpp>
#include <VisionCore/Buffers/Image2D.hpp>

/**
 * TODO:
 * 
 * This is only 2D. There can be also: glTexImage1D / glTexImage3D / GL_TEXTURE_CUBE_MAP etc
 * 
 * Add Pyramid-MultiLevel support.
 * 
 * More TexParameters (get/set)
 * 
 * Texture with Buffer data - GL_TEXTURE_BUFFER
 * 
 * glTexStorage2D  glTextureStorage2D ??
 * 
 * glTextureView — initialize a texture as a data alias of another texture's data store
 * 
 * glGenerateTextureMipmap
 * 
 * Image: glBindImageTexture​.
 */

namespace vc
{

namespace wrapgl
{
    
class TextureBase
{
public:
    inline TextureBase();
    inline ~TextureBase();
    
    inline void create(GLenum int_format = GL_RGBA);
    inline void destroy();
    inline bool isValid() const;
    
    inline GLuint id() const;
    
    inline GLenum intFormat() const;
    
    inline static void bind(const GLenum unit);
protected:
    GLuint texid;
    GLenum internal_format;
};

class Texture2DBase : public TextureBase
{
public:
    typedef ScopeBinder<Texture2DBase> Binder;
    
    inline Texture2DBase();
    
    template<typename T>
    inline void upload(const Buffer2DView<T,TargetHost>& buf, GLenum data_format = vc::wrapgl::internal::GLTextureFormats<T>::Format);
    inline void upload(const GLvoid* data, GLenum data_format = GL_RED, GLenum data_type = GL_FLOAT);
    
    template<typename T>
    inline void download(Buffer2DView<T,TargetHost>& buf, GLenum data_format = vc::wrapgl::internal::GLTextureFormats<T>::Format);
    inline void download(GLvoid* data, GLenum data_format = GL_RED, GLenum data_type = GL_FLOAT);
    
    inline void setSamplingLinear();
    inline void setSamplingNearestNeighbour();
    inline void setWrapClamp();
    inline void setWrapClampToEdge();
    inline void setWrapClampToBorder();
    inline void setWrapRepeat();
    inline void setWrapMirroredRepeat();
    inline void setDepthParameters();
    
    inline void setBorderColor(float3 color);
    inline void setBorderColor(float4 color);
    inline void setBorderColor(const Eigen::Matrix<float,3,1>& color);
    inline void setBorderColor(const Eigen::Matrix<float,4,1>& color);
    inline void setBorderColor(float r, float g, float b, float a = 1.0f);
    
    using TextureBase::bind;
    inline void bind() const;
    inline void unbind() const;
    
    inline GLint width() const;
    inline GLint height() const;
    
protected:
    GLint texw;
    GLint texh;
};
    
class Texture2D : public Texture2DBase
{
public:
    typedef typename Texture2DBase::Binder Binder;
    
    inline Texture2D();
    inline Texture2D(GLint w, GLint h, GLenum int_format = GL_RGBA32F, GLvoid* data = nullptr, int border = 0);
    inline ~Texture2D();
    
    template<typename T>
    inline void create(const Buffer2DView<T,TargetHost>& buf, GLenum int_format = GL_RGBA32F, int border = 0);
    inline void create(GLint w, GLint h, GLenum int_format = GL_RGBA32F,  GLvoid* data = nullptr, int border = 0);
    
    inline void destroy();
};

}

template<typename T, typename Target>
class GPUTexture2DFromOpenGL { };

template<typename T, typename Target>
class Buffer2DFromOpenGLTexture { };

/**
 * NOTE:
 * Host Buffer2D From OpenGL texture.
 * Note glMapBuffer doesn't work here, so we have to fake it.
 * It is Buffer2DManaged but copies the texture data.
 */
template<typename T>
class Buffer2DFromOpenGLTexture<T,TargetHost> : public Buffer2DManaged<T,TargetHost>
{
public:
    typedef Buffer2DView<T,TargetHost> ViewT;
    
    Buffer2DFromOpenGLTexture() = delete;
    
    inline Buffer2DFromOpenGLTexture(wrapgl::Texture2D& gltex) : 
        Buffer2DManaged<T,TargetHost>(gltex.width(),gltex.height())
    {        
        // copy the OpenGL texture
        gltex.bind();
        gltex.download(*this, wrapgl::internal::GLChannelTraits<vc::type_traits<T>::ChannelCount>::opengl_data_format);
    }
    
    inline ~Buffer2DFromOpenGLTexture()
    {
        
    }
    
    Buffer2DFromOpenGLTexture(const Buffer2DFromOpenGLTexture<T,TargetDeviceCUDA>& img) = delete;
    
    inline Buffer2DFromOpenGLTexture(Buffer2DFromOpenGLTexture<T,TargetDeviceCUDA>&& img) : ViewT(std::move(img))
    {
      
    }
    
    Buffer2DFromOpenGLTexture<T,TargetDeviceCUDA>& operator=(const Buffer2DFromOpenGLTexture<T,TargetDeviceCUDA>& img) = delete;
    
    inline Buffer2DFromOpenGLTexture<T,TargetDeviceCUDA>& operator=(Buffer2DFromOpenGLTexture<T,TargetDeviceCUDA>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

}

#ifdef VISIONCORE_HAVE_CUDA
#include <VisionCore/CUDAException.hpp>

#include <VisionCore/Buffers/CUDATexture.hpp>

namespace vc
{
    
// some functions wrapper here as cuda_gl_interop header leaks horrible stuff
namespace internal
{
    cudaGraphicsResource* registerOpenGLTexture(GLenum textype, GLuint id, unsigned int flags = cudaGraphicsMapFlagsNone);
}

template<typename T>
class GPUTexture2DFromOpenGL<T,TargetDeviceCUDA> : public GPUTexture2DView<T,TargetDeviceCUDA>
{
public:
    typedef GPUTexture2DView<T,TargetDeviceCUDA> ViewT;
    
    GPUTexture2DFromOpenGL() = delete;
    
    inline GPUTexture2DFromOpenGL(wrapgl::Texture2D& gltex);
    
    inline ~GPUTexture2DFromOpenGL();
    
    inline GPUTexture2DFromOpenGL(const GPUTexture2DFromOpenGL<T,TargetDeviceCUDA>& img) = delete;
    
    inline GPUTexture2DFromOpenGL(GPUTexture2DFromOpenGL<T,TargetDeviceCUDA>&& img)
      : ViewT(std::move(img)), resDesc(img.resDesc), texDesc(img.texDesc)
    {
        img.texref = 0;
    }
    
    inline GPUTexture2DFromOpenGL<T,TargetDeviceCUDA>& operator=(const GPUTexture2DFromOpenGL<T,TargetDeviceCUDA>&) = delete;
    
    inline GPUTexture2DFromOpenGL<T,TargetDeviceCUDA>& operator=(GPUTexture2DFromOpenGL<T,TargetDeviceCUDA>&& img)
    {
        ViewT::operator=(std::move(img));
        resDesc = img.resDesc;
        texDesc = img.texDesc;
        return *this;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
    
private:
    cudaGraphicsResource* cuda_res;
    cudaArray* array;
    cudaResourceDesc resDesc;
    cudaTextureDesc texDesc;
};

/**
 * CUDA Buffer2D From OpenGL texture.
 * It is Buffer2DManaged but copies the texture data.
 */
template<typename T>
class Buffer2DFromOpenGLTexture<T,TargetDeviceCUDA> : public Buffer2DManaged<T, TargetDeviceCUDA>
{
public:
    typedef Buffer2DView<T, TargetDeviceCUDA> ViewT;
    
    Buffer2DFromOpenGLTexture() = delete;
    
    inline Buffer2DFromOpenGLTexture(wrapgl::Texture2D& gltex);
    
    inline ~Buffer2DFromOpenGLTexture() { }
    
    Buffer2DFromOpenGLTexture(const Buffer2DFromOpenGLTexture<T,TargetDeviceCUDA>& img) = delete;
    
    inline Buffer2DFromOpenGLTexture(Buffer2DFromOpenGLTexture<T,TargetDeviceCUDA>&& img) : ViewT(std::move(img)) { }
    
    Buffer2DFromOpenGLTexture<T,TargetDeviceCUDA>& operator=(const Buffer2DFromOpenGLTexture<T,TargetDeviceCUDA>&) = delete;
    
    inline Buffer2DFromOpenGLTexture<T,TargetDeviceCUDA>& operator=(Buffer2DFromOpenGLTexture<T,TargetDeviceCUDA>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }

    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
private:
    cudaGraphicsResource* cuda_res;
};

}

#endif // VISIONCORE_HAVE_CUDA

#ifdef VISIONCORE_HAVE_OPENCL
namespace vc
{
    
template<typename T>
class GPUTexture2DFromOpenGL<T,TargetDeviceOpenCL> : public Image2DView<T,TargetDeviceOpenCL>
{
public:
    typedef Image2DView<T,TargetDeviceOpenCL> ViewT;
    
    GPUTexture2DFromOpenGL() = delete;
    
    inline GPUTexture2DFromOpenGL(wrapgl::Texture2D& gltex) : ViewT() { }
    
    inline ~GPUTexture2DFromOpenGL() { if(ViewT::isValid()) { } }
    
    inline GPUTexture2DFromOpenGL(const GPUTexture2DFromOpenGL<T,TargetDeviceOpenCL>& img) = delete;
    
    inline GPUTexture2DFromOpenGL(GPUTexture2DFromOpenGL<T,TargetDeviceOpenCL>&& img) : ViewT(std::move(img)) { }
    
    inline GPUTexture2DFromOpenGL<T,TargetDeviceOpenCL>& operator=(const GPUTexture2DFromOpenGL<T,TargetDeviceOpenCL>&) = delete;
    
    inline GPUTexture2DFromOpenGL<T,TargetDeviceOpenCL>& operator=(GPUTexture2DFromOpenGL<T,TargetDeviceOpenCL>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};
    
}
#endif // VISIONCORE_HAVE_OPENCL

#include <VisionCore/WrapGL/impl/WrapGLTexture_impl.hpp>

#endif // VISIONCORE_WRAPGL_TEXTURE_HPP
