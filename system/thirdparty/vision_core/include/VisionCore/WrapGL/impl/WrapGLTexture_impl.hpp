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

#ifndef VISIONCORE_WRAPGL_TEXTURE_IMPL_HPP
#define VISIONCORE_WRAPGL_TEXTURE_IMPL_HPP

inline vc::wrapgl::TextureBase::TextureBase() : texid(0), internal_format((GLenum)0)
{
    
}

inline vc::wrapgl::TextureBase::~TextureBase()
{
    destroy();
}

inline void vc::wrapgl::TextureBase::create(GLenum int_format)
{
    if(isValid()) { destroy(); }
    
    internal_format = int_format;
    glGenTextures(1,&texid);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::TextureBase::destroy()
{
    if(isValid()) 
    {
        glDeleteTextures(1,&texid);
        WRAPGL_CHECK_ERROR();
        internal_format = (GLenum)0;
        texid = 0;
    }
}

inline GLuint vc::wrapgl::TextureBase::id() const 
{ 
    return texid; 
}

inline GLenum vc::wrapgl::TextureBase::intFormat() const 
{ 
    return internal_format; 
}

inline bool vc::wrapgl::TextureBase::isValid() const 
{ 
    return texid != 0; 
}

inline void vc::wrapgl::TextureBase::bind(const GLenum unit)
{
    glActiveTexture(unit);
    WRAPGL_CHECK_ERROR();
}

inline vc::wrapgl::Texture2DBase::Texture2DBase() : TextureBase(), texw(0), texh(0)
{
    
}

template<typename T>
inline void vc::wrapgl::Texture2DBase::upload(const Buffer2DView<T,TargetHost>& buf, GLenum data_format)
{
    upload(buf.ptr(), data_format, internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type);
}

inline void vc::wrapgl::Texture2DBase::upload(const GLvoid* data, GLenum data_format, GLenum data_type)
{
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,texw,texh,data_format,data_type,data);
    WRAPGL_CHECK_ERROR();
}

template<typename T>
inline void vc::wrapgl::Texture2DBase::download(Buffer2DView<T,TargetHost>& buf, GLenum data_format)
{
    download(buf.ptr(), data_format, internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type);
}

inline void vc::wrapgl::Texture2DBase::download(GLvoid* data, GLenum data_format, GLenum data_type)
{
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    WRAPGL_CHECK_ERROR();
    glGetTexImage(GL_TEXTURE_2D, 0, data_format, data_type, data);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Texture2DBase::setSamplingLinear()
{
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (GLint)GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)GL_LINEAR);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Texture2DBase::setSamplingNearestNeighbour()
{
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (GLint)GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)GL_NEAREST);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Texture2DBase::setWrapClamp()
{
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint)GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)GL_CLAMP);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Texture2DBase::setWrapClampToEdge()
{
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint)GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)GL_CLAMP_TO_EDGE);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Texture2DBase::setWrapClampToBorder()
{
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint)GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)GL_CLAMP_TO_BORDER);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Texture2DBase::setWrapRepeat()
{
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint)GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)GL_REPEAT);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Texture2DBase::setWrapMirroredRepeat()
{
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint)GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)GL_MIRRORED_REPEAT);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Texture2DBase::setDepthParameters()
{
    glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, (GLint)GL_INTENSITY);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, (GLint)GL_COMPARE_R_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, (GLint)GL_LEQUAL);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Texture2DBase::setBorderColor(float3 color)
{
    setBorderColor(color.x, color.y, color.z, 1.0f);
}

inline void vc::wrapgl::Texture2DBase::setBorderColor(float4 color)
{
    setBorderColor(color.x, color.y, color.z, color.w);
}

inline void vc::wrapgl::Texture2DBase::setBorderColor(const Eigen::Matrix<float,3,1>& color)
{
    setBorderColor(color(0), color(1), color(2), 1.0f);
}

inline void vc::wrapgl::Texture2DBase::setBorderColor(const Eigen::Matrix<float,4,1>& color)
{
    setBorderColor(color(0), color(1), color(2), color(3));
}

inline void vc::wrapgl::Texture2DBase::setBorderColor(float r, float g, float b, float a)
{
    GLfloat params[4] = {r,g,b,a};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, params);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Texture2DBase::bind() const
{
    glBindTexture(GL_TEXTURE_2D, texid);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Texture2DBase::unbind() const
{
    glBindTexture(GL_TEXTURE_2D, 0);
    WRAPGL_CHECK_ERROR();
}

inline GLint vc::wrapgl::Texture2DBase::width() const 
{ 
    return texw; 
}

inline GLint vc::wrapgl::Texture2DBase::height() const 
{ 
    return texh; 
}

inline vc::wrapgl::Texture2D::Texture2D() : Texture2DBase()
{
    
}

inline vc::wrapgl::Texture2D::Texture2D(GLint w, GLint h, GLenum int_format, GLvoid* data, int border)
    : Texture2DBase()
{
    create(w, h, int_format, data, border);
}

inline vc::wrapgl::Texture2D::~Texture2D() 
{
    destroy();
}

template<typename T>
inline void vc::wrapgl::Texture2D::create(const Buffer2DView<T,TargetHost>& buf, GLenum int_format, int border) 
{
    create(buf.width(), buf.height(), int_format, (GLvoid*)buf.ptr(), border);
}

inline void vc::wrapgl::Texture2D::create(GLint w, GLint h, GLenum int_format,  GLvoid* data, int border)
{
    if(isValid()) { destroy(); }
    
    TextureBase::create(int_format);
    
    texw = w;
    texh = h;

    bind();
    
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (GLint)GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint)GL_REPEAT);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)GL_REPEAT);
    
    glTexStorage2D(GL_TEXTURE_2D, 1, internal_format, texw, texh );
    WRAPGL_CHECK_ERROR();
    
    unbind();
}

inline void vc::wrapgl::Texture2D::destroy()
{
    TextureBase::destroy();
}
 
#ifdef VISIONCORE_HAVE_CUDA
template<typename T>
inline vc::GPUTexture2DFromOpenGL<T,vc::TargetDeviceCUDA>::GPUTexture2DFromOpenGL(wrapgl::Texture2D& gltex) : ViewT(), cuda_res(0)
{
    cuda_res = internal::registerOpenGLTexture(GL_TEXTURE_2D, gltex.id());
    
    cudaError_t err = cudaGraphicsMapResources(1, &cuda_res);
    if(err != cudaSuccess) { throw CUDAException(err, "Error mapping OpenGL texture"); }

    err = cudaGraphicsSubResourceGetMappedArray(&array, cuda_res, 0, 0);
    if(err != cudaSuccess) { throw CUDAException(err, "Error getting cudaArray from OpenGL texture"); }
    
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;
    
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    texDesc.addressMode[1]   = cudaAddressModeWrap;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.normalizedCoords = 0;
    
    err = cudaCreateTextureObject(&(ViewT::handle()), &resDesc, &texDesc, NULL);
    if(err != cudaSuccess)
    {
        throw CUDAException(err, "Cannot create texture object");
    }
}

template<typename T>
inline vc::GPUTexture2DFromOpenGL<T,vc::TargetDeviceCUDA>::~GPUTexture2DFromOpenGL()
{
    if(cuda_res) 
    {
        cudaError_t err;
        
        err = cudaDestroyTextureObject(ViewT::handle());
        assert(err == cudaSuccess);
        
        err = cudaGraphicsUnmapResources(1, &cuda_res);
        assert(err == cudaSuccess);

        err = cudaGraphicsUnregisterResource(cuda_res);
        assert(err == cudaSuccess);
    }
}
    
template<typename T>
inline vc::Buffer2DFromOpenGLTexture<T,vc::TargetDeviceCUDA>::Buffer2DFromOpenGLTexture(wrapgl::Texture2D& gltex)
  : Buffer2DManaged<T,vc::TargetDeviceCUDA>(gltex.width(),gltex.height())
{    
    cudaArray* textPtr = nullptr;
    
    cudaGraphicsResource* cuda_res = internal::registerOpenGLTexture(GL_TEXTURE_2D, gltex.id());
    if(cuda_res == nullptr) { throw std::runtime_error("Cannot register the texture"); }
  
    cudaError_t err = cudaGraphicsMapResources(1, &cuda_res);
    if( err != cudaSuccess ) { throw CUDAException(err, "Unable to map CUDA resources"); }
    
    err = cudaGraphicsSubResourceGetMappedArray(&textPtr, cuda_res, 0, 0);
    if( err != cudaSuccess ) { throw CUDAException(err, "Unable to get mapped array"); }
    
    err = cudaMemcpy2DFromArray(ViewT::memptr, ViewT::line_pitch, textPtr, 0, 0, 
                                ViewT::xsize * sizeof(T), ViewT::ysize, cudaMemcpyDeviceToDevice);
    if( err != cudaSuccess ) { throw CUDAException(err, "Unable memcpy from Array"); }
    
    err = cudaGraphicsUnmapResources(1, &cuda_res);
    if( err != cudaSuccess ) { throw CUDAException(err, "Unable unmap CUDA resources"); }
    
    err = cudaGraphicsUnregisterResource(cuda_res);
    if( err != cudaSuccess ) { throw CUDAException(err, "Unable unregister CUDA resources"); }
}
#endif // VISIONCORE_HAVE_CUDA


#endif // VISIONCORE_WRAPGL_TEXTURE_IMPL_HPP
