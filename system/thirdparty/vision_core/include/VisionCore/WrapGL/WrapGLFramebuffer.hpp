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
 * Frame/Render buffer objects.
 * ****************************************************************************
 */

#ifndef VISIONCORE_WRAPGL_FRAME_BUFFER_HPP
#define VISIONCORE_WRAPGL_FRAME_BUFFER_HPP

#include <array>

#include <VisionCore/WrapGL/WrapGLCommon.hpp>
#include <VisionCore/WrapGL/WrapGLTexture.hpp>

namespace vc
{

namespace wrapgl
{
    
class RenderBuffer
{
public:    
    typedef ScopeBinder<RenderBuffer> Binder;
  
    inline RenderBuffer();
    inline RenderBuffer(GLint w, GLint h, GLenum internal_format = GL_DEPTH_COMPONENT24);
    inline ~RenderBuffer();
    
    inline void create(GLint w, GLint h, GLenum internal_format = GL_DEPTH_COMPONENT24);
    inline void destroy();
    inline bool isValid() const;
    
    inline void bind() const;
    inline void unbind() const;
    
    template<typename T>
    inline void download(Buffer2DView<T,TargetHost>& buf, GLenum data_format = GL_DEPTH_COMPONENT);
    
    inline void download(GLvoid* data, GLenum data_format = GL_DEPTH_COMPONENT, GLenum data_type = GL_FLOAT);
    inline GLuint id() const;
    inline GLint width() const;
    inline GLint height() const;
private:
    GLuint rbid;
    GLint rbw;
    GLint rbh;
};

class FrameBuffer
{
    constexpr static std::size_t MAX_ATTACHMENTS = 8;
    static std::array<GLenum,MAX_ATTACHMENTS> attachment_buffers;
public:    
    typedef ScopeBinder<FrameBuffer> Binder;
    
    inline FrameBuffer();
    inline ~FrameBuffer();
    
    inline bool isValid() const;
    inline GLuint id() const;
    
    inline GLenum attach(Texture2D& tex);
    
    inline unsigned int colorAttachmentCount() const;

    // This needs GL_DEPTH_COMPONENT24 / GL_DEPTH texture.
    inline GLenum attachDepth(Texture2D& tex);
    inline GLenum attachDepth(RenderBuffer& rb);
    
    inline void bind() const;
    inline void unbind() const;
    
    inline void drawInto() const;
    inline static GLenum checkComplete();
    
    inline void clearBuffer(unsigned int idx, float* val);
    inline void clearDepthBuffer(float val);
private:
    GLuint fbid;
    unsigned int attachments;
};

}
    
}

#include <VisionCore/WrapGL/impl/WrapGLFramebuffer_impl.hpp>

#endif // VISIONCORE_WRAPGL_FRAME_BUFFER_HPP
