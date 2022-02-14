/**
 * ****************************************************************************
 * Copyright (c) 2017, Robert Lukierski.
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

#ifndef VISIONCORE_WRAPGL_FRAME_BUFFER_IMPL_HPP
#define VISIONCORE_WRAPGL_FRAME_BUFFER_IMPL_HPP
  
inline vc::wrapgl::RenderBuffer::RenderBuffer() : rbid(0), rbw(0), rbh(0)
{
    
}

inline vc::wrapgl::RenderBuffer::~RenderBuffer()
{
    destroy();
}
    
inline vc::wrapgl::RenderBuffer::RenderBuffer(GLint w, GLint h, GLenum internal_format) : rbid(0)
{
    create(w,h,internal_format);
}

inline void vc::wrapgl::RenderBuffer::create(GLint w, GLint h, GLenum internal_format)
{
    destroy();
    
    rbw = w;
    rbh = h;
    
    glGenRenderbuffers(1, &rbid);
    WRAPGL_CHECK_ERROR();
    bind();
    glRenderbufferStorage(GL_RENDERBUFFER, internal_format, rbw, rbh);
    WRAPGL_CHECK_ERROR();
    unbind();
}

inline void vc::wrapgl::RenderBuffer::destroy()
{
    if(rbid != 0)
    {
        glDeleteRenderbuffers(1, &rbid);
        WRAPGL_CHECK_ERROR();
        rbid = 0;
    }
}

inline bool vc::wrapgl::RenderBuffer::isValid() const 
{ 
    return rbid != 0; 
}

inline void vc::wrapgl::RenderBuffer::bind() const
{
    glBindRenderbuffer(GL_RENDERBUFFER, rbid);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::RenderBuffer::unbind() const
{
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    WRAPGL_CHECK_ERROR();
}

template<typename T>
inline void vc::wrapgl::RenderBuffer::download(Buffer2DView<T,TargetHost>& buf, GLenum data_format)
{
    download(buf.ptr(), data_format, internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type);
}

inline void vc::wrapgl::RenderBuffer::download(GLvoid* data, GLenum data_format, GLenum data_type)
{
    glReadPixels(0,0,width(),height(),data_format,data_type,data);
}

inline GLuint vc::wrapgl::RenderBuffer::id() const 
{ 
    return rbid; 
}

inline GLint vc::wrapgl::RenderBuffer::width() const 
{ 
    return rbw; 
}

inline GLint vc::wrapgl::RenderBuffer::height() const 
{ 
    return rbh; 
}

inline vc::wrapgl::FrameBuffer::FrameBuffer() : attachments(0)
{
    glGenFramebuffers(1, &fbid);
    WRAPGL_CHECK_ERROR();
}

inline vc::wrapgl::FrameBuffer::~FrameBuffer()
{
    glDeleteFramebuffers(1, &fbid); 
    WRAPGL_CHECK_ERROR();
    fbid = 0; 
}

inline GLenum vc::wrapgl::FrameBuffer::attach(Texture2D& tex)
{
    const GLenum color_attachment = GL_COLOR_ATTACHMENT0 + attachments;
    glFramebufferTexture2D(GL_FRAMEBUFFER, color_attachment, GL_TEXTURE_2D, tex.id(), 0);
    WRAPGL_CHECK_ERROR();
    attachments++;
    return color_attachment;
}

inline GLenum vc::wrapgl::FrameBuffer::attachDepth(Texture2D& tex)
{
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex.id(), 0);
    WRAPGL_CHECK_ERROR();
    return GL_DEPTH_ATTACHMENT;
}

inline GLenum vc::wrapgl::FrameBuffer::attachDepth(RenderBuffer& rb)
{
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb.id());
    WRAPGL_CHECK_ERROR();
    return GL_DEPTH_ATTACHMENT;
}

inline bool vc::wrapgl::FrameBuffer::isValid() const 
{ 
    return fbid != 0; 
}

inline void vc::wrapgl::FrameBuffer::bind() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, fbid);   
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::FrameBuffer::unbind() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::FrameBuffer::drawInto() const
{
    glDrawBuffers( attachments, attachment_buffers.data() );
    WRAPGL_CHECK_ERROR();
}

inline GLenum vc::wrapgl::FrameBuffer::checkComplete() 
{
    const GLenum ret = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    WRAPGL_CHECK_ERROR();
    return ret;
}

inline void vc::wrapgl::FrameBuffer::clearBuffer(unsigned int idx, float* val)
{
    glClearBufferfv(GL_COLOR, idx, val);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::FrameBuffer::clearDepthBuffer(float val)
{
    glClearBufferfv(GL_DEPTH, 0, &val);
    WRAPGL_CHECK_ERROR();
}

inline GLuint vc::wrapgl::FrameBuffer::id() const 
{ 
    return fbid; 
}

inline unsigned int vc::wrapgl::FrameBuffer::colorAttachmentCount() const 
{ 
    return attachments; 
}

#endif // VISIONCORE_WRAPGL_FRAME_BUFFER_IMPL_HPP
