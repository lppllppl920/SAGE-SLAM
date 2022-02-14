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
 * Transform feedback.
 * ****************************************************************************
 */

#ifndef VISIONCORE_WRAPGL_TRANSFORM_FEEDBACK_IMPL_HPP
#define VISIONCORE_WRAPGL_TRANSFORM_FEEDBACK_IMPL_HPP
    
inline vc::wrapgl::TransformFeedback::TransformFeedback() : tbid(0)
{
    create();
}

inline vc::wrapgl::TransformFeedback::~TransformFeedback() 
{
    destroy();
}
    
inline void vc::wrapgl::TransformFeedback::create()
{
    destroy();
    
    glGenTransformFeedbacks(1, &tbid);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::TransformFeedback::destroy()
{
    if(tbid != 0)
    {
        glDeleteTransformFeedbacks(1, &tbid);
        WRAPGL_CHECK_ERROR();
        tbid = 0;
    }
}

inline bool vc::wrapgl::TransformFeedback::isValid() const 
{ 
    return tbid != 0; 
}

inline void vc::wrapgl::TransformFeedback::bind() const
{
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tbid);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::TransformFeedback::unbind() const
{
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::TransformFeedback::draw(GLenum mode) const
{
    glDrawTransformFeedback(mode, tbid);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::TransformFeedback::draw(GLenum mode, GLsizei instcnt) const
{
    glDrawTransformFeedbackInstanced(mode, tbid, instcnt);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::TransformFeedback::begin(GLenum primode)
{
    glBeginTransformFeedback(primode);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::TransformFeedback::end()
{
    glEndTransformFeedback();
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::TransformFeedback::pause()
{
    glPauseTransformFeedback();
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::TransformFeedback::resume()
{
    glPauseTransformFeedback();
    WRAPGL_CHECK_ERROR();
}

inline GLuint vc::wrapgl::TransformFeedback::id() const 
{ 
    return tbid; 
}

#endif // VISIONCORE_WRAPGL_TRANSFORM_FEEDBACK_IMPL_HPP
