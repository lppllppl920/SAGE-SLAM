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
 * Transform feedback.
 * ****************************************************************************
 */

#ifndef VISIONCORE_WRAPGL_TRANSFORM_FEEDBACK_HPP
#define VISIONCORE_WRAPGL_TRANSFORM_FEEDBACK_HPP

#include <VisionCore/WrapGL/WrapGLCommon.hpp>

namespace vc
{

namespace wrapgl
{
    
class TransformFeedback
{
public:    
    typedef ScopeBinder<TransformFeedback> Binder;
    
    inline TransformFeedback();
    inline ~TransformFeedback();
    
    inline void create();
    inline void destroy();
    inline bool isValid() const;
    
    inline void bind() const;
    inline void unbind() const;
    
    inline void draw(GLenum mode = GL_POINTS) const;
    inline void draw(GLenum mode, GLsizei instcnt) const;
    
    inline static void begin(GLenum primode);
    inline static void end();
    inline static void pause();
    inline static void resume();
    
    inline GLuint id() const;
private:
    GLuint tbid;
};
    
}

}

#include <VisionCore/WrapGL/impl/WrapGLTransformFeedback_impl.hpp>

#endif // VISIONCORE_WRAPGL_TRANSFORM_FEEDBACK_HPP
