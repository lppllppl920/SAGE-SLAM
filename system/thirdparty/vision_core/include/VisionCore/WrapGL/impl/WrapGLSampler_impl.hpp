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
 * Texture Samplers.
 * ****************************************************************************
 */

#ifndef VISIONCORE_WRAPGL_SAMPLER_IMPL_HPP
#define VISIONCORE_WRAPGL_SAMPLER_IMPL_HPP

inline vc::wrapgl::Sampler::Sampler() : sid(0)
{
    create(0);
}

inline vc::wrapgl::Sampler::Sampler(GLuint texu) : sid(0)
{
    create(texu);
}

inline vc::wrapgl::Sampler::~Sampler()
{
    destroy();
}

inline void vc::wrapgl::Sampler::create(GLuint texu)
{
    destroy();
    
    glGenSamplers(1, &sid);
    WRAPGL_CHECK_ERROR();
    texunit = texu;
}

inline void vc::wrapgl::Sampler::destroy()
{
    if(sid != 0)
    {
        glDeleteSamplers(1, &sid);
        WRAPGL_CHECK_ERROR();
        sid = 0;
    }
}

inline bool vc::wrapgl::Sampler::isValid() const 
{ 
    return sid != 0; 
}

inline void vc::wrapgl::Sampler::bind() const
{
    glBindSampler(texunit, sid);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Sampler::unbind() const
{
    glBindSampler(texunit, 0);
    WRAPGL_CHECK_ERROR();
}

template<typename T>
inline T vc::wrapgl::Sampler::get(GLenum param)
{
    T ret;
    glGetSamplerParameterfv(sid, param, &ret);
    WRAPGL_CHECK_ERROR();
    return ret;
}

template<typename T>
inline void vc::wrapgl::Sampler::set(GLenum param, T val)
{
    glSamplerParameterf(sid, param, val);
    WRAPGL_CHECK_ERROR();
}

inline GLuint vc::wrapgl::Sampler::id() const 
{ 
    return sid; 
}

#endif // VISIONCORE_WRAPGL_SAMPLER_IMPL_HPP
