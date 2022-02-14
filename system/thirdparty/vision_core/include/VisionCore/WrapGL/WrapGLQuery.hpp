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
 * Queries.
 * ****************************************************************************
 */

#ifndef VISIONCORE_WRAPGL_QUERY_HPP
#define VISIONCORE_WRAPGL_QUERY_HPP

#include <VisionCore/WrapGL/WrapGLCommon.hpp>

namespace vc
{

namespace wrapgl
{
    
class Query
{
public:
    typedef ScopeBinder<Query> Binder;
    
    enum class Target
    {
        eSamplesPassed = GL_SAMPLES_PASSED, 
        eAnySamplesPasses = GL_ANY_SAMPLES_PASSED, 
        eTransformFeedbackPrimitivesWritten = GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, 
        eTimeElapsed = GL_TIME_ELAPSED, 
        eTimestamp = GL_TIMESTAMP
    };
    
    enum class Parameter
    {
        eCurrent = GL_CURRENT_QUERY,
        eCounterBits = GL_QUERY_COUNTER_BITS,
        eResult = GL_QUERY_RESULT, 
        eResultNoWait = GL_QUERY_RESULT_NO_WAIT,
        eResultAvailable = GL_QUERY_RESULT_AVAILABLE
    };
    
    inline Query();
    inline ~Query();
    
    inline void create();
    inline void destroy();
    inline bool isValid() const;
    
    // 
    inline void begin(Target target) const;
    inline void end(Target target) const;
    
    inline void begin(Target target, GLuint idx) const;
    inline void end(Target target, GLuint idx) const;
    
    inline GLint get(Target target, Parameter pname);
    inline GLint get(Target target, GLuint index, Parameter pname);
    template<typename T>
    inline void getObject(Parameter pname, T* params = nullptr);
    
    inline void queryTimestamp();
    
    inline GLuint id() const;
private:
    GLuint qid;
};

template<> struct ScopeBinder<Query>
{
    ScopeBinder() = delete;
    ScopeBinder(const ScopeBinder&) = delete;
    ScopeBinder(ScopeBinder&&) = delete;
    ScopeBinder& operator=(const ScopeBinder&) = delete;
    ScopeBinder& operator=(ScopeBinder&&) = delete;
    
    inline ScopeBinder(const Query& aobj, typename Query::Target tgt);
    inline ~ScopeBinder();
    
    const Query& obj;
    typename Query::Target target;
};

}
    
}

#include <VisionCore/WrapGL/impl/WrapGLQuery_impl.hpp>

#endif // VISIONCORE_WRAPGL_QUERY_HPP
