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
 * Queries.
 * ****************************************************************************
 */

#ifndef VISIONCORE_WRAPGL_QUERY_IMPL_HPP
#define VISIONCORE_WRAPGL_QUERY_IMPL_HPP

inline vc::wrapgl::Query::Query() : qid(0)
{
    create();
}

inline vc::wrapgl::Query::~Query()
{
    destroy();
}

inline void vc::wrapgl::Query::create()
{
    destroy();
    
    glGenQueries(1, &qid);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Query::destroy()
{
    if(qid != 0)
    {
        glDeleteQueries(1, &qid);
        WRAPGL_CHECK_ERROR();
        qid = 0;
    }
}

inline bool vc::wrapgl::Query::isValid() const 
{ 
    return qid != 0; 
}

inline void vc::wrapgl::Query::begin(Target target) const
{
    glBeginQuery(static_cast<GLenum>(target), qid);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Query::end(Target target) const
{
    glEndQuery(static_cast<GLenum>(target));
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Query::begin(Target target, GLuint idx) const
{
    glBeginQueryIndexed(static_cast<GLenum>(target), idx, qid);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Query::end(Target target, GLuint idx) const
{
    glEndQueryIndexed(static_cast<GLenum>(target), idx);
    WRAPGL_CHECK_ERROR();
}

inline GLint vc::wrapgl::Query::get(Target target, Parameter pname)
{
    GLint ret = 0;
    glGetQueryiv(static_cast<GLenum>(target), static_cast<GLenum>(pname), &ret);
    WRAPGL_CHECK_ERROR();
    return ret;
}

inline GLint vc::wrapgl::Query::get(Target target, GLuint index, Parameter pname)
{
    GLint ret = 0;
    glGetQueryIndexediv(static_cast<GLenum>(target), index, static_cast<GLenum>(pname), &ret);
    WRAPGL_CHECK_ERROR();
    return ret;
}

namespace internal
{

template<typename T> struct GetQueryObject {};

template<> struct GetQueryObject<GLint>
{ static inline void run(GLuint id, GLenum pname, GLint* params) { glGetQueryObjectiv(id,pname,params); } };
template<> struct GetQueryObject<GLuint>
{ static inline void run(GLuint id, GLenum pname, GLuint* params) { glGetQueryObjectuiv(id,pname,params); } };
template<> struct GetQueryObject<GLint64>
{ static inline void run(GLuint id, GLenum pname, GLint64* params) { glGetQueryObjecti64v(id,pname,params); } };
template<> struct GetQueryObject<GLuint64>
{ static inline void run(GLuint id, GLenum pname, GLuint64* params) { glGetQueryObjectui64v(id,pname,params); } };

}

template<typename T>
inline void vc::wrapgl::Query::getObject(Parameter pname, T* params)
{
    ::internal::GetQueryObject<T>::run(qid, static_cast<GLenum>(pname), params);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Query::queryTimestamp()
{
    glQueryCounter(qid, GL_TIMESTAMP);
    WRAPGL_CHECK_ERROR();
}

inline GLuint vc::wrapgl::Query::id() const 
{ 
    return qid; 
}

inline vc::wrapgl::ScopeBinder<vc::wrapgl::Query>::ScopeBinder(const vc::wrapgl::Query& aobj, 
                                                               typename vc::wrapgl::Query::Target tgt) : obj(aobj), target(tgt)
{
  obj.begin(target);
}

inline vc::wrapgl::ScopeBinder<vc::wrapgl::Query>::~ScopeBinder()
{
  obj.end(target);
}

#endif // VISIONCORE_WRAPGL_QUERY_IMPL_HPP
