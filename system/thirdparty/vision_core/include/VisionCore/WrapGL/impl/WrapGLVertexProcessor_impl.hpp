/**
 * ****************************************************************************
 * Copyright (c) 2018, Robert Lukierski.
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
 * Utility class to process images/buffers (texture-in/texture-out).
 * ****************************************************************************
 */

#ifndef VISIONCORE_WRAPGL_VERTEX_PROCESSOR_IMPL_HPP
#define VISIONCORE_WRAPGL_VERTEX_PROCESSOR_IMPL_HPP

#include <memory>
#include <functional>

#include <VisionCore/WrapGL/WrapGL.hpp>

inline vc::wrapgl::VertexProcessor::VertexProcessor() : 
    prog(std::make_shared<vc::wrapgl::Program>()), 
    feedback(), 
    query()
{
    
}

inline vc::wrapgl::VertexProcessor::VertexProcessor(ProgPtr p) : 
    prog(p), 
    feedback(), 
    query()
{
  
}

inline vc::wrapgl::VertexProcessor::~VertexProcessor()
{

}

inline vc::wrapgl::VertexProcessor::CompileRetT vc::wrapgl::VertexProcessor::addShaderFromSourceFile(vc::wrapgl::Program::Type type,
                                                                                                     const std::string& fn, 
                                                                                                     const IncPathVecT& inc_path)
{
    return prog->addShaderFromSourceFile(type, fn, inc_path);
}

inline vc::wrapgl::VertexProcessor::CompileRetT vc::wrapgl::VertexProcessor::addShaderFromSourceCode(vc::wrapgl::Program::Type type,
                                                                                                     const std::string& src, 
                                                                                                     const IncPathVecT& inc_path)
{
    return prog->addShaderFromSourceCode(type, src, inc_path);
}

inline void vc::wrapgl::VertexProcessor::runPre()
{
    glEnable(GL_RASTERIZER_DISCARD); 
    WRAPGL_CHECK_ERROR();
}

inline std::size_t vc::wrapgl::VertexProcessor::runPost(std::size_t count)
{
    feedback.begin(GL_POINTS);
    
    {
        vc::wrapgl::Query::Binder bind_query(query, vc::wrapgl::Query::Target::eTransformFeedbackPrimitivesWritten);
        
        glDrawArrays(GL_POINTS, 0, count);
        WRAPGL_CHECK_ERROR();
    }
    
    feedback.end();
    
    GLuint pri_count = 0;
    query.getObject<GLuint>(vc::wrapgl::Query::Parameter::eResult, &pri_count);
    
    glDisable(GL_RASTERIZER_DISCARD);
    WRAPGL_CHECK_ERROR();
    
    glFinish();
    WRAPGL_CHECK_ERROR();
    
    return pri_count;
}

#endif // VISIONCORE_WRAPGL_VERTEX_PROCESSOR_IMPL_HPP
