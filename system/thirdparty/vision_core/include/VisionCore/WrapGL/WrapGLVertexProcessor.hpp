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

#ifndef VISIONCORE_WRAPGL_VERTEX_PROCESSOR_HPP
#define VISIONCORE_WRAPGL_VERTEX_PROCESSOR_HPP

#include <cstdint>
#include <cstddef>
#include <string>
#include <memory>
#include <functional>

#include <VisionCore/WrapGL/WrapGLProgram.hpp>
#include <VisionCore/WrapGL/WrapGLTransformFeedback.hpp>
#include <VisionCore/WrapGL/WrapGLQuery.hpp>

namespace vc
{
  
namespace wrapgl
{
  
class VertexProcessor
{
public:
    typedef typename vc::wrapgl::Program::IncPathVecT IncPathVecT;
    typedef typename vc::wrapgl::Program::CompileRetT CompileRetT;
    typedef std::shared_ptr<vc::wrapgl::Program> ProgPtr;
    typedef std::shared_ptr<const vc::wrapgl::Program> CProgPtr;
    
    VertexProcessor();
    VertexProcessor(ProgPtr p);
    virtual ~VertexProcessor();
    
    CompileRetT addShaderFromSourceFile(vc::wrapgl::Program::Type type, const std::string& fn, 
                                        const IncPathVecT& inc_path = IncPathVecT());
    
    CompileRetT addShaderFromSourceCode(vc::wrapgl::Program::Type type, const std::string& src, 
                                        const IncPathVecT& inc_path = IncPathVecT());
    
    CompileRetT link() { return prog->link(); }

    template<typename Fun>
    int run(std::size_t count, Fun f)
    {
        if(!prog->isLinked())
        {
            return -1;
        }
        
        vc::wrapgl::Program::Binder bind_prog(*prog);
        vc::wrapgl::TransformFeedback::Binder bind_tf(feedback);
        
        runPre();
        
        f(*prog);
        
        const std::size_t v = runPost(count);
        
        return v;
    }
    
    CProgPtr getProgram() const { return prog; }
    ProgPtr getProgram() { return prog; }
    
    const vc::wrapgl::TransformFeedback& getTransformFeedback() const { return feedback; }
    vc::wrapgl::TransformFeedback& getTransformFeedback() { return feedback; }
private:
    void runPre();
    std::size_t runPost(std::size_t count);
    
    ProgPtr prog;
    vc::wrapgl::TransformFeedback feedback;
    vc::wrapgl::Query query;
};

}

}


#include <VisionCore/WrapGL/impl/WrapGLVertexProcessor_impl.hpp>

#endif // VISIONCORE_WRAPGL_VERTEX_PROCESSOR_HPP
