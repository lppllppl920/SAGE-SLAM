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

#ifndef VISIONCORE_WRAPGL_PIXEL_PROCESSOR_IMPL_HPP
#define VISIONCORE_WRAPGL_PIXEL_PROCESSOR_IMPL_HPP

#include <memory>
#include <functional>

#include <VisionCore/WrapGL/WrapGL.hpp>

inline vc::wrapgl::PixelProcessor::PixelProcessor(std::size_t w, std::size_t h) : 
    width(w), height(h),
    prog(std::make_shared<vc::wrapgl::Program>()),
    rb(width,height)
{
    addDefaultShaders();
}

inline vc::wrapgl::PixelProcessor::PixelProcessor(ProgPtr p, std::size_t w, std::size_t h) : 
    width(w), height(h),
    prog(p),
    rb(width,height)
{
    addDefaultShaders();
}

inline vc::wrapgl::PixelProcessor::~PixelProcessor()
{

}

inline vc::wrapgl::PixelProcessor::CompileRetT vc::wrapgl::PixelProcessor::addShaderFromSourceFile(const std::string& fn, 
                                                                                                   const IncPathVecT& inc_path)
{
    return prog->addShaderFromSourceFile(vc::wrapgl::Program::Type::Fragment, fn, inc_path);
}

inline vc::wrapgl::PixelProcessor::CompileRetT vc::wrapgl::PixelProcessor::addShaderFromSourceCode(const std::string& src, 
                                                                                                   const IncPathVecT& inc_path)
{
    return prog->addShaderFromSourceCode(vc::wrapgl::Program::Type::Fragment, src, inc_path);
}

inline void vc::wrapgl::PixelProcessor::runPre()
{
    glPushAttrib(GL_VIEWPORT_BIT);
    WRAPGL_CHECK_ERROR();
    
    glViewport(0, 0, width, height);
    WRAPGL_CHECK_ERROR();
    
    glClearColor(0, 0, 0, 0);
    WRAPGL_CHECK_ERROR();
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::PixelProcessor::runPost()
{
    glDrawArrays(GL_POINTS, 0, 1);
    WRAPGL_CHECK_ERROR();
    
    glPopAttrib();
    WRAPGL_CHECK_ERROR();
    
    glFinish();
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::PixelProcessor::addDefaultShaders()
{
    auto r1 = prog->addShaderFromSourceCode(vc::wrapgl::Program::Type::Vertex,"#version 330\nvoid main() { }\n");
    if(!r1.first) { throw std::runtime_error(r1.second); }
    
    auto r2 = prog->addShaderFromSourceCode(vc::wrapgl::Program::Type::Geometry,
        "#version 330 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices = 4) out;\n"
        "out vec2 texcoord;\n"
        "void main()\n"
        "{\n"
        "    gl_Position = vec4(1.0, 1.0, 0.0, 1.0); texcoord = vec2(1.0, 1.0); EmitVertex();\n"
        "    gl_Position = vec4(-1.0, 1.0, 0.0, 1.0); texcoord = vec2(0.0, 1.0); EmitVertex();\n"
        "    gl_Position = vec4(1.0,-1.0, 0.0, 1.0); texcoord = vec2(1.0, 0.0); EmitVertex();\n"
        "    gl_Position = vec4(-1.0,-1.0, 0.0, 1.0); texcoord = vec2(0.0, 0.0); EmitVertex();\n" 
        "    EndPrimitive();\n"
        "}\n");
    
    if(!r2.first) { throw std::runtime_error(r2.second); }
    
    vc::wrapgl::FrameBuffer::Binder bind_fb(fb);
    fb.attachDepth(rb);
}

#endif // VISIONCORE_WRAPGL_PIXEL_PROCESSOR_IMPL_HPP
