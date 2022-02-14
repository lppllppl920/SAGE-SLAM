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

#ifndef VISIONCORE_WRAPGL_PIXEL_PROCESSOR_HPP
#define VISIONCORE_WRAPGL_PIXEL_PROCESSOR_HPP

#include <cstdint>
#include <cstddef>
#include <string>
#include <memory>
#include <functional>

#include <VisionCore/WrapGL/WrapGLProgram.hpp>
#include <VisionCore/WrapGL/WrapGLFramebuffer.hpp>

namespace vc
{
  
namespace wrapgl
{
  
class PixelProcessor
{
public:  
    typedef typename vc::wrapgl::Program::IncPathVecT IncPathVecT;
    typedef typename vc::wrapgl::Program::CompileRetT CompileRetT;
    typedef std::shared_ptr<vc::wrapgl::Program> ProgPtr;
    typedef std::shared_ptr<const vc::wrapgl::Program> CProgPtr;
    
    PixelProcessor(std::size_t w, std::size_t h = 1);
    PixelProcessor(ProgPtr p, std::size_t w, std::size_t h = 1);
    virtual ~PixelProcessor();    
    
    CompileRetT addShaderFromSourceFile(const std::string& fn, const IncPathVecT& inc_path = IncPathVecT());
    CompileRetT addShaderFromSourceCode(const std::string& src, const IncPathVecT& inc_path = IncPathVecT());
    CompileRetT link() { return prog->link(); }
    
    /*
     * Use in vec2 texcoord;
     */
    template<typename Fun>
    bool run(Fun f)
    {
        if(!prog->isLinked())
        {
            return false;
        }
        
        vc::wrapgl::FrameBuffer::Binder bind_fb(fb);
        
        runPre();
        
        vc::wrapgl::Program::Binder bind_prog(*prog);
        
        f(*prog);
        
        runPost();
        
        return true;
    }
    
    CProgPtr getProgram() const { return prog; }
    ProgPtr getProgram() { return prog; }
    
    const vc::wrapgl::FrameBuffer& getFrameBuffer() const { return fb; }
    vc::wrapgl::FrameBuffer& getFrameBuffer() { return fb; }
    
    std::size_t getWidth() const { return width; }
    std::size_t getHeight() const { return height; }
private:
    void runPre();
    void runPost();
    void addDefaultShaders();
    
    std::size_t width, height;
    ProgPtr prog;
    vc::wrapgl::FrameBuffer fb;
    vc::wrapgl::RenderBuffer rb;
};

}

}


#include <VisionCore/WrapGL/impl/WrapGLPixelProcessor_impl.hpp>

#endif // VISIONCORE_WRAPGL_PIXEL_PROCESSOR_HPP
