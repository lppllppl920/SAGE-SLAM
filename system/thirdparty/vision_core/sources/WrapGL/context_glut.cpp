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
 * (Free)GLUT Context.
 * ****************************************************************************
 */

#include <VisionCore/WrapGL/WrapGLContext.hpp>

#include <GL/freeglut.h>

// FreeGLUT Callbacks
extern "C"
{
    
void onGLUTDisplay()
{
}
    
}

struct vc::wrapgl::FreeGLUTContext::Pimpl
{
    Pimpl(glbinding::ContextHandle& h) : hndl(h)
    {
        
    }
    
    ~Pimpl()
    {
        
    }
    
    bool create(const std::string& window_title, std::size_t w, std::size_t h, bool double_buff)
    {
        if( glutGet(GLUT_INIT_STATE) == 0)
        {
            int argc = 0;
            glutInit(&argc, 0);
            glutInitDisplayMode(double_buff == true ? GLUT_DOUBLE : 0);
        }
        
        glutInitWindowSize(w,h);
        hndl = glutCreateWindow(window_title.c_str());
        
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
        
        glutDisplayFunc(&onGLUTDisplay);
        
#if 0
        glutKeyboardFunc(&process::Keyboard);
        glutKeyboardUpFunc(&process::KeyboardUp);
        glutReshapeFunc(&process::Resize);
        glutMouseFunc(&process::Mouse);
        glutMotionFunc(&process::MouseMotion);
        glutPassiveMotionFunc(&process::PassiveMouseMotion);
        glutSpecialFunc(&process::SpecialFunc);
        glutSpecialUpFunc(&process::SpecialFuncUp);
#endif
        return hndl != 0;
    }
    
    void finishFrame()
    {
        glutSwapBuffers();
        glutMainLoopEvent();
    }
    
    void shutdown()
    {
        glutDestroyWindow(hndl);
    }
    
    bool shouldQuit()
    {
        return glutGetWindow() == 0;
    }
    
    void switchContext()
    {
        
    }
    
    glbinding::ContextHandle& hndl;
};

vc::wrapgl::FreeGLUTContext::FreeGLUTContext() : pp(new Pimpl(hndl))
{
    
}

vc::wrapgl::FreeGLUTContext::FreeGLUTContext(const std::string& window_title, std::size_t w, std::size_t h, bool double_buff) : pp(new Pimpl(hndl))
{
    create(window_title, w, h, double_buff);
}

vc::wrapgl::FreeGLUTContext::~FreeGLUTContext()
{
    
}


void vc::wrapgl::FreeGLUTContext::finishFrame()
{
    pp->finishFrame();
}

void vc::wrapgl::FreeGLUTContext::shutdown()
{
    pp->shutdown();
}

bool vc::wrapgl::FreeGLUTContext::shouldQuit()
{
    return pp->shouldQuit();
}

bool vc::wrapgl::FreeGLUTContext::create_impl(const std::string& window_title, std::size_t w, std::size_t h, bool double_buff)
{
    return pp->create(window_title, w, h, double_buff);
}

void vc::wrapgl::FreeGLUTContext::switchContext_impl()
{
    pp->switchContext();
}
