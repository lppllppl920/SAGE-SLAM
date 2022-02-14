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
 * Windowless Context.
 * ****************************************************************************
 */

#include <VisionCore/WrapGL/WrapGLContext.hpp>

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <X11/X.h>
#include <X11/Xlib.h>
#define __gl_h_
#include <GL/glx.h>

typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
typedef Bool (*glXMakeContextCurrentARBProc)(Display*, GLXDrawable, GLXDrawable, GLXContext);
static glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;
static glXMakeContextCurrentARBProc glXMakeContextCurrentARB = 0;

struct vc::wrapgl::WindowlessContext::Pimpl
{
    Pimpl(glbinding::ContextHandle& h) : hndl(h), should_quit(false)
    {
        
    }
    
    ~Pimpl()
    {
        
    }
    
    bool create(const std::string& window_title, std::size_t w, std::size_t h, bool double_buff)
    {
        static int visual_attribs[] = 
        {
            None
        };
        int context_attribs[] = 
        {
            GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
            GLX_CONTEXT_MINOR_VERSION_ARB, 5,
            GLX_CONTEXT_FLAGS_ARB, GLX_CONTEXT_DEBUG_BIT_ARB,
            GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
            None
        };
        
        int fbcount = 0;
        
        // open display 
        if(!(display = XOpenDisplay(0)))
        {
            return false;
        }
        
        // get framebuffer configs, any is usable (might want to add proper attribs) 
        if ( !(fbc = glXChooseFBConfig(display, DefaultScreen(display), visual_attribs, &fbcount) ) )
        {
            return false;
        }
        
        // get the required extensions 
        glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)glXGetProcAddressARB( (const GLubyte *) "glXCreateContextAttribsARB");
        glXMakeContextCurrentARB = (glXMakeContextCurrentARBProc)glXGetProcAddressARB( (const GLubyte *) "glXMakeContextCurrent");
        if(!(glXCreateContextAttribsARB && glXMakeContextCurrentARB))
        {
            XFree(fbc);
            return false;
        }
        
        // create a context using glXCreateContextAttribsARB 
        if(!(ctx = glXCreateContextAttribsARB(display, fbc[0], 0, True, context_attribs)))
        {
            XFree(fbc);
            return false;
        }
        
        // create temporary pbuffer 
        int pbuffer_attribs[] = 
        {
            GLX_PBUFFER_WIDTH, (int)w,
            GLX_PBUFFER_HEIGHT, (int)h,
            None
        };
        
        pbuf = glXCreatePbuffer(display, fbc[0], pbuffer_attribs);
        
        XFree(fbc);
        XSync(display, False);
        
        // try to make it the current context 
        if(!glXMakeContextCurrent(display, pbuf, pbuf, ctx))
        {
            // some drivers does not support context without default framebuffer, so fallback on using the default window.
            if(!glXMakeContextCurrent(display, DefaultRootWindow(display), DefaultRootWindow(display), ctx))
            {
                return false;
            }
        }
        
        hndl = (glbinding::ContextHandle)display;
        
        return hndl != 0;
    }
    
    void finishFrame()
    {
        glXSwapBuffers( display , pbuf );
    }
    
    void shutdown()
    {
        glXMakeCurrent( display, 0, 0 );
        glXDestroyContext( display, ctx );

        XCloseDisplay( display );
    }
    
    bool shouldQuit()
    {
        return should_quit;
    }
    
    void switchContext()
    {
        
    }
    
    glbinding::ContextHandle& hndl;
    Display *display = 0;
    GLXFBConfig* fbc = 0;
    GLXContext ctx = 0;
    GLXPbuffer pbuf;
    bool should_quit;
};

vc::wrapgl::WindowlessContext::WindowlessContext() : pp(new Pimpl(hndl))
{
    
}

vc::wrapgl::WindowlessContext::~WindowlessContext()
{
    
}

void vc::wrapgl::WindowlessContext::finishFrame()
{
    pp->finishFrame();
}

void vc::wrapgl::WindowlessContext::shutdown()
{
    pp->shutdown();
}

bool vc::wrapgl::WindowlessContext::shouldQuit()
{
    return pp->shouldQuit();
}

bool vc::wrapgl::WindowlessContext::create_impl(const std::string& window_title, std::size_t w, std::size_t h, bool double_buff)
{
    return pp->create(window_title, w, h, double_buff);
}

void vc::wrapgl::WindowlessContext::switchContext_impl()
{
    pp->switchContext();
}
