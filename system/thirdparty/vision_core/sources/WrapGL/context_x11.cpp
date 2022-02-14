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
 * Basic X11 Context.
 * ****************************************************************************
 */

#include <VisionCore/WrapGL/WrapGLContext.hpp>

#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#define __gl_h_
#include <GL/glx.h>

const long EVENT_MASKS = ButtonPressMask|ButtonReleaseMask|StructureNotifyMask|ButtonMotionMask|PointerMotionMask|KeyPressMask|KeyReleaseMask;

#define GLX_CONTEXT_MAJOR_VERSION_ARB       0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB       0x2092
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);

static bool ctxErrorOccurred = false;
static int ctxErrorHandler( Display *dpy, XErrorEvent *ev )
{
    ctxErrorOccurred = true;
    return 0;
}

static bool isExtensionSupported(const char *extList, const char *extension)
{
    const char *start;
    const char *where, *terminator;
    
    /* Extension names should not have spaces. */
    where = strchr(extension, ' ');
    if (where || *extension == '\0')
        return false;
    
    for (start=extList;;) {
        where = strstr(start, extension);
        
        if (!where)
            break;
        
        terminator = where + strlen(extension);
        
        if ( where == start || *(where - 1) == ' ' )
            if ( *terminator == ' ' || *terminator == '\0' )
                return true;
            
            start = terminator;
    }
    
    return false;
}

struct vc::wrapgl::X11Context::Pimpl
{
    Pimpl(glbinding::ContextHandle& h) : hndl(h), should_quit(false)
    {
        
    }
    
    ~Pimpl()
    {
        
    }
    
    bool create(const std::string& window_title, std::size_t w, std::size_t h, bool double_buff)
    {
        int glx_sample_buffers = 1;
        int glx_samples = 4;
        
        display = XOpenDisplay(NULL);
        
        if (!display) 
        {
            throw std::runtime_error("Failed to open X display");
        }
        
        // Desired attributes
        static int visual_attribs[] =
        {
            GLX_X_RENDERABLE    , True,
            GLX_DRAWABLE_TYPE   , GLX_WINDOW_BIT,
            GLX_RENDER_TYPE     , GLX_RGBA_BIT,
            GLX_X_VISUAL_TYPE   , GLX_TRUE_COLOR,
            GLX_RED_SIZE        , 8,
            GLX_GREEN_SIZE      , 8,
            GLX_BLUE_SIZE       , 8,
            GLX_ALPHA_SIZE      , 8,
            GLX_DEPTH_SIZE      , 24,
            GLX_STENCIL_SIZE    , 8,
            GLX_DOUBLEBUFFER    , double_buff ? True : False,
            GLX_SAMPLE_BUFFERS  , glx_sample_buffers,
            GLX_SAMPLES         , glx_sample_buffers > 0 ? glx_samples : 0,
            None
        };
        
        
        int glx_major, glx_minor;
        if ( !glXQueryVersion( display, &glx_major, &glx_minor ) || ( ( glx_major == 1 ) && ( glx_minor < 3 ) ) || ( glx_major < 1 ) )
        {
            // FBConfigs were added in GLX version 1.3.
            throw std::runtime_error("Invalid GLX version. Require GLX >= 1.3");
        }
        
        int fbcount;
        GLXFBConfig* fbc = glXChooseFBConfig(display, DefaultScreen(display), visual_attribs, &fbcount);
        if (!fbc) 
        {
            throw std::runtime_error("Unable to retrieve framebuffer options");
        }
        
        int best_fbc = -1;
        int worst_fbc = -1;
        int best_num_samp = -1;
        int worst_num_samp = 999;
        
        // Enumerate framebuffer options, storing the best and worst that match our attribs
        for (int i=0; i<fbcount; ++i)
        {
            XVisualInfo *vi = glXGetVisualFromFBConfig( display, fbc[i] );
            
            if ( vi )
            {
                int samp_buf, samples;
                glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLE_BUFFERS, &samp_buf );
                glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLES       , &samples  );
                
                if ( (best_fbc < 0) || (samp_buf>0 && samples>best_num_samp) )
                {
                    best_fbc = i, best_num_samp = samples;
                }
                
                if ( (worst_fbc < 0) || (samp_buf>0 && samples<worst_num_samp) )
                {
                    worst_fbc = i, worst_num_samp = samples;
                }
            }
            
            XFree( vi );
        }
        
        // Select the minimum suitable option. The 'best' is often too slow.
        GLXFBConfig bestFbc = fbc[ worst_fbc ];
        XFree( fbc );
        
        // Get a visual
        XVisualInfo *vi = glXGetVisualFromFBConfig( display, bestFbc );
        
        // Create colourmap
        XSetWindowAttributes swa;
        swa.colormap = cmap = XCreateColormap( display, RootWindow( display, vi->screen ), vi->visual, AllocNone );
        swa.background_pixmap = None ;
        swa.border_pixel      = 0;
        swa.event_mask        = StructureNotifyMask;
        
        // Create window
        win = XCreateWindow( display, RootWindow( display, vi->screen ), 0, 0, w, h, 0, vi->depth, InputOutput, vi->visual, CWBorderPixel|CWColormap|CWEventMask, &swa );
        
        XFree( vi );
        
        if ( !win ) 
        {
            throw std::runtime_error("Failed to create window." );
        }
        
        //wmDeleteMessage = XInternAtom(display, "WM_DELETE_WINDOW", False);
        //XSetWMProtocols(display, win, &wmDeleteMessage, 1);
        
        XStoreName( display, win, window_title.c_str() );
        XMapWindow( display, win );
        
        // Request to be notified of these events
        XSelectInput(display, win, EVENT_MASKS );
        
        // Get the default screen's GLX extension list
        const char *glxExts = glXQueryExtensionsString( display, DefaultScreen( display ) );
        
        glXCreateContextAttribsARBProc glXCreateContextAttribsARB =(glXCreateContextAttribsARBProc) glXGetProcAddressARB((const GLubyte *) "glXCreateContextAttribsARB");
        
        // Install an X error handler so the application won't exit if GL 3.0
        // context allocation fails. Handler is global and shared across all threads.
        ctxErrorOccurred = false;
        int (*oldHandler)(Display*, XErrorEvent*) = XSetErrorHandler(&ctxErrorHandler);
        
        if ( isExtensionSupported( glxExts, "GLX_ARB_create_context" ) && glXCreateContextAttribsARB )
        {
            int context_attribs[] = {
                GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
                GLX_CONTEXT_MINOR_VERSION_ARB, 0,
                //GLX_CONTEXT_FLAGS_ARB        , GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
                None
            };
            
            ctx = glXCreateContextAttribsARB( display, bestFbc, 0, True, context_attribs );
            
            // Sync to ensure any errors generated are processed.
            XSync( display, False );
            if ( ctxErrorOccurred || !ctx ) 
            {
                ctxErrorOccurred = false;
                // Fall back to old-style 2.x context. Implementations will return the newest
                // context version compatible with OpenGL versions less than version 3.0.
                context_attribs[1] = 1;  // GLX_CONTEXT_MAJOR_VERSION_ARB = 1
                context_attribs[3] = 0;  // GLX_CONTEXT_MINOR_VERSION_ARB = 0
                ctx = glXCreateContextAttribsARB( display, bestFbc, 0, True, context_attribs );
            }
        } 
        else 
        {
            // Fallback to GLX 1.3 Context
            ctx = glXCreateNewContext( display, bestFbc, GLX_RGBA_TYPE, 0, True );
        }
        
        // Sync to ensure any errors generated are processed.
        XSync( display, False );
        
        // Restore the original error handler
        XSetErrorHandler( oldHandler );
        
        if ( ctxErrorOccurred || !ctx )
        {
            throw std::runtime_error("Failed to create an OpenGL context");
        }
        
        // Verifying that context is a direct context
        if ( ! glXIsDirect ( display, ctx ) ) 
        {
            // Indirect GLX rendering context obtained\n");
        }
        
        glXMakeCurrent( display, win, ctx );
        
        hndl = win;
        
        return hndl != 0;
    }
    
    void finishFrame()
    {
        glXSwapBuffers ( display, win );
        XEvent ev;
        while(XCheckWindowEvent(display,win,EVENT_MASKS,&ev))
        {
            switch(ev.type)
            {
                case ClientMessage:
                    std::cerr << "Got CM" << std::endl;
                    if((Atom)ev.xclient.data.l[0] == wmDeleteMessage) 
                    {
                        std::cerr << "Need to quit" << std::endl;
                        shutdown();
                        should_quit = true;
                    }
                    break;
                case ConfigureNotify:
                    // TODO pangolin::process::Resize(ev.xconfigure.width, ev.xconfigure.height);
                    break;
                case ButtonPress:
                case ButtonRelease:
                {
                    const int button = ev.xbutton.button-1;
                    const int mask = Button1Mask << button;
                    // TODO pangolin::process::Mouse(button,ev.xbutton.state & mask,ev.xbutton.x, ev.xbutton.y);
                    break;
                }
                case MotionNotify:
                    if(ev.xmotion.state & (Button1Mask|Button2Mask|Button3Mask) ) 
                    {
                        // TODO pangolin::process::MouseMotion(ev.xmotion.x, ev.xmotion.y);
                    }
                    else
                    {
                        // TODO pangolin::process::PassiveMouseMotion(ev.xmotion.x, ev.xmotion.y);
                    }
                    break;
                case KeyPress:
                case KeyRelease:
                    int key;
                    char ch;
                    KeySym sym;
                    
                    if( XLookupString(&ev.xkey,&ch,1,&sym,0) == 0) 
                    {
                        switch (sym) 
                        {
#if 0
                            case XK_F1:        key = PANGO_SPECIAL + PANGO_KEY_F1         ; break;
                            case XK_F2:        key = PANGO_SPECIAL + PANGO_KEY_F2         ; break;
                            case XK_F3:        key = PANGO_SPECIAL + PANGO_KEY_F3         ; break;
                            case XK_F4:        key = PANGO_SPECIAL + PANGO_KEY_F4         ; break;
                            case XK_F5:        key = PANGO_SPECIAL + PANGO_KEY_F5         ; break;
                            case XK_F6:        key = PANGO_SPECIAL + PANGO_KEY_F6         ; break;
                            case XK_F7:        key = PANGO_SPECIAL + PANGO_KEY_F7         ; break;
                            case XK_F8:        key = PANGO_SPECIAL + PANGO_KEY_F8         ; break;
                            case XK_F9:        key = PANGO_SPECIAL + PANGO_KEY_F9         ; break;
                            case XK_F10:       key = PANGO_SPECIAL + PANGO_KEY_F10        ; break;
                            case XK_F11:       key = PANGO_SPECIAL + PANGO_KEY_F11        ; break;
                            case XK_F12:       key = PANGO_SPECIAL + PANGO_KEY_F12        ; break;
                            case XK_Left:      key = PANGO_SPECIAL + PANGO_KEY_LEFT       ; break;
                            case XK_Up:        key = PANGO_SPECIAL + PANGO_KEY_UP         ; break;
                            case XK_Right:     key = PANGO_SPECIAL + PANGO_KEY_RIGHT      ; break;
                            case XK_Down:      key = PANGO_SPECIAL + PANGO_KEY_DOWN       ; break;
                            case XK_Page_Up:   key = PANGO_SPECIAL + PANGO_KEY_PAGE_UP    ; break;
                            case XK_Page_Down: key = PANGO_SPECIAL + PANGO_KEY_PAGE_DOWN  ; break;
                            case XK_Home:      key = PANGO_SPECIAL + PANGO_KEY_HOME       ; break;
                            case XK_End:       key = PANGO_SPECIAL + PANGO_KEY_END        ; break;
                            case XK_Insert:    key = PANGO_SPECIAL + PANGO_KEY_INSERT     ; break;
#endif
                            case XK_Shift_L:
                            case XK_Shift_R:
                                key = -1;
                                if(ev.type==KeyPress) 
                                {
                                    // TODO pangolin::context->mouse_state |=  pangolin::KeyModifierShift;
                                }
                                else
                                {
                                    // TODO pangolin::context->mouse_state &= ~pangolin::KeyModifierShift;
                                }
                                break;
                            case XK_Control_L:
                            case XK_Control_R:
                                key = -1;
                                if(ev.type==KeyPress) 
                                {
                                    // TODO pangolin::context->mouse_state |=  pangolin::KeyModifierCtrl;
                                }
                                else
                                {
                                    // TODO pangolin::context->mouse_state &= ~pangolin::KeyModifierCtrl;
                                }
                                break;
                            case XK_Alt_L:
                            case XK_Alt_R:
                                key = -1;
                                if(ev.type == KeyPress) 
                                {
                                    // TODO pangolin::context->mouse_state |=  pangolin::KeyModifierAlt;
                                }
                                else
                                {
                                    // TODO pangolin::context->mouse_state &= ~pangolin::KeyModifierAlt;
                                }
                                break;
                            case XK_Super_L:
                            case XK_Super_R:
                                key = -1;
                                if(ev.type == KeyPress) 
                                {
                                    // TODO pangolin::context->mouse_state |=  pangolin::KeyModifierCmd;
                                }
                                else
                                {
                                    // TODO pangolin::context->mouse_state &= ~pangolin::KeyModifierCmd;
                                }
                                break;
                                
                            default: key = -1; break;
                        }
                    }
                    else
                    {
                        key = ch;
                    }
                    
                    if(key >=0) 
                    {
                        if(ev.type == KeyPress) 
                        {
                            // TODO pangolin::process::Keyboard(key, ev.xkey.x, ev.xkey.y);
                        }
                        else
                        {
                            // TODO pangolin::process::KeyboardUp(key, ev.xkey.x, ev.xkey.y);
                        }
                    }
                    
                    break;
            }
        }
    }
    
    void shutdown()
    {
        glXMakeCurrent( display, 0, 0 );
        glXDestroyContext( display, ctx );
        
        XDestroyWindow( display, win );
        XFreeColormap( display, cmap );
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
    Window win = 0;
    GLXContext ctx = 0;
    Colormap cmap;
    Atom wmDeleteMessage;
    bool should_quit;
};

vc::wrapgl::X11Context::X11Context() : pp(new Pimpl(hndl))
{
    
}

vc::wrapgl::X11Context::X11Context(const std::string& window_title, std::size_t w, std::size_t h, bool double_buff) : pp(new Pimpl(hndl))
{
    create(window_title, w, h, double_buff);
}

vc::wrapgl::X11Context::~X11Context()
{
    
}


void vc::wrapgl::X11Context::finishFrame()
{
    pp->finishFrame();
}

void vc::wrapgl::X11Context::shutdown()
{
    pp->shutdown();
}

bool vc::wrapgl::X11Context::shouldQuit()
{
    return pp->shouldQuit();
}

bool vc::wrapgl::X11Context::create_impl(const std::string& window_title, std::size_t w, std::size_t h, bool double_buff)
{
    return pp->create(window_title, w, h, double_buff);
}

void vc::wrapgl::X11Context::switchContext_impl()
{
    pp->switchContext();
}
