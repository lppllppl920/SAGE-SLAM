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
 * Context management.
 * ****************************************************************************
 */

#ifndef VISIONCORE_WRAPGL_CONTEXT_HPP
#define VISIONCORE_WRAPGL_CONTEXT_HPP

#include <memory>

#include <VisionCore/WrapGL/WrapGLCommon.hpp>

#ifdef VISIONCORE_HAVE_GLBINDING

namespace vc
{

namespace wrapgl
{
    
class Context
{
public:
    virtual ~Context() { }
    
    bool create(const std::string& window_title, std::size_t w, std::size_t h, bool double_buff = true)
    {
        bool ok = create_impl(window_title, w, h, double_buff);
        
        if(ok)
        {
            glbinding::Binding::initialize(hndl);
        }
        
        return ok;
    }
    
    inline bool isCreated() const { return hndl != nullptr; }
    
    virtual void finishFrame() = 0;
    virtual bool shouldQuit() = 0;
    virtual void shutdown() = 0;
    
    void switchContext()
    {
        switchContext_impl();
        glbinding::Binding::useContext(hndl);
    }
protected:
    virtual bool create_impl(const std::string& window_title, std::size_t w, std::size_t h, bool double_buff = true) = 0;
    virtual void switchContext_impl() = 0;
    glbinding::ContextHandle hndl;
};

class FreeGLUTContext : public Context
{
struct Pimpl;
public:
    FreeGLUTContext();
    FreeGLUTContext(const std::string& window_title, std::size_t w, std::size_t h, bool double_buff = true);
    virtual ~FreeGLUTContext();
    virtual void finishFrame();
    virtual bool shouldQuit();
    virtual void shutdown();
private:
    virtual bool create_impl(const std::string& window_title, std::size_t w, std::size_t h, bool double_buff = true);
    virtual void switchContext_impl();
    
    std::unique_ptr<Pimpl> pp;
};

class X11Context : public Context
{
    struct Pimpl;
public:
    X11Context();
    X11Context(const std::string& window_title, std::size_t w, std::size_t h, bool double_buff = true);
    virtual ~X11Context();
    virtual void finishFrame();
    virtual bool shouldQuit();
    virtual void shutdown();
private:
    virtual bool create_impl(const std::string& window_title, std::size_t w, std::size_t h, bool double_buff = true);
    virtual void switchContext_impl();
    
    std::unique_ptr<Pimpl> pp;
};

class WindowlessContext : public Context
{
    struct Pimpl;
public:
    WindowlessContext();
    virtual ~WindowlessContext();
    virtual void finishFrame();
    virtual bool shouldQuit();
    virtual void shutdown();
private:
    virtual bool create_impl(const std::string& window_title, std::size_t w, std::size_t h, bool double_buff = true);
    virtual void switchContext_impl();
    
    std::unique_ptr<Pimpl> pp;
};
    
}

}

#endif // VISIONCORE_HAVE_GLBINDING

#endif // VISIONCORE_WRAPGL_CONTEXT_HPP
