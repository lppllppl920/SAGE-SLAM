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
 * Video4Linux Driver.
 * ****************************************************************************
 */

#ifndef V4L_DRIVER_HPP
#define V4L_DRIVER_HPP

#include <stdint.h>
#include <stddef.h>
#include <stdexcept>
#include <cstring>
#include <string>
#include <sstream>
#include <memory>
#include <chrono>
#include <functional>

#include <CameraDrivers.hpp>

namespace drivers
{
    
    
namespace camera
{

/**
 * V4L Camera Driver.
 * @todo
 */
class V4L: public CameraDriverBase
{
public:
    struct V4LAPIPimpl;
    
    class V4LException : public std::exception
    {
    public:
        V4LException(const char* file, int line);
        
        virtual ~V4LException() throw() { }
        
        virtual const char* what() const throw()
        {
            return errormsg.c_str();
        }
    private:
        std::string errormsg;
    };
    
    V4L();
    virtual ~V4L();
    
    void open(const std::string& path);
    void close();
    
    void start();
    void stop();
    
    void setModeAndFramerate(EVideoMode vmode, EFrameRate framerate);
    void setCustomMode(EPixelFormat pixfmt, unsigned int width, unsigned int height, unsigned int offset_x = 0, unsigned int offset_y = 0, uint16_t format7mode = 0);
    
    bool getFeaturePower(EFeature fidx);
    void setFeaturePower(EFeature fidx, bool b);
    bool getFeatureAuto(EFeature fidx);
    void setFeatureAuto(EFeature fidx, bool b);
    uint32_t getFeatureValue(EFeature fidx);
    float getFeatureValueAbs(EFeature fidx);
    uint32_t getFeatureMin(EFeature fidx);
    uint32_t getFeatureMax(EFeature fidx);
    void setFeatureValue(EFeature fidx, uint32_t val);
    void setFeatureValueAbs(EFeature fidx, float val);
    void setFeature(EFeature fidx, bool power, bool automatic, uint32_t val);
    
    std::size_t getWidth() const { return m_width; }
    std::size_t getHeight() const { return m_height; }
    EPixelFormat getPixelFormat() const { return m_pixfmt; }

private:
    void image_release(void* img);
    
    virtual bool isOpenedImpl() const;
    virtual bool isStartedImpl() const { return is_running; }
    virtual bool captureFrameImpl(FrameBuffer* cf1, FrameBuffer* cf2, FrameBuffer* cf3, FrameBuffer* cf4, int64_t timeout = 0);   
    
    std::unique_ptr<V4LAPIPimpl> m_pimpl;
    std::size_t m_width, m_height;
    EPixelFormat m_pixfmt;
    bool is_running;
};

}

}

#endif // V4L_DRIVER_HPP
