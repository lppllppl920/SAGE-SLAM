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
 * OpenNI2 Driver.
 * ****************************************************************************
 */

#ifndef OPENNI_DRIVER_HPP
#define OPENNI_DRIVER_HPP

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

#ifdef CAMERA_DRIVERS_HAVE_CAMERA_MODELS
#include <CameraModels/CameraModels.hpp>
#endif // CAMERA_DRIVERS_HAVE_CAMERA_MODELS

namespace openni
{
class Recorder;    
}

namespace drivers
{

namespace camera
{
    
/**
 * OpenNI Driver.
 * 
 * Supports:
 * AUTO_EXPOSURE - automatic exposure
 * WHITE_BALANCE - automatic white balance
 * SHUTTER - manual exposure
 * GAIN - manual gain
 */
class OpenNI : public CameraDriverBase
{
public:
    struct OpenNIAPIPimpl;
    
    class OpenNIException : public std::exception
    {
    public:
        OpenNIException(int status, const char* file, int line);
        
        virtual ~OpenNIException() throw() { }
        
        virtual const char* what() const throw()
        {
            return errormsg.c_str();
        }
    private:
        std::string errormsg;
    };
    
    class OpenNIRecorder
    {
    public:
        OpenNIRecorder();
        ~OpenNIRecorder();
        
        bool create(const char* fn);
        void destroy();
        bool isValid() const;
        
        bool attachRGB(OpenNI& parent);
        bool attachDepth(OpenNI& parent);
        bool attachIR(OpenNI& parent);
        
        bool start();
        void stop();
        bool isStarted() const { return is_recording; }
    private:
        std::unique_ptr<openni::Recorder> recorder;
        bool is_recording;
    };
    
    OpenNI();
    virtual ~OpenNI();
    
    void open(unsigned int idx = 0);
    void close();
    
    void start(bool depth = true, bool rgb = true, bool ir = false);
    void stop();
    
    void setDepthMode(std::size_t w, std::size_t h, unsigned int fps, bool submm = false, bool register_to_color = false);
    void setRGBMode(std::size_t w, std::size_t h, unsigned int fps, EPixelFormat pixfmt);
    void setIRMode(std::size_t w, std::size_t h, unsigned int fps, EPixelFormat pixfmt);
    
    std::size_t getRGBWidth() const;
    std::size_t getRGBHeight() const;
    EPixelFormat getRGBPixelFormat() const;
    
    std::size_t getDepthWidth() const;
    std::size_t getDepthHeight() const;
    EPixelFormat getDepthPixelFormat() const;
    
    std::size_t getIRWidth() const;
    std::size_t getIRHeight() const;
    EPixelFormat getIRPixelFormat() const;
    
    void setDepthCallback(FrameCallback c);
    void unsetDepthCallback();
    void setRGBCallback(FrameCallback c);
    void unsetRGBCallback();
    void setIRCallback(FrameCallback c);
    void unsetIRCallback();
    
    void setSynchronization(bool b);
    float getBaseline() const { return baseline; }
    float getDepthFocalLength();
    float getColorFocalLength();
    float getIRFocalLength();
    
    bool getDepthMirroring();
    bool getRGBMirroring();
    bool getIRMirroring();
    
    void setDepthMirroring(bool b);
    void setRGBMirroring(bool b);
    void setIRMirroring(bool b);
    
    bool getEmitter();
    void setEmitter(bool v);
    
    void getRGBIntrinsics(float& fx, float& fy, float& u0, float& v0) const;
    void getDepthIntrinsics(float& fx, float& fy, float& u0, float& v0) const;
    void getIRIntrinsics(float& fx, float& fy, float& u0, float& v0) const;
    
#ifdef CAMERA_DRIVERS_HAVE_CAMERA_MODELS
    void getRGBIntrinsics(cammod::PinholeDisparity<float>& cam) const;
    void getDepthIntrinsics(cammod::PinholeDisparity<float>& cam) const;
    void getIRIntrinsics(cammod::PinholeDisparity<float>& cam) const;
#endif // CAMERA_DRIVERS_HAVE_CAMERA_MODELS
    
    virtual bool getFeaturePower(EFeature fidx) { return true; }
    virtual bool getFeatureAuto(EFeature fidx);
    virtual void setFeatureAuto(EFeature fidx, bool b);
    virtual uint32_t getFeatureValue(EFeature fidx) { return (uint32_t)getFeatureValueAbs(fidx); }
    virtual float getFeatureValueAbs(EFeature fidx);
    virtual uint32_t getFeatureMin(EFeature fidx);
    virtual uint32_t getFeatureMax(EFeature fidx);
    virtual void setFeatureValue(EFeature fidx, uint32_t val) { setFeatureValueAbs(fidx, (float)val); }
    virtual void setFeatureValueAbs(EFeature fidx, float val);
private:
    friend class OpenNIRecorder;
    void image_release(void* img);
    
    virtual bool isOpenedImpl() const;
    virtual bool isStartedImpl() const { return is_running; }
    virtual bool captureFrameImpl(FrameBuffer* cf1, FrameBuffer* cf2, FrameBuffer* cf3, FrameBuffer* cf4, int64_t timeout = 0);   
    
    std::unique_ptr<OpenNIAPIPimpl> m_pimpl;
    bool is_running;
    bool is_running_depth, is_running_rgb, is_running_ir;
    float baseline;
};

}

}

#endif // OPENNI_DRIVER_HPP
