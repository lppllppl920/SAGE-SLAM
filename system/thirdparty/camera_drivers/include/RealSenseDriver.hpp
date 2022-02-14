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
 * Intel RealSense Driver.
 * ****************************************************************************
 */

#ifndef REALSENSE_DRIVER_HPP
#define REALSENSE_DRIVER_HPP

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

namespace drivers
{

namespace camera
{
    
/**
 * RealSense Driver.
 * 
 * Supports:
 * BRIGHTNESS - brightness
 * AUTO_EXPOSURE - automatic exposure
 * SHARPNESS - sharpness
 * WHITE_BALANCE - (auto)white balance
 * HUE - hue
 * SATURATION  -saturation
 * GAMMA - gamma
 * SHUTTER - manual exposure
 * GAIN - gain
 * TEMPERATURE - backlight compensation
 */
class RealSense : public CameraDriverBase
{
public:
    struct RealSenseAPIPimpl;
    
    class RealSenseException : public std::exception
    {
    public:
        RealSenseException(int status, const char* file, int line);
        
        virtual ~RealSenseException() throw() { }
        
        virtual const char* what() const throw()
        {
            return errormsg.c_str();
        }
    private:
        std::string errormsg;
    };
    
    RealSense();
    virtual ~RealSense();
    
    void open(unsigned int idx = 0);
    void close();
    
    void start();
    void stop();
    
    void setDepthMode(std::size_t w, std::size_t h, unsigned int fps, bool aligned_to_color = false, bool aligned_to_rectified_color = false);
    void setRGBMode(std::size_t w, std::size_t h, unsigned int fps, EPixelFormat pixfmt, bool rectified = false, bool aligned_to_depth = false);
    void setIR1Mode(std::size_t w, std::size_t h, unsigned int fps);
    void setIR2Mode(std::size_t w, std::size_t h, unsigned int fps);
    
    void setDepthCallback(FrameCallback c);
    void unsetDepthCallback();
    void setRGBCallback(FrameCallback c);
    void unsetRGBCallback();
    void setIR1Callback(FrameCallback c);
    void unsetIR1Callback();
    void setIR2Callback(FrameCallback c);
    void unsetIR2Callback();
    void setMotionCallback(MotionCallback c);
    void unsetMotionCallback();
    void setEventCallback(EventCallback c);
    void unsetEventCallback();

    std::size_t getRGBWidth() const;
    std::size_t getRGBHeight() const;
    EPixelFormat getRGBPixelFormat() const;
    
    std::size_t getDepthWidth() const;
    std::size_t getDepthHeight() const;
    EPixelFormat getDepthPixelFormat() const;
    
    std::size_t getIR1Width() const;
    std::size_t getIR1Height() const;
    EPixelFormat getIR1PixelFormat() const;
    
    std::size_t getIR2Width() const;
    std::size_t getIR2Height() const;
    EPixelFormat getIR2PixelFormat() const;
    
    float getDepthScale() const;
    
    void getRGBIntrinsics(float& fx, float& fy, float& u0, float& v0, std::array<float,5>* dist = nullptr) const;
    void getDepthIntrinsics(float& fx, float& fy, float& u0, float& v0, std::array<float,5>* dist = nullptr) const;
    void getIR1Intrinsics(float& fx, float& fy, float& u0, float& v0, std::array<float,5>* dist = nullptr) const;
    void getIR2Intrinsics(float& fx, float& fy, float& u0, float& v0, std::array<float,5>* dist = nullptr) const;
    
    void getExtrinsicsDepthToColor(float& tx, float& ty, float&tz, std::array<float,9>* rotMat = nullptr) const;
    void getExtrinsicsColorToDepth(float& tx, float& ty, float&tz, std::array<float,9>* rotMat = nullptr) const;
    
#ifdef CAMERA_DRIVERS_HAVE_CAMERA_MODELS
    void getRGBIntrinsics(cammod::PinholeDisparity<float>& cam) const;
    void getRGBIntrinsics(cammod::PinholeDisparityBrownConrady<float>& cam) const;
    void getDepthIntrinsics(cammod::PinholeDisparity<float>& cam) const;
    void getDepthIntrinsics(cammod::PinholeDisparityBrownConrady<float>& cam) const;
    void getIR1Intrinsics(cammod::PinholeDisparity<float>& cam) const;
    void getIR1Intrinsics(cammod::PinholeDisparityBrownConrady<float>& cam) const;
    void getIR2Intrinsics(cammod::PinholeDisparity<float>& cam) const;
    void getIR2Intrinsics(cammod::PinholeDisparityBrownConrady<float>& cam) const;
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
    virtual bool isOpenedImpl() const;
    virtual bool isStartedImpl() const { return is_running; }
    virtual bool captureFrameImpl(FrameBuffer* cf1, FrameBuffer* cf2, FrameBuffer* cf3, FrameBuffer* cf4, int64_t timeout = 0);   
    
    std::unique_ptr<RealSenseAPIPimpl> m_pimpl;
    bool is_running;
};

}

}

#endif // REALSENSE_DRIVER_HPP
