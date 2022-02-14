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
 * KinectOne Driver.
 * ****************************************************************************
 */

#ifndef KINECTONE_DRIVER_HPP
#define KINECTONE_DRIVER_HPP

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
 * KinectOne Driver.
 * @todo
 */
class KinectOne : public CameraDriverBase
{
public:
    struct KOAPIPimpl;
    
    KinectOne();
    virtual ~KinectOne();
    
    void open(unsigned int idx = 0, bool depth = true, bool rgb = true, bool registered = true);
    void close();
    
    void start();
    void stop();

    std::size_t getRGBWidth() const;
    std::size_t getRGBHeight() const;
    EPixelFormat getRGBPixelFormat() const;
    
    std::size_t getDepthWidth() const;
    std::size_t getDepthHeight() const;
    EPixelFormat getDepthPixelFormat() const;
    
    void getRegisteredIntrinsics(float& fx, float& fy, float& u0, float& v0) const;
    
#ifdef CAMERA_DRIVERS_HAVE_CAMERA_MODELS
    void getRegistereIntrinsics(cammod::PinholeDisparity<float>& cam) const;
#endif // CAMERA_DRIVERS_HAVE_CAMERA_MODELS
private:
    void image_release(void* img);
    void image_release_nothing(void* img);
    
    virtual bool isOpenedImpl() const;
    virtual bool isStartedImpl() const { return is_running; }
    virtual bool captureFrameImpl(FrameBuffer* cf1, FrameBuffer* cf2, FrameBuffer* cf3, FrameBuffer* cf4, int64_t timeout = 0);   
    
    std::unique_ptr<KOAPIPimpl> m_pimpl;
    bool is_running;
};

}

}

#endif // KINECTONE_DRIVER_HPP
