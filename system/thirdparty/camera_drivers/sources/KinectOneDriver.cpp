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

#include <atomic>
#include <KinectOneDriver.hpp>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>

struct drivers::camera::KinectOne::KOAPIPimpl
{
    KOAPIPimpl() : 
        rgb_width(1920), 
        rgb_height(1080), 
        depth_width(512), 
        depth_height(424), 
        dev(nullptr), listener(nullptr), registration(nullptr), pipeline(nullptr),
        frame_types(libfreenect2::Frame::Color | libfreenect2::Frame::Depth),
        undistorted(512, 424, 4),
        registered(512, 424, 4),
        should_register(false)
    {

    }
    
    std::size_t rgb_width, rgb_height;
    std::size_t depth_width, depth_height;
    
    libfreenect2::Freenect2 freenect2;
    std::unique_ptr<libfreenect2::Freenect2Device> dev;
    std::unique_ptr<libfreenect2::SyncMultiFrameListener> listener;
    std::unique_ptr<libfreenect2::Registration> registration;
    std::unique_ptr<libfreenect2::PacketPipeline> pipeline;
    
    unsigned int frame_types;
    libfreenect2::Frame undistorted, registered;
    bool should_register;
    
    void setRegistrationMode(bool on)
    {
        if(on)
        {
            rgb_width = 512;
            rgb_height = 424;
        }
        else
        {
            rgb_width = 1920;
            rgb_height = 1080;
        }
        
        should_register = on;
    }
};

struct FrameTuple
{
    FrameTuple() : count(2)  { }
    libfreenect2::FrameMap frames;
    std::atomic<unsigned int> count;
};

drivers::camera::KinectOne::KinectOne() : CameraDriverBase(), m_pimpl(new KOAPIPimpl()), is_running(false)
{
    
}

drivers::camera::KinectOne::~KinectOne()
{
    if(isOpened()) 
    { 
        close();
    }
}

void drivers::camera::KinectOne::image_release(void* img)
{
    FrameTuple* ft = static_cast<FrameTuple*>(img);
    ft->count--;
    if(ft->count == 0)
    {
        if(m_pimpl->listener.get() != nullptr)
        {
            m_pimpl->listener->release(ft->frames);
        }
        delete ft;
    }
}

void drivers::camera::KinectOne::image_release_nothing(void* img)
{
    
}

void drivers::camera::KinectOne::open(unsigned int idx, bool depth, bool rgb, bool registered)
{
    if(m_pimpl->freenect2.enumerateDevices() == 0)
    {
        throw std::runtime_error("No devices");
    }
    
    m_pimpl->setRegistrationMode(registered);
    m_pimpl->pipeline = std::unique_ptr<libfreenect2::PacketPipeline>(new libfreenect2::OpenGLPacketPipeline());
    m_pimpl->dev = std::unique_ptr<libfreenect2::Freenect2Device>(m_pimpl->freenect2.openDevice(idx, m_pimpl->pipeline.get())); // TODO FIXME
    //m_pimpl->dev = std::unique_ptr<libfreenect2::Freenect2Device>(m_pimpl->freenect2.openDefaultDevice()); // TODO FIXME
    
    if(m_pimpl->dev.get() != nullptr)
    {
        if(depth)
        {
            m_pimpl->frame_types |= libfreenect2::Frame::Depth;
        }
        else
        {
            m_pimpl->frame_types &= ~libfreenect2::Frame::Depth;
        }
        
        if(rgb)
        {
            m_pimpl->frame_types |= libfreenect2::Frame::Color;
        }
        else
        {
            m_pimpl->frame_types &= ~libfreenect2::Frame::Color;
        }
                
        m_pimpl->listener = std::unique_ptr<libfreenect2::SyncMultiFrameListener>(new libfreenect2::SyncMultiFrameListener(m_pimpl->frame_types));
        m_pimpl->dev->setColorFrameListener(m_pimpl->listener.get());
        m_pimpl->dev->setIrAndDepthFrameListener(m_pimpl->listener.get());
    }
    else
    {
        throw std::runtime_error("Cannot open device");
    }
}

bool drivers::camera::KinectOne::isOpenedImpl() const
{
    return m_pimpl->dev.get() != nullptr;
}

void drivers::camera::KinectOne::close()
{
    if(m_pimpl->dev.get() != nullptr)
    {
        m_pimpl->dev->close();
        m_pimpl->dev.reset();
        m_pimpl->listener.reset();
        m_pimpl->pipeline.reset();
    }
}

void drivers::camera::KinectOne::start()
{
    m_pimpl->dev->start();
    if(m_pimpl->should_register)
    {
        m_pimpl->registration = std::unique_ptr<libfreenect2::Registration>(new libfreenect2::Registration(m_pimpl->dev->getIrCameraParams(), m_pimpl->dev->getColorCameraParams()));
    }
    is_running = true;
}

void drivers::camera::KinectOne::stop()
{
    is_running = false;
    m_pimpl->dev->stop();
    if(m_pimpl->should_register)
    {
        m_pimpl->registration.reset();
    }
}

std::size_t drivers::camera::KinectOne::getRGBWidth() const { return m_pimpl->rgb_width; }
std::size_t drivers::camera::KinectOne::getRGBHeight() const { return m_pimpl->rgb_height; }
drivers::camera::EPixelFormat drivers::camera::KinectOne::getRGBPixelFormat() const { return EPixelFormat::PIXEL_FORMAT_RGBA8; }

std::size_t drivers::camera::KinectOne::getDepthWidth() const { return m_pimpl->depth_width; }
std::size_t drivers::camera::KinectOne::getDepthHeight() const { return m_pimpl->depth_height; }
drivers::camera::EPixelFormat drivers::camera::KinectOne::getDepthPixelFormat() const { return EPixelFormat::PIXEL_FORMAT_DEPTH_F32_M; }

// DEPTH, RGB, Infrared
bool drivers::camera::KinectOne::captureFrameImpl(FrameBuffer* cf1, FrameBuffer* cf2, FrameBuffer* cf3, FrameBuffer* cf4, int64_t timeout)
{
    FrameTuple* ft = new FrameTuple();
    
    bool was_timeout = false;
    
    if(timeout == 0)
    {
        m_pimpl->listener->waitForNewFrame(ft->frames);
    }
    else
    {
        was_timeout = !m_pimpl->listener->waitForNewFrame(ft->frames, timeout / 1000000);
    }
    
    std::chrono::high_resolution_clock::time_point tp = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds d = tp.time_since_epoch();
    
    if(was_timeout)
    {
        delete ft;
        return false;
    }
    
    libfreenect2::Frame *rgb = ft->frames[libfreenect2::Frame::Color];
    libfreenect2::Frame *depth = ft->frames[libfreenect2::Frame::Depth];
    
    if(!m_pimpl->should_register)
    {
        cf1->create(ft, std::bind(&drivers::camera::KinectOne::image_release, this, std::placeholders::_1), depth->data, depth->width, depth->height, EPixelFormat::PIXEL_FORMAT_DEPTH_F32_M);
        cf1->setFrameCounter(depth->sequence);
        cf1->setTimeStamp(depth->timestamp * 10 * 1000000);
        cf1->setPCTimeStamp(d.count());
        
        cf2->create(ft, std::bind(&drivers::camera::KinectOne::image_release, this, std::placeholders::_1), rgb->data, rgb->width, rgb->height, EPixelFormat::PIXEL_FORMAT_RGBA8);
        cf2->setFrameCounter(rgb->sequence);
        cf2->setTimeStamp(rgb->timestamp * 10 * 1000000);
        cf2->setPCTimeStamp(d.count());
    }
    else
    {
        m_pimpl->registration->apply(rgb, depth, &m_pimpl->undistorted, &m_pimpl->registered);
        
        cf1->create(&m_pimpl->undistorted, std::bind(&drivers::camera::KinectOne::image_release_nothing, this, std::placeholders::_1), 
                    m_pimpl->undistorted.data, m_pimpl->undistorted.width, m_pimpl->undistorted.height, EPixelFormat::PIXEL_FORMAT_DEPTH_F32_M);
        cf1->setFrameCounter(depth->sequence);
        cf1->setTimeStamp(depth->timestamp * 10 * 1000000);
        cf1->setPCTimeStamp(d.count());
        
        cf2->create(&m_pimpl->registered, std::bind(&drivers::camera::KinectOne::image_release_nothing, this, std::placeholders::_1), 
                    m_pimpl->registered.data, m_pimpl->registered.width, m_pimpl->registered.height, EPixelFormat::PIXEL_FORMAT_RGBA8);
        cf2->setFrameCounter(rgb->sequence);
        cf2->setTimeStamp(rgb->timestamp * 10 * 1000000);
        cf2->setPCTimeStamp(d.count());
        
        m_pimpl->listener->release(ft->frames);
        delete ft;
    }
    
    return true;
}

void drivers::camera::KinectOne::getRegisteredIntrinsics(float& fx, float& fy, float& u0, float& v0) const
{
    libfreenect2::Freenect2Device::IrCameraParams params = m_pimpl->dev->getIrCameraParams();
    fx = params.fx;
    fy = params.fy;
    u0 = params.cx;
    v0 = params.cy;
}

#ifdef CAMERA_DRIVERS_HAVE_CAMERA_MODELS
void drivers::camera::KinectOne::getRegistereIntrinsics(cammod::PinholeDisparity<float>& cam) const
{
    libfreenect2::Freenect2Device::IrCameraParams params = m_pimpl->dev->getIrCameraParams();
    cam = cammod::PinholeDisparity<float>(params.fx, params.fy, params.cx, params.cy, (float)m_pimpl->depth_width, (float)m_pimpl->depth_height);
}
#endif // CAMERA_DRIVERS_HAVE_CAMERA_MODELS
