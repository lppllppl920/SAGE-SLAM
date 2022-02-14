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

#include <atomic>
#include <cmath>

#include <OpenNIDriver.hpp>

#include <OpenNI.h>
#include <PS1080.h>

#include <boost/format.hpp>

drivers::camera::OpenNI::OpenNIException::OpenNIException(int status, const char* file, int line)
{
    std::string msg;
    
    switch(status)
    {
        case openni::STATUS_ERROR: msg = "Error"; break;
        case openni::STATUS_NOT_IMPLEMENTED: msg = "Not implemented"; break;
        case openni::STATUS_NOT_SUPPORTED: msg = "Not supported"; break;
        case openni::STATUS_BAD_PARAMETER: msg = "Bad parameter"; break;
        case openni::STATUS_OUT_OF_FLOW: msg = "Out of flow"; break;
        case openni::STATUS_NO_DEVICE: msg = "No device"; break;
        case openni::STATUS_TIME_OUT: msg = "Timeout"; break;
        default:
            msg = "Unknown"; 
    }
    
    std::stringstream ss;
    ss << "ONI Error " << msg << "(" << openni::OpenNI::getExtendedError() << ") , " << " at " << file << ":" << line;
    errormsg = ss.str();
}

#define ONI_CHECK_ERROR(err_code) { if(err_code != openni::STATUS_OK) { throw OpenNIException(err_code, __FILE__, __LINE__); } }

static drivers::camera::EPixelFormat OpenNIToPixelFormat(const openni::VideoFrameRef& frame)
{
    drivers::camera::EPixelFormat pixfmt = drivers::camera::EPixelFormat::PIXEL_FORMAT_UNSUPPORTED;
    
    switch(frame.getVideoMode().getPixelFormat())
    {
        case openni::PIXEL_FORMAT_DEPTH_1_MM:
            pixfmt = drivers::camera::EPixelFormat::PIXEL_FORMAT_DEPTH_U16_1MM;
            break;
        case openni::PIXEL_FORMAT_DEPTH_100_UM:
            pixfmt = drivers::camera::EPixelFormat::PIXEL_FORMAT_DEPTH_U16_100UM;
            break;
        case openni::PIXEL_FORMAT_RGB888:
            pixfmt = drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB8;
            break;
        case openni::PIXEL_FORMAT_GRAY8:
            pixfmt = drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO8;
            break;
        case openni::PIXEL_FORMAT_GRAY16:
            pixfmt = drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO16;
            break;
            
        case openni::PIXEL_FORMAT_YUV422:
        case openni::PIXEL_FORMAT_SHIFT_9_2:
        case openni::PIXEL_FORMAT_SHIFT_9_3:
        case openni::PIXEL_FORMAT_JPEG:
        case openni::PIXEL_FORMAT_YUYV:
        default:
            pixfmt = drivers::camera::EPixelFormat::PIXEL_FORMAT_UNSUPPORTED;
    }
    
    return pixfmt;
}

class FrameToCallbackListener : public openni::VideoStream::NewFrameListener
{
public:
    FrameToCallbackListener()
    {
        
    }
    
    virtual ~FrameToCallbackListener()
    {
        
    }
    
    drivers::camera::FrameCallback Callback;
    
    virtual void onNewFrame(openni::VideoStream& vs)
    {
        openni::VideoFrameRef frame;
        
        vs.readFrame(&frame);
        
        std::chrono::high_resolution_clock::time_point tp = std::chrono::high_resolution_clock::now();
        std::chrono::nanoseconds d = tp.time_since_epoch();
        
        drivers::camera::EPixelFormat pixfmt = OpenNIToPixelFormat(frame);
        
        drivers::camera::FrameBuffer fb((uint8_t*)frame.getData(), frame.getWidth(), frame.getHeight(), pixfmt, frame.getStrideInBytes());
        
        fb.setFrameCounter(frame.getFrameIndex());
        fb.setTimeStamp(frame.getTimestamp());
        fb.setPCTimeStamp(d.count());
        
        if(Callback)
        {
            Callback(fb);
        }
    }
};

struct drivers::camera::OpenNI::OpenNIAPIPimpl
{
    OpenNIAPIPimpl() 
    {
        streams = new openni::VideoStream*[3];
        streams[0] = nullptr;
        streams[1] = nullptr;
        streams[2] = nullptr;
    }
    
    ~OpenNIAPIPimpl()
    {
        delete[] streams;
    }
    
    void frame_release(void* img)
    {
        openni::VideoFrameRef* frame = static_cast<openni::VideoFrameRef*>(img);
        delete frame;
    }
    
    openni::Array<openni::DeviceInfo> devlist;
    openni::Device dev;
    openni::VideoMode vm_depth;
    openni::VideoMode vm_rgb;
    openni::VideoMode vm_ir;
    openni::VideoStream vs_depth;
    openni::VideoStream vs_rgb;
    openni::VideoStream vs_ir;
    FrameToCallbackListener listen_depth;
    FrameToCallbackListener listen_rgb;
    FrameToCallbackListener listen_ir;
    openni::VideoStream** streams;
};


drivers::camera::OpenNI::OpenNI() : CameraDriverBase(), m_pimpl(new OpenNIAPIPimpl()), is_running(false), is_running_depth(false), is_running_rgb(false), is_running_ir(false)
{
    openni::Status status = openni::OpenNI::initialize();
    ONI_CHECK_ERROR(status);
    
    openni::OpenNI::enumerateDevices(&m_pimpl->devlist);
}
   
drivers::camera::OpenNI::~OpenNI()
{
    close();
    openni::OpenNI::shutdown();
}

void drivers::camera::OpenNI::image_release(void* img)
{
    // TODO FIXME
}

static void __attribute__((__unused__)) printVideoMode(int i, const openni::VideoMode& vm)
{
    std::string pixfmt;
    
    switch(vm.getPixelFormat())
    {
        case openni::PIXEL_FORMAT_DEPTH_1_MM: pixfmt = "DEPTH_1MM"; break;
        case openni::PIXEL_FORMAT_DEPTH_100_UM: pixfmt = "DEPTH_100UM"; break;
        case openni::PIXEL_FORMAT_SHIFT_9_2: pixfmt = "SHIFT_92"; break;
        case openni::PIXEL_FORMAT_SHIFT_9_3: pixfmt = "SHIFT_93"; break;
        case openni::PIXEL_FORMAT_RGB888: pixfmt = "RGB888"; break;
        case openni::PIXEL_FORMAT_YUV422: pixfmt = "YUV422"; break;
        case openni::PIXEL_FORMAT_GRAY8: pixfmt = "GRAY8"; break;
        case openni::PIXEL_FORMAT_GRAY16: pixfmt = "GRAY16"; break;
        case openni::PIXEL_FORMAT_JPEG: pixfmt = "JPEG"; break;
        case openni::PIXEL_FORMAT_YUYV: pixfmt = "YUYV"; break;
        default: pixfmt = "UNKNOWN";
    }
}

void drivers::camera::OpenNI::open(unsigned int idx)
{
    if(isOpened()) { return; }
    
    if((int)idx >= m_pimpl->devlist.getSize())
    {
        throw std::runtime_error("No such device");
    }
    
    const auto& di = m_pimpl->devlist[idx];
    
    openni::Status status = m_pimpl->dev.open(di.getUri());
    ONI_CHECK_ERROR(status);
    
    status = m_pimpl->vs_depth.create(m_pimpl->dev, openni::SENSOR_DEPTH);
    ONI_CHECK_ERROR(status);
    
    status = m_pimpl->vs_rgb.create(m_pimpl->dev, openni::SENSOR_COLOR);
    ONI_CHECK_ERROR(status);
    
    status = m_pimpl->vs_ir.create(m_pimpl->dev, openni::SENSOR_IR);
    ONI_CHECK_ERROR(status);
    
    std::stringstream ss;
    ss << "0x" << std::hex << m_pimpl->dev.getDeviceInfo().getUsbVendorId() << ":0x" << m_pimpl->dev.getDeviceInfo().getUsbProductId();
    m_cinfo.SerialNumber = ss.str();
    m_cinfo.InterfaceType = CameraInfo::CameraInterface::USB;
    m_cinfo.IsColorCamera = true;
    m_cinfo.ModelName = std::string(m_pimpl->dev.getDeviceInfo().getName());
    m_cinfo.VendorName = std::string(m_pimpl->dev.getDeviceInfo().getVendor());
    m_cinfo.SensorInfo = std::string("RGBD");
    m_cinfo.SensorResolution = std::string("");
    m_cinfo.DriverName = std::string("OpenNI2");
    ss.str("");
    ss << openni::OpenNI::getVersion().major << "." << openni::OpenNI::getVersion().minor << "." << openni::OpenNI::getVersion().maintenance;
    m_cinfo.FirmwareVersion = ss.str();
    ss.str("");
    ss << openni::OpenNI::getVersion().build;
    m_cinfo.FirmwareBuildTime = ss.str();
    
    if(m_pimpl->vs_depth.isPropertySupported(XN_STREAM_PROPERTY_EMITTER_DCMOS_DISTANCE))
    {
        double dbaseline;
        status = m_pimpl->vs_depth.getProperty(XN_STREAM_PROPERTY_EMITTER_DCMOS_DISTANCE, &dbaseline);
        ONI_CHECK_ERROR(status);
        baseline = static_cast<float>(dbaseline * 0.01f);  // baseline from cm -> meters
    }
    
}

bool drivers::camera::OpenNI::isOpenedImpl() const
{
    return m_pimpl->dev.isValid();
}

void drivers::camera::OpenNI::close()
{
    if(!isOpened()) { return; }
    
    m_pimpl->vs_depth.destroy();
    m_pimpl->vs_rgb.destroy();
    m_pimpl->vs_ir.destroy();
    m_pimpl->dev.close();
}

void drivers::camera::OpenNI::start(bool depth, bool rgb, bool ir)
{
    if(isStarted()) { return; }
    
    openni::Status status;
    
    if(depth)
    {
        status = m_pimpl->vs_depth.setMirroringEnabled(false);
        ONI_CHECK_ERROR(status);
        status = m_pimpl->vs_depth.start();
        ONI_CHECK_ERROR(status);
    }
    
    is_running_depth = depth;
    
    if(rgb)
    {
        status = m_pimpl->vs_rgb.setMirroringEnabled(false);
        ONI_CHECK_ERROR(status);
        status = m_pimpl->vs_rgb.start();
        ONI_CHECK_ERROR(status);
    }
    
    is_running_rgb = rgb;
    
    if(ir)
    {
        status = m_pimpl->vs_ir.setMirroringEnabled(false);
        ONI_CHECK_ERROR(status);
        status = m_pimpl->vs_ir.start();
        ONI_CHECK_ERROR(status);
    }
    
    is_running_ir = ir;
    
    is_running = true;
}

void drivers::camera::OpenNI::stop()
{
    if(!isStarted()) { return; }
    
    m_pimpl->vs_depth.stop();
    m_pimpl->vs_rgb.stop();
    m_pimpl->vs_ir.stop();
    
    is_running_depth = is_running_rgb = is_running_ir = false;
    is_running = false;
}

void drivers::camera::OpenNI::setDepthMode(std::size_t w, std::size_t h, unsigned int fps, bool submm, bool register_to_color)
{
    m_pimpl->vm_depth.setFps(fps);
    m_pimpl->vm_depth.setResolution(w,h);
    if(submm)
    {
        m_pimpl->vm_depth.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_100_UM);
    }
    else
    {
        m_pimpl->vm_depth.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);
    }
    
    if(register_to_color)
    {
        m_pimpl->dev.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
    }
    
    openni::Status status = m_pimpl->vs_depth.setVideoMode(m_pimpl->vm_depth);
    ONI_CHECK_ERROR(status);
}

void drivers::camera::OpenNI::setRGBMode(std::size_t w, std::size_t h, unsigned int fps, drivers::camera::EPixelFormat pixfmt)
{
    m_pimpl->vm_rgb.setFps(fps);
    m_pimpl->vm_rgb.setResolution(w,h);
    switch(pixfmt)
    {
        case EPixelFormat::PIXEL_FORMAT_MONO8:
            m_pimpl->vm_rgb.setPixelFormat(openni::PIXEL_FORMAT_GRAY8);
            break;
        case EPixelFormat::PIXEL_FORMAT_MONO16:
            m_pimpl->vm_rgb.setPixelFormat(openni::PIXEL_FORMAT_GRAY16);
            break;
        case EPixelFormat::PIXEL_FORMAT_RGB8:
            m_pimpl->vm_rgb.setPixelFormat(openni::PIXEL_FORMAT_RGB888);
            break;
        default:
            throw std::runtime_error("Unsupported pixel format");
    }
    
    openni::Status status = m_pimpl->vs_rgb.setVideoMode(m_pimpl->vm_rgb);
    ONI_CHECK_ERROR(status);
}

void drivers::camera::OpenNI::setIRMode(std::size_t w, std::size_t h, unsigned int fps, drivers::camera::EPixelFormat pixfmt)
{
    m_pimpl->vm_ir.setFps(fps);
    m_pimpl->vm_ir.setResolution(w,h);
    switch(pixfmt)
    {
        case EPixelFormat::PIXEL_FORMAT_MONO8:
            m_pimpl->vm_ir.setPixelFormat(openni::PIXEL_FORMAT_GRAY8);
            break;
        case EPixelFormat::PIXEL_FORMAT_MONO16:
            m_pimpl->vm_ir.setPixelFormat(openni::PIXEL_FORMAT_GRAY16);
            break;
        case EPixelFormat::PIXEL_FORMAT_RGB8:
            m_pimpl->vm_ir.setPixelFormat(openni::PIXEL_FORMAT_RGB888);
            break;
        default:
            throw std::runtime_error("Unsupported pixel format");
    }
    
    openni::Status status = m_pimpl->vs_ir.setVideoMode(m_pimpl->vm_ir);
    ONI_CHECK_ERROR(status);
}

std::size_t drivers::camera::OpenNI::getRGBWidth() const
{
    openni::VideoMode vm = m_pimpl->vs_rgb.getVideoMode();
    return vm.getResolutionX();
}

std::size_t drivers::camera::OpenNI::getRGBHeight() const
{
    openni::VideoMode vm = m_pimpl->vs_rgb.getVideoMode();
    return vm.getResolutionY();
}

drivers::camera::EPixelFormat drivers::camera::OpenNI::getRGBPixelFormat() const
{
    openni::VideoMode vm = m_pimpl->vs_rgb.getVideoMode();
    switch(vm.getPixelFormat())
    {
        case openni::PIXEL_FORMAT_GRAY8: return EPixelFormat::PIXEL_FORMAT_MONO8;
        case openni::PIXEL_FORMAT_GRAY16: return EPixelFormat::PIXEL_FORMAT_MONO16;
        case openni::PIXEL_FORMAT_RGB888: return EPixelFormat::PIXEL_FORMAT_RGB8;
        default: return EPixelFormat::PIXEL_FORMAT_UNSUPPORTED;
    }
}

std::size_t drivers::camera::OpenNI::getDepthWidth() const
{
    openni::VideoMode vm = m_pimpl->vs_depth.getVideoMode();
    return vm.getResolutionX();
}

std::size_t drivers::camera::OpenNI::getDepthHeight() const
{
    openni::VideoMode vm = m_pimpl->vs_depth.getVideoMode();
    return vm.getResolutionY();
}

drivers::camera::EPixelFormat drivers::camera::OpenNI::getDepthPixelFormat() const
{
    openni::VideoMode vm = m_pimpl->vs_depth.getVideoMode();
    switch(vm.getPixelFormat())
    {
        case openni::PIXEL_FORMAT_DEPTH_1_MM: return EPixelFormat::PIXEL_FORMAT_DEPTH_U16_1MM;
        case openni::PIXEL_FORMAT_DEPTH_100_UM: return EPixelFormat::PIXEL_FORMAT_DEPTH_U16_100UM;
        default: return EPixelFormat::PIXEL_FORMAT_UNSUPPORTED;
    }
}

std::size_t drivers::camera::OpenNI::getIRWidth() const
{
    openni::VideoMode vm = m_pimpl->vs_ir.getVideoMode();
    return vm.getResolutionX();
}

std::size_t drivers::camera::OpenNI::getIRHeight() const
{
    openni::VideoMode vm = m_pimpl->vs_ir.getVideoMode();
    return vm.getResolutionY();
}

drivers::camera::EPixelFormat drivers::camera::OpenNI::getIRPixelFormat() const
{
    return EPixelFormat::PIXEL_FORMAT_MONO16;
}

void drivers::camera::OpenNI::setDepthCallback(drivers::camera::FrameCallback c)
{
    m_pimpl->listen_depth.Callback = c;
    openni::Status status = m_pimpl->vs_depth.addNewFrameListener(&m_pimpl->listen_depth);
    ONI_CHECK_ERROR(status);
}

void drivers::camera::OpenNI::unsetDepthCallback()
{
    m_pimpl->vs_depth.removeNewFrameListener(&m_pimpl->listen_depth);
}

void drivers::camera::OpenNI::setRGBCallback(drivers::camera::FrameCallback c)
{
    m_pimpl->listen_rgb.Callback = c;
    openni::Status status = m_pimpl->vs_rgb.addNewFrameListener(&m_pimpl->listen_rgb);
    ONI_CHECK_ERROR(status);
}

void drivers::camera::OpenNI::unsetRGBCallback()
{
    m_pimpl->vs_rgb.removeNewFrameListener(&m_pimpl->listen_rgb);
}

void drivers::camera::OpenNI::setIRCallback(drivers::camera::FrameCallback c)
{
    m_pimpl->listen_ir.Callback = c;
    openni::Status status = m_pimpl->vs_ir.addNewFrameListener(&m_pimpl->listen_ir);
    ONI_CHECK_ERROR(status);
}

void drivers::camera::OpenNI::unsetIRCallback()
{
    m_pimpl->vs_ir.removeNewFrameListener(&m_pimpl->listen_ir);
}

void drivers::camera::OpenNI::setSynchronization(bool b)
{
    openni::Status status = m_pimpl->dev.setDepthColorSyncEnabled(b);
    ONI_CHECK_ERROR(status);
}

float drivers::camera::OpenNI::getColorFocalLength()
{
    int frameWidth = m_pimpl->vs_rgb.getVideoMode().getResolutionX();
    float hFov = m_pimpl->vs_rgb.getHorizontalFieldOfView();
    float calculatedFocalLengthX = frameWidth / (2.0f * std::tan (hFov / 2.0f));
    return calculatedFocalLengthX;
}

float drivers::camera::OpenNI::getDepthFocalLength()
{
    int frameWidth = m_pimpl->vs_depth.getVideoMode().getResolutionX();
    float hFov = m_pimpl->vs_depth.getHorizontalFieldOfView();
    float calculatedFocalLengthX = frameWidth / (2.0f * std::tan (hFov / 2.0f));
    return calculatedFocalLengthX;
}

float drivers::camera::OpenNI::getIRFocalLength()
{
    int frameWidth = m_pimpl->vs_ir.getVideoMode().getResolutionX();
    float hFov = m_pimpl->vs_ir.getHorizontalFieldOfView();
    float calculatedFocalLengthX = frameWidth / (2.0f * std::tan (hFov / 2.0f));
    return calculatedFocalLengthX;
}

void drivers::camera::OpenNI::getRGBIntrinsics(float& fx, float& fy, float& u0, float& v0) const
{
    const int frameWidth = m_pimpl->vs_rgb.getVideoMode().getResolutionX();
    const int frameHeight = m_pimpl->vs_rgb.getVideoMode().getResolutionY();
    const float hFov = m_pimpl->vs_rgb.getHorizontalFieldOfView();
    const float vFov = m_pimpl->vs_rgb.getVerticalFieldOfView();
    fx = frameWidth / (2.0f * std::tan (hFov / 2.0f));
    fy = frameHeight / (2.0f * std::tan (vFov / 2.0f));
    u0 = (float)frameWidth / 2.0f;
    v0 = (float)frameHeight / 2.0f;
}

void drivers::camera::OpenNI::getDepthIntrinsics(float& fx, float& fy, float& u0, float& v0) const
{
    const int frameWidth = m_pimpl->vs_depth.getVideoMode().getResolutionX();
    const int frameHeight = m_pimpl->vs_depth.getVideoMode().getResolutionY();
    const float hFov = m_pimpl->vs_depth.getHorizontalFieldOfView();
    const float vFov = m_pimpl->vs_depth.getVerticalFieldOfView();
    fx = frameWidth / (2.0f * std::tan (hFov / 2.0f));
    fy = frameHeight / (2.0f * std::tan (vFov / 2.0f));
    u0 = (float)frameWidth / 2.0f;
    v0 = (float)frameHeight / 2.0f;
}

void drivers::camera::OpenNI::getIRIntrinsics(float& fx, float& fy, float& u0, float& v0) const
{
    const int frameWidth = m_pimpl->vs_ir.getVideoMode().getResolutionX();
    const int frameHeight = m_pimpl->vs_ir.getVideoMode().getResolutionY();
    const float hFov = m_pimpl->vs_ir.getHorizontalFieldOfView();
    const float vFov = m_pimpl->vs_ir.getVerticalFieldOfView();
    fx = frameWidth / (2.0f * std::tan (hFov / 2.0f));
    fy = frameHeight / (2.0f * std::tan (vFov / 2.0f));
    u0 = (float)frameWidth / 2.0f;
    v0 = (float)frameHeight / 2.0f;
}
    
#ifdef CAMERA_DRIVERS_HAVE_CAMERA_MODELS
void drivers::camera::OpenNI::getRGBIntrinsics(cammod::PinholeDisparity<float>& cam) const
{
    const int frameWidth = m_pimpl->vs_rgb.getVideoMode().getResolutionX();
    const int frameHeight = m_pimpl->vs_rgb.getVideoMode().getResolutionY();
    float hFov = m_pimpl->vs_rgb.getHorizontalFieldOfView();
    float vFov = m_pimpl->vs_rgb.getVerticalFieldOfView();
    const float fx = frameWidth / (2.0f * std::tan (hFov / 2.0f));
    const float fy = frameHeight / (2.0f * std::tan (vFov / 2.0f));
    const float u0 = (float)frameWidth / 2.0f;
    const float v0 = (float)frameHeight / 2.0f;
    cam = cammod::PinholeDisparity<float>(fx,fy,u0,v0,(float)frameWidth,(float)frameHeight);
}
void drivers::camera::OpenNI::getDepthIntrinsics(cammod::PinholeDisparity<float>& cam) const
{
    const int frameWidth = m_pimpl->vs_depth.getVideoMode().getResolutionX();
    const int frameHeight = m_pimpl->vs_depth.getVideoMode().getResolutionY();
    const float hFov = m_pimpl->vs_depth.getHorizontalFieldOfView();
    const float vFov = m_pimpl->vs_depth.getVerticalFieldOfView();
    const float fx = frameWidth / (2.0f * std::tan (hFov / 2.0f));
    const float fy = frameHeight / (2.0f * std::tan (vFov / 2.0f));
    const float u0 = (float)frameWidth / 2.0f;
    const float v0 = (float)frameHeight / 2.0f;
    cam = cammod::PinholeDisparity<float>(fx,fy,u0,v0,(float)frameWidth,(float)frameHeight);
}
void drivers::camera::OpenNI::getIRIntrinsics(cammod::PinholeDisparity<float>& cam) const
{
    const int frameWidth = m_pimpl->vs_ir.getVideoMode().getResolutionX();
    const int frameHeight = m_pimpl->vs_ir.getVideoMode().getResolutionY();
    const float hFov = m_pimpl->vs_ir.getHorizontalFieldOfView();
    const float vFov = m_pimpl->vs_ir.getVerticalFieldOfView();
    const float fx = frameWidth / (2.0f * std::tan (hFov / 2.0f));
    const float fy = frameHeight / (2.0f * std::tan (vFov / 2.0f));
    const float u0 = (float)frameWidth / 2.0f;
    const float v0 = (float)frameHeight / 2.0f;
    cam = cammod::PinholeDisparity<float>(fx,fy,u0,v0,(float)frameWidth,(float)frameHeight);
}
#endif // CAMERA_DRIVERS_HAVE_CAMERA_MODELS

bool drivers::camera::OpenNI::getDepthMirroring()
{
    return m_pimpl->vs_depth.getMirroringEnabled();
}

bool drivers::camera::OpenNI::getRGBMirroring()
{
    return m_pimpl->vs_rgb.getMirroringEnabled();
}

bool drivers::camera::OpenNI::getIRMirroring()
{
    return m_pimpl->vs_ir.getMirroringEnabled();
}

void drivers::camera::OpenNI::setDepthMirroring(bool b)
{
    m_pimpl->vs_depth.setMirroringEnabled(b);
}

void drivers::camera::OpenNI::setRGBMirroring(bool b)
{
    m_pimpl->vs_rgb.setMirroringEnabled(b);
}

void drivers::camera::OpenNI::setIRMirroring(bool b)
{
    m_pimpl->vs_ir.setMirroringEnabled(b);
}

bool drivers::camera::OpenNI::getEmitter()
{
    if(m_pimpl->dev.isPropertySupported(XN_MODULE_PROPERTY_EMITTER_STATUS))
    {
        XnEmitterData ED;
        
        openni::Status status = m_pimpl->dev.getProperty<XnEmitterData>(XN_MODULE_PROPERTY_EMITTER_STATUS, &ED);
        ONI_CHECK_ERROR(status);
        
        return ED.m_State;
    }
    else
    {
        return false;
    }
}

void drivers::camera::OpenNI::setEmitter(bool v)
{
    if(m_pimpl->dev.isPropertySupported(XN_MODULE_PROPERTY_EMITTER_STATE))
    {
        typedef       unsigned long long      XnUInt64;
        openni::Status status = m_pimpl->dev.setProperty<XnUInt64>(XN_MODULE_PROPERTY_EMITTER_STATE, v ? (XnUInt64)TRUE : (XnUInt64)FALSE);
        ONI_CHECK_ERROR(status);
    }
    else
    {
        throw std::runtime_error("Emitter Property Not Supported");
    }
}

bool drivers::camera::OpenNI::captureFrameImpl(FrameBuffer* cf1, FrameBuffer* cf2, FrameBuffer* cf3, FrameBuffer* cf4, int64_t timeout)
{
    int changedIndex, waitStreamCount = 0;
    
    if(is_running_ir)
    {
        waitStreamCount = 1;
        m_pimpl->streams[0] = &(m_pimpl->vs_ir);
    }
    else
    {
        if(is_running_depth && is_running_rgb)
        {
            m_pimpl->streams[0] = &(m_pimpl->vs_depth);
            m_pimpl->streams[1] = &(m_pimpl->vs_rgb);
            waitStreamCount = 2;
        }
        else if(is_running_depth && !is_running_rgb)
        {
            m_pimpl->streams[0] = &(m_pimpl->vs_depth);
            waitStreamCount = 1;
        }
        else if(!is_running_depth && is_running_rgb)
        {
            m_pimpl->streams[0] = &(m_pimpl->vs_rgb);
            waitStreamCount = 1;
        }
    }
    
    openni::Status rc = openni::OpenNI::waitForAnyStream(m_pimpl->streams, waitStreamCount, &changedIndex, timeout > 0 ? timeout / 1000000 : openni::TIMEOUT_FOREVER);
    if (rc != openni::STATUS_OK) 
    { 
        return false; 
    }
    
    std::chrono::high_resolution_clock::time_point tp = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds d = tp.time_since_epoch();
    
    auto set_frame = [&](openni::VideoStream& vs, FrameBuffer* fb)
    {
        openni::VideoFrameRef* ni_frame = new openni::VideoFrameRef();
        vs.readFrame(ni_frame);
        
        if(ni_frame->isValid() && (fb != nullptr))
        {
            drivers::camera::EPixelFormat out_pixfmt = OpenNIToPixelFormat(*ni_frame);
            fb->create(ni_frame, std::bind(&drivers::camera::OpenNI::OpenNIAPIPimpl::frame_release, m_pimpl.get(), std::placeholders::_1),
                       (uint8_t*)ni_frame->getData(), 
                       ni_frame->getWidth(), ni_frame->getHeight(), out_pixfmt, ni_frame->getStrideInBytes());
            fb->setFrameCounter(ni_frame->getFrameIndex());
            fb->setTimeStamp(ni_frame->getTimestamp());
            fb->setPCTimeStamp(d.count());
        }
    };
    
    // handle IR separately
    if(is_running_ir)
    {
        set_frame(m_pimpl->vs_ir, cf1);
    }
    else
    {
        if(is_running_depth && is_running_rgb)
        {
            // Depth
            set_frame(m_pimpl->vs_depth, cf1);
                        
            // RGB
            set_frame(m_pimpl->vs_rgb, cf2);
        }
        else if(!is_running_depth && is_running_rgb)
        {
            // RGB
            set_frame(m_pimpl->vs_rgb, cf1);
        }
        else if(is_running_depth && !is_running_rgb)
        {
            // Depth
            set_frame(m_pimpl->vs_depth, cf1);
        }
        else
        {
            // nothing?
            return false;
        }        
    }
    
    return true;
}

drivers::camera::OpenNI::OpenNIRecorder::OpenNIRecorder() : recorder(new openni::Recorder()), is_recording(false)
{

}

drivers::camera::OpenNI::OpenNIRecorder::~OpenNIRecorder()
{

}

bool drivers::camera::OpenNI::OpenNIRecorder::create(const char* fn)
{
    is_recording = false;
    return recorder->create(fn) == openni::STATUS_OK;
}

void drivers::camera::OpenNI::OpenNIRecorder::destroy()
{
    is_recording = false;
    recorder->destroy();
}

bool drivers::camera::OpenNI::OpenNIRecorder::isValid() const
{
    return recorder->isValid();
}

bool drivers::camera::OpenNI::OpenNIRecorder::attachRGB(drivers::camera::OpenNI& parent)
{
    if(parent.is_running_rgb)
    {
        return recorder->attach(parent.m_pimpl->vs_rgb) == openni::STATUS_OK;
    }
    else
    {
        return false;
    }
}

bool drivers::camera::OpenNI::OpenNIRecorder::attachDepth(drivers::camera::OpenNI& parent)
{
    if(parent.is_running_depth)
    {
        return recorder->attach(parent.m_pimpl->vs_depth) == openni::STATUS_OK;
    }
    else
    {
        return false;
    }
}

bool drivers::camera::OpenNI::OpenNIRecorder::attachIR(drivers::camera::OpenNI& parent)
{
    if(parent.is_running_ir)
    {
        return recorder->attach(parent.m_pimpl->vs_ir) == openni::STATUS_OK;
    }
    else
    {
        return false;
    }
}

bool drivers::camera::OpenNI::OpenNIRecorder::start()
{
    bool ok = (recorder->start() == openni::STATUS_OK);
    if(ok)
    {
        is_recording = true;
    }
    return ok;
}

void drivers::camera::OpenNI::OpenNIRecorder::stop()
{
    is_recording = false;
    recorder->stop();
}

bool drivers::camera::OpenNI::getFeatureAuto(EFeature fidx)
{
    switch(fidx)
    {
        case EFeature::EXPOSURE: return m_pimpl->vs_rgb.getCameraSettings()->getAutoExposureEnabled();
        case EFeature::WHITE_BALANCE: return m_pimpl->vs_rgb.getCameraSettings()->getAutoWhiteBalanceEnabled();
        default: return false;
    }
}

void drivers::camera::OpenNI::setFeatureAuto(EFeature fidx, bool b)
{
    switch(fidx)
    {
        case EFeature::EXPOSURE: m_pimpl->vs_rgb.getCameraSettings()->setAutoExposureEnabled(b); break;
        case EFeature::WHITE_BALANCE: m_pimpl->vs_rgb.getCameraSettings()->setAutoWhiteBalanceEnabled(b); break;
        default: return;
    }
}

float drivers::camera::OpenNI::getFeatureValueAbs(EFeature fidx)
{
    switch(fidx)
    {
        case EFeature::EXPOSURE: return m_pimpl->vs_rgb.getCameraSettings()->getAutoExposureEnabled();
        case EFeature::WHITE_BALANCE: return m_pimpl->vs_rgb.getCameraSettings()->getAutoWhiteBalanceEnabled();
        case EFeature::SHUTTER: return m_pimpl->vs_rgb.getCameraSettings()->getExposure();
        case EFeature::GAIN: return m_pimpl->vs_rgb.getCameraSettings()->getGain();
        default: return 0.0f;
    }
}

uint32_t drivers::camera::OpenNI::getFeatureMin(EFeature fidx)
{
    switch(fidx)
    {
        case EFeature::EXPOSURE: return 0;
        case EFeature::WHITE_BALANCE: return 0;
        case EFeature::SHUTTER: return 0;
        case EFeature::GAIN: return 0;
        default: return 0;
    }
}

uint32_t drivers::camera::OpenNI::getFeatureMax(EFeature fidx)
{
    switch(fidx)
    {
        case EFeature::EXPOSURE: return 1;
        case EFeature::WHITE_BALANCE: return 1;
        case EFeature::SHUTTER: return 100;
        case EFeature::GAIN: return 100;
        default: return 0;
    }
}

void drivers::camera::OpenNI::setFeatureValueAbs(EFeature fidx, float val)
{
    switch(fidx)
    {
        case EFeature::EXPOSURE: m_pimpl->vs_rgb.getCameraSettings()->setAutoExposureEnabled(val > 0.0f ? true : false); break;
        case EFeature::WHITE_BALANCE: m_pimpl->vs_rgb.getCameraSettings()->setAutoWhiteBalanceEnabled(val > 0.0f ? true : false); break;
        case EFeature::SHUTTER: m_pimpl->vs_rgb.getCameraSettings()->setExposure((int)val); break;
        case EFeature::GAIN: m_pimpl->vs_rgb.getCameraSettings()->setGain((int)val); break;
        default: return;
    }
}
