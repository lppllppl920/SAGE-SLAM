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

#include <atomic>

#include <RealSenseDriver.hpp>

#include <librealsense/rs.hpp>

drivers::camera::RealSense::RealSenseException::RealSenseException(int status, const char* file, int line)
{
    std::stringstream ss;
    ss << "RS Error " << status << " , " << " at " << file << ":" << line;
    errormsg = ss.str();
}

#define RS_CHECK_ERROR(err_code) { if(err_code != openni::STATUS_OK) { throw OpenNIException(err_code, __FILE__, __LINE__); } }

static inline bool getRSOption(drivers::camera::EFeature fidx, rs::option& opt, bool isauto = false)
{
    switch(fidx)
    {
        case drivers::camera::EFeature::BRIGHTNESS: opt = rs::option::color_brightness; return true;
        case drivers::camera::EFeature::EXPOSURE: opt = rs::option::color_enable_auto_exposure; return true;
        case drivers::camera::EFeature::SHARPNESS: opt = rs::option::color_sharpness; return true;
        case drivers::camera::EFeature::WHITE_BALANCE: if(!isauto) { opt = rs::option::color_white_balance; } else { opt = rs::option::color_white_balance; } return true;
        case drivers::camera::EFeature::HUE: opt = rs::option::color_hue; return true;
        case drivers::camera::EFeature::SATURATION: opt = rs::option::color_saturation; return true;
        case drivers::camera::EFeature::GAMMA: opt = rs::option::color_gamma; return true;
        case drivers::camera::EFeature::SHUTTER: opt = rs::option::color_exposure; return true;
        case drivers::camera::EFeature::GAIN: opt = rs::option::color_gain; return true;
        case drivers::camera::EFeature::TEMPERATURE: opt = rs::option::color_backlight_compensation; return true;
        default: return false;
    }
}


struct drivers::camera::RealSense::RealSenseAPIPimpl
{
    RealSenseAPIPimpl() : color_valid(false), depth_valid(false), ir1_valid(false), ir2_valid(false), device(nullptr)
    {
        
    }
    
    ~RealSenseAPIPimpl()
    {
        
    }
    
    void frame_release(void* img)
    {
        
    }
    
    bool color_valid, depth_valid, ir1_valid, ir2_valid;
    rs::stream default_color_stream;
    rs::stream default_depth_stream;
    rs::stream default_ir1_stream;
    rs::stream default_ir2_stream;
    
    rs::context ctx;
    rs::device* device;
    rs::intrinsics int_color;
    rs::intrinsics int_depth;
    rs::extrinsics ext_depth_to_color;
};


drivers::camera::RealSense::RealSense() : CameraDriverBase(), m_pimpl(new RealSenseAPIPimpl()), is_running(false)
{
    
}
   
drivers::camera::RealSense::~RealSense()
{
    close();
}

void drivers::camera::RealSense::open(unsigned int idx)
{
    if(isOpened()) { return; }
    
    if(m_pimpl->ctx.get_device_count() == 0) 
    {
        throw std::runtime_error("No RealSense devices");
    }
        
    m_pimpl->device = m_pimpl->ctx.get_device(idx);
    
    m_cinfo.SerialNumber = std::string(m_pimpl->device->get_serial());
    m_cinfo.InterfaceType = CameraInfo::CameraInterface::USB;
    m_cinfo.IsColorCamera = true;
    m_cinfo.ModelName = std::string(m_pimpl->device->get_name());
    m_cinfo.VendorName = std::string("Intel");
    m_cinfo.SensorInfo = std::string(m_pimpl->device->get_usb_port_id());
    m_cinfo.SensorResolution = std::string("NA");
    m_cinfo.DriverName = std::string("libRealSense");
    m_cinfo.FirmwareVersion = std::string(m_pimpl->device->get_firmware_version());
    if(m_pimpl->device->supports(rs::camera_info::build_date))
    {
        m_cinfo.FirmwareBuildTime = std::string(m_pimpl->device->get_info(rs::camera_info::build_date));
    }
    else
    {
        m_cinfo.FirmwareBuildTime = std::string("NA");
    }
}

bool drivers::camera::RealSense::isOpenedImpl() const
{
    return m_pimpl->device != nullptr;
}

void drivers::camera::RealSense::close()
{
    if(!isOpened()) { return; }
    
    if(is_running)
    {
        stop();
    }
    
    if(m_pimpl->depth_valid)
    {
        m_pimpl->device->disable_stream(rs::stream::depth);
        m_pimpl->depth_valid = false;
    }
    
    if(m_pimpl->color_valid)
    {
        m_pimpl->device->disable_stream(rs::stream::color);
        m_pimpl->color_valid = false;
    }
    
    if(m_pimpl->ir1_valid)
    {
        m_pimpl->device->disable_stream(rs::stream::infrared);
        m_pimpl->ir1_valid = false;
    }
    
    if(m_pimpl->ir2_valid)
    {
        m_pimpl->device->disable_stream(rs::stream::infrared2);
        m_pimpl->ir2_valid = false;
    }
    
    m_pimpl->device = nullptr;
}

void drivers::camera::RealSense::start()
{
    if(isStarted()) { return; }
    
    m_pimpl->device->start();
    
    is_running = true;
}

void drivers::camera::RealSense::stop()
{
    if(!isStarted()) { return; }
    
    is_running = false;
    
    m_pimpl->device->stop();
}

void drivers::camera::RealSense::setDepthMode(std::size_t w, std::size_t h, unsigned int fps, bool aligned_to_color, bool aligned_to_rectified_color)
{
    if(!isOpened()) { return; }
    
    if(aligned_to_rectified_color)
    {
        m_pimpl->default_depth_stream = rs::stream::depth_aligned_to_rectified_color;
    }
    else if(aligned_to_color)
    {
        m_pimpl->default_depth_stream = rs::stream::depth_aligned_to_color;
    }
    else
    {
        m_pimpl->default_depth_stream = rs::stream::depth;
    }
    
    m_pimpl->device->enable_stream(rs::stream::depth, w, h, rs::format::z16, fps);
    
    m_pimpl->depth_valid = true;
}

void drivers::camera::RealSense::setRGBMode(std::size_t w, std::size_t h, unsigned int fps, drivers::camera::EPixelFormat pixfmt, bool rectified, bool aligned_to_depth)
{
    if(!isOpened()) { return; }
    
    if(aligned_to_depth)
    {
        m_pimpl->default_color_stream = rs::stream::color_aligned_to_depth;
    }
    else if(rectified)
    {
        m_pimpl->default_color_stream = rs::stream::rectified_color;
    }
    else
    {
        m_pimpl->default_color_stream = rs::stream::color;
    }
    
    rs::format rsfmt = rs::format::rgb8;
    
    switch(pixfmt)
    {
        case EPixelFormat::PIXEL_FORMAT_MONO8:
            rsfmt = rs::format::y8;
            break;
        case EPixelFormat::PIXEL_FORMAT_MONO16:
            rsfmt = rs::format::y16;
            break;
        case EPixelFormat::PIXEL_FORMAT_RGB8:
            rsfmt = rs::format::rgb8;
            break;
        case EPixelFormat::PIXEL_FORMAT_BGR8:
            rsfmt = rs::format::bgr8;
            break;
        default:
            throw std::runtime_error("Unsupported pixel format");
    }
    
    m_pimpl->device->enable_stream(rs::stream::color, w, h, rsfmt, fps);
    
    m_pimpl->color_valid = true;
}

void drivers::camera::RealSense::setIR1Mode(std::size_t w, std::size_t h, unsigned int fps)
{
    if(!isOpened()) { return; }
    
    m_pimpl->default_ir1_stream = rs::stream::infrared;
    
    m_pimpl->device->enable_stream(rs::stream::infrared, w, h, rs::format::y16, fps);
    
    m_pimpl->ir1_valid = true;
}

void drivers::camera::RealSense::setIR2Mode(std::size_t w, std::size_t h, unsigned int fps)
{
    if(!isOpened()) { return; }
    
    m_pimpl->default_ir2_stream = rs::stream::infrared2;
    
    m_pimpl->device->enable_stream(rs::stream::infrared2, w, h, rs::format::y16, fps);
    
    m_pimpl->ir2_valid = true;
}

std::size_t drivers::camera::RealSense::getRGBWidth() const
{
    if(m_pimpl->color_valid)
    {
        return m_pimpl->device->get_stream_width(rs::stream::color);
    }
    else
    {
        return 0;
    }
}

std::size_t drivers::camera::RealSense::getRGBHeight() const
{
    if(m_pimpl->color_valid)
    {
        return m_pimpl->device->get_stream_height(rs::stream::color);
    }
    else
    {
        return 0;
    }
}

drivers::camera::EPixelFormat drivers::camera::RealSense::getRGBPixelFormat() const
{
    return drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB8;
}

std::size_t drivers::camera::RealSense::getDepthWidth() const
{
    if(m_pimpl->depth_valid)
    {
        return m_pimpl->device->get_stream_width(rs::stream::depth);
    }
    else
    {
        return 0;
    }
}

std::size_t drivers::camera::RealSense::getDepthHeight() const
{
    if(m_pimpl->depth_valid)
    {
        return m_pimpl->device->get_stream_height(rs::stream::depth);
    }
    else
    {
        return 0;
    }
}

drivers::camera::EPixelFormat drivers::camera::RealSense::getDepthPixelFormat() const
{
    return drivers::camera::EPixelFormat::PIXEL_FORMAT_DEPTH_U16;
}

std::size_t drivers::camera::RealSense::getIR1Width() const
{
    if(m_pimpl->ir1_valid)
    {
        return m_pimpl->device->get_stream_height(rs::stream::infrared);
    }
    else
    {
        return 0;
    }
}

std::size_t drivers::camera::RealSense::getIR1Height() const
{
    if(m_pimpl->ir1_valid)
    {
        return m_pimpl->device->get_stream_height(rs::stream::infrared);
    }
    else
    {
        return 0;
    }
}

drivers::camera::EPixelFormat drivers::camera::RealSense::getIR1PixelFormat() const
{
    return drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO16;
}

std::size_t drivers::camera::RealSense::getIR2Width() const
{
    if(m_pimpl->ir2_valid)
    {
        return m_pimpl->device->get_stream_height(rs::stream::infrared2);
    }
    else
    {
        return 0;
    }
}

std::size_t drivers::camera::RealSense::getIR2Height() const
{
    if(m_pimpl->ir2_valid)
    {
        return m_pimpl->device->get_stream_height(rs::stream::infrared2);
    }
    else
    {
        return 0;
    }
}

drivers::camera::EPixelFormat drivers::camera::RealSense::getIR2PixelFormat() const
{
    return drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO16;
}

float drivers::camera::RealSense::getDepthScale() const
{
    return m_pimpl->device->get_depth_scale();
}

void drivers::camera::RealSense::getRGBIntrinsics(float& fx, float& fy, float& u0, float& v0, std::array<float,5>* dist) const
{
    rs::intrinsics ints = m_pimpl->device->get_stream_intrinsics(m_pimpl->default_color_stream);
    
    fx = ints.fx;
    fy = ints.fy;
    u0 = ints.ppx;
    v0 = ints.ppy;
    
    if(dist != nullptr)
    {
        for(std::size_t i = 0 ; i < 5 ; ++i) 
        {
            dist->at(i) = ints.coeffs[i];
        }
    }
}

void drivers::camera::RealSense::getDepthIntrinsics(float& fx, float& fy, float& u0, float& v0, std::array<float,5>* dist) const
{
    rs::intrinsics ints = m_pimpl->device->get_stream_intrinsics(m_pimpl->default_depth_stream);
    
    fx = ints.fx;
    fy = ints.fy;
    u0 = ints.ppx;
    v0 = ints.ppy;
    
    if(dist != nullptr)
    {
        for(std::size_t i = 0 ; i < 5 ; ++i) 
        {
            dist->at(i) = ints.coeffs[i];
        }
    }
}

void drivers::camera::RealSense::getIR1Intrinsics(float& fx, float& fy, float& u0, float& v0, std::array<float,5>* dist) const
{
    rs::intrinsics ints = m_pimpl->device->get_stream_intrinsics(m_pimpl->default_ir1_stream);
    
    fx = ints.fx;
    fy = ints.fy;
    u0 = ints.ppx;
    v0 = ints.ppy;
    
    if(dist != nullptr)
    {
        for(std::size_t i = 0 ; i < 5 ; ++i) 
        {
            dist->at(i) = ints.coeffs[i];
        }
    }
}

void drivers::camera::RealSense::getIR2Intrinsics(float& fx, float& fy, float& u0, float& v0, std::array<float,5>* dist) const
{
    rs::intrinsics ints = m_pimpl->device->get_stream_intrinsics(m_pimpl->default_ir2_stream);
    
    fx = ints.fx;
    fy = ints.fy;
    u0 = ints.ppx;
    v0 = ints.ppy;
    
    if(dist != nullptr)
    {
        for(std::size_t i = 0 ; i < 5 ; ++i) 
        {
            dist->at(i) = ints.coeffs[i];
        }
    }
}

void drivers::camera::RealSense::getExtrinsicsDepthToColor(float& tx, float& ty, float&tz, std::array<float,9>* rotMat) const
{
    rs::extrinsics exts = m_pimpl->device->get_extrinsics(m_pimpl->default_depth_stream, m_pimpl->default_color_stream);
    tx = exts.translation[0];
    ty = exts.translation[1];
    tz = exts.translation[2];
    if(rotMat != nullptr)
    {
        for(std::size_t i = 0 ; i < 9 ; ++i) { rotMat->at(i) = exts.rotation[i]; }
    }
}

void drivers::camera::RealSense::getExtrinsicsColorToDepth(float& tx, float& ty, float&tz, std::array<float,9>* rotMat) const
{
    rs::extrinsics exts = m_pimpl->device->get_extrinsics(m_pimpl->default_color_stream, m_pimpl->default_depth_stream);
    tx = exts.translation[0];
    ty = exts.translation[1];
    tz = exts.translation[2];
    if(rotMat != nullptr)
    {
        for(std::size_t i = 0 ; i < 9 ; ++i) { rotMat->at(i) = exts.rotation[i]; }
    }
}

#ifdef CAMERA_DRIVERS_HAVE_CAMERA_MODELS
void drivers::camera::RealSense::getRGBIntrinsics(cammod::PinholeDisparity<float>& cam) const
{
    rs::intrinsics ints = m_pimpl->device->get_stream_intrinsics(m_pimpl->default_color_stream);
    cam = cammod::PinholeDisparity<float>(ints.fx, ints.fy, ints.ppx, ints.ppy, (float)getRGBWidth(), (float)getRGBHeight());
}

void drivers::camera::RealSense::getRGBIntrinsics(cammod::PinholeDisparityBrownConrady<float>& cam) const
{
    rs::intrinsics ints = m_pimpl->device->get_stream_intrinsics(m_pimpl->default_color_stream);
    cam = cammod::PinholeDisparityBrownConrady<float>(ints.fx, ints.fy, ints.ppx, ints.ppy, 
                                            ints.coeffs[0], ints.coeffs[1], ints.coeffs[2], ints.coeffs[3], ints.coeffs[4], 
                                            (float)getRGBWidth(), (float)getRGBHeight());
}

void drivers::camera::RealSense::getDepthIntrinsics(cammod::PinholeDisparity<float>& cam) const
{
    rs::intrinsics ints = m_pimpl->device->get_stream_intrinsics(m_pimpl->default_depth_stream);
    cam = cammod::PinholeDisparity<float>(ints.fx, ints.fy, ints.ppx, ints.ppy, (float)getDepthWidth(), (float)getDepthHeight());
}

void drivers::camera::RealSense::getDepthIntrinsics(cammod::PinholeDisparityBrownConrady<float>& cam) const
{
    rs::intrinsics ints = m_pimpl->device->get_stream_intrinsics(m_pimpl->default_depth_stream);
    cam = cammod::PinholeDisparityBrownConrady<float>(ints.fx, ints.fy, ints.ppx, ints.ppy, 
                                            ints.coeffs[0], ints.coeffs[1], ints.coeffs[2], ints.coeffs[3], ints.coeffs[4], 
                                            (float)getDepthWidth(), (float)getDepthHeight());
}

void drivers::camera::RealSense::getIR1Intrinsics(cammod::PinholeDisparity<float>& cam) const
{
    rs::intrinsics ints = m_pimpl->device->get_stream_intrinsics(m_pimpl->default_ir1_stream);
    cam = cammod::PinholeDisparity<float>(ints.fx, ints.fy, ints.ppx, ints.ppy, (float)getIR1Width(), (float)getIR1Height());
}

void drivers::camera::RealSense::getIR1Intrinsics(cammod::PinholeDisparityBrownConrady<float>& cam) const
{
    rs::intrinsics ints = m_pimpl->device->get_stream_intrinsics(m_pimpl->default_ir1_stream);
    cam = cammod::PinholeDisparityBrownConrady<float>(ints.fx, ints.fy, ints.ppx, ints.ppy, 
                                            ints.coeffs[0], ints.coeffs[1], ints.coeffs[2], ints.coeffs[3], ints.coeffs[4], 
                                            (float)getIR1Width(), (float)getIR1Height());
}

void drivers::camera::RealSense::getIR2Intrinsics(cammod::PinholeDisparity<float>& cam) const
{
    rs::intrinsics ints = m_pimpl->device->get_stream_intrinsics(m_pimpl->default_ir2_stream);
    cam = cammod::PinholeDisparity<float>(ints.fx, ints.fy, ints.ppx, ints.ppy, (float)getIR2Width(), (float)getIR2Height());
}

void drivers::camera::RealSense::getIR2Intrinsics(cammod::PinholeDisparityBrownConrady<float>& cam) const
{
    rs::intrinsics ints = m_pimpl->device->get_stream_intrinsics(m_pimpl->default_ir2_stream);
    cam = cammod::PinholeDisparityBrownConrady<float>(ints.fx, ints.fy, ints.ppx, ints.ppy, 
                                            ints.coeffs[0], ints.coeffs[1], ints.coeffs[2], ints.coeffs[3], ints.coeffs[4], 
                                            (float)getIR2Width(), (float)getIR2Height());
}
#endif // CAMERA_DRIVERS_HAVE_CAMERA_MODELS

bool drivers::camera::RealSense::getFeatureAuto(EFeature fidx)
{
    rs::option opt;
    if(!getRSOption(fidx,opt)) { return false; }
    
    if(m_pimpl->device->supports_option(opt))
    {
        return m_pimpl->device->get_option(opt) > 0.0f;
    }
    
    return false;
}

void drivers::camera::RealSense::setFeatureAuto(EFeature fidx, bool b)
{
    rs::option opt;
    if(!getRSOption(fidx,opt)) { return; }
    
    if(m_pimpl->device->supports_option(opt))
    {
        m_pimpl->device->set_option(opt, b == true ? 1.0 : 0.0);
    }
}

float drivers::camera::RealSense::getFeatureValueAbs(EFeature fidx)
{
    rs::option opt;
    if(!getRSOption(fidx,opt)) { return 0.0f; }
    
    if(m_pimpl->device->supports_option(opt))
    {
        return m_pimpl->device->get_option(opt);
    }
    
    return 0.0f;
}

uint32_t drivers::camera::RealSense::getFeatureMin(EFeature fidx)
{
    rs::option opt;
    if(!getRSOption(fidx,opt)) { return 0; }
    
    double vmin, vmax, vstep;
    
    if(m_pimpl->device->supports_option(opt))
    {
        m_pimpl->device->get_option_range(opt, vmin, vmax, vstep);
        return (uint32_t)vmin;
    }
    
    return 0.0f;
}

uint32_t drivers::camera::RealSense::getFeatureMax(EFeature fidx)
{
    rs::option opt;
    if(!getRSOption(fidx,opt)) { return 0; }
    
    double vmin, vmax, vstep;
    
    if(m_pimpl->device->supports_option(opt))
    {
        m_pimpl->device->get_option_range(opt, vmin, vmax, vstep);
        return (uint32_t)vmax;
    }
    
    return 0.0f;
}

void drivers::camera::RealSense::setFeatureValueAbs(EFeature fidx, float val)
{
    rs::option opt;
    if(!getRSOption(fidx,opt)) { return; }
    
    if(m_pimpl->device->supports_option(opt))
    {
        m_pimpl->device->set_option(opt, val);
    }
}

bool drivers::camera::RealSense::captureFrameImpl(FrameBuffer* cf1, FrameBuffer* cf2, FrameBuffer* cf3, 
                                                  FrameBuffer* cf4, int64_t timeout)
{
    // TODO FIXME timeout
    m_pimpl->device->wait_for_frames();
    
    std::chrono::high_resolution_clock::time_point tp = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds d = tp.time_since_epoch();
    
    auto setupFrame = [&](FrameBuffer* cf, rs::stream s, drivers::camera::EPixelFormat pf)
    {
        if(cf != nullptr)
        {
            cf->create(m_pimpl->device->get_stream_width(s), m_pimpl->device->get_stream_height(s), pf);
            cf->copyFrom(m_pimpl->device->get_frame_data(s));
            cf->setFrameCounter(m_pimpl->device->get_frame_number(s));
            cf->setTimeStamp((uint64_t)(m_pimpl->device->get_frame_timestamp(s) * 1000000.0));
            cf->setPCTimeStamp(d.count());
        }
    };

    if(m_pimpl->color_valid && m_pimpl->depth_valid && m_pimpl->ir1_valid && m_pimpl->ir2_valid) // 4 R D I1 I2
    {
        setupFrame(cf1, m_pimpl->default_depth_stream, getDepthPixelFormat());
        setupFrame(cf2, m_pimpl->default_color_stream, getRGBPixelFormat());
        setupFrame(cf3, m_pimpl->default_ir1_stream, getIR1PixelFormat());
        setupFrame(cf4, m_pimpl->default_ir2_stream, getIR2PixelFormat());
    }
    else if(m_pimpl->color_valid && m_pimpl->depth_valid && m_pimpl->ir1_valid && !m_pimpl->ir2_valid) // 3 R D I1
    {
        setupFrame(cf1, m_pimpl->default_depth_stream, getDepthPixelFormat());
        setupFrame(cf2, m_pimpl->default_color_stream, getRGBPixelFormat());
        setupFrame(cf3, m_pimpl->default_ir1_stream, getIR1PixelFormat());
    }
    else if(m_pimpl->color_valid && m_pimpl->depth_valid && !m_pimpl->ir1_valid && m_pimpl->ir2_valid) // 3 R D I2
    {
        setupFrame(cf1, m_pimpl->default_depth_stream, getDepthPixelFormat());
        setupFrame(cf2, m_pimpl->default_color_stream, getRGBPixelFormat());
        setupFrame(cf3, m_pimpl->default_ir2_stream, getIR2PixelFormat());
    }
    else if(m_pimpl->color_valid && !m_pimpl->depth_valid && m_pimpl->ir1_valid && m_pimpl->ir2_valid) // 3 R I1 I2
    {
        setupFrame(cf1, m_pimpl->default_color_stream, getRGBPixelFormat());
        setupFrame(cf2, m_pimpl->default_ir1_stream, getIR1PixelFormat());
        setupFrame(cf3, m_pimpl->default_ir2_stream, getIR2PixelFormat());
    }
    else if(!m_pimpl->color_valid && m_pimpl->depth_valid && m_pimpl->ir1_valid && m_pimpl->ir2_valid) // 3 D I1 I2
    {
        setupFrame(cf1, m_pimpl->default_depth_stream, getDepthPixelFormat());
        setupFrame(cf2, m_pimpl->default_ir1_stream, getIR1PixelFormat());
        setupFrame(cf3, m_pimpl->default_ir2_stream, getIR2PixelFormat());
    }
    else if(m_pimpl->color_valid && m_pimpl->depth_valid && !m_pimpl->ir1_valid && !m_pimpl->ir2_valid) // 2 R D
    {
        setupFrame(cf1, m_pimpl->default_depth_stream, getDepthPixelFormat());
        setupFrame(cf2, m_pimpl->default_color_stream, getRGBPixelFormat());
    }
    else if(m_pimpl->color_valid && !m_pimpl->depth_valid && m_pimpl->ir1_valid && !m_pimpl->ir2_valid) // 2 R I1
    {
        setupFrame(cf1, m_pimpl->default_color_stream, getRGBPixelFormat());
        setupFrame(cf2, m_pimpl->default_ir1_stream, getIR1PixelFormat());
    }
    else if(m_pimpl->color_valid && !m_pimpl->depth_valid && !m_pimpl->ir1_valid && m_pimpl->ir2_valid) // 2 R I2
    {
        setupFrame(cf1, m_pimpl->default_color_stream, getRGBPixelFormat());
        setupFrame(cf2, m_pimpl->default_ir2_stream, getIR2PixelFormat());
    }
    else if(!m_pimpl->color_valid && m_pimpl->depth_valid && m_pimpl->ir1_valid && !m_pimpl->ir2_valid) // 2 D I1
    {
        setupFrame(cf1, m_pimpl->default_depth_stream, getDepthPixelFormat());
        setupFrame(cf2, m_pimpl->default_ir1_stream, getIR1PixelFormat());
    }
    else if(!m_pimpl->color_valid && m_pimpl->depth_valid && !m_pimpl->ir1_valid && m_pimpl->ir2_valid) // 2 D I2
    {
        setupFrame(cf1, m_pimpl->default_depth_stream, getDepthPixelFormat());
        setupFrame(cf2, m_pimpl->default_ir2_stream, getIR2PixelFormat());
    }
    else if(!m_pimpl->color_valid && !m_pimpl->depth_valid && m_pimpl->ir1_valid && m_pimpl->ir2_valid) // 2 I1 I2
    {
        setupFrame(cf1, m_pimpl->default_ir1_stream, getIR1PixelFormat());
        setupFrame(cf2, m_pimpl->default_ir2_stream, getIR2PixelFormat());
    }
    else if(m_pimpl->color_valid && !m_pimpl->depth_valid && !m_pimpl->ir1_valid && !m_pimpl->ir2_valid) // 1 R
    {
        setupFrame(cf1, m_pimpl->default_color_stream, getRGBPixelFormat());
    }
    else if(!m_pimpl->color_valid && m_pimpl->depth_valid && !m_pimpl->ir1_valid && !m_pimpl->ir2_valid) // 1 D
    {
        setupFrame(cf1, m_pimpl->default_depth_stream, getDepthPixelFormat());
    }
    else if(!m_pimpl->color_valid && !m_pimpl->depth_valid && m_pimpl->ir1_valid && !m_pimpl->ir2_valid) // 1 I1
    {
        setupFrame(cf1, m_pimpl->default_ir1_stream, getIR1PixelFormat());
    }
    else if(!m_pimpl->color_valid && !m_pimpl->depth_valid && !m_pimpl->ir1_valid && m_pimpl->ir2_valid) // 1 I2
    {
        setupFrame(cf1, m_pimpl->default_ir2_stream, getIR2PixelFormat());
    }
        
    return true;
}
