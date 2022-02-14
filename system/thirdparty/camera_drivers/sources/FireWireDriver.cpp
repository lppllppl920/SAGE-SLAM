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
 * DC1394 Driver.
 * ****************************************************************************
 */
#include <FireWireDriver.hpp>

// DC1394
#include <sys/select.h>
#include <dc1394/dc1394.h>

#define FIREWIRE_CHECK_ERROR(err_code) { if(err_code != DC1394_SUCCESS) { throw FireWireException(err_code, __FILE__, __LINE__); } }

static inline dc1394video_mode_t VideoModeToDC1394(drivers::camera::EVideoMode v)
{
    switch(v)
    {
        case drivers::camera::EVideoMode::VIDEOMODE_640x480RGB   : return DC1394_VIDEO_MODE_640x480_RGB8;
        case drivers::camera::EVideoMode::VIDEOMODE_640x480Y8    : return DC1394_VIDEO_MODE_640x480_MONO8;
        case drivers::camera::EVideoMode::VIDEOMODE_640x480Y16   : return DC1394_VIDEO_MODE_640x480_MONO16;
        case drivers::camera::EVideoMode::VIDEOMODE_800x600RGB   : return DC1394_VIDEO_MODE_800x600_RGB8;
        case drivers::camera::EVideoMode::VIDEOMODE_800x600Y8    : return DC1394_VIDEO_MODE_800x600_MONO8;
        case drivers::camera::EVideoMode::VIDEOMODE_800x600Y16   : return DC1394_VIDEO_MODE_800x600_MONO16;
        case drivers::camera::EVideoMode::VIDEOMODE_1024x768RGB  : return DC1394_VIDEO_MODE_1024x768_RGB8;
        case drivers::camera::EVideoMode::VIDEOMODE_1024x768Y8   : return DC1394_VIDEO_MODE_1024x768_MONO8;
        case drivers::camera::EVideoMode::VIDEOMODE_1024x768Y16  : return DC1394_VIDEO_MODE_1024x768_MONO16;
        case drivers::camera::EVideoMode::VIDEOMODE_1280x960RGB  : return DC1394_VIDEO_MODE_1280x960_RGB8;
        case drivers::camera::EVideoMode::VIDEOMODE_1280x960Y8   : return DC1394_VIDEO_MODE_1280x960_MONO8;
        case drivers::camera::EVideoMode::VIDEOMODE_1280x960Y16  : return DC1394_VIDEO_MODE_1280x960_MONO16;
        case drivers::camera::EVideoMode::VIDEOMODE_1600x1200RGB : return DC1394_VIDEO_MODE_1600x1200_RGB8;
        case drivers::camera::EVideoMode::VIDEOMODE_1600x1200Y8  : return DC1394_VIDEO_MODE_1600x1200_MONO8;
        case drivers::camera::EVideoMode::VIDEOMODE_1600x1200Y16 : return DC1394_VIDEO_MODE_1600x1200_MONO16;
        case drivers::camera::EVideoMode::VIDEOMODE_CUSTOM       : return DC1394_VIDEO_MODE_FORMAT7_0;
        default: return (dc1394video_mode_t)0;
    }
}

static inline dc1394framerate_t FrameRateToDC1394(drivers::camera::EFrameRate v)
{
    switch(v)
    {
        case drivers::camera::EFrameRate::FRAMERATE_15     : return DC1394_FRAMERATE_15;
        case drivers::camera::EFrameRate::FRAMERATE_30     : return DC1394_FRAMERATE_30;
        case drivers::camera::EFrameRate::FRAMERATE_60     : return DC1394_FRAMERATE_60;
        case drivers::camera::EFrameRate::FRAMERATE_120    : return DC1394_FRAMERATE_120;
        case drivers::camera::EFrameRate::FRAMERATE_240    : return DC1394_FRAMERATE_240;
        case drivers::camera::EFrameRate::FRAMERATE_CUSTOM : return (dc1394framerate_t)0;
        default: return (dc1394framerate_t)0;
    }
}

static inline dc1394feature_t FeatureToDC1394(drivers::camera::EFeature v)
{
    switch(v)
    {
        case drivers::camera::EFeature::BRIGHTNESS   : return DC1394_FEATURE_BRIGHTNESS;
        case drivers::camera::EFeature::EXPOSURE     : return DC1394_FEATURE_EXPOSURE;
        case drivers::camera::EFeature::SHARPNESS    : return DC1394_FEATURE_SHARPNESS;
        case drivers::camera::EFeature::WHITE_BALANCE: return DC1394_FEATURE_WHITE_BALANCE;
        case drivers::camera::EFeature::HUE          : return DC1394_FEATURE_HUE;
        case drivers::camera::EFeature::SATURATION   : return DC1394_FEATURE_SATURATION;
        case drivers::camera::EFeature::GAMMA        : return DC1394_FEATURE_GAMMA;
        case drivers::camera::EFeature::IRIS         : return DC1394_FEATURE_IRIS;
        case drivers::camera::EFeature::FOCUS        : return DC1394_FEATURE_FOCUS;
        case drivers::camera::EFeature::ZOOM         : return DC1394_FEATURE_ZOOM;
        case drivers::camera::EFeature::PAN          : return DC1394_FEATURE_PAN;
        case drivers::camera::EFeature::TILT         : return DC1394_FEATURE_TILT;
        case drivers::camera::EFeature::SHUTTER      : return DC1394_FEATURE_SHUTTER;
        case drivers::camera::EFeature::GAIN         : return DC1394_FEATURE_GAIN;
        case drivers::camera::EFeature::TRIGGER_MODE : return DC1394_FEATURE_TRIGGER;
        case drivers::camera::EFeature::TRIGGER_DELAY: return DC1394_FEATURE_TRIGGER_DELAY;
        case drivers::camera::EFeature::FRAME_RATE   : return DC1394_FEATURE_FRAME_RATE;
        case drivers::camera::EFeature::TEMPERATURE  : return DC1394_FEATURE_TEMPERATURE;
        default: return (dc1394feature_t)0;
    }
}

static inline dc1394speed_t ISOSpeedToDC1394(drivers::camera::FireWire::EISOSpeed v)
{
    switch(v)
    {
        case drivers::camera::FireWire::EISOSpeed::ISO_SPEED_100  : return DC1394_ISO_SPEED_100;
        case drivers::camera::FireWire::EISOSpeed::ISO_SPEED_200  : return DC1394_ISO_SPEED_200;
        case drivers::camera::FireWire::EISOSpeed::ISO_SPEED_400  : return DC1394_ISO_SPEED_400;
        case drivers::camera::FireWire::EISOSpeed::ISO_SPEED_800  : return DC1394_ISO_SPEED_800;
        case drivers::camera::FireWire::EISOSpeed::ISO_SPEED_1600 : return DC1394_ISO_SPEED_1600;
        case drivers::camera::FireWire::EISOSpeed::ISO_SPEED_3200 : return DC1394_ISO_SPEED_3200;
        default: return (dc1394speed_t)0;
    }
}

drivers::camera::FireWire::FireWireException::FireWireException(uint32_t errcode, const char* file, int line)
{ 
    std::stringstream ss;
    ss << "FW Error " << errcode << " and " << file << ":" << line;
    errormsg = ss.str();
}

struct drivers::camera::FireWire::FWAPIPimpl
{
    FWAPIPimpl() : m_maxnumcam(0), fw_handle(0), fw_camera(0), fw_list(0), fd(0) { FD_ZERO(&fds); }
    unsigned int m_maxnumcam;
    dc1394_t* fw_handle;
    dc1394camera_t* fw_camera;
    dc1394camera_list_t* fw_list;
    int fd;
    fd_set fds;
    struct timeval tval; 
};
    
drivers::camera::FireWire::FireWire() : 
    CameraDriverBase(), m_pimpl(new FWAPIPimpl()), m_pixfmt(EPixelFormat::PIXEL_FORMAT_MONO8), iso_speed(drivers::camera::FireWire::EISOSpeed::ISO_SPEED_400), dma_buf(10), m_camid(0), is_running(false)
{
    m_pimpl->fw_handle = dc1394_new();
    if(m_pimpl->fw_handle == 0)
    {
        throw std::runtime_error("Cannot open DC1394");
    }
    
    dc1394error_t err;
    
    // list cameras
    err = dc1394_camera_enumerate(m_pimpl->fw_handle, &m_pimpl->fw_list);
    FIREWIRE_CHECK_ERROR(err);
    
    if (m_pimpl->fw_list->num == 0) { throw std::runtime_error("No cameras"); }
    
    m_pimpl->m_maxnumcam = m_pimpl->fw_list->num;
}

drivers::camera::FireWire::~FireWire()
{
    close();
    
    dc1394_camera_free_list(m_pimpl->fw_list);
    
    dc1394_free(m_pimpl->fw_handle);
    m_pimpl->fw_handle = 0;
}

void drivers::camera::FireWire::image_release(void* img)
{
    dc1394error_t err;
    dc1394video_frame_t* frame = static_cast<dc1394video_frame_t*>(img);
    err = dc1394_capture_enqueue(m_pimpl->fw_camera, frame);
    FIREWIRE_CHECK_ERROR(err);
}

void drivers::camera::FireWire::open(uint32_t id)
{
    if(isOpened()) { return; }
    
    if(id >= m_pimpl->m_maxnumcam) { throw std::runtime_error("Wrong ID"); }
    
    dc1394error_t err;
    const uint64_t cam_guid = m_pimpl->fw_list->ids[id].guid;
    
    // open the camera
    m_pimpl->fw_camera = dc1394_camera_new(m_pimpl->fw_handle, cam_guid);
    if(!m_pimpl->fw_camera) { throw std::runtime_error("Error opening the camera"); }
    
    // Attempt to stop camera if it is already running
    dc1394switch_t is_iso_on = DC1394_OFF;
    
    err = dc1394_video_get_transmission(m_pimpl->fw_camera, &is_iso_on);
    FIREWIRE_CHECK_ERROR(err);

    // turn it off
    if(is_iso_on == DC1394_ON)
    {
        err = dc1394_video_set_transmission(m_pimpl->fw_camera, DC1394_OFF);
        FIREWIRE_CHECK_ERROR(err);
    }
    
    // reset camera
    err = dc1394_camera_reset(m_pimpl->fw_camera);
    FIREWIRE_CHECK_ERROR(err);
    
    m_camid = id;
    
    // fill Camera Info
    std::stringstream ss;
    ss << m_pimpl->fw_camera->unit_spec_ID;
    m_cinfo.SerialNumber = ss.str();
    m_cinfo.InterfaceType = CameraInfo::CameraInterface::IEEE1394;
    m_cinfo.IsColorCamera = true; // we don't know
    m_cinfo.ModelName = std::string(m_pimpl->fw_camera->model);
    m_cinfo.VendorName = std::string(m_pimpl->fw_camera->vendor);
    m_cinfo.SensorInfo = std::string("");
    m_cinfo.SensorResolution = std::string("");
    m_cinfo.DriverName = std::string("DC1394");
    ss.str("");
    ss << m_pimpl->fw_camera->unit_sw_version << "." << m_pimpl->fw_camera->unit_sub_sw_version;
    m_cinfo.FirmwareVersion = ss.str();
    m_cinfo.FirmwareBuildTime = std::string("");
}

bool drivers::camera::FireWire::isOpenedImpl() const
{
    return m_pimpl->fw_camera != 0;
}

void drivers::camera::FireWire::close()
{
    if(!isOpened()) { return; }
    
    dc1394error_t err = dc1394_capture_stop(m_pimpl->fw_camera);
    FIREWIRE_CHECK_ERROR(err);

    dc1394_camera_free(m_pimpl->fw_camera);
    m_pimpl->fw_camera = 0;
}

void drivers::camera::FireWire::setModeAndFramerate(EVideoMode vmode, EFrameRate framerate)
{
    dc1394error_t err;
    dc1394speed_t dc_iso_speed = ISOSpeedToDC1394(iso_speed);
    
    if(dc_iso_speed >= (int)DC1394_ISO_SPEED_800)
    {
        // enable 1394B mode first (if possible)
        err = dc1394_video_set_operation_mode(m_pimpl->fw_camera, DC1394_OPERATION_MODE_1394B);
        if(err != DC1394_SUCCESS)
        {
            // limit then
            dc_iso_speed = DC1394_ISO_SPEED_400;
        }
    }
    
    // set iso
    err = dc1394_video_set_iso_speed(m_pimpl->fw_camera, dc_iso_speed);
    FIREWIRE_CHECK_ERROR(err);
    
    err = dc1394_video_set_mode(m_pimpl->fw_camera, VideoModeToDC1394(vmode));
    FIREWIRE_CHECK_ERROR(err);
    
    // get width & height
    err = dc1394_get_image_size_from_video_mode(m_pimpl->fw_camera, VideoModeToDC1394(vmode), (uint32_t*)&m_width, (uint32_t*)&m_height);
    FIREWIRE_CHECK_ERROR(err);
    
    // set framerate
    err = dc1394_video_set_framerate(m_pimpl->fw_camera, FrameRateToDC1394(framerate));
    FIREWIRE_CHECK_ERROR(err);
    
    // set buffer count
    err = dc1394_capture_setup(m_pimpl->fw_camera, dma_buf, DC1394_CAPTURE_FLAGS_DEFAULT);
    FIREWIRE_CHECK_ERROR(err);
    
    m_pixfmt = drivers::camera::VideoModeToPixelFormat(vmode);
} 

void drivers::camera::FireWire::setCustomMode(drivers::camera::EPixelFormat pixfmt, unsigned int width, unsigned int height, unsigned int offset_x, unsigned int offset_y, uint16_t format7mode)
{
    dc1394error_t err;
    dc1394speed_t dc_iso_speed = ISOSpeedToDC1394(iso_speed);
    
    if(dc_iso_speed >= (int)DC1394_ISO_SPEED_800)
    {
        // enable 1394B mode first (if possible)
        err = dc1394_video_set_operation_mode(m_pimpl->fw_camera, DC1394_OPERATION_MODE_1394B);
        if(err != DC1394_SUCCESS)
        {
            // limit then
            dc_iso_speed = DC1394_ISO_SPEED_400;
        }
    }
    
    // set iso
    err = dc1394_video_set_iso_speed(m_pimpl->fw_camera, dc_iso_speed);
    FIREWIRE_CHECK_ERROR(err);
    
    err = dc1394_video_set_mode(m_pimpl->fw_camera, (dc1394video_mode_t)((uint16_t)DC1394_VIDEO_MODE_FORMAT7_0 + format7mode));
    FIREWIRE_CHECK_ERROR(err);
    
    dc1394color_coding_t dcpixfmt;
    switch(pixfmt)
    {
        case drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO8: dcpixfmt = DC1394_COLOR_CODING_MONO8; break;
        case drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO16: dcpixfmt = DC1394_COLOR_CODING_MONO16; break;
        case drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB8: dcpixfmt = DC1394_COLOR_CODING_RGB8; break;
        default: throw std::runtime_error("Pixel format not supported");
    }
    
    err = dc1394_format7_set_roi(m_pimpl->fw_camera, (dc1394video_mode_t)((uint16_t)DC1394_VIDEO_MODE_FORMAT7_0 + format7mode),
                                 dcpixfmt,
                                 DC1394_USE_MAX_AVAIL, // use max packet size
                                 offset_x, offset_y, // left, top
                                 width, height);  // width, height
    FIREWIRE_CHECK_ERROR(err);
    
    m_width = width;
    m_height = height;
    m_pixfmt = pixfmt;
    
    // set buffer count
    err = dc1394_capture_setup(m_pimpl->fw_camera, dma_buf, DC1394_CAPTURE_FLAGS_DEFAULT);
    FIREWIRE_CHECK_ERROR(err);
}

void drivers::camera::FireWire::start()
{
    if(isStarted()) { return; }
    
    dc1394error_t err;

    FD_SET(dc1394_capture_get_fileno(m_pimpl->fw_camera), &m_pimpl->fds);
    
    // start transmission
    err = dc1394_video_set_transmission(m_pimpl->fw_camera, DC1394_ON);
    FIREWIRE_CHECK_ERROR(err);
    
    is_running = true;
}

void drivers::camera::FireWire::stop()
{
    if(!isStarted()) { return; }
    
    dc1394error_t err;
    
    if(FD_ISSET(dc1394_capture_get_fileno(m_pimpl->fw_camera), &m_pimpl->fds))
    {
        FD_CLR(dc1394_capture_get_fileno(m_pimpl->fw_camera), &m_pimpl->fds);
    } 
    
    // stop transmission
    err = dc1394_video_set_transmission(m_pimpl->fw_camera, DC1394_OFF);
    FIREWIRE_CHECK_ERROR(err);
    
    is_running = false;
}

bool drivers::camera::FireWire::captureFrameImpl(FrameBuffer* cf1, FrameBuffer* cf2, FrameBuffer* cf3, FrameBuffer* cf4, int64_t timeout)
{
    dc1394error_t err;
    dc1394video_frame_t *frame = 0;
    
    if(timeout != 0) 
    {
        m_pimpl->tval.tv_sec = timeout / 1000000000; // sec
        m_pimpl->tval.tv_usec = (timeout % 1000000000) / 1000; // usec
        
        // wait for it
        if(select(FD_SETSIZE, &m_pimpl->fds, NULL, NULL, &m_pimpl->tval) == 1) 
        {
            // capture frame
            err = dc1394_capture_dequeue(m_pimpl->fw_camera, DC1394_CAPTURE_POLICY_POLL, &frame);
            if(err != DC1394_SUCCESS)
            {
                return false;
            }
        }       
        else
        {
            return false;
        }
    }
    else // blocking
    {
        err = dc1394_capture_dequeue(m_pimpl->fw_camera, DC1394_CAPTURE_POLICY_WAIT, &frame);
        if(err != DC1394_SUCCESS)
        {
            return false;
        }
    }

    std::chrono::high_resolution_clock::time_point tp = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds d = tp.time_since_epoch();
    
    drivers::camera::EPixelFormat pixfmt;
    switch(frame->color_coding)
    {
        case DC1394_COLOR_CODING_MONO8:
            pixfmt = drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO8;
            break;
        case DC1394_COLOR_CODING_MONO16:
            pixfmt = drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO16;
            break;
        case DC1394_COLOR_CODING_RGB8:
            pixfmt = drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB8;
            break;
        default: // unknown
            throw std::runtime_error("Unsupported mode");
            break;
    }
    
    if(cf1 != nullptr)
    {
        cf1->create(frame, std::bind(&drivers::camera::FireWire::image_release, this, std::placeholders::_1), frame->image, frame->size[0], frame->size[1], pixfmt, frame->stride);
        cf1->setPCTimeStamp(d.count());
        cf1->setTimeStamp(frame->timestamp);
        cf1->setGain(0);
        cf1->setShutter(0);
        cf1->setBrightness(0);
        cf1->setExposure(0);
        cf1->setWhiteBalance(0);
        cf1->setFrameCounter(0);
    }

    return true;
}

bool drivers::camera::FireWire::getFeaturePower(EFeature fidx)
{
    dc1394error_t err;

    dc1394switch_t ison = DC1394_OFF;
    
    err = dc1394_feature_get_power(m_pimpl->fw_camera, FeatureToDC1394(fidx), &ison);
    FIREWIRE_CHECK_ERROR(err);
    
    return ison == DC1394_ON;
}

void drivers::camera::FireWire::setFeaturePower(EFeature fidx, bool b)
{
    dc1394error_t err;
    
    err = dc1394_feature_set_power(m_pimpl->fw_camera, FeatureToDC1394(fidx), b == true ? DC1394_ON : DC1394_OFF);
    FIREWIRE_CHECK_ERROR(err);
}

bool drivers::camera::FireWire::getFeatureAuto(EFeature fidx)
{
    dc1394error_t err;

    dc1394feature_mode_t fmode;
    err = dc1394_feature_get_mode(m_pimpl->fw_camera, FeatureToDC1394(fidx), &fmode);
    FIREWIRE_CHECK_ERROR(err);
    
    return fmode == DC1394_FEATURE_MODE_AUTO;
}

void drivers::camera::FireWire::setFeatureAuto(EFeature fidx, bool b)
{
    dc1394error_t err;
    
    err = dc1394_feature_set_mode(m_pimpl->fw_camera, FeatureToDC1394(fidx), b == true ? DC1394_FEATURE_MODE_AUTO : DC1394_FEATURE_MODE_MANUAL);
    FIREWIRE_CHECK_ERROR(err);
}

uint32_t drivers::camera::FireWire::getFeatureValue(EFeature fidx)
{
    dc1394error_t err;
    uint32_t ret = 0;
    
    err = dc1394_feature_get_value(m_pimpl->fw_camera, FeatureToDC1394(fidx), &ret);
    FIREWIRE_CHECK_ERROR(err);
    
    return ret;
}

float drivers::camera::FireWire::getFeatureValueAbs(EFeature fidx)
{
    dc1394error_t err;
    float ret = 0.0f;
    
    err = dc1394_feature_get_absolute_value(m_pimpl->fw_camera, FeatureToDC1394(fidx), &ret);
    FIREWIRE_CHECK_ERROR(err);
    
    return ret;
}

uint32_t drivers::camera::FireWire::getFeatureMin(EFeature fidx)
{
    dc1394error_t err;
    
    uint32_t vmin, vmax;
    
    err = dc1394_feature_get_boundaries(m_pimpl->fw_camera, FeatureToDC1394(fidx), &vmin, &vmax);
    FIREWIRE_CHECK_ERROR(err);
    
    return vmin;
}

uint32_t drivers::camera::FireWire::getFeatureMax(EFeature fidx)
{
    dc1394error_t err;
    
    uint32_t vmin, vmax;
    
    err = dc1394_feature_get_boundaries(m_pimpl->fw_camera, FeatureToDC1394(fidx), &vmin, &vmax);
    FIREWIRE_CHECK_ERROR(err);
    
    return vmax;
}

void drivers::camera::FireWire::setFeatureValue(EFeature fidx, uint32_t val)
{
    dc1394error_t err;
    
    err = dc1394_feature_set_value(m_pimpl->fw_camera, FeatureToDC1394(fidx), val);
    FIREWIRE_CHECK_ERROR(err);
}

void drivers::camera::FireWire::setFeatureValueAbs(EFeature fidx, float val)
{
    dc1394error_t err;
    
    err = dc1394_feature_set_absolute_value(m_pimpl->fw_camera, FeatureToDC1394(fidx), val);
    FIREWIRE_CHECK_ERROR(err);
}


