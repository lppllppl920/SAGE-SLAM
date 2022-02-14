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
 * PointGrey FlyCapture2 Driver.
 * ****************************************************************************
 */

#include <PointGreyDriver.hpp>

// FlyCapture2
// Not my code, not my problem
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-pedantic"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#include <FlyCapture2.h>
#include <FlyCapture2GUI.h>
#pragma GCC diagnostic pop

#define PG_CHECK_ERROR(err_code) { if(err_code != FlyCapture2::PGRERROR_OK) { throw PGException(&err_code, __FILE__, __LINE__); } }

// --------------------------------------------------------------------------
// Enum mappers
// --------------------------------------------------------------------------

static inline FlyCapture2::VideoMode VideoModeToFlyCapture(drivers::camera::EVideoMode v)
{
    switch(v)
    {
        case drivers::camera::EVideoMode::VIDEOMODE_640x480RGB   : return FlyCapture2::VIDEOMODE_640x480RGB;
        case drivers::camera::EVideoMode::VIDEOMODE_640x480Y8    : return FlyCapture2::VIDEOMODE_640x480Y8;
        case drivers::camera::EVideoMode::VIDEOMODE_640x480Y16   : return FlyCapture2::VIDEOMODE_640x480Y16;
        case drivers::camera::EVideoMode::VIDEOMODE_800x600RGB   : return FlyCapture2::VIDEOMODE_800x600RGB;
        case drivers::camera::EVideoMode::VIDEOMODE_800x600Y8    : return FlyCapture2::VIDEOMODE_800x600Y8;
        case drivers::camera::EVideoMode::VIDEOMODE_800x600Y16   : return FlyCapture2::VIDEOMODE_800x600Y16;
        case drivers::camera::EVideoMode::VIDEOMODE_1024x768RGB  : return FlyCapture2::VIDEOMODE_1024x768RGB;
        case drivers::camera::EVideoMode::VIDEOMODE_1024x768Y8   : return FlyCapture2::VIDEOMODE_1024x768Y8;
        case drivers::camera::EVideoMode::VIDEOMODE_1024x768Y16  : return FlyCapture2::VIDEOMODE_1024x768Y16;
        case drivers::camera::EVideoMode::VIDEOMODE_1280x960RGB  : return FlyCapture2::VIDEOMODE_1280x960RGB;
        case drivers::camera::EVideoMode::VIDEOMODE_1280x960Y8   : return FlyCapture2::VIDEOMODE_1280x960Y8;
        case drivers::camera::EVideoMode::VIDEOMODE_1280x960Y16  : return FlyCapture2::VIDEOMODE_1280x960Y16;
        case drivers::camera::EVideoMode::VIDEOMODE_1600x1200RGB : return FlyCapture2::VIDEOMODE_1600x1200RGB;
        case drivers::camera::EVideoMode::VIDEOMODE_1600x1200Y8  : return FlyCapture2::VIDEOMODE_1600x1200Y8;
        case drivers::camera::EVideoMode::VIDEOMODE_1600x1200Y16 : return FlyCapture2::VIDEOMODE_1600x1200Y16;
        case drivers::camera::EVideoMode::VIDEOMODE_CUSTOM       : return FlyCapture2::VIDEOMODE_FORMAT7;
        default: return FlyCapture2::NUM_VIDEOMODES;
    }
}

static inline FlyCapture2::FrameRate FrameRateToFlyCapture(drivers::camera::EFrameRate v)
{
    switch(v)
    {
        case drivers::camera::EFrameRate::FRAMERATE_15     : return FlyCapture2::FRAMERATE_15;
        case drivers::camera::EFrameRate::FRAMERATE_30     : return FlyCapture2::FRAMERATE_30;
        case drivers::camera::EFrameRate::FRAMERATE_60     : return FlyCapture2::FRAMERATE_60;
        case drivers::camera::EFrameRate::FRAMERATE_120    : return FlyCapture2::FRAMERATE_120;
        case drivers::camera::EFrameRate::FRAMERATE_240    : return FlyCapture2::FRAMERATE_240;
        case drivers::camera::EFrameRate::FRAMERATE_CUSTOM : return FlyCapture2::FRAMERATE_FORMAT7;
        default: return FlyCapture2::NUM_FRAMERATES;
    }
}

static inline FlyCapture2::PropertyType FeatureToFlyCapture(drivers::camera::EFeature v)
{
    switch(v)
    {
        case drivers::camera::EFeature::BRIGHTNESS   : return FlyCapture2::BRIGHTNESS;
        case drivers::camera::EFeature::EXPOSURE     : return FlyCapture2::AUTO_EXPOSURE;
        case drivers::camera::EFeature::SHARPNESS    : return FlyCapture2::SHARPNESS;
        case drivers::camera::EFeature::WHITE_BALANCE: return FlyCapture2::WHITE_BALANCE;
        case drivers::camera::EFeature::HUE          : return FlyCapture2::HUE;
        case drivers::camera::EFeature::SATURATION   : return FlyCapture2::SATURATION;
        case drivers::camera::EFeature::GAMMA        : return FlyCapture2::GAMMA;
        case drivers::camera::EFeature::IRIS         : return FlyCapture2::IRIS;
        case drivers::camera::EFeature::FOCUS        : return FlyCapture2::FOCUS;
        case drivers::camera::EFeature::ZOOM         : return FlyCapture2::ZOOM;
        case drivers::camera::EFeature::PAN          : return FlyCapture2::PAN;
        case drivers::camera::EFeature::TILT         : return FlyCapture2::TILT;
        case drivers::camera::EFeature::SHUTTER      : return FlyCapture2::SHUTTER;
        case drivers::camera::EFeature::GAIN         : return FlyCapture2::GAIN;
        case drivers::camera::EFeature::TRIGGER_MODE : return FlyCapture2::TRIGGER_MODE;
        case drivers::camera::EFeature::TRIGGER_DELAY: return FlyCapture2::TRIGGER_DELAY;
        case drivers::camera::EFeature::FRAME_RATE   : return FlyCapture2::FRAME_RATE;
        case drivers::camera::EFeature::TEMPERATURE  : return FlyCapture2::TEMPERATURE;
        default: return FlyCapture2::UNSPECIFIED_PROPERTY_TYPE;
    }
}

drivers::camera::PointGrey::PGException::PGException(const FlyCapture2::Error* fcrc, const char* file, int line)
{ 
    std::stringstream ss;
    ss << "PG FC2 Error " << fcrc->GetType() << " , " << fcrc->GetDescription() << " , " << fcrc->GetFilename() << ":" << fcrc->GetLine() << " and " << file << ":" << line;
    errormsg = ss.str();
}

struct drivers::camera::PointGrey::PGAPIPimpl
{
    PGAPIPimpl() : m_maxnumcam(0) { }
    FlyCapture2::BusManager m_busman;
    unsigned int m_maxnumcam;
    FlyCapture2::PGRGuid m_uid;
    FlyCapture2::Camera m_cam;
    FlyCapture2::CameraInfo m_cam_info;
};
    
drivers::camera::PointGrey::PointGrey() : CameraDriverBase(), m_pimpl(new PGAPIPimpl()), m_width(0), m_height(0), m_pixfmt(EPixelFormat::PIXEL_FORMAT_MONO8), m_camid(0), is_running(false)
{
    FlyCapture2::Error fcrc;
    
    fcrc = m_pimpl->m_busman.GetNumOfCameras(&m_pimpl->m_maxnumcam);
    PG_CHECK_ERROR(fcrc);
}

drivers::camera::PointGrey::~PointGrey()
{
    close();
}

void drivers::camera::PointGrey::image_release(void* img)
{
    delete static_cast<FlyCapture2::Image*>(img);
}

void drivers::camera::PointGrey::open(uint32_t id)
{
    if(id >= m_pimpl->m_maxnumcam) { throw std::runtime_error("Wrong ID"); }
    
    FlyCapture2::Error fcrc;
    
    fcrc = m_pimpl->m_busman.GetCameraFromIndex(id, &m_pimpl->m_uid);
    PG_CHECK_ERROR(fcrc);
    
    m_camid = id;
    
    fcrc = m_pimpl->m_cam.Connect(&m_pimpl->m_uid);
    PG_CHECK_ERROR(fcrc);
    
    fcrc = m_pimpl->m_cam.GetCameraInfo(&m_pimpl->m_cam_info);
    PG_CHECK_ERROR(fcrc);
    
    FlyCapture2::FC2Config cur_cfg;
    
    fcrc = m_pimpl->m_cam.GetConfiguration(&cur_cfg);
    PG_CHECK_ERROR(fcrc);
    
    // fill Camera Info
    std::stringstream ss;
    ss << m_pimpl->m_cam_info.sensorInfo;
    m_cinfo.SerialNumber = ss.str();
    switch(m_pimpl->m_cam_info.interfaceType)
    {
        case FlyCapture2::INTERFACE_IEEE1394:
            m_cinfo.InterfaceType = CameraInfo::CameraInterface::IEEE1394;
            break;
        case FlyCapture2::INTERFACE_USB2:
        case FlyCapture2::INTERFACE_USB3:
            m_cinfo.InterfaceType = CameraInfo::CameraInterface::USB;
            break;
        case FlyCapture2::INTERFACE_GIGE:
            m_cinfo.InterfaceType = CameraInfo::CameraInterface::GIGE;
            break;
        case FlyCapture2::INTERFACE_UNKNOWN:
        default:
            m_cinfo.InterfaceType = CameraInfo::CameraInterface::UNKNOWN;
            break;
    }
    m_cinfo.IsColorCamera = m_pimpl->m_cam_info.isColorCamera;
    m_cinfo.ModelName = std::string(m_pimpl->m_cam_info.modelName);
    m_cinfo.VendorName = std::string(m_pimpl->m_cam_info.vendorName);
    m_cinfo.SensorInfo = std::string(m_pimpl->m_cam_info.sensorInfo);
    m_cinfo.SensorResolution = std::string(m_pimpl->m_cam_info.sensorResolution);
    m_cinfo.DriverName = std::string(m_pimpl->m_cam_info.driverName);
    m_cinfo.FirmwareVersion = std::string(m_pimpl->m_cam_info.firmwareVersion);
    m_cinfo.FirmwareBuildTime = std::string(m_pimpl->m_cam_info.firmwareBuildTime);
}

bool drivers::camera::PointGrey::isOpenedImpl() const
{
    return m_pimpl->m_cam.IsConnected();
}

void drivers::camera::PointGrey::close()
{
    if(isOpened())
    {
        m_pimpl->m_cam.Disconnect();
    }
}

void drivers::camera::PointGrey::setModeAndFramerate(EVideoMode vmode, EFrameRate framerate)
{
    FlyCapture2::Error fcrc = m_pimpl->m_cam.SetVideoModeAndFrameRate(VideoModeToFlyCapture(vmode), FrameRateToFlyCapture(framerate));
    PG_CHECK_ERROR(fcrc);

    m_width = drivers::camera::VideoModeToWidth(vmode);
    m_height = drivers::camera::VideoModeToHeight(vmode);
    m_pixfmt = drivers::camera::VideoModeToPixelFormat(vmode);
} 

void drivers::camera::PointGrey::setCustomMode(drivers::camera::EPixelFormat pixfmt, unsigned int width, unsigned int height, unsigned int offset_x, unsigned int offset_y, uint16_t format7mode)
{
    FlyCapture2::Error fcrc;
    
    // Get Format7 information
    FlyCapture2::Format7Info fmt7Info;
    bool supported = false;
    FlyCapture2::Mode fmt7Mode = (FlyCapture2::Mode)((int)FlyCapture2::MODE_0 + format7mode);
    fmt7Info.mode = fmt7Mode;
    fcrc = m_pimpl->m_cam.GetFormat7Info( &fmt7Info, &supported );
    PG_CHECK_ERROR(fcrc);
    
    if(!supported)
    {
        throw std::runtime_error("PointGreyCamera::setFormat7 Format 7 mode not supported on this camera.");
    }
    
    // Make Format7 Configuration
    FlyCapture2::Format7ImageSettings fmt7ImageSettings;
    fmt7ImageSettings.mode = fmt7Mode;
    FlyCapture2::PixelFormat fmt7PixFmt;
    switch(pixfmt)
    {
        case drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO8: fmt7PixFmt = FlyCapture2::PixelFormat::PIXEL_FORMAT_MONO8; break;
        case drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO16: fmt7PixFmt = FlyCapture2::PixelFormat::PIXEL_FORMAT_MONO16; break;
        case drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB8: fmt7PixFmt = FlyCapture2::PixelFormat::PIXEL_FORMAT_RGB8; break;
        default: throw std::runtime_error("Pixel format not supported");
    }
    fmt7ImageSettings.pixelFormat = fmt7PixFmt;
    
    // Check Width
    if(fmt7Info.imageHStepSize > 0)
    {
        width = width / fmt7Info.imageHStepSize * fmt7Info.imageHStepSize; // Locks the width into an appropriate multiple using an integer divide
    }
    if(width == 0)
    {
        fmt7ImageSettings.width = fmt7Info.maxWidth;
    } 
    else if(width > fmt7Info.maxWidth)
    {
        width = fmt7Info.maxWidth;
        fmt7ImageSettings.width = fmt7Info.maxWidth;
    } 
    else 
    {
        fmt7ImageSettings.width = width;
    }
    
    // Check Height
    if(fmt7Info.imageVStepSize > 0)
    {
        height = height / fmt7Info.imageVStepSize * fmt7Info.imageVStepSize; // Locks the height into an appropriate multiple using an integer divide
    }
    if(height == 0)
    {
        fmt7ImageSettings.height = fmt7Info.maxHeight;
    } 
    else if(height > fmt7Info.maxHeight)
    {
        height = fmt7Info.maxHeight;
        fmt7ImageSettings.height = fmt7Info.maxHeight;
    } 
    else 
    {
        fmt7ImageSettings.height = height;
    }
    
    // Check OffsetX
    if(fmt7Info.offsetHStepSize > 0)
    {
        offset_x = offset_x / fmt7Info.offsetHStepSize * fmt7Info.offsetHStepSize;  // Locks the X offset into an appropriate multiple using an integer divide
    }
    if(offset_x > (fmt7Info.maxWidth - fmt7ImageSettings.width))
    {
        offset_x = fmt7Info.maxWidth - fmt7ImageSettings.width;
    }
    fmt7ImageSettings.offsetX  = offset_x;
    
    // Check OffsetY
    if(fmt7Info.offsetVStepSize > 0)
    {
        offset_y = offset_y / fmt7Info.offsetVStepSize * fmt7Info.offsetVStepSize;  // Locks the X offset into an appropriate multiple using an integer divide
    }
    if(offset_y > fmt7Info.maxHeight - fmt7ImageSettings.height)
    {
        offset_y = fmt7Info.maxHeight - fmt7ImageSettings.height;
    }
    fmt7ImageSettings.offsetY  = offset_y;
    
    // Validate the settings to make sure that they are valid
    FlyCapture2::Format7PacketInfo fmt7PacketInfo;
    bool valid;
    fcrc = m_pimpl->m_cam.ValidateFormat7Settings(&fmt7ImageSettings, &valid, &fmt7PacketInfo );
    PG_CHECK_ERROR(fcrc);
    if (!valid)
    {
        throw std::runtime_error("PointGreyCamera::setFormat7 Format 7 Settings Not Valid.");
    }
    
    m_width = fmt7ImageSettings.width;
    m_height = fmt7ImageSettings.height;
    m_pixfmt = pixfmt;
    
    // Stop the camera to allow settings to change.
    fcrc = m_pimpl->m_cam.SetFormat7Configuration(&fmt7ImageSettings, fmt7PacketInfo.recommendedBytesPerPacket);
    PG_CHECK_ERROR(fcrc);   
    
    // switch to format 7
    //fcrc = m_pimpl->m_cam.SetVideoModeAndFrameRate(FlyCapture2::VIDEOMODE_FORMAT7, FlyCapture2::FRAMERATE_FORMAT7);
    //PG_CHECK_ERROR(fcrc);
}

void drivers::camera::PointGrey::start()
{
    if(is_running) { return; }
    
    FlyCapture2::Error fcrc = m_pimpl->m_cam.StartCapture();
    PG_CHECK_ERROR(fcrc);
    is_running = true;
}

void drivers::camera::PointGrey::start(drivers::camera::FrameCallback c)
{
    if(is_running) { return; }
    
    cb = c;
    FlyCapture2::Error fcrc = m_pimpl->m_cam.StartCapture(pgCallback, this);
    PG_CHECK_ERROR(fcrc);
    is_running = true;
}

void drivers::camera::PointGrey::pgCallback(FlyCapture2::Image* img, const void* cbdata)
{
    const drivers::camera::PointGrey* driver = (const drivers::camera::PointGrey*)cbdata;
    FrameBuffer fb;
    
    std::chrono::high_resolution_clock::time_point tp = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds d = tp.time_since_epoch();
    
    drivers::camera::EPixelFormat pixfmt;
    switch(img->GetPixelFormat())
    {
        case FlyCapture2::PIXEL_FORMAT_MONO8:
            pixfmt = drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO8;
            break;
        case FlyCapture2::PIXEL_FORMAT_MONO16:
            pixfmt = drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO16;
            break;
        case FlyCapture2::PIXEL_FORMAT_RGB8:
            pixfmt = drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB8;
            break;
        default: // unknown
            throw std::runtime_error("Unsupported mode");
            break;
    }
    
    fb.create(img->GetData(), img->GetCols(), img->GetRows(), pixfmt, img->GetStride());
    fb.setPCTimeStamp(d.count());
    fb.setTimeStamp(img->GetMetadata().embeddedTimeStamp);
    fb.setGain(img->GetMetadata().embeddedGain);
    fb.setShutter(img->GetMetadata().embeddedShutter);
    fb.setBrightness(img->GetMetadata().embeddedBrightness);
    fb.setExposure(img->GetMetadata().embeddedExposure);
    fb.setWhiteBalance(img->GetMetadata().embeddedWhiteBalance);
    fb.setFrameCounter(img->GetMetadata().embeddedFrameCounter);
    
    if(driver->cb)
    {
        driver->cb(fb);
    }
}

void drivers::camera::PointGrey::stop()
{
    if(!is_running) { return; }
    
    FlyCapture2::Error fcrc = m_pimpl->m_cam.StopCapture();
    PG_CHECK_ERROR(fcrc);
    is_running = false;
}

bool drivers::camera::PointGrey::captureFrameImpl(FrameBuffer* cf1, FrameBuffer* cf2, FrameBuffer* cf3, FrameBuffer* cf4, int64_t timeout)
{
    FlyCapture2::Image* m_rawimage = new FlyCapture2::Image();
    
    FlyCapture2::Error fcrc = m_pimpl->m_cam.RetrieveBuffer( m_rawimage );
    if(fcrc != FlyCapture2::PGRERROR_OK) 
    { 
        if((fcrc == FlyCapture2::PGRERROR_TIMEOUT) || (fcrc == FlyCapture2::PGRERROR_ISOCH_NOT_STARTED) || (fcrc == FlyCapture2::PGRERROR_NOT_CONNECTED))
        {
            return false; 
        }
        else
        {
            PG_CHECK_ERROR(fcrc);
        }
    } 
    
    std::chrono::high_resolution_clock::time_point tp = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds d = tp.time_since_epoch();
    
    drivers::camera::EPixelFormat pixfmt;
    switch(m_rawimage->GetPixelFormat())
    {
        case FlyCapture2::PIXEL_FORMAT_MONO8:
            pixfmt = drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO8;
            break;
        case FlyCapture2::PIXEL_FORMAT_MONO16:
            pixfmt = drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO16;
            break;
        case FlyCapture2::PIXEL_FORMAT_RGB8:
            pixfmt = drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB8;
            break;
        default: // unknown
            throw std::runtime_error("Unsupported mode");
            break;
    }
    
    if(cf1 != nullptr)
    {
        cf1->create(m_rawimage, std::bind(&drivers::camera::PointGrey::image_release, this, std::placeholders::_1), m_rawimage->GetData(), m_rawimage->GetCols(), m_rawimage->GetRows(), pixfmt, m_rawimage->GetStride());
        
        cf1->setPCTimeStamp(d.count());
        cf1->setTimeStamp(m_rawimage->GetMetadata().embeddedTimeStamp);
        cf1->setGain(m_rawimage->GetMetadata().embeddedGain);
        cf1->setShutter(m_rawimage->GetMetadata().embeddedShutter);
        cf1->setBrightness(m_rawimage->GetMetadata().embeddedBrightness);
        cf1->setExposure(m_rawimage->GetMetadata().embeddedExposure);
        cf1->setWhiteBalance(m_rawimage->GetMetadata().embeddedWhiteBalance);
        cf1->setFrameCounter(m_rawimage->GetMetadata().embeddedFrameCounter);
    }
    else
    {
        delete m_rawimage;
    }

    return true;
}

bool drivers::camera::PointGrey::getFeaturePower(EFeature fidx)
{
    FlyCapture2::PropertyType propt = FeatureToFlyCapture(fidx);
    
    FlyCapture2::Property p(propt);
    
    FlyCapture2::Error fcrc = m_pimpl->m_cam.GetProperty(&p);
    PG_CHECK_ERROR(fcrc);
    
    return p.onOff;
}

void drivers::camera::PointGrey::setFeaturePower(EFeature fidx, bool b)
{
    FlyCapture2::PropertyType propt = FeatureToFlyCapture(fidx);
    
    FlyCapture2::Property p(propt);
    
    FlyCapture2::Error fcrc = m_pimpl->m_cam.GetProperty(&p);
    PG_CHECK_ERROR(fcrc);
    
    p.onOff = b;
    
    fcrc = m_pimpl->m_cam.SetProperty(&p);
    PG_CHECK_ERROR(fcrc);
}

bool drivers::camera::PointGrey::getFeatureAuto(EFeature fidx)
{
    FlyCapture2::PropertyType propt = FeatureToFlyCapture(fidx);
    
    FlyCapture2::Property p(propt);
    
    FlyCapture2::Error fcrc = m_pimpl->m_cam.GetProperty(&p);
    PG_CHECK_ERROR(fcrc);
    
    return p.autoManualMode;
}

void drivers::camera::PointGrey::setFeatureAuto(EFeature fidx, bool b)
{
    FlyCapture2::PropertyType propt = FeatureToFlyCapture(fidx);
    
    FlyCapture2::Property p(propt);
    
    FlyCapture2::Error fcrc = m_pimpl->m_cam.GetProperty(&p);
    PG_CHECK_ERROR(fcrc);
    
    p.autoManualMode = b;
    
    fcrc = m_pimpl->m_cam.SetProperty(&p);
    PG_CHECK_ERROR(fcrc);
}

uint32_t drivers::camera::PointGrey::getFeatureValue(EFeature fidx)
{
    FlyCapture2::PropertyType propt = FeatureToFlyCapture(fidx);
    
    FlyCapture2::Property p(propt);
    
    FlyCapture2::Error fcrc = m_pimpl->m_cam.GetProperty(&p);
    PG_CHECK_ERROR(fcrc);
    
    return p.valueA;
}

float drivers::camera::PointGrey::getFeatureValueAbs(EFeature fidx)
{
    FlyCapture2::PropertyType propt = FeatureToFlyCapture(fidx);
    
    FlyCapture2::Property p(propt);
    
    FlyCapture2::Error fcrc = m_pimpl->m_cam.GetProperty(&p);
    PG_CHECK_ERROR(fcrc);
    
    return p.absValue;
}

uint32_t drivers::camera::PointGrey::getFeatureMin(EFeature fidx)
{
    FlyCapture2::PropertyType propt = FeatureToFlyCapture(fidx);
    
    FlyCapture2::PropertyInfo pinfo(propt);
    
    FlyCapture2::Error fcrc = m_pimpl->m_cam.GetPropertyInfo(&pinfo);
    PG_CHECK_ERROR(fcrc);
    
    return pinfo.min;
}

uint32_t drivers::camera::PointGrey::getFeatureMax(EFeature fidx)
{
    FlyCapture2::PropertyType propt = FeatureToFlyCapture(fidx);
    
    FlyCapture2::PropertyInfo pinfo(propt);
    
    FlyCapture2::Error fcrc = m_pimpl->m_cam.GetPropertyInfo(&pinfo);
    PG_CHECK_ERROR(fcrc);
    
    return pinfo.max;
}

void drivers::camera::PointGrey::setFeatureValue(EFeature fidx, uint32_t val)
{
    FlyCapture2::PropertyType propt = FeatureToFlyCapture(fidx);
    
    FlyCapture2::Property p(propt);
    
    FlyCapture2::Error fcrc = m_pimpl->m_cam.GetProperty(&p);
    PG_CHECK_ERROR(fcrc);
    p.valueA = val;
    
    fcrc = m_pimpl->m_cam.SetProperty(&p);
    PG_CHECK_ERROR(fcrc);
}

void drivers::camera::PointGrey::setFeatureValueAbs(EFeature fidx, float val)
{
    FlyCapture2::PropertyType propt = FeatureToFlyCapture(fidx);
    
    FlyCapture2::Property p(propt);
    
    FlyCapture2::Error fcrc = m_pimpl->m_cam.GetProperty(&p);
    PG_CHECK_ERROR(fcrc);
    p.absValue = val;
    p.absControl = true;
    
    fcrc = m_pimpl->m_cam.SetProperty(&p);
    PG_CHECK_ERROR(fcrc);
}

void drivers::camera::PointGrey::setFeature(EFeature fidx, bool power, bool automatic, uint32_t val)
{
    FlyCapture2::PropertyType propt = FeatureToFlyCapture(fidx);
    
    FlyCapture2::Property p(propt);
    
    FlyCapture2::Error fcrc = m_pimpl->m_cam.GetProperty(&p);
    PG_CHECK_ERROR(fcrc);
    
    p.onOff = power;
    p.autoManualMode = automatic;
    p.valueA = val;
    p.onePush = false;
    p.absControl = false;
    
    fcrc = m_pimpl->m_cam.SetProperty(&p);
    PG_CHECK_ERROR(fcrc);
}

