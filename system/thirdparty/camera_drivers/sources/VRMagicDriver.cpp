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
 * VRMagic Driver.
 * ****************************************************************************
 */

// VRMagicSDK
// Not my code, not my problem
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-pedantic"
#pragma GCC diagnostic ignored "-Woverflow"
#include <vrmusbcamcpp.h>
#pragma GCC diagnostic pop

#include <VRMagicDriver.hpp>

// --------------------------------------------------------------------------
// Enum mappers
// --------------------------------------------------------------------------

static constexpr VRmUsbCamCPP::ColorFormat EPixelFormatToColorFormat[] = {
    VRM_GRAY_8,
    VRM_GRAY_16,
    VRM_BGR_3X8,
};

static inline VRmUsbCamCPP::ColorFormat PixelFormatToVRFormat(drivers::camera::EPixelFormat v)
{
    if((int)v <= (int)drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB8)
    {
        return EPixelFormatToColorFormat[(int)v];
    }
    else
    {
        throw std::runtime_error("PixelFormat not supported");
    }
}

static constexpr VRmPropId RPCFeatureToVRMProperty[] = {
    /* BRIGHTNESS   */  VRM_PROPID_CAM_BRIGHTNESS_I,
    /* AUTO_EXPOSURE*/  VRM_PROPID_PLUGIN_AUTO_EXPOSURE_B,
    /* SHARPNESS    */  (VRmPropId)0x0, // UNSUPPORTED
    /* WHITE_BALANCE*/  VRM_PROPID_PLUGIN_AUTO_WHITE_BALANCE_FILTER_B,
    /* HUE          */  VRM_PROPID_CAM_HUE_I,
    /* SATURATION   */  VRM_PROPID_CAM_SATURATION_I,
    /* GAMMA        */  VRM_PROPID_FILTER_MASTER_GAMMA_F,
    /* IRIS         */  (VRmPropId)0x0, // UNSUPPORTED
    /* FOCUS        */  (VRmPropId)0x0, // UNSUPPORTED
    /* ZOOM         */  (VRmPropId)0x0, // UNSUPPORTED
    /* PAN          */  (VRmPropId)0x0, // UNSUPPORTED
    /* TILT         */  (VRmPropId)0x0, // UNSUPPORTED
    /* SHUTTER      */  VRM_PROPID_CAM_EXPOSURE_TIME1_F,
    /* GAIN         */  VRM_PROPID_CAM_GAIN_MONOCHROME_I,
    /* TRIGGER_MODE */  VRM_PROPID_CAM_TRIGGER_POLARITY_E,
    /* TRIGGER_DELAY*/  VRM_PROPID_CAM_TRIGGER_DELAY_F,
    /* FRAME_RATE   */  VRM_PROPID_CAM_FRAMERATE_MAX_F,
    /* TEMPERATURE  */  VRM_PROPID_CAM_TEMPERATURE_F
};

struct drivers::camera::VRMagic::VRMAPIPimpl
{
    VRMAPIPimpl()  { }
    uint64_t timeout_us;
    std::vector<VRmUsbCamCPP::DeviceKeyPtr> devlist;
    VRmUsbCamCPP::DevicePtr device;
    VRmUsbCamCPP::ImageFormat imgfmt;
    VRmUsbCamCPP::ImageFormat desired_format;
};
    
drivers::camera::VRMagic::VRMagic() : CameraDriverBase(), m_pimpl(new VRMAPIPimpl())
{
    m_pimpl->timeout_us = 5000000;
    m_pimpl->devlist = VRmUsbCamCPP::VRmUsbCam::get_DeviceKeyList();
    m_pimpl->device.reset();
}

drivers::camera::VRMagic::~VRMagic()
{
    close();
}

void drivers::camera::VRMagic::image_release(void* img)
{
    
}

void drivers::camera::VRMagic::open(uint32_t id)
{
    if(isOpened()) { return; }
    
    m_pimpl->device = VRmUsbCamCPP::VRmUsbCam::OpenDevice(m_pimpl->devlist[id]);
    
    m_pimpl->imgfmt = m_pimpl->device->get_SourceFormat(m_pimpl->device->get_SensorPortList()[0]);
}

bool drivers::camera::VRMagic::isOpenedImpl() const
{
    if(m_pimpl->device.get() != nullptr) 
    {
        return true;
    }
    else 
    {
        return false;
    }
}

void drivers::camera::VRMagic::close()
{
    if(!isOpened()) { return; }
    
    m_pimpl->device.reset();
}

void drivers::camera::VRMagic::setModeAndFramerate(EVideoMode vmode, EFrameRate framerate)
{
    m_width = drivers::camera::VideoModeToWidth(vmode);
    m_height = drivers::camera::VideoModeToHeight(vmode);
    m_pixfmt = drivers::camera::VideoModeToPixelFormat(vmode);
    auto fps = drivers::camera::FrameRateToFPS(framerate);
    setCustomMode(m_pixfmt, m_width, m_height);
    setFeaturePower(EFeature::FRAME_RATE, true);
    setFeatureValue(EFeature::FRAME_RATE, fps);
} 

void drivers::camera::VRMagic::setCustomMode(drivers::camera::EPixelFormat pixfmt, unsigned int width, unsigned int height, unsigned int offset_x, unsigned int offset_y, uint16_t format7mode)
{
    VRmUsbCamCPP::ColorFormat vr_cf = PixelFormatToVRFormat(pixfmt);
    m_pimpl->desired_format.set_Size(VRmUsbCamCPP::SizeI(width, height));
    m_pimpl->desired_format.set_ColorFormat(vr_cf);
    m_pimpl->desired_format.set_ImageModifier(VRM_STANDARD);
    m_width = width;
    m_height = height;
    m_pixfmt = pixfmt;
    
    bool color_format_acceptable = false;
    bool width_acceptable = false;
    bool height_acceptable = false;
    
    std::vector<VRmUsbCamCPP::ImageFormat> targetFormatList = m_pimpl->device->get_TargetFormatList(m_pimpl->device->get_SensorPortList()[0]);
    for ( size_t i = 0; i < targetFormatList.size(); ++i )
    {
        if(targetFormatList[i].get_ColorFormat() == vr_cf)
        {
            color_format_acceptable = true;
            break;
        }
    }
    
    for ( size_t i = 0; i < targetFormatList.size(); ++i )
    {
        auto size = targetFormatList[i].get_Size();
        
        if((int)width <= size.m_width)
        {
            width_acceptable = true;
            break;
        }
    }
    
    for ( size_t i = 0; i < targetFormatList.size(); ++i )
    {
        auto size = targetFormatList[i].get_Size();
        
        if((int)height <= size.m_height)
        {
            height_acceptable = true;
            break;
        }
    }
    
    if(!(color_format_acceptable && width_acceptable && height_acceptable))
    {
        throw std::runtime_error("Format not supported");
    }
}

void drivers::camera::VRMagic::start()
{
    if(isStarted()) { return; }
    
    m_pimpl->device->ResetFrameCounter();
    m_pimpl->device->Start();
    is_running = true;
}

void drivers::camera::VRMagic::stop()
{
    if(!isStarted()) { return; }
    
    m_pimpl->device->Stop();
    is_running = false;
}

bool drivers::camera::VRMagic::captureFrameImpl(FrameBuffer* cf1, FrameBuffer* cf2, FrameBuffer* cf3, FrameBuffer* cf4, int64_t timeout)   
{
    VRmUsbCamCPP::ImagePtr p_source_img;
    int frames_dropped;
    try
    {
        // wait for new frame
        p_source_img = m_pimpl->device->LockNextImage(m_pimpl->device->get_SensorPortList()[0], &frames_dropped, timeout / 1000000);
        
        std::chrono::high_resolution_clock::time_point tp = std::chrono::high_resolution_clock::now();
        std::chrono::nanoseconds d = tp.time_since_epoch();
        
        if(cf1 != nullptr)
        {
            // allocate if different
            cf1->create(m_pimpl->desired_format.get_Size().m_width, m_pimpl->desired_format.get_Size().m_height, m_pixfmt);
            
            // wrap around VR SDK Structures
            VRmUsbCamCPP::ImagePtr p_target_img = VRmUsbCamCPP::VRmUsbCam::SetImage(m_pimpl->desired_format, cf1->getData(), cf1->getStride());
            
            // convert if neccessary
            VRmUsbCamCPP::VRmUsbCam::ConvertImage(p_source_img, p_target_img);
            
            // now we can unlock source image
            m_pimpl->device->UnlockNextImage(p_source_img);
            
            // fill extra variables
            cf1->setPCTimeStamp(d.count());
            cf1->setTimeStamp(p_target_img->get_TimeStamp());
            cf1->setFrameCounter(p_target_img->get_FrameCounter());
            cf1->setFrameRate(boost::get<float>(m_pimpl->device->get_PropertyValue(VRM_PROPID_GRAB_FRAMERATE_AVERAGE_F)));
            cf1->setFramesDropped(frames_dropped);
            
            // now we don't need the wrapper
            // TODO FIXME hope it doesn't deallocate our stuff
            p_target_img.reset();
        }
        
        return true;
    }
    catch(const VRmUsbCamCPP::Exception& ex)
    {
        switch(ex.get_Number())
        {
            case (int)VRM_ERROR_CODE_FUNCTION_CALL_TIMEOUT:
            case (int)VRM_ERROR_CODE_TRIGGER_TIMEOUT:
            case (int)VRM_ERROR_CODE_TRIGGER_STALL:
                break;                
            case (int)VRM_ERROR_CODE_GENERIC_ERROR:
            default:
                throw;
        }
    }
    
    return false;
}

bool drivers::camera::VRMagic::getFeaturePower(EFeature fidx)
{
    // TODO FIXME
    return false;
}

void drivers::camera::VRMagic::setFeaturePower(EFeature fidx, bool b)
{
    // TODO FIXME
}

bool drivers::camera::VRMagic::getFeatureAuto(EFeature fidx)
{
    // TODO FIXME
    return false;
}

void drivers::camera::VRMagic::setFeatureAuto(EFeature fidx, bool b)
{
    // TODO FIXME
}

uint32_t drivers::camera::VRMagic::getFeatureValue(EFeature fidx)
{
    // TODO FIXME
    return 0;
}

float drivers::camera::VRMagic::getFeatureValueAbs(EFeature fidx)
{
    // TODO FIXME
    return 0.0f;
}

uint32_t drivers::camera::VRMagic::getFeatureMin(EFeature fidx)
{
    // TODO FIXME
    return 0;
}

uint32_t drivers::camera::VRMagic::getFeatureMax(EFeature fidx)
{
    // TODO FIXME
    return 0;
}

void drivers::camera::VRMagic::setFeatureValue(EFeature fidx, uint32_t val)
{
    // TODO FIXME
}

void drivers::camera::VRMagic::setFeatureValueAbs(EFeature fidx, float val)
{

}

void drivers::camera::VRMagic::setFeature(EFeature fidx, bool power, bool automatic, uint32_t val)
{
    // TODO FIXME
}
