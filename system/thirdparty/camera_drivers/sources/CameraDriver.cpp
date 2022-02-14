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
 * Camera Driver Abstraction.
 * ****************************************************************************
 */

#include <CameraDrivers.hpp>

struct PixelFormatInfo
{
    const char*     Name;
    std::size_t     ChannelCount;
    std::size_t     BytesPerChannel;
};

static const PixelFormatInfo PixelFormatInfoMap[] =
{
    {"PIXEL_FORMAT_MONO8", 1, sizeof(uint8_t)},
    {"PIXEL_FORMAT_MONO16", 1, sizeof(uint16_t)},
    {"PIXEL_FORMAT_RGB8", 3, sizeof(uint8_t)},
    {"PIXEL_FORMAT_BGR8", 3, sizeof(uint8_t)},
    {"PIXEL_FORMAT_RGBA8", 4, sizeof(uint8_t)},
    {"PIXEL_FORMAT_BGRA8", 4, sizeof(uint8_t)},
    {"PIXEL_FORMAT_MONO32F", 1, sizeof(float)},
    {"PIXEL_FORMAT_RGB32F", 3, sizeof(float)},
    {"PIXEL_FORMAT_DEPTH_U16", 1, sizeof(uint16_t)},
    {"PIXEL_FORMAT_DEPTH_U16_1MM", 1, sizeof(uint8_t)},
    {"PIXEL_FORMAT_DEPTH_U16_100UM", 1, sizeof(uint8_t)},
    {"PIXEL_FORMAT_DEPTH_F32_M", 1, sizeof(float)},
};

struct VideoModeInfo
{
    const char*                     Name;
    std::size_t                     Width;
    std::size_t                     Height;
    drivers::camera::EPixelFormat   PixelFormat;
};

static const VideoModeInfo VideoModeInfoMap[] =
{
    {"VIDEOMODE_640x480RGB", 640, 480, drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB8 },
    {"VIDEOMODE_640x480Y8", 640, 480, drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO8 },
    {"VIDEOMODE_640x480Y16", 640, 480, drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO16 },
    {"VIDEOMODE_800x600RGB", 800, 600, drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB8 },
    {"VIDEOMODE_800x600Y8", 800, 600, drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO8 },
    {"VIDEOMODE_800x600Y16", 800, 600, drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO16 },
    {"VIDEOMODE_1024x768RGB", 1024, 768, drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB8 },
    {"VIDEOMODE_1024x768Y8", 1024, 768, drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO8 },
    {"VIDEOMODE_1024x768Y16", 1024, 768, drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO16 },
    {"VIDEOMODE_1280x960RGB", 1280, 960, drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB8 },
    {"VIDEOMODE_1280x960Y8", 1280, 960, drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO8 },
    {"VIDEOMODE_1280x960Y16", 1280, 960, drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO16 },
    {"VIDEOMODE_1600x1200RGB", 1600, 1200, drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB8 },
    {"VIDEOMODE_1600x1200Y8", 1600, 1200, drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO8 },
    {"VIDEOMODE_1600x1200Y16", 1600, 1200, drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO16 },
    {"VIDEOMODE_CUSTOM", 0, 0, drivers::camera::EPixelFormat::PIXEL_FORMAT_UNSUPPORTED },
};

struct FrameRateInfo
{
    const char* Name;
    std::size_t FPS;
};

static const FrameRateInfo FrameRateInfoMap[] =
{
    {"FRAMERATE_15", 15},
    {"FRAMERATE_30", 30},
    {"FRAMERATE_60", 60},
    {"FRAMERATE_120", 120},
    {"FRAMERATE_240", 240},
    {"FRAMERATE_CUSTOM", 0},
};

struct FeatureInfo
{
    const char* Name;
    bool        CanAuto;
    bool        CanAbsolute;
};

static const FeatureInfo FeatureInfoMap[] =
{
    {"BRIGHTNESS", false, true},
    {"EXPOSURE", true, true},
    {"SHARPNESS", false, true},
    {"WHITE_BALANCE", true, true},
    {"HUE", false, true},
    {"SATURATION", false, true},
    {"GAMMA", false, true},
    {"IRIS", false, true},
    {"FOCUS", false, true},
    {"ZOOM", false, true},
    {"PAN", false, true},
    {"TILT", false, true},
    {"SHUTTER", false, true},
    {"GAIN", true, true},
    {"TRIGGER_MODE", false, true},
    {"TRIGGER_DELAY", false, true},
    {"FRAME_RATE", true, true},
    {"TEMPERATURE", false, true},
};

const char* drivers::camera::PixelFormatToString(drivers::camera::EPixelFormat v)
{
    if((int)v < (int)drivers::camera::EPixelFormat::PIXEL_FORMAT_UNSUPPORTED)
    {
        const PixelFormatInfo& pfi = PixelFormatInfoMap[(int)v];
        return pfi.Name;
    }
    else
    {
        return "Unsupported";
    }
}

std::size_t drivers::camera::PixelFormatToBytesPerPixel(drivers::camera::EPixelFormat v)
{
    if((int)v < (int)drivers::camera::EPixelFormat::PIXEL_FORMAT_UNSUPPORTED)
    {
        const PixelFormatInfo& pfi = PixelFormatInfoMap[(int)v];
        return pfi.ChannelCount * pfi.BytesPerChannel;
    }
    else
    {
        return 0;
    }
}

std::size_t drivers::camera::PixelFormatToChannelCount(drivers::camera::EPixelFormat v)
{
    if((int)v < (int)drivers::camera::EPixelFormat::PIXEL_FORMAT_UNSUPPORTED)
    {
        const PixelFormatInfo& pfi = PixelFormatInfoMap[(int)v];
        return pfi.ChannelCount;
    }
    else
    {
        return 0;
    }
}

const char* drivers::camera::VideoModeToString(drivers::camera::EVideoMode v)
{
    if((int)v < (int)drivers::camera::EVideoMode::VIDEOMODE_UNSUPPORTED)
    {
        const VideoModeInfo& vmi = VideoModeInfoMap[(int)v];
        return vmi.Name;
    }
    else
    {
        return "Unsupported";
    }
}

std::size_t drivers::camera::VideoModeToWidth(drivers::camera::EVideoMode v)
{
    if((int)v < (int)drivers::camera::EVideoMode::VIDEOMODE_UNSUPPORTED)
    {
        const VideoModeInfo& vmi = VideoModeInfoMap[(int)v];
        return vmi.Width;
    }
    else 
    { 
        return 0; 
    }
}

std::size_t drivers::camera::VideoModeToHeight(drivers::camera::EVideoMode v)
{
    if((int)v < (int)drivers::camera::EVideoMode::VIDEOMODE_UNSUPPORTED)
    {
        const VideoModeInfo& vmi = VideoModeInfoMap[(int)v];
        return vmi.Height;
    }
    else
    {
        return 0;
    }
}

drivers::camera::EPixelFormat drivers::camera::VideoModeToPixelFormat(drivers::camera::EVideoMode v)
{
    if((int)v < (int)drivers::camera::EVideoMode::VIDEOMODE_UNSUPPORTED)
    {
        const VideoModeInfo& vmi = VideoModeInfoMap[(int)v];
        return vmi.PixelFormat;
    }
    else 
    { 
        return drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO8; 
    }
}

std::size_t drivers::camera::VideoModeToChannels(drivers::camera::EVideoMode v)
{
    return drivers::camera::PixelFormatToChannelCount(drivers::camera::VideoModeToPixelFormat(v));
}

const char* drivers::camera::FrameRateToString(drivers::camera::EFrameRate fr)
{
    if((int)fr < (int)drivers::camera::EFrameRate::FRAMERATE_UNSUPPORTED)
    {
        const FrameRateInfo& fri = FrameRateInfoMap[(int)fr];
        return fri.Name;
    }
    else
    {
        return "Unsupported";
    }
}

std::size_t drivers::camera::FrameRateToFPS(drivers::camera::EFrameRate fr)
{
    if((int)fr < (int)drivers::camera::EFrameRate::FRAMERATE_UNSUPPORTED)
    {
        const FrameRateInfo& fri = FrameRateInfoMap[(int)fr];
        return fri.FPS;
    }
    else 
    { 
        return 0; 
    }
}

const char* drivers::camera::FeatureToString(drivers::camera::EFeature ef)
{
    if((int)ef < (int)drivers::camera::EFeature::UNSUPPORTED)
    {
        const FeatureInfo& fi = FeatureInfoMap[(int)ef];
        return fi.Name;
    }
    else
    {
        return "Unsupported";
    }
}

void drivers::camera::FrameBuffer::create(drivers::camera::EVideoMode vm)
{
    const VideoModeInfo& vmi = VideoModeInfoMap[(int)vm];
    create(vmi.Width, vmi.Height, vmi.PixelFormat, PixelFormatToChannelCount(vmi.PixelFormat));
}

void drivers::camera::FrameBuffer::create(std::size_t awidth, std::size_t aheight, drivers::camera::EPixelFormat apixfmt, std::size_t astride)
{
    width = awidth;
    height = aheight;
    pixfmt = apixfmt;
    bpp = PixelFormatToBytesPerPixel(pixfmt);
    if(astride == 0) { stride = width * PixelFormatToBytesPerPixel(pixfmt); } else { stride = astride; }
    unsigned int new_data_size = stride * height;
    if(new_data_size != data_size) // only reallocate when necessary
    {
        // release what we might have now
        release();
        
        data_size = new_data_size;
        create_byte_array(data_size);
    }
}

void drivers::camera::FrameBuffer::create(uint8_t* abuffer, std::size_t awidth, std::size_t aheight, drivers::camera::EPixelFormat apixfmt, std::size_t astride)
{
    // release what we might have now, a must for external buffers
    release();
    
    deleter = std::bind(&FrameBuffer::delete_nothing, this, std::placeholders::_1);
    associated_buffer = abuffer;
    memaccess = abuffer;
    width = awidth;
    height = aheight;
    pixfmt = apixfmt;
    bpp = PixelFormatToBytesPerPixel(pixfmt);
    if(astride == 0) { stride = width * PixelFormatToBytesPerPixel(pixfmt); } else { stride = astride; }
    data_size = stride * height;
}

void drivers::camera::FrameBuffer::create(void* extobject, std::function<void (void*)> adeleter, uint8_t* abuffer, std::size_t awidth, std::size_t aheight, drivers::camera::EPixelFormat apixfmt, std::size_t astride)
{
    // release what we might have now, a must for external buffers
    release();
    
    associated_buffer = extobject;
    deleter = adeleter;
    memaccess = abuffer;
    width = awidth;
    height = aheight;
    pixfmt = apixfmt;
    bpp = PixelFormatToBytesPerPixel(pixfmt);
    if(astride == 0) { stride = width * PixelFormatToBytesPerPixel(pixfmt); } else { stride = astride; }
    data_size = stride * height;
}

void drivers::camera::FrameBuffer::release() 
{ 
    if(isValid())
    {
        if(deleter) 
        { 
            deleter(associated_buffer);  
        }
        
        memaccess = nullptr; 
        
    }
}
