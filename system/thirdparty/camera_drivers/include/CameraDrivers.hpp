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

#ifndef CAMERA_DRIVERS_HPP
#define CAMERA_DRIVERS_HPP

#include <stdint.h>
#include <stddef.h>
#include <stdexcept>
#include <cstring>
#include <string>
#include <sstream>
#include <memory>
#include <chrono>
#include <functional>

#ifdef CAMERA_DRIVERS_HAVE_CEREAL
#include <cereal/cereal.hpp>
#endif // CAMERA_DRIVERS_HAVE_CEREAL

namespace drivers
{
    
namespace camera
{

enum class EPixelFormat
{
    PIXEL_FORMAT_MONO8 = 0,
    PIXEL_FORMAT_MONO16,
    PIXEL_FORMAT_RGB8,
    PIXEL_FORMAT_BGR8,
    PIXEL_FORMAT_RGBA8,
    PIXEL_FORMAT_BGRA8,
    PIXEL_FORMAT_MONO32F,
    PIXEL_FORMAT_RGB32F,
    PIXEL_FORMAT_DEPTH_U16,
    PIXEL_FORMAT_DEPTH_U16_1MM,
    PIXEL_FORMAT_DEPTH_U16_100UM,
    PIXEL_FORMAT_DEPTH_F32_M,
    PIXEL_FORMAT_UNSUPPORTED
};

const char* PixelFormatToString(EPixelFormat epf);
std::size_t PixelFormatToBytesPerPixel(EPixelFormat epf);
std::size_t PixelFormatToChannelCount(EPixelFormat epf);
    
enum class EVideoMode
{
    VIDEOMODE_640x480RGB = 0,
    VIDEOMODE_640x480Y8,
    VIDEOMODE_640x480Y16,
    VIDEOMODE_800x600RGB, 
    VIDEOMODE_800x600Y8, 
    VIDEOMODE_800x600Y16,
    VIDEOMODE_1024x768RGB,
    VIDEOMODE_1024x768Y8,
    VIDEOMODE_1024x768Y16,
    VIDEOMODE_1280x960RGB,
    VIDEOMODE_1280x960Y8,
    VIDEOMODE_1280x960Y16,
    VIDEOMODE_1600x1200RGB,
    VIDEOMODE_1600x1200Y8,
    VIDEOMODE_1600x1200Y16,
    VIDEOMODE_CUSTOM,
    VIDEOMODE_UNSUPPORTED
};

const char* VideoModeToString(EVideoMode v);
std::size_t VideoModeToWidth(EVideoMode v);
std::size_t VideoModeToHeight(EVideoMode v);
std::size_t VideoModeToChannels(EVideoMode v);
EPixelFormat VideoModeToPixelFormat(EVideoMode v);

enum class EFrameRate
{
    FRAMERATE_15 = 0,
    FRAMERATE_30,
    FRAMERATE_60,
    FRAMERATE_120,
    FRAMERATE_240,
    FRAMERATE_CUSTOM,
    FRAMERATE_UNSUPPORTED
};

const char* FrameRateToString(EFrameRate fr);
std::size_t FrameRateToFPS(EFrameRate fr);

enum class EFeature
{
    BRIGHTNESS = 0,
    EXPOSURE,
    SHARPNESS, 
    WHITE_BALANCE,
    HUE,
    SATURATION,
    GAMMA,
    IRIS,
    FOCUS,
    ZOOM,
    PAN,
    TILT,
    SHUTTER,
    GAIN,
    TRIGGER_MODE,
    TRIGGER_DELAY,
    FRAME_RATE, 
    TEMPERATURE,
    UNSUPPORTED
};

const char* FeatureToString(EFeature ef);

/**
 * Generic frame buffer class.
 */
class FrameBuffer
{
public:
    /**
     * No image.
     */
    inline FrameBuffer() : 
        associated_buffer(nullptr),
        memaccess(nullptr),
        data_size(0), 
        width(0), 
        height(0), 
        pixfmt(EPixelFormat::PIXEL_FORMAT_MONO8),
        stride(0), 
        pcTimeStamp(0),
        embeddedTimeStamp(0), 
        embeddedGain(0), 
        embeddedShutter(0), 
        embeddedBrightness(0), 
        embeddedExposure(0), 
        embeddedWhiteBalance(0), 
        embeddedFrameCounter(0),
        embeddedFrameRate(0.0f),
        embeddedFramesDropped(0),
        video_source("")
    { 
        bpp = PixelFormatToBytesPerPixel(pixfmt);
    }
    
    /**
     * Copy creates on the heap?
     */
    inline FrameBuffer(const FrameBuffer& other) : 
        associated_buffer(nullptr),
        memaccess(nullptr),
        data_size(other.data_size), 
        width(other.width), 
        height(other.height), 
        pixfmt(other.pixfmt), 
        stride(other.stride), 
        pcTimeStamp(other.pcTimeStamp),
        embeddedTimeStamp(other.embeddedTimeStamp), 
        embeddedGain(other.embeddedGain), 
        embeddedShutter(other.embeddedShutter), 
        embeddedBrightness(other.embeddedBrightness), 
        embeddedExposure(other.embeddedExposure), 
        embeddedWhiteBalance(other.embeddedWhiteBalance), 
        embeddedFrameCounter(other.embeddedFrameCounter),
        embeddedFrameRate(other.embeddedFrameRate),
        embeddedFramesDropped(other.embeddedFramesDropped),
        video_source(other.video_source)
    {
        bpp = PixelFormatToBytesPerPixel(pixfmt);
        if(stride == 0) { stride = width * PixelFormatToBytesPerPixel(pixfmt); }
        if(other.isValid())
        {
            
            create_byte_array(data_size);
            memcpy(memaccess, other.memaccess, data_size);
        }
    }

    /**
     * Move constructor.
     */
    inline FrameBuffer(FrameBuffer&& other) : 
        deleter(std::move(other.deleter)),
        associated_buffer(other.associated_buffer),
        memaccess(other.memaccess),
        data_size(other.data_size), 
        width(other.width), 
        height(other.height), 
        pixfmt(other.pixfmt), 
        stride(other.stride), 
        pcTimeStamp(other.pcTimeStamp),
        embeddedTimeStamp(other.embeddedTimeStamp), 
        embeddedGain(other.embeddedGain), 
        embeddedShutter(other.embeddedShutter), 
        embeddedBrightness(other.embeddedBrightness), 
        embeddedExposure(other.embeddedExposure), 
        embeddedWhiteBalance(other.embeddedWhiteBalance), 
        embeddedFrameCounter(other.embeddedFrameCounter),
        embeddedFrameRate(other.embeddedFrameRate),
        embeddedFramesDropped(other.embeddedFramesDropped),
        video_source(other.video_source)
    {
        // invalidate other
        other.associated_buffer = nullptr;
        other.memaccess = nullptr;
        other.width = other.height = other.stride = other.data_size = 0;
        bpp = PixelFormatToBytesPerPixel(pixfmt);
    }

    /**
     * Create from VideoMode.
     */
    inline FrameBuffer(EVideoMode vm) : associated_buffer(nullptr), memaccess(nullptr) 
    { 
        create(vm); 
    }
    
    /**
     * Create on the heap with manual parameters.
     */
    inline FrameBuffer(std::size_t awidth, std::size_t aheight, 
                       EPixelFormat apixfmt = EPixelFormat::PIXEL_FORMAT_MONO8, 
                       std::size_t astride = 0) : associated_buffer(nullptr), memaccess(nullptr)
    {
        create(awidth, aheight, apixfmt, astride); 
    }
    
    /**
     * Create with external buffer.
     */
    inline FrameBuffer(uint8_t* abuffer, std::size_t awidth, std::size_t aheight, 
                       EPixelFormat apixfmt = EPixelFormat::PIXEL_FORMAT_MONO8, 
                       std::size_t astride = 0) : associated_buffer(nullptr), memaccess(nullptr)
    { 
        create(abuffer, awidth, aheight, apixfmt, astride);
    }
    
    /**
     * Create with an external object and its deleter.
     */
    inline FrameBuffer(void* extobject, std::function<void (void*)> adeleter, 
                       uint8_t* abuffer, std::size_t awidth, std::size_t aheight, 
                       EPixelFormat apixfmt = EPixelFormat::PIXEL_FORMAT_MONO8, 
                       std::size_t astride = 0) : associated_buffer(nullptr), memaccess(nullptr)
    {
        create(extobject, adeleter, abuffer, awidth, aheight, apixfmt, astride);
    }
    
    virtual ~FrameBuffer() 
    { 
        release(); 
    }
    
    /**
     * Copy creates on the heap?
     */
    inline FrameBuffer operator=(const FrameBuffer& other)
    {
        associated_buffer = nullptr;
        memaccess = nullptr;
        data_size = other.data_size;
        width = other.width;
        height = other.height;
        pixfmt = other.pixfmt;
        stride = other.stride;
        pcTimeStamp = other.pcTimeStamp;
        embeddedTimeStamp = other.embeddedTimeStamp;
        embeddedGain = other.embeddedGain;
        embeddedShutter = other.embeddedShutter;
        embeddedBrightness = other.embeddedBrightness;
        embeddedExposure = other.embeddedExposure;
        embeddedWhiteBalance = other.embeddedWhiteBalance;
        embeddedFrameCounter = other.embeddedFrameCounter;       
        embeddedFrameRate = other.embeddedFrameRate;
        embeddedFramesDropped = other.embeddedFramesDropped;
        video_source = other.video_source;
        
        bpp = PixelFormatToBytesPerPixel(pixfmt);
        if(stride == 0) { stride = width * PixelFormatToBytesPerPixel(pixfmt); }
        if(other.isValid())
        {
            create_byte_array(data_size);
            memcpy(memaccess, other.memaccess, data_size);
        }
        
        return *this;
    }

    /**
     * Move operator.
     */
    inline FrameBuffer& operator=(FrameBuffer&& other)
    {
        deleter = std::move(other.deleter);
        associated_buffer = other.associated_buffer;
        memaccess = other.memaccess;
        data_size = other.data_size;
        width = other.width;
        height = other.height;
        pixfmt = other.pixfmt; 
        stride = other.stride;
        pcTimeStamp = other.pcTimeStamp;
        embeddedTimeStamp = other.embeddedTimeStamp;
        embeddedGain = other.embeddedGain;
        embeddedShutter = other.embeddedShutter;
        embeddedBrightness = other.embeddedBrightness;
        embeddedExposure = other.embeddedExposure;
        embeddedWhiteBalance = other.embeddedWhiteBalance;
        embeddedFrameCounter = other.embeddedFrameCounter;
        embeddedFrameRate = other.embeddedFrameRate;
        embeddedFramesDropped = other.embeddedFramesDropped;
        video_source = other.video_source;
        
        // invalidate other
        other.associated_buffer = nullptr;
        other.memaccess = nullptr;
        other.width = other.height = other.stride = other.data_size = 0;
        bpp = PixelFormatToBytesPerPixel(pixfmt);
        
        return *this;
    }

    /**
     * Create on the heap from VideMode
     */
    void create(EVideoMode vm);
    
    /**
     * Create on the heap from manual parameters.
     */
    void create(std::size_t awidth, std::size_t aheight, 
                EPixelFormat apixfmt = EPixelFormat::PIXEL_FORMAT_MONO8, 
                std::size_t astride = 0);
    
    /**
     * Create from memory owned by someone else (no delete).
     */
    void create(uint8_t* abuffer, std::size_t awidth, std::size_t aheight, 
                EPixelFormat apixfmt = EPixelFormat::PIXEL_FORMAT_MONO8, 
                std::size_t astride = 0);
    
    /**
     * Create with an external object and its deleter.
     */
    void create(void* extobject, std::function<void (void*)> adeleter, 
                uint8_t* abuffer, std::size_t awidth, std::size_t aheight, 
                EPixelFormat apixfmt = EPixelFormat::PIXEL_FORMAT_MONO8, 
                std::size_t astride = 0);
    
    /**
     * Release resources.
     */
    void release();
    
    inline bool isValid() const { return memaccess != nullptr; }
    
    inline uint8_t* getData() { return memaccess; }
    inline const uint8_t* getData() const { return memaccess; }
    
    inline void copyFrom(const void* ptr)
    {
        memcpy(getData(), ptr, getDataSize());
    }
    
    template<typename T>
    inline T& getPixel(std::size_t x, std::size_t y)
    {
        return *ptr<T>(x,y);
    }
    
    template<typename T>
    inline const T& getPixel(std::size_t x, std::size_t y) const
    {
        return *ptr<T>(x,y);
    }
    
    template<typename T>
    inline T& getPixel(std::size_t idx)
    {
        return *((reinterpret_cast<T*>(getData())) + idx); 
    }
    
    template<typename T>
    inline const T& getPixel(std::size_t idx) const
    {
        return *((reinterpret_cast<const T*>(getData())) + idx);
    }
    
    inline std::size_t getDataSize() const { return data_size; }
    inline std::size_t getWidth() const { return width; }
    inline std::size_t getHeight() const { return height; }
    inline EPixelFormat getPixelFormat() const { return pixfmt; }
    inline std::size_t getStridePixels() const { return stride / bpp; } // in pixels
    inline std::size_t getStride() const { return stride; } // in bytes
    
    inline uint64_t getPCTimeStamp() const { return pcTimeStamp; }
    inline void setPCTimeStamp(uint64_t v) { pcTimeStamp = v; }
    inline uint64_t getTimeStamp() const { return embeddedTimeStamp; }
    inline void setTimeStamp(uint64_t v) { embeddedTimeStamp = v; }
    inline unsigned int getGain() const { return embeddedGain; }
    inline void setGain(unsigned int v) { embeddedGain = v; }
    inline unsigned int getShutter() const { return embeddedShutter; }
    inline void setShutter(unsigned int v) { embeddedShutter = v; }
    inline unsigned int getBrightness() const { return embeddedBrightness; }
    inline void setBrightness(unsigned int v) { embeddedBrightness = v; }
    inline unsigned int getExposure() const { return embeddedExposure; }
    inline void setExposure(unsigned int v) { embeddedExposure = v; }
    inline unsigned int getWhiteBalance() const { return embeddedWhiteBalance; }
    inline void setWhiteBalance(unsigned int v) { embeddedWhiteBalance = v; }
    inline unsigned int getFrameCounter() const { return embeddedFrameCounter; }
    inline void setFrameCounter(unsigned int v) { embeddedFrameCounter = v; }
    inline float getFrameRate() const { return embeddedFrameRate; }
    inline void setFrameRate(float v) { embeddedFrameRate = v; }
    inline unsigned int getFramesDropped() const { return embeddedFramesDropped; }
    inline void setFramesDropped(unsigned int v) { embeddedFramesDropped = v; }
    
    inline const std::string& getVideoSource() const { return video_source; }
    inline void setVideoSource(const std::string& s) const { video_source = s; }
    inline void setVideoSource(const std::string& s) { video_source = s; }
    
#ifdef CAMERA_DRIVERS_HAVE_CEREAL
    template<class Archive>
    void save(Archive & archive, std::uint32_t const version) const
    {
        archive(cereal::make_nvp("DataSize", data_size));
        if(data_size > 0)
        {
            archive(cereal::binary_data(getData(), getDataSize()));
        }
        
        archive(cereal::make_nvp("Width", width));
        archive(cereal::make_nvp("Height", height));
        archive(cereal::make_nvp("PixelFormat", pixfmt));
        archive(cereal::make_nvp("Stride", stride));
        archive(cereal::make_nvp("PCTimeStamp", pcTimeStamp));
        archive(cereal::make_nvp("TimeStamp", embeddedTimeStamp));
        archive(cereal::make_nvp("Gain", embeddedGain));
        archive(cereal::make_nvp("Shutter", embeddedShutter));
        archive(cereal::make_nvp("Brightness", embeddedBrightness));
        archive(cereal::make_nvp("Exposure", embeddedExposure));
        archive(cereal::make_nvp("WhiteBalance", embeddedWhiteBalance));
        archive(cereal::make_nvp("FrameCounter", embeddedFrameCounter));
        archive(cereal::make_nvp("FrameRate", embeddedFrameRate));
        archive(cereal::make_nvp("FramesDropped", embeddedFramesDropped));
        archive(cereal::make_nvp("VideoSource", video_source));
    }
    
    template<class Archive>
    void load(Archive & archive, std::uint32_t const version)
    {
        archive(cereal::make_nvp("DataSize", data_size));
        if(data_size > 0)
        {
            release(); // remove old stuff if any
            create_byte_array(data_size);
            archive(cereal::binary_data(getData(), getDataSize()));
        }
        
        archive(cereal::make_nvp("Width", width));
        archive(cereal::make_nvp("Height", height));
        archive(cereal::make_nvp("PixelFormat", pixfmt));
        archive(cereal::make_nvp("Stride", stride));
        archive(cereal::make_nvp("PCTimeStamp", pcTimeStamp));
        archive(cereal::make_nvp("TimeStamp", embeddedTimeStamp));
        archive(cereal::make_nvp("Gain", embeddedGain));
        archive(cereal::make_nvp("Shutter", embeddedShutter));
        archive(cereal::make_nvp("Brightness", embeddedBrightness));
        archive(cereal::make_nvp("Exposure", embeddedExposure));
        archive(cereal::make_nvp("WhiteBalance", embeddedWhiteBalance));
        archive(cereal::make_nvp("FrameCounter", embeddedFrameCounter));
        archive(cereal::make_nvp("FrameRate", embeddedFrameRate));
        archive(cereal::make_nvp("FramesDropped", embeddedFramesDropped));
        archive(cereal::make_nvp("VideoSource", video_source));
    }
#endif // CAMERA_DRIVERS_HAVE_CEREAL
private:
    template<typename T>
    inline T* ptr(std::size_t x, std::size_t y)
    {
        return (T*)( ((uint8_t*)getData()) + y * getStride()) + x;
    }
    
    template<typename T>
    inline const T* ptr(std::size_t x, std::size_t y) const
    {
        return (const T*)( ((const uint8_t*)getData()) + y * getStride()) + x;
    }
    
    void create_byte_array(std::size_t nbytes)
    {
        memaccess = new uint8_t[nbytes];
        associated_buffer = memaccess;
        deleter = std::bind(&FrameBuffer::delete_byte_array, this, std::placeholders::_1);
    }
    
    void delete_byte_array(void* ab)
    {
        delete[] static_cast<uint8_t*>(ab);
    }
    
    void delete_nothing(void*) {  }
    
    std::function<void (void*)> deleter;
    void* associated_buffer;
    uint8_t* memaccess;
    std::size_t data_size;
    std::size_t width;
    std::size_t height;
    EPixelFormat pixfmt;
    std::size_t stride;
    uint64_t pcTimeStamp;
    uint64_t embeddedTimeStamp;
    unsigned int embeddedGain;
    unsigned int embeddedShutter;
    unsigned int embeddedBrightness;
    unsigned int embeddedExposure;
    unsigned int embeddedWhiteBalance;
    unsigned int embeddedFrameCounter;
    float embeddedFrameRate;
    unsigned int embeddedFramesDropped;
    mutable std::string video_source;
    std::size_t bpp;
};

/**
 * Callback type for video streams.
 */
typedef std::function<void(const FrameBuffer&)> FrameCallback;

/**
 * Generic event class.
 */
class EventData
{
public:
    uint64_t Timestamp;   /// [ns]
    uint64_t FrameNumber; /// 
    uint64_t EventSource; /// [user defined]
};

/**
 * Callback type for events.
 */
typedef std::function<void(const EventData&)> EventCallback;

/**
 * Generic accelerometer data class.
 */
class AccelerometerData
{
public:
    bool                IsValid;
    EventData           Event;
    std::array<float,3> Accelerometer;  /// [m/s^2] 
    float               Temperature;    /// [degC] 
};

/**
 * Callback type for accelerometer.
 */
typedef std::function<void(const AccelerometerData&)> AccelerometerCallback;

/**
 * Generic gyroscope data class.
 */
class GyroscopeData
{
public:
    bool                IsValid;
    EventData           Event;
    std::array<float,3> Gyroscope;      /// [rad/s]
    float               Temperature;    /// [degC] 
};

/**
 * Callback type for gyroscope.
 */
typedef std::function<void(const GyroscopeData&)> GyroscopeCallback;

/**
 * Generic compass data class.
 */
class CompassData
{
public:
    bool                IsValid;
    EventData           Event;
    std::array<float,3> Compass;        /// [uT]
    float               Temperature;    /// [degC] 
};

/**
 * Callback type for compass.
 */
typedef std::function<void(const CompassData&)> CompassCallback;

/**
 * Generic motion class.
 */
class MotionData
{
public:
    bool                IsValid;
    EventData           Event;
    std::array<float,3> Accelerometer;  /// [m/s^2] 
    std::array<float,3> Gyroscope;      /// [rad/s]
    std::array<float,3> Compass;        /// [uT]
    float               Temperature;    /// [degC] 
};

/**
 * Callback type for motion.
 */
typedef std::function<void(const MotionData&)> MotionCallback;

/**
 * Camera Information & Configuration
 */
struct CameraInfo
{
    enum class CameraInterface
    {
        UNKNOWN = 0,
        USB,
        IEEE1394,
        GIGE,
        SOC
    };
    
    std::string SerialNumber;
    CameraInterface InterfaceType;
    bool IsColorCamera;
    std::string ModelName;
    std::string VendorName;
    std::string SensorInfo;
    std::string SensorResolution;
    std::string DriverName;
    std::string FirmwareVersion;
    std::string FirmwareBuildTime;
    
    CameraInfo() : 
        SerialNumber(""), InterfaceType(CameraInfo::CameraInterface::UNKNOWN),
        IsColorCamera(false), ModelName(""), VendorName(""), SensorInfo(""), SensorResolution(""),
        DriverName(""), FirmwareVersion(""), FirmwareBuildTime("")
    {
        
    }
};

/**
 * Abstract camera interface.
 */
class CameraDriverBase
{
public:
    virtual ~CameraDriverBase() { }

    bool isOpened() const
    { 
        return isOpenedImpl();
    }

    bool isStarted() const
    {
        if(!isOpened()) { return false; }
        
        return isStartedImpl();
    }
    
    virtual bool getFeaturePower(EFeature) { return false; }
    virtual void setFeaturePower(EFeature, bool) { }
    virtual bool getFeatureAuto(EFeature) { return false; }
    virtual void setFeatureAuto(EFeature, bool) { }
    virtual uint32_t getFeatureValue(EFeature){ return 0; }
    virtual float getFeatureValueAbs(EFeature){ return 0.0f; }
    virtual uint32_t getFeatureMin(EFeature) { return 0; }
    virtual uint32_t getFeatureMax(EFeature) { return 0; }
    virtual void setFeatureValue(EFeature, uint32_t) { }
    virtual void setFeatureValueAbs(EFeature, float) { }
    virtual void setFeature(EFeature fidx, bool power, bool automatic, uint32_t val) 
    {
        setFeaturePower(fidx, power);
        setFeatureAuto(fidx, automatic);
        setFeatureValue(fidx, val);
    }
    
    /**
     * By convention, if the streams are enabled the order should be as follows:
     * Depth , RGBD , Infrared1, Infrared2
     */
    
    template<typename _Rep = int64_t, typename _Period = std::ratio<1>>
    bool captureFrame(FrameBuffer& cf1, const std::chrono::duration<_Rep, _Period>& timeout = std::chrono::seconds(0))
    {
        return captureFrameImpl(&cf1, nullptr, nullptr, nullptr, std::chrono::duration_cast<std::chrono::nanoseconds>(timeout).count());
    }
    
    template<typename _Rep = int64_t, typename _Period = std::ratio<1>>
    bool captureFrame(FrameBuffer& cf1, FrameBuffer& cf2, const std::chrono::duration<_Rep, _Period>& timeout = std::chrono::seconds(0))
    {
        return captureFrameImpl(&cf1, &cf2, nullptr, nullptr, std::chrono::duration_cast<std::chrono::nanoseconds>(timeout).count());
    }
    
    template<typename _Rep = int64_t, typename _Period = std::ratio<1>>
    bool captureFrame(FrameBuffer& cf1, FrameBuffer& cf2, FrameBuffer& cf3, const std::chrono::duration<_Rep, _Period>& timeout = std::chrono::seconds(0))
    {
        return captureFrameImpl(&cf1, &cf2, &cf3, nullptr, std::chrono::duration_cast<std::chrono::nanoseconds>(timeout).count());
    }
    
    template<typename _Rep = int64_t, typename _Period = std::ratio<1>>
    bool captureFrame(FrameBuffer& cf1, FrameBuffer& cf2, FrameBuffer& cf3, FrameBuffer& cf4, const std::chrono::duration<_Rep, _Period>& timeout = std::chrono::seconds(0))
    {
        return captureFrameImpl(&cf1, &cf2, &cf3, &cf4, std::chrono::duration_cast<std::chrono::nanoseconds>(timeout).count());
    }

    const CameraInfo& cameraInformation() const { return m_cinfo; }
protected:
    virtual bool isOpenedImpl() const = 0;
    virtual bool isStartedImpl() const = 0;
    virtual bool captureFrameImpl(FrameBuffer* cf1, FrameBuffer* cf2, FrameBuffer* cf3, FrameBuffer* cf4, int64_t timeout = 0) = 0;
    CameraInfo m_cinfo;
};

}

}

#endif // CAMERA_DRIVERS_HPP
