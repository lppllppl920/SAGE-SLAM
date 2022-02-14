/**
 * ****************************************************************************
 * Copyright (c) 2016, Robert Lukierski.
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
 * Image IO.
 * ****************************************************************************
 */

#include <VisionCore/IO/ImageIO.hpp>

#include <VisionCore/Buffers/Buffer2D.hpp>
#include <VisionCore/Buffers/Image2D.hpp>

#ifdef VISIONCORE_HAVE_CAMERA_DRIVERS
#include <CameraDrivers.hpp>
#endif // VISIONCORE_HAVE_CAMERA_DRIVERS

#ifdef VISIONCORE_HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#endif // VISIONCORE_HAVE_OPENCV

struct OpenCVBackend { };


template<typename T, typename Backend>
struct ImageIOProxy
{
    static inline T load(const std::string& fn) { throw std::runtime_error("No IO backend"); }
    static inline void save(const std::string& fn, const T& input) { throw std::runtime_error("No IO backend"); }
};

/**
 * OpenCV Backend.
 */
#ifdef VISIONCORE_HAVE_OPENCV
template<>
struct ImageIOProxy<cv::Mat,OpenCVBackend>
{
    static inline cv::Mat load(const std::string& fn)
    {
        cv::Mat ret = cv::imread(fn, cv::IMREAD_ANYDEPTH);
        
        if(!ret.data)
        {
            throw std::runtime_error("File not found");
        }
        
        return ret;
    }
    
    static inline void save(const std::string& fn, const cv::Mat& input)
    {
        cv::imwrite(fn, input);
    }
};

#ifdef VISIONCORE_HAVE_CAMERA_DRIVERS

static inline drivers::camera::EPixelFormat OpenCVType2PixelFormat(int ocv_type)
{
    switch(ocv_type)
    {
        case CV_8UC1:  return drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO8;
        case CV_16UC1: return drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO16;
        case CV_8UC3:  return drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB8;
        case CV_32FC1: return drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO32F;
        case CV_32FC3: return drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB32F;
        default: throw std::runtime_error("Pixel Format not supported");
    }
}

static inline int PixelFomat2OpenCVType(drivers::camera::EPixelFormat pf)
{
    switch(pf)
    {
        case drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO8: return CV_8UC1;
        case drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO16: return CV_16UC1;
        case drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB8: return CV_8UC3;
        case drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO32F: return CV_32FC1;
        case drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB32F: return CV_32FC3;
        default: throw std::runtime_error("Pixel Format not supported");
    }
}

template<>
struct ImageIOProxy<drivers::camera::FrameBuffer, OpenCVBackend>
{
    static inline drivers::camera::FrameBuffer load(const std::string& fn)
    {
        cv::Mat cv_tmp = cv::imread(fn);
        
        if(!cv_tmp.data)
        {
            throw std::runtime_error("File not found");
        }
        
        drivers::camera::FrameBuffer ret(cv_tmp.cols, cv_tmp.rows, OpenCVType2PixelFormat(cv_tmp.type()));
        cv::Mat cv_wrapper(ret.getHeight(), ret.getWidth(), cv_tmp.type(), ret.getData());
        
        cv_tmp.copyTo(cv_wrapper);
        
        return ret;
    }
    
    static inline void save(const std::string& fn, const drivers::camera::FrameBuffer& input)
    {
        cv::Mat cv_wrapper(input.getHeight(), input.getWidth(), PixelFomat2OpenCVType(input.getPixelFormat()), (void*)input.getData());
        cv::imwrite(fn, cv_wrapper);
    }
};
#endif // VISIONCORE_HAVE_CAMERA_DRIVERS

template<typename T>
struct ImageIOProxy<vc::Image2DView<T, vc::TargetHost>,OpenCVBackend>
{
    static inline void save(const std::string& fn, const vc::Image2DView<T, vc::TargetHost>& input)
    {
        cv::Mat cv_proxy(input.height(), input.width(), CV_MAKETYPE(vc::internal::OpenCVType<typename vc::type_traits<T>::ChannelType>::TypeCode, vc::type_traits<T>::ChannelCount), (void*)input.ptr());
        cv::imwrite(fn, cv_proxy);
    }
};

template<typename T>
struct ImageIOProxy<vc::Image2DManaged<T, vc::TargetHost>,OpenCVBackend>
{
    static inline vc::Image2DManaged<T, vc::TargetHost> load(const std::string& fn)
    {
        int flag = 0;
        if(vc::type_traits<T>::ChannelCount == 1)
        {
            flag = cv::IMREAD_GRAYSCALE;
        }
        else
        {
            flag = cv::IMREAD_COLOR;
        }
        
        cv::Mat cv_tmp = cv::imread(fn, flag | cv::IMREAD_ANYDEPTH);
        
        if(!cv_tmp.data)
        {
            throw std::runtime_error("File not found");
        }
        
        if((vc::type_traits<T>::ChannelCount != cv_tmp.channels()) || (CV_MAKETYPE(vc::internal::OpenCVType<typename vc::type_traits<T>::ChannelType>::TypeCode, vc::type_traits<T>::ChannelCount) != cv_tmp.type()))
        {
            throw std::runtime_error("Wrong channel count / pixel type");
        }
        
        vc::Image2DManaged<T, vc::TargetHost> ret(cv_tmp.cols, cv_tmp.rows);
        cv::Mat cv_proxy(ret.height(), ret.width(), 
                         CV_MAKETYPE(vc::internal::OpenCVType<typename vc::type_traits<T>::ChannelType>::TypeCode,
                                     vc::type_traits<T>::ChannelCount), 
                         (void*)ret.ptr());
        
        // copy
        cv_tmp.copyTo(cv_proxy);
        
        return ret;
    }
};

template<typename T>
struct ImageIOProxy<vc::Buffer2DView<T, vc::TargetHost>,OpenCVBackend>
{
    static inline void save(const std::string& fn, const vc::Buffer2DView<T, vc::TargetHost>& input)
    {
        cv::Mat cv_proxy(input.height(), input.width(), 
                         CV_MAKETYPE(vc::internal::OpenCVType<typename vc::type_traits<T>::ChannelType>::TypeCode,
                                     vc::type_traits<T>::ChannelCount), 
                         (void*)input.ptr());
        cv::imwrite(fn, cv_proxy);
    }
};

template<typename T>
struct ImageIOProxy<vc::Buffer2DManaged<T, vc::TargetHost>,OpenCVBackend>
{
    static inline vc::Buffer2DManaged<T, vc::TargetHost> load(const std::string& fn)
    {
        int flag = 0;
        if(vc::type_traits<T>::ChannelCount == 1)
        {
            flag = cv::IMREAD_GRAYSCALE;
        }
        else
        {
            flag = cv::IMREAD_COLOR;
        }
        cv::Mat cv_tmp = cv::imread(fn, flag | cv::IMREAD_ANYDEPTH);
        
        if(!cv_tmp.data)
        {
            throw std::runtime_error("File not found");
        }
        
        if((vc::type_traits<T>::ChannelCount != cv_tmp.channels()) || 
           (CV_MAKETYPE(vc::internal::OpenCVType<typename vc::type_traits<T>::ChannelType>::TypeCode, 
                        vc::type_traits<T>::ChannelCount) != cv_tmp.type()))
        {
            throw std::runtime_error("Wrong channel count / pixel type");
        }
        
        vc::Buffer2DManaged<T, vc::TargetHost> ret(cv_tmp.cols, cv_tmp.rows);
        cv::Mat cv_proxy(ret.height(), ret.width(), 
                         CV_MAKETYPE(vc::internal::OpenCVType<typename vc::type_traits<T>::ChannelType>::TypeCode,
                                     vc::type_traits<T>::ChannelCount), 
                         (void*)ret.ptr());
        
        // copy
        cv_tmp.copyTo(cv_proxy);
        
        return ret;
    }
};
#endif // VISIONCORE_HAVE_OPENCV

template<typename T>
T vc::io::loadImage(const std::string& fn)
{
    return ImageIOProxy<T,OpenCVBackend>::load(fn);
}

template<typename T>
void vc::io::saveImage(const std::string& fn, const T& input)
{
    ImageIOProxy<T,OpenCVBackend>::save(fn, input);
}

#ifdef VISIONCORE_HAVE_OPENCV
template cv::Mat vc::io::loadImage<cv::Mat>(const std::string& fn);
template void vc::io::saveImage<cv::Mat>(const std::string& fn, const cv::Mat& input);
#endif // VISIONCORE_HAVE_OPENCV

template vc::Image2DManaged<float, vc::TargetHost> vc::io::loadImage<vc::Image2DManaged<float, vc::TargetHost>>(const std::string& fn);
template void vc::io::saveImage<vc::Image2DView<float, vc::TargetHost>>(const std::string& fn, const vc::Image2DView<float, vc::TargetHost>& input);

template vc::Image2DManaged<uint8_t, vc::TargetHost> vc::io::loadImage<vc::Image2DManaged<uint8_t, vc::TargetHost>>(const std::string& fn);
template void vc::io::saveImage<vc::Image2DView<uint8_t, vc::TargetHost>>(const std::string& fn, const vc::Image2DView<uint8_t, vc::TargetHost>& input);

template vc::Image2DManaged<uint16_t, vc::TargetHost> vc::io::loadImage<vc::Image2DManaged<uint16_t, vc::TargetHost>>(const std::string& fn);
template void vc::io::saveImage<vc::Image2DView<uint16_t, vc::TargetHost>>(const std::string& fn, const vc::Image2DView<uint16_t, vc::TargetHost>& input);

template vc::Image2DManaged<uchar3, vc::TargetHost> vc::io::loadImage<vc::Image2DManaged<uchar3, vc::TargetHost>>(const std::string& fn);
template void vc::io::saveImage<vc::Image2DView<uchar3, vc::TargetHost>>(const std::string& fn, const vc::Image2DView<uchar3, vc::TargetHost>& input);

template vc::Image2DManaged<float3, vc::TargetHost> vc::io::loadImage<vc::Image2DManaged<float3, vc::TargetHost>>(const std::string& fn);
template void vc::io::saveImage<vc::Image2DView<float3, vc::TargetHost>>(const std::string& fn, const vc::Image2DView<float3, vc::TargetHost>& input);

template vc::Buffer2DManaged<float, vc::TargetHost> vc::io::loadImage<vc::Buffer2DManaged<float, vc::TargetHost>>(const std::string& fn);
template void vc::io::saveImage<vc::Buffer2DView<float, vc::TargetHost>>(const std::string& fn, const vc::Buffer2DView<float, vc::TargetHost>& input);

template vc::Buffer2DManaged<uint8_t, vc::TargetHost> vc::io::loadImage<vc::Buffer2DManaged<uint8_t, vc::TargetHost>>(const std::string& fn);
template void vc::io::saveImage<vc::Buffer2DView<uint8_t, vc::TargetHost>>(const std::string& fn, const vc::Buffer2DView<uint8_t, vc::TargetHost>& input);

template vc::Buffer2DManaged<uint16_t, vc::TargetHost> vc::io::loadImage<vc::Buffer2DManaged<uint16_t, vc::TargetHost>>(const std::string& fn);
template void vc::io::saveImage<vc::Buffer2DView<uint16_t, vc::TargetHost>>(const std::string& fn, const vc::Buffer2DView<uint16_t, vc::TargetHost>& input);

template vc::Buffer2DManaged<uchar3, vc::TargetHost> vc::io::loadImage<vc::Buffer2DManaged<uchar3, vc::TargetHost>>(const std::string& fn);
template void vc::io::saveImage<vc::Buffer2DView<uchar3, vc::TargetHost>>(const std::string& fn, const vc::Buffer2DView<uchar3, vc::TargetHost>& input);

template vc::Buffer2DManaged<float3, vc::TargetHost> vc::io::loadImage<vc::Buffer2DManaged<float3, vc::TargetHost>>(const std::string& fn);
template void vc::io::saveImage<vc::Buffer2DView<float3, vc::TargetHost>>(const std::string& fn, const vc::Buffer2DView<float3, vc::TargetHost>& input);

#ifdef VISIONCORE_HAVE_CAMERA_DRIVERS
template drivers::camera::FrameBuffer vc::io::loadImage<drivers::camera::FrameBuffer>(const std::string& fn);
template void vc::io::saveImage<drivers::camera::FrameBuffer>(const std::string& fn, const drivers::camera::FrameBuffer& input);
#endif // VISIONCORE_HAVE_CAMERA_DRIVERS
