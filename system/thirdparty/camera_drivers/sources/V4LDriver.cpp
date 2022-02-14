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
 * Video4Linux Driver.
 * ****************************************************************************
 */

#include <atomic>

#include <V4LDriver.hpp>

#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <asm/types.h>
#include <linux/videodev2.h>
#include <libv4l1.h>
#include <libv4l2.h>

drivers::camera::V4L::V4LException::V4LException(const char* file, int line)
{
    std::stringstream ss;
    ss << "V4L Error " << strerror(errno) << " , " << errno << " , " << " at " << file << ":" << line;
    errormsg = ss.str();
}

#define V4L_THROW_ERROR() { throw drivers::camera::V4L::V4LException(__FILE__, __LINE__); } 

struct V4LFrameBuffer 
{
    V4LFrameBuffer(drivers::camera::V4L::V4LAPIPimpl& parent, std::size_t idx);
    ~V4LFrameBuffer();
    
    drivers::camera::V4L::V4LAPIPimpl& parent;
    std::size_t length;
    void*       mmapbuf;
};

struct drivers::camera::V4L::V4LAPIPimpl
{
    V4LAPIPimpl() : fd(-1)
    {
        FD_ZERO (&fds);
        FD_SET (*fd, &fds);

        tv.tv_sec = 2;
        tv.tv_usec = 0;
    }
    
    ~V4LAPIPimpl()
    {
        
    }
    
    //a blocking wrapper of the ioctl function
    int ioctl(int request, void *arg)
    {
        int r;
        
        do r = v4l2_ioctl(fd, request, arg);
        while(-1 == r && EINTR == errno);
        
        return r;
    }
    
    void fcc2s(char *str, uint32_t val)
    {
        str[0] = val & 0xff;
        str[1] = (val >> 8) & 0xff;
        str[2] = (val >> 16) & 0xff;
        str[3] = (val >> 24) & 0xff;
    }
    
    uint32_t s2fcc(const char* str)
    {
        return ( (uint32_t) ((( str[3] ) << 24) | (uint32_t(str[2]) << 16) | (uint32_t(str[1]) << 8) | uint32_t(str[0])) );
    }
    
    bool requestBuffers(std::size_t count)
    {
        buf_count = count;
        
        struct v4l2_requestbuffers req;
        memset(&req, 0, sizeof(v4l2_requestbuffers));
        
        req.count               = buf_count;
        req.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory              = V4L2_MEMORY_MMAP;
        
        if(ioctl(VIDIOC_REQBUFS, &req) == -1)
        {
            return false;
        }
        
        if(req.count < buf_count)
        {
            return false;
        }
        
        for(std::size_t i = 0 ; i < buf_count ; ++i)
        {
            std::shared_ptr<V4LFrameBuffer> fb = std::make_shared<V4LFrameBuffer>(*this, i);
            buffers.push_back(fb);
        }
        
        for(std::size_t i = 0 ; i < buf_count ; ++i)
        {
            buffers[i]->queue(i);
        }
        
        return true;
    }
    
    bool releaseBuffers()
    {
        buffers.clear();
        
        struct v4l2_requestbuffers req;
        memset(&req, 0, sizeof(v4l2_requestbuffers));
        
        req.count               = 0;
        req.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory              = V4L2_MEMORY_MMAP;
        
        if(ioctl(VIDIOC_REQBUFS, &req) == -1)
        {
            return false;
        }
        
        buf_count = 0;
        
        return true;
    }
    
    bool queue(std::size_t idx)
    {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        
        buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory      = V4L2_MEMORY_MMAP;
        buf.index       = idx;
        
        if(ioctl(VIDIOC_QBUF, &buf) == -1)
        {
            return false;
        }
        
        return true;
    }
    
    int dequeue()
    {
        struct v4l2_buffer buf;//needed for memory mapping
        memset(&buf, 0, sizeof(v4l2_buffer));
        
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        
        if(ioctl(VIDIOC_DQBUF, &buf) == -1)
        {
            return -1;
        }
        
        return buf.index;
    }
    
    bool set_control(__u32 id, __s32 value)
    {
        struct v4l2_queryctrl queryctrl;
        struct v4l2_control control;
        
        memset(&queryctrl, 0, sizeof(queryctrl));
        queryctrl.id = id;
        
        if(ioctl(VIDIOC_QUERYCTRL, &queryctrl) == -1) 
        {
            if(errno != EINVAL) 
            {
                return false;
            } 
            else 
            {
                // not supported
                return false;
            }
        } 
        else if (queryctrl.flags & V4L2_CTRL_FLAG_DISABLED) 
        {
            // not supported
            return false;
        } 
        else 
        {
            memset(&control, 0, sizeof (control));
            control.id = id;
            control.value = value;
            
            if(ioctl(VIDIOC_S_CTRL, &control) == -1) 
            {
                return false;
            }
        }
        
        return true;
    }
    
    int fd;
    std::size_t buf_count;
    std::vector<std::shared_ptr<V4LFrameBuffer>> buffers;
    fd_set fds;
    struct timeval tv;
};

V4LFrameBuffer::V4LFrameBuffer(drivers::camera::V4L::V4LAPIPimpl& p, std::size_t idx) : parent(p), length(0), mmapbuf(0)
{
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    
    buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory      = V4L2_MEMORY_MMAP;
    buf.index       = idx;
    
    if(parent.ioctl(VIDIOC_QUERYBUF, &buf) == -1)
    {
        V4L_THROW_ERROR();
    }
    
    mmapbuf = v4l2_mmap(NULL,
                        buf.length,
                        PROT_READ | PROT_WRITE /* required */,
                        MAP_SHARED /* recommended */,
                        parent.fd,
                        buf.m.offset);
    
    if(mmapbuf == MAP_FAILED)
    {
        V4L_THROW_ERROR();
    }
    
    length = buf.length;
}

V4LFrameBuffer::~V4LFrameBuffer()
{
    if(v4l2_munmap(mmapbuf, length) == -1)
    {
        // error
    }
}

drivers::camera::V4L::V4L() : CameraDriverBase(), m_pimpl(new V4LAPIPimpl()), is_running(false)
{
    
}
   
drivers::camera::V4L::~V4L()
{
    close();
}

void drivers::camera::V4L::image_release(void* img)
{
    // TODO FIXME
}

void drivers::camera::V4L::open(const std::string& path)
{
    if(isOpened()) { return; }   
    
    m_pimpl->fd = v4l2_open (path.c_str(), O_RDWR | O_NONBLOCK, 0);
    if(m_pimpl->fd == -1)
    {
        V4L_THROW_ERROR();
    }
    
    struct v4l2_capability cap;
    
    if(m_pimpl->ioctl(VIDIOC_QUERYCAP, &cap) == -1)
    {
        V4L_THROW_ERROR();
    }
    
    struct v4l2_format fmt;
    char pixel_str[5] = {0};
    
    memset(&fmt, 0, sizeof(struct v4l2_format));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    
    if(m_pimpl->ioctl(VIDIOC_G_FMT, &fmt) == -1)
    {
        V4L_THROW_ERROR();
    }
    
    m_pimpl->fcc2s(pixel_str, fmt.fmt.pix.pixelformat);
}

bool drivers::camera::V4L::isOpenedImpl() const
{
    return m_pimpl->fd != -1;
}

void drivers::camera::V4L::close()
{
    if(!isOpened()) { return; }
    
    if(v4l2_close (m_pimpl->fd) == -1)
    {
        V4L_THROW_ERROR();
    }
    
    m_pimpl->fd = -1;
}

void drivers::camera::V4L::start()
{
    if(isStarted()) { return; }
    
    enum v4l2_buf_type type;
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    
    //start the capture from the device
    if(m_pimpl->ioctl(VIDIOC_STREAMON, &type) == -1)
    {
        V4L_THROW_ERROR();
    }
        
    is_running = true;
}

void drivers::camera::V4L::stop()
{
    if(!isStarted()) { return; }
    
    is_running = false;
    
    enum v4l2_buf_type type;
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    
    //this call to xioctl allows to stop the stream from the capture device
    if(m_pimpl->ioctl(VIDIOC_STREAMOFF, &type) == -1)
    {
        V4L_THROW_ERROR();
    }
}

void drivers::camera::V4L::setModeAndFramerate(drivers::camera::EVideoMode vmode, drivers::camera::EFrameRate framerate)
{
    m_width = drivers::camera::VideoModeToWidth(vmode);
    m_height = drivers::camera::VideoModeToHeight(vmode);
    m_pixfmt = drivers::camera::VideoModeToPixelFormat(vmode);
    auto fps = drivers::camera::FrameRateToFPS(framerate);
    setCustomMode(m_pixfmt, m_width, m_height);
    setFeaturePower(EFeature::FRAME_RATE, true);
    setFeatureValue(EFeature::FRAME_RATE, fps);
}

void drivers::camera::V4L::setCustomMode(drivers::camera::EPixelFormat pixfmt, unsigned int width, unsigned int height, unsigned int offset_x, unsigned int offset_y, uint16_t format7mode)
{

}

bool drivers::camera::V4L::captureFrameImpl(FrameBuffer* cf1, FrameBuffer* cf2, FrameBuffer* cf3, FrameBuffer* cf4, int64_t timeout)
{
    int r = -1;
    
    if(timeout > 0)
    {
        m_pimpl->tv.tv_sec = timeout / 1000000000; // sec
        m_pimpl->tv.tv_usec = (timeout % 1000000000) / 1000; // usec
        
        r = select(m_pimpl->fd + 1, &(m_pimpl->fds), NULL, NULL, &(m_pimpl->tv));
    }
    else
    {
        r = select(m_pimpl->fd + 1, &(m_pimpl->fds), NULL, NULL, NULL);
    }
    
    if(r == -1)
    {
        V4L_THROW_ERROR();
    }
    
    if(r == 0)
    {
        return false;
    }
    
    std::chrono::high_resolution_clock::time_point tp = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds d = tp.time_since_epoch();
    
    if(cf1 != nullptr)
    {
        int idx = m_pimpl->dequeue();
        
        if(idx < 0)
        {
            V4L_THROW_ERROR();
        }
        
        cf1->create((void*)idx, std::bind(&drivers::camera::V4L::image_release, this, std::placeholders::_1), m_pimpl->buffers[idx], frame->size[0], frame->size[1], pixfmt, frame->stride);
        cf1->setPCTimeStamp(d.count());
        cf1->setTimeStamp(0);
        cf1->setGain(0);
        cf1->setShutter(0);
        cf1->setBrightness(0);
        cf1->setExposure(0);
        cf1->setWhiteBalance(0);
        cf1->setFrameCounter(0);
    }
    
    return true;
}

bool drivers::camera::V4L::getFeaturePower(EFeature fidx)
{
    
}

void drivers::camera::V4L::setFeaturePower(EFeature fidx, bool b)
{
    
}

bool drivers::camera::V4L::getFeatureAuto(EFeature fidx)
{
    
}

void drivers::camera::V4L::setFeatureAuto(EFeature fidx, bool b)
{
    
}

uint32_t drivers::camera::V4L::getFeatureValue(EFeature fidx)
{
    
}

float drivers::camera::V4L::getFeatureValueAbs(EFeature fidx)
{
    
}

uint32_t drivers::camera::V4L::getFeatureMin(EFeature fidx)
{
    
}

uint32_t drivers::camera::V4L::getFeatureMax(EFeature fidx)
{
    
}

void drivers::camera::V4L::setFeatureValue(EFeature fidx, uint32_t val)
{
    
}

void drivers::camera::V4L::setFeatureValueAbs(EFeature fidx, float val)
{
    
}

void drivers::camera::V4L::setFeature(EFeature fidx, bool power, bool automatic, uint32_t val)
{

}


