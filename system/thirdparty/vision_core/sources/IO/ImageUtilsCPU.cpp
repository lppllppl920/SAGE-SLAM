/**
 * 
 * Image Utils.
 * Miscellaneous.
 * 
 * Copyright (c) Robert Lukierski 2015. All rights reserved.
 * Author: Robert Lukierski.
 * 
 */

#include <VisionCore/Image/BufferOps.hpp>

#include <numeric>
#include <VisionCore/Platform.hpp>

#include <Eigen/Dense>

#include <VisionCore/LaunchUtils.hpp>
#include <VisionCore/Math/LossFunctions.hpp>
#include <Image/JoinSplitHelpers.hpp>

template<typename T, typename Target>
void vc::image::rescaleBufferInplace(vc::Buffer1DView< T, Target>& buf_in, T alpha, T beta, T clamp_min, T clamp_max)
{
    std::transform(buf_in.ptr(), buf_in.ptr() + buf_in.size(), buf_in.ptr(), [&](T val) -> T 
    {
        return clamp(val * alpha + beta, clamp_min, clamp_max); 
    });
}

template<typename T, typename Target>
void vc::image::rescaleBufferInplace(vc::Buffer2DView<T, Target>& buf_in, T alpha, T beta, T clamp_min, T clamp_max)
{
    rescaleBuffer(buf_in, buf_in, alpha, beta, clamp_min, clamp_max);
}

template<typename T, typename Target>
void vc::image::rescaleBufferInplaceMinMax(vc::Buffer2DView<T, Target>& buf_in, T vmin, T vmax, T clamp_min, T clamp_max)
{
    rescaleBuffer(buf_in, buf_in, T(1.0f) / (vmax - vmin), -vmin * (T(1.0)/(vmax - vmin)), clamp_min, clamp_max);
}

template<typename T1, typename T2, typename Target>
void vc::image::rescaleBuffer(const vc::Buffer2DView<T1, Target>& buf_in, 
                              vc::Buffer2DView<T2, Target>& buf_out, 
                              float alpha, float beta, float clamp_min, float clamp_max)
{
    vc::launchParallelFor(buf_in.width(), buf_in.height(), [&](std::size_t x, std::size_t y)
    {
        if(buf_in.inBounds(x,y) && buf_out.inBounds(x,y))
        {
            const T2 val = convertPixel<T1,T2>(buf_in(x,y));
            buf_out(x,y) = clamp(val * alpha + beta, clamp_min, clamp_max); 
        }
    });
}

template<typename T, typename Target>
void vc::image::normalizeBufferInplace(vc::Buffer2DView< T, Target >& buf_in)
{
    const T min_val = calcBufferMin(buf_in);
    const T max_val = calcBufferMax(buf_in);

    rescaleBufferInplace(buf_in, T(1.0f) / (max_val - min_val), -min_val * (T(1.0)/(max_val - min_val)));
}

template<typename T, typename Target>
void vc::image::clampBuffer(vc::Buffer1DView<T, Target>& buf_io, T a, T b)
{
    vc::launchParallelFor(buf_io.size(), [&](std::size_t idx)
    {
        if(buf_io.inBounds(idx))
        {
            buf_io(idx) = clamp(buf_io(idx), a, b); 
        }
    });
}

template<typename T, typename Target>
void vc::image::clampBuffer(vc::Buffer2DView<T, Target>& buf_io, T a, T b)
{
    vc::launchParallelFor(buf_io.width(), buf_io.height(), [&](std::size_t x, std::size_t y)
    {
        if(buf_io.inBounds(x,y))
        {
            buf_io(x,y) = clamp(buf_io(x,y), a, b); 
        }
    });
}

template<typename T, typename Target>
T vc::image::calcBufferMin(const vc::Buffer1DView< T, Target >& buf_in)
{
    const T* ret = std::min_element(buf_in.ptr(), buf_in.ptr() + buf_in.size());
    return *ret;
}

template<typename T, typename Target>
T vc::image::calcBufferMax(const vc::Buffer1DView< T, Target >& buf_in)
{
    const T* ret = std::max_element(buf_in.ptr(), buf_in.ptr() + buf_in.size());
    return *ret;
}

template<typename T, typename Target>
T vc::image::calcBufferMean(const vc::Buffer1DView< T, Target >& buf_in)
{
    T sum = std::accumulate(buf_in.ptr(), buf_in.ptr() + buf_in.size(), vc::zero<T>());
    return sum / (T)buf_in.size();
}

template<typename T, typename Target>
T vc::image::calcBufferMin(const vc::Buffer2DView< T, Target >& buf_in)
{
    T minval = std::numeric_limits<T>::max();
    
    for(std::size_t y = 0 ; y < buf_in.height() ; ++y)
    {
        for(std::size_t x = 0 ; x < buf_in.width() ; ++x)
        {
            minval = std::min(minval, buf_in(x,y));
        }
    }
    
    return minval;
}

template<typename T, typename Target>
T vc::image::calcBufferMax(const vc::Buffer2DView< T, Target >& buf_in)
{
    T maxval = std::numeric_limits<T>::lowest();
    
    for(std::size_t y = 0 ; y < buf_in.height() ; ++y)
    {
        for(std::size_t x = 0 ; x < buf_in.width() ; ++x)
        {
            maxval = std::max(maxval, buf_in(x,y));
        }
    }
    
    return maxval;
}

template<typename T, typename Target>
T vc::image::calcBufferMean(const vc::Buffer2DView< T, Target >& buf_in)
{
    T sum = vc::zero<T>();
    
    for(std::size_t y = 0 ; y < buf_in.height() ; ++y)
    {
        for(std::size_t x = 0 ; x < buf_in.width() ; ++x)
        {
            sum += buf_in(x,y);
        }
    }
    
    return sum / (T)buf_in.area();
}

template<typename T, typename Target>
void vc::image::downsampleHalf(const vc::Buffer2DView<T, Target>& buf_in, vc::Buffer2DView<T, Target>& buf_out)
{
    if(!( (buf_in.width()/2 == buf_out.width()) && (buf_in.height()/2 == buf_out.height())))
    {
        throw std::runtime_error("In/Out dimensions don't match");
    }
    
    vc::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
            const T* tl = buf_in.ptr(2*x,2*y);
            const T* bl = buf_in.ptr(2*x,2*y+1);
            
            buf_out(x,y) = (T)(*tl + *(tl+1) + *bl + *(bl+1)) / 4;
    });
}

template<typename T, typename Target>
void vc::image::downsampleHalfNoInvalid(const vc::Buffer2DView<T, Target>& buf_in, vc::Buffer2DView<T, Target>& buf_out)
{
    if(!( (buf_in.width()/2 == buf_out.width()) && (buf_in.height()/2 == buf_out.height())))
    {
        throw std::runtime_error("In/Out dimensions don't match");
    }
    
    vc::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        const T* tl = buf_in.ptr(2*x,2*y);
        const T* bl = buf_in.ptr(2*x,2*y+1);
        const T v1 = *tl;
        const T v2 = *(tl+1);
        const T v3 = *bl;
        const T v4 = *(bl+1);
        
        int n = 0;
        T sum = 0;
        
        if(vc::isvalid(v1)) { sum += v1; n++; }
        if(vc::isvalid(v2)) { sum += v2; n++; }
        if(vc::isvalid(v3)) { sum += v3; n++; }
        if(vc::isvalid(v4)) { sum += v4; n++; }
        
        buf_out(x,y) = n > 0 ? (T)(sum / (T)n) : vc::getInvalid<T>();
    });
}

template<typename T, typename Target>
void vc::image::leaveQuarter(const vc::Buffer2DView<T, Target>& buf_in, vc::Buffer2DView<T, Target>& buf_out)
{
    dim3 gridDim, blockDim;
    
    if(!( (buf_in.width()/2 == buf_out.width()) && (buf_in.height()/2 == buf_out.height())))
    {
        throw std::runtime_error("In/Out dimensions don't match");
    }
    
    vc::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        buf_out(x,y) = buf_in(2*x,2*y);
    });
}

template<typename TCOMP, typename Target>
void vc::image::join(const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_in1, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_in2, vc::Buffer2DView<TCOMP, Target>& buf_out)
{
    assert((buf_out.width() == buf_in1.width()) && (buf_out.height() == buf_in1.height()));
    assert((buf_in1.width() == buf_in2.width()) && (buf_in1.height() == buf_in2.height()));
    
    vc::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        ::internal::JoinSplitHelper<TCOMP>::join(buf_in1(x,y), buf_in2(x,y), buf_out(x,y));
    });
}

template<typename TCOMP, typename Target>
void vc::image::join(const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_in1, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_in2, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_in3, vc::Buffer2DView<TCOMP, Target>& buf_out)
{
    assert((buf_out.width() == buf_in1.width()) && (buf_out.height() == buf_in1.height()));
    assert((buf_in1.width() == buf_in2.width()) && (buf_in1.height() == buf_in2.height()));
    assert((buf_in2.width() == buf_in3.width()) && (buf_in2.height() == buf_in3.height()));
    
    vc::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        ::internal::JoinSplitHelper<TCOMP>::join(buf_in1(x,y), buf_in2(x,y), buf_in3(x,y), buf_out(x,y));
    });
}

template<typename TCOMP, typename Target>
void vc::image::join(const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_in1, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_in2, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_in3, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_in4, vc::Buffer2DView<TCOMP, Target>& buf_out)
{
    assert((buf_out.width() == buf_in1.width()) && (buf_out.height() == buf_in1.height()));
    assert((buf_in1.width() == buf_in2.width()) && (buf_in1.height() == buf_in2.height()));
    assert((buf_in2.width() == buf_in3.width()) && (buf_in2.height() == buf_in3.height()));
    assert((buf_in3.width() == buf_in4.width()) && (buf_in3.height() == buf_in4.height()));
    
    vc::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        ::internal::JoinSplitHelper<TCOMP>::join(buf_in1(x,y), buf_in2(x,y), buf_in3(x,y), buf_in4(x,y), buf_out(x,y));
    });
}

template<typename TCOMP, typename Target>
void vc::image::split(const vc::Buffer2DView<TCOMP, Target>& buf_in, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_out1, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_out2)
{
    assert((buf_in.width() == buf_out1.width()) && (buf_in.height() == buf_out1.height()));
    assert((buf_out1.width() == buf_out2.width()) && (buf_out1.height() == buf_out2.height()));
    
    vc::launchParallelFor(buf_in.width(), buf_in.height(), [&](std::size_t x, std::size_t y)
    {
        ::internal::JoinSplitHelper<TCOMP>::split(buf_in(x,y), buf_out1(x,y), buf_out2(x,y));
    });
}

template<typename TCOMP, typename Target>
void vc::image::split(const vc::Buffer2DView<TCOMP, Target>& buf_in, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_out1, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_out2, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_out3)
{
    assert((buf_in.width() == buf_out1.width()) && (buf_in.height() == buf_out1.height()));
    assert((buf_out1.width() == buf_out2.width()) && (buf_out1.height() == buf_out2.height()));
    assert((buf_out2.width() == buf_out3.width()) && (buf_out2.height() == buf_out3.height()));
    
    vc::launchParallelFor(buf_in.width(), buf_in.height(), [&](std::size_t x, std::size_t y)
    {
        ::internal::JoinSplitHelper<TCOMP>::split(buf_in(x,y), buf_out1(x,y), buf_out2(x,y), buf_out3(x,y));
    });
}

template<typename TCOMP, typename Target>
void vc::image::split(const vc::Buffer2DView<TCOMP, Target>& buf_in, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_out1, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_out2, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_out3, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, Target>& buf_out4)
{
    assert((buf_in.width() == buf_out1.width()) && (buf_in.height() == buf_out1.height()));
    assert((buf_out1.width() == buf_out2.width()) && (buf_out1.height() == buf_out2.height()));
    assert((buf_out2.width() == buf_out3.width()) && (buf_out2.height() == buf_out3.height()));
    assert((buf_out3.width() == buf_out4.width()) && (buf_out3.height() == buf_out4.height()));
    
    vc::launchParallelFor(buf_in.width(), buf_in.height(), [&](std::size_t x, std::size_t y)
    {
        ::internal::JoinSplitHelper<TCOMP>::split(buf_in(x,y), buf_out1(x,y), buf_out2(x,y), buf_out3(x,y), buf_out4(x,y));
    });
}

/**
 * fillBuffer
 */
template<typename T, typename Target>
void vc::image::fillBuffer(vc::Buffer1DView<T, Target>& buf_in, const typename vc::type_traits<T>::ChannelType& v)
{
    std::transform(buf_in.ptr(), buf_in.ptr() + buf_in.size(), buf_in.ptr(), [&](T val) -> T 
    {
        return vc::internal::type_dispatcher_helper<T>::fill(v);
    });
}

/**
 * fillBuffer
 */
template<typename T, typename Target>
void vc::image::fillBuffer(vc::Buffer2DView<T, Target>& buf_in, const typename vc::type_traits<T>::ChannelType& v)
{
    vc::launchParallelFor(buf_in.width(), buf_in.height(), [&](std::size_t x, std::size_t y)
    {
        buf_in(x,y) = vc::internal::type_dispatcher_helper<T>::fill(v);
    });
}

/**
 * Invert Buffer
 */
template<typename T, typename Target>
void vc::image::invertBuffer(vc::Buffer1DView<T, Target>& buf_io)
{
    //typedef typename vc::type_traits<T>::ChannelType Scalar;
    
    std::transform(buf_io.ptr(), buf_io.ptr() + buf_io.size(), buf_io.ptr(), [&](T val) -> T 
    {
        return ::internal::JoinSplitHelper<T>::invertedValue(val);
    });
}

/**
 * Invert Buffer
 */
template<typename T, typename Target>
void vc::image::invertBuffer(vc::Buffer2DView<T, Target>& buf_io)
{
    vc::launchParallelFor(buf_io.width(), buf_io.height(), [&](std::size_t x, std::size_t y)
    {
        buf_io(x,y) = ::internal::JoinSplitHelper<T>::invertedValue(buf_io(x,y));
    });
}

/**
 * Threshold Buffer
 */
template<typename T, typename Target>
void vc::image::thresholdBuffer(const vc::Buffer2DView<T, Target>& buf_in, vc::Buffer2DView<T, Target>& buf_out, T thr, T val_below, T val_above)
{
    vc::launchParallelFor(buf_in.width(), buf_in.height(), [&](std::size_t x, std::size_t y)
    {
        if(buf_in.inBounds(x,y)) // is valid
        {
            const T& val = buf_in(x,y);
            if(val < thr)
            {
                buf_out(x,y) = val_below;
            }
            else
            {
                buf_out(x,y) = val_above;
            }
        }
    });
}

/**
 * Threshold Buffer
 */
template<typename T, typename Target>
void vc::image::thresholdBuffer(const vc::Buffer2DView<T, Target>& buf_in, vc::Buffer2DView<T, Target>& buf_out, T thr, T val_below, T val_above, T minval, T maxval, bool saturation)
{
    vc::launchParallelFor(buf_in.width(), buf_in.height(), [&](std::size_t x, std::size_t y)
    {
        if(buf_in.inBounds(x,y)) // is valid
        {
            T val = buf_in(x,y);
            
            if(saturation)
            {
                val = clamp(val, minval, maxval);
            }
            
            const T relative_val = (val - minval) / (maxval - minval);
            
            if(relative_val < thr)
            {
                buf_out(x,y) = val_below;
            }
            else
            {
                buf_out(x,y) = val_above;
            }
        }
    });
}

/**
 * Flip X.
 */
template<typename T, typename Target>
void vc::image::flipXBuffer(const vc::Buffer2DView<T, Target>& buf_in, vc::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in.width() == buf_out.width()) && (buf_in.height() == buf_out.height()));
    
    vc::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        if(buf_out.inBounds(x,y)) // is valid
        {
            const std::size_t nx = ((buf_in.width() - 1) - x);
            if(buf_in.inBounds(nx,y))
            {
                buf_out(x,y) = buf_in(nx,y);
            }
        }
    });
}

/**
 * Flip Y.
 */
template<typename T, typename Target>
void vc::image::flipYBuffer(const vc::Buffer2DView<T, Target>& buf_in, vc::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in.width() == buf_out.width()) && (buf_in.height() == buf_out.height()));
    
    vc::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        if(buf_out.inBounds(x,y)) // is valid
        {
            const std::size_t ny = ((buf_in.height() - 1) - y);
            if(buf_in.inBounds(x,ny))
            {
                buf_out(x,y) = buf_in(x,ny);
            }
        }
    });
}

template<typename T, typename Target>
void vc::image::bufferSubstract(const vc::Buffer2DView<T, Target>& buf_in1, const vc::Buffer2DView<T, Target>& buf_in2, vc::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in1.width() == buf_out.width()) && (buf_in1.height() == buf_out.height()));
    assert((buf_in2.width() == buf_out.width()) && (buf_in2.height() == buf_out.height()));
    
    vc::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        if(buf_out.inBounds(x,y)) // is valid
        {
            buf_out(x,y) = buf_in1(x,y) - buf_in2(x,y);
        }
    });
}

template<typename T, typename Target>
void vc::image::bufferSubstractL1(const vc::Buffer2DView<T, Target>& buf_in1, const vc::Buffer2DView<T, Target>& buf_in2, vc::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in1.width() == buf_out.width()) && (buf_in1.height() == buf_out.height()));
    assert((buf_in2.width() == buf_out.width()) && (buf_in2.height() == buf_out.height()));
    
    vc::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        if(buf_out.inBounds(x,y)) // is valid
        {
            buf_out(x,y) = vc::math::lossL1(buf_in1(x,y) - buf_in2(x,y));
        }
    });
}

template<typename T, typename Target>
void vc::image::bufferSubstractL2(const vc::Buffer2DView<T, Target>& buf_in1, const vc::Buffer2DView<T, Target>& buf_in2, vc::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in1.width() == buf_out.width()) && (buf_in1.height() == buf_out.height()));
    assert((buf_in2.width() == buf_out.width()) && (buf_in2.height() == buf_out.height()));
    
    vc::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        if(buf_out.inBounds(x,y)) // is valid
        {
            buf_out(x,y) = vc::math::lossL2(buf_in1(x,y) - buf_in2(x,y));
        }
    });
}

template<typename T, typename Target>
T vc::image::bufferSum(const vc::Buffer1DView<T, Target>& buf_in, const T& initial, unsigned int tpb)
{
    return vc::launchParallelReduce(buf_in.size(), initial,
    [&](const std::size_t i, T& v)
    {
        v += buf_in(i);
    },
    [&](const T& v1, const T& v2)
    {
        return v1 + v2;
    });
}

template<typename T, typename Target>
T vc::image::bufferSum(const vc::Buffer2DView<T, Target>& buf_in, const T& initial, unsigned int tpb)
{
    return vc::launchParallelReduce(buf_in.width(), buf_in.height(), initial,
    [&](const std::size_t x, const std::size_t y, T& v)
    {
        v += buf_in(x,y);
    },
    [&](const T& v1, const T& v2)
    {
        return v1 + v2;
    });
}

#define JOIN_SPLIT_FUNCTIONS2(TCOMP) \
template void vc::image::join(const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetHost>& buf_in1, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetHost>& buf_in2, vc::Buffer2DView<TCOMP, vc::TargetHost>& buf_out); \
template void vc::image::split(const vc::Buffer2DView<TCOMP, vc::TargetHost>& buf_in, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetHost>& buf_out1, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetHost>& buf_out2); 

#define JOIN_SPLIT_FUNCTIONS3(TCOMP) \
template void vc::image::join(const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetHost>& buf_in1, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetHost>& buf_in2, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetHost>& buf_in3, vc::Buffer2DView<TCOMP, vc::TargetHost>& buf_out); \
template void vc::image::split(const vc::Buffer2DView<TCOMP, vc::TargetHost>& buf_in, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetHost>& buf_out1, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetHost>& buf_out2, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetHost>& buf_out3); 

#define JOIN_SPLIT_FUNCTIONS4(TCOMP) \
template void vc::image::join(const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetHost>& buf_in1, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetHost>& buf_in2, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetHost>& buf_in3, const vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetHost>& buf_in4, vc::Buffer2DView<TCOMP, vc::TargetHost>& buf_out); \
template void vc::image::split(const vc::Buffer2DView<TCOMP, vc::TargetHost>& buf_in, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetHost>& buf_out1, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetHost>& buf_out2, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetHost>& buf_out3, vc::Buffer2DView<typename vc::type_traits<TCOMP>::ChannelType, vc::TargetHost>& buf_out4); \

// instantiations

JOIN_SPLIT_FUNCTIONS2(Eigen::Vector2f)
JOIN_SPLIT_FUNCTIONS3(Eigen::Vector3f)
JOIN_SPLIT_FUNCTIONS4(Eigen::Vector4f)
JOIN_SPLIT_FUNCTIONS2(Eigen::Vector2d)
JOIN_SPLIT_FUNCTIONS3(Eigen::Vector3d)
JOIN_SPLIT_FUNCTIONS4(Eigen::Vector4d)

JOIN_SPLIT_FUNCTIONS2(float2)
JOIN_SPLIT_FUNCTIONS3(float3)
JOIN_SPLIT_FUNCTIONS4(float4)

// statistics

#define MIN_MAX_MEAN_THR_FUNCS(BUF_TYPE) \
template BUF_TYPE vc::image::calcBufferMin<BUF_TYPE, vc::TargetHost>(const vc::Buffer1DView< BUF_TYPE, vc::TargetHost >& buf_in); \
template BUF_TYPE vc::image::calcBufferMax<BUF_TYPE, vc::TargetHost>(const vc::Buffer1DView< BUF_TYPE, vc::TargetHost >& buf_in); \
template BUF_TYPE vc::image::calcBufferMean<BUF_TYPE, vc::TargetHost>(const vc::Buffer1DView< BUF_TYPE, vc::TargetHost >& buf_in); \
template BUF_TYPE vc::image::calcBufferMin<BUF_TYPE, vc::TargetHost>(const vc::Buffer2DView< BUF_TYPE, vc::TargetHost >& buf_in); \
template BUF_TYPE vc::image::calcBufferMax<BUF_TYPE, vc::TargetHost>(const vc::Buffer2DView< BUF_TYPE, vc::TargetHost >& buf_in); \
template BUF_TYPE vc::image::calcBufferMean<BUF_TYPE, vc::TargetHost>(const vc::Buffer2DView< BUF_TYPE, vc::TargetHost >& buf_in); \
template void vc::image::thresholdBuffer<BUF_TYPE, vc::TargetHost>(const vc::Buffer2DView< BUF_TYPE, vc::TargetHost>& buf_in, vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_out, BUF_TYPE thr, BUF_TYPE val_below, BUF_TYPE val_above); \
template void vc::image::thresholdBuffer<BUF_TYPE, vc::TargetHost>(const vc::Buffer2DView< BUF_TYPE, vc::TargetHost>& buf_in, vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_out, BUF_TYPE thr, BUF_TYPE val_below, BUF_TYPE val_above, BUF_TYPE minval, BUF_TYPE maxval, bool saturation);

MIN_MAX_MEAN_THR_FUNCS(float)
MIN_MAX_MEAN_THR_FUNCS(uint8_t)
MIN_MAX_MEAN_THR_FUNCS(uint16_t)

// various

#define SIMPLE_TYPE_FUNCS(BUF_TYPE) \
template void vc::image::leaveQuarter<BUF_TYPE, vc::TargetHost>(const vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_in, vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_out); \
template void vc::image::downsampleHalf<BUF_TYPE, vc::TargetHost>(const vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_in, vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_out); \
template void vc::image::fillBuffer(vc::Buffer1DView<BUF_TYPE, vc::TargetHost>& buf_in, const typename vc::type_traits<BUF_TYPE>::ChannelType& v); \
template void vc::image::fillBuffer(vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_in, const typename vc::type_traits<BUF_TYPE>::ChannelType& v); \
template void vc::image::invertBuffer(vc::Buffer1DView<BUF_TYPE, vc::TargetHost>& buf_io); \
template void vc::image::invertBuffer(vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_io);  \
template void vc::image::flipXBuffer(const vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_in, vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_out); \
template void vc::image::flipYBuffer(const vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_in, vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_out); \
template BUF_TYPE vc::image::bufferSum(const vc::Buffer1DView<BUF_TYPE, vc::TargetHost>& buf_in, const BUF_TYPE& initial, unsigned int tpb);\
template BUF_TYPE vc::image::bufferSum(const vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_in, const BUF_TYPE& initial, unsigned int tpb);

SIMPLE_TYPE_FUNCS(uint8_t)
SIMPLE_TYPE_FUNCS(uint16_t)
SIMPLE_TYPE_FUNCS(uchar3)
SIMPLE_TYPE_FUNCS(uchar4)
SIMPLE_TYPE_FUNCS(float)
SIMPLE_TYPE_FUNCS(float3)
SIMPLE_TYPE_FUNCS(float4)
SIMPLE_TYPE_FUNCS(Eigen::Vector3f)
SIMPLE_TYPE_FUNCS(Eigen::Vector4f)

#define CLAMP_FUNC_TYPES(BUF_TYPE) \
template void vc::image::clampBuffer(vc::Buffer1DView<BUF_TYPE, vc::TargetHost>& buf_io, BUF_TYPE a, BUF_TYPE b); \
template void vc::image::clampBuffer(vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_io, BUF_TYPE a, BUF_TYPE b);

CLAMP_FUNC_TYPES(uint8_t)
CLAMP_FUNC_TYPES(uint16_t)
CLAMP_FUNC_TYPES(float)
CLAMP_FUNC_TYPES(float2)
CLAMP_FUNC_TYPES(float3)
CLAMP_FUNC_TYPES(float4)

#define STUPID_FUNC_TYPES(BUF_TYPE)\
template void vc::image::bufferSubstract(const vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_in1, const vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_in2, vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_out);\
template void vc::image::bufferSubstractL1(const vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_in1, const vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_in2, vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_out);\
template void vc::image::bufferSubstractL2(const vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_in1, const vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_in2, vc::Buffer2DView<BUF_TYPE, vc::TargetHost>& buf_out);
STUPID_FUNC_TYPES(float)

template void vc::image::rescaleBufferInplace<float, vc::TargetHost>(vc::Buffer1DView< float, vc::TargetHost >& buf_in, float alpha, float beta, float clamp_min, float clamp_max);
template void vc::image::rescaleBufferInplace<float, vc::TargetHost>(vc::Buffer2DView< float, vc::TargetHost >& buf_in, float alpha, float beta, float clamp_min, float clamp_max);
template void vc::image::rescaleBufferInplaceMinMax<float, vc::TargetHost>(vc::Buffer2DView< float, vc::TargetHost >& buf_in, float alpha, float beta, float clamp_min, float clamp_max);
template void vc::image::normalizeBufferInplace<float, vc::TargetHost>(vc::Buffer2DView< float, vc::TargetHost >& buf_in);

template void vc::image::rescaleBuffer<uint8_t, float, vc::TargetHost>(const vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_in, vc::Buffer2DView<float, vc::TargetHost>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void vc::image::rescaleBuffer<uint16_t, float, vc::TargetHost>(const vc::Buffer2DView<uint16_t, vc::TargetHost>& buf_in, vc::Buffer2DView<float, vc::TargetHost>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void vc::image::rescaleBuffer<uint32_t, float, vc::TargetHost>(const vc::Buffer2DView<uint32_t, vc::TargetHost>& buf_in, vc::Buffer2DView<float, vc::TargetHost>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void vc::image::rescaleBuffer<float, float, vc::TargetHost>(const vc::Buffer2DView<float, vc::TargetHost>& buf_in, vc::Buffer2DView<float, vc::TargetHost>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);

template void vc::image::rescaleBuffer<float, uint8_t, vc::TargetHost>(const vc::Buffer2DView<float, vc::TargetHost>& buf_in, vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void vc::image::rescaleBuffer<float, uint16_t, vc::TargetHost>(const vc::Buffer2DView<float, vc::TargetHost>& buf_in, vc::Buffer2DView<uint16_t, vc::TargetHost>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void vc::image::rescaleBuffer<float, uint32_t, vc::TargetHost>(const vc::Buffer2DView<float, vc::TargetHost>& buf_in, vc::Buffer2DView<uint32_t, vc::TargetHost>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);

template void vc::image::downsampleHalfNoInvalid<uint8_t, vc::TargetHost>(const vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_in, vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_out);
template void vc::image::downsampleHalfNoInvalid<uint16_t, vc::TargetHost>(const vc::Buffer2DView<uint16_t, vc::TargetHost>& buf_in, vc::Buffer2DView<uint16_t, vc::TargetHost>& buf_out);
template void vc::image::downsampleHalfNoInvalid<float, vc::TargetHost>(const vc::Buffer2DView<float, vc::TargetHost>& buf_in, vc::Buffer2DView<float, vc::TargetHost>& buf_out);
