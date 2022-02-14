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
 * Launching parallel processing.
 * ****************************************************************************
 */

#ifndef VISIONCORE_LAUNCH_UTILS_HPP
#define VISIONCORE_LAUNCH_UTILS_HPP

#include <VisionCore/Platform.hpp>
#include <VisionCore/Buffers/Buffer1D.hpp>
#include <VisionCore/Buffers/Buffer2D.hpp>
#include <VisionCore/Buffers/Buffer3D.hpp>

#ifdef VISIONCORE_HAVE_TBB
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#endif // VISIONCORE_HAVE_TBB

namespace vc
{
    
namespace detail
{
    template<typename T>
    inline int gcd(T a, T b)
    {
        const T amodb = a%b;
        return amodb ? gcd(b, amodb) : b;
    }
    
    inline unsigned int calculateBlockDim(unsigned int dim, unsigned int blocks = 32)
    {
        return detail::gcd<unsigned int>(dim, blocks);
    }
    
    inline unsigned int calculateBlockDimOver(unsigned int dim, unsigned int blocks = 32)
    {
      return blocks;
    }
    
    inline unsigned int calculateGridDim(unsigned int dim, unsigned int blocks = 32)
    {
      return dim / blocks;
    }
    
    inline unsigned int calculateGridDimOver(unsigned int dim, unsigned int blocks = 32)
    {
      return ceil(dim / (double)blocks);
    }
}

template<typename T>
inline void InitDimFromBuffer(dim3& blockDim, dim3& gridDim, const Buffer1DView<T,TargetDeviceCUDA>& image, int blockx = 32)
{
    blockDim = dim3(detail::calculateBlockDim(image.size(),blockx), 1, 1);
    gridDim =  dim3(detail::calculateGridDim(image.size(),blockDim.x), 1, 1);
}

template<typename T>
inline void InitDimFromBufferOver(dim3& blockDim, dim3& gridDim, const Buffer1DView<T,TargetDeviceCUDA>& image, 
                                  int blockx = 32)
{
    blockDim = dim3(detail::calculateBlockDimOver(image.size(), blockx), 1, 1);
    gridDim =  dim3(detail::calculateGridDimOver(image.size(), blockDim.x), 1, 1);
}

template<typename T>
inline void InitDimFromBuffer(dim3& blockDim, dim3& gridDim, const Buffer2DView<T,TargetDeviceCUDA>& image, 
                              unsigned blockx = 32, unsigned blocky = 32)
{
    blockDim = dim3(detail::calculateBlockDim(image.width(),blockx), 
                    detail::calculateBlockDim(image.height(),blocky), 1);
    gridDim =  dim3(detail::calculateGridDim(image.width(),blockDim.x), 
                    detail::calculateGridDim(image.height(),blockDim.y), 1);
}

template<typename T>
inline void InitDimFromBufferOver(dim3& blockDim, dim3& gridDim, const Buffer2DView<T,TargetDeviceCUDA>& image, 
                                  int blockx = 32, int blocky = 32)
{
    blockDim = dim3(detail::calculateBlockDimOver(image.width(), blockx), 
                    detail::calculateBlockDimOver(image.height(), blocky), 1);
    gridDim =  dim3( detail::calculateGridDimOver(image.width(), blockDim.x), 
                     detail::calculateGridDimOver(image.height(), blockDim.y), 1);
}

template<typename T>
inline void InitDimFromBuffer(dim3& blockDim, dim3& gridDim, const Buffer3DView<T,TargetDeviceCUDA>& image, 
                              unsigned blockx = 32, unsigned blocky = 32, unsigned blockz = 32)
{
    blockDim = dim3(detail::calculateBlockDim(image.width(),blockx),
                    detail::calculateBlockDim(image.height(),blocky), 
                    detail::calculateBlockDim(image.depth(),blockz));
    gridDim =  dim3(detail::calculateGridDim(image.width(),blockDim.x),
                    detail::calculateGridDim(image.height(),blockDim.y),
                    detail::calculateGridDim(image.depth(),blockDim.z));
}

template<typename T>
inline void InitDimFromBufferOver(dim3& blockDim, dim3& gridDim, const Buffer3DView<T,TargetDeviceCUDA>& image, 
                                  int blockx = 32, int blocky = 32, int blockz = 32)
{
    blockDim = dim3(detail::calculateBlockDimOver(image.width(),blockx),
                    detail::calculateBlockDimOver(image.height(),blocky), 
                    detail::calculateBlockDimOver(image.depth(),blockz));
    gridDim =  dim3(detail::calculateGridDimOver(image.width(),blockDim.x),
                    detail::calculateGridDimOver(image.height(),blockDim.y),
                    detail::calculateGridDimOver(image.depth(),blockDim.z));
}

inline void InitDimFromDimensions2D(dim3& blockDim, dim3& gridDim, int dimx, int dimy, int blockx = 32, int blocky = 32)
{
  blockDim = dim3(detail::calculateBlockDim(dimx, blockx), 
                  detail::calculateBlockDim(dimy, blocky), 1);
  gridDim =  dim3( detail::calculateGridDim(dimx, blockDim.x), 
                   detail::calculateGridDim(dimy, blockDim.y), 1);
}

inline void InitDimFromDimensions2DOver(dim3& blockDim, dim3& gridDim, int dimx, int dimy, int blockx = 32, int blocky = 32)
{
    blockDim = dim3(detail::calculateBlockDimOver(dimx, blockx), 
                    detail::calculateBlockDimOver(dimy, blocky), 1);
    gridDim =  dim3( detail::calculateGridDimOver(dimx, blockDim.x), 
                    detail::calculateGridDimOver(dimy, blockDim.y), 1);
}

/// TBB!
#ifdef VISIONCORE_HAVE_TBB
template<typename PerItemFunction>
static inline void launchParallelFor(std::size_t dim, PerItemFunction pif)
{
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, dim), [&](const tbb::blocked_range<std::size_t>& r)
    {
        for(std::size_t i = r.begin() ; i != r.end() ; ++i )
        {
            pif(i);
        }
    });
}

template<typename PerItemFunction>
static inline void launchParallelFor(std::size_t dimx, std::size_t dimy, PerItemFunction pif)
{
    tbb::parallel_for(tbb::blocked_range2d<std::size_t>(0, dimy, 0, dimx), [&](const tbb::blocked_range2d<std::size_t>& r)
    {
        for(std::size_t y = r.rows().begin() ; y != r.rows().end() ; ++y )
        {
            for(std::size_t x = r.cols().begin() ; x != r.cols().end() ; ++x ) 
            {
                pif(x,y);
            }
        }
    });
}

template<typename VT, typename PerItemFunction, typename JoinFunction>
static inline VT launchParallelReduce(std::size_t dim, const VT& initial, PerItemFunction pif, JoinFunction jf)
{
    return tbb::parallel_reduce(tbb::blocked_range<std::size_t>(0, dim), initial,
    [&](const tbb::blocked_range<std::size_t>& r, const VT& v)
    {
        VT ret = v;
        for(std::size_t i = r.begin() ; i != r.end() ; ++i )
        {
            pif(i, ret);
        }
        return ret;
    },
    [&](const VT& v1, const VT& v2)
    {
        return jf(v1, v2);
    }
    );
}

template<typename VT, typename PerItemFunction, typename JoinFunction>
static inline VT launchParallelReduce(std::size_t dimx, std::size_t dimy, const VT& initial, PerItemFunction pif, JoinFunction jf)
{
    return tbb::parallel_reduce(tbb::blocked_range2d<std::size_t>(0, dimy, 0, dimx), initial,
    [&](const tbb::blocked_range2d<std::size_t>& r, const VT& v)
    {
        VT ret = v;
        
        for(std::size_t y = r.rows().begin() ; y != r.rows().end() ; ++y )
        {
            for(std::size_t x = r.cols().begin() ; x != r.cols().end() ; ++x ) 
            {
                pif(x,y,ret);
            }
        }
        return ret;
    },
    [&](const VT& v1, const VT& v2)
    {
        return jf(v1, v2);
    }
    );
}

#else // VISIONCORE_HAVE_TBB

/// no TBB = no parallelism at all

template<typename PerItemFunction>
static inline void launchParallelFor(std::size_t dim, PerItemFunction pif)
{
    for(std::size_t i = 0 ; i < dim ; ++i )
    {
        pif(i);
    }
}

template<typename PerItemFunction>
static inline void launchParallelFor(std::size_t dimx, std::size_t dimy, PerItemFunction pif)
{
    for(std::size_t y = 0 ; y < dimy ; ++y)
    {
        for(std::size_t x = 0 ; x < dimy ; ++x)
        {
            pif(x,y);
        }
    }
}

template<typename VT, typename PerItemFunction, typename JoinFunction>
static inline VT launchParallelReduce(std::size_t dim, const VT& initial, PerItemFunction pif, JoinFunction jf)
{
    VT ret = initial;
    
    for(std::size_t i = 0 ; i < dim ; ++i )
    {
        pif(i, ret);
    }
    
    return ret;
}

template<typename VT, typename PerItemFunction, typename JoinFunction>
static inline VT launchParallelReduce(std::size_t dimx, std::size_t dimy, const VT& initial, PerItemFunction pif, JoinFunction jf)
{
    VT ret = initial;
    
    for(std::size_t y = 0 ; y < dimy ; ++y)
    {
        for(std::size_t x = 0 ; x < dimy ; ++x)
        {
            pif(x,y, ret);
        }
    }
    
    return ret;
}

#endif // VISIONCORE_HAVE_TBB

}
#endif // VISIONCORE_LAUNCH_UTILS_HPP
