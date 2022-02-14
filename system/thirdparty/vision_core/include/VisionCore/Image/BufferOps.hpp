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
 * Various operations on buffers.
 * ****************************************************************************
 */

#ifndef VISIONCORE_IMAGE_BUFFEROPS_HPP
#define VISIONCORE_IMAGE_BUFFEROPS_HPP

#include <type_traits>

#include <VisionCore/Buffers/Buffer1D.hpp>
#include <VisionCore/Buffers/Buffer2D.hpp>
#include <VisionCore/Buffers/Image2D.hpp>
#include <VisionCore/Buffers/ImagePyramid.hpp>

/**
 * @note No dimension checking for now, also Thrust is non-pitched.
 */

namespace vc
{

namespace image
{
    
/**
 * Rescale element-wise and clamp.
 * out = in * alpha + beta
 */
template<typename T, typename Target>
void rescaleBufferInplace(Buffer1DView<T, Target>& buf_in, T alpha, T beta = 0.0f, T clamp_min = 0.0f, T clamp_max = 1.0f);

/**
 * Rescale element-wise and clamp.
 * out = in * alpha + beta
 */
template<typename T, typename Target>
void rescaleBufferInplace(Buffer2DView<T, Target>& buf_in, T alpha, T beta = 0.0f, T clamp_min = 0.0f, T clamp_max = 1.0f);

/**
 * Rescale element-wise and clamp.
 * out = in * alpha + beta
 */
template<typename T, typename Target>
void rescaleBufferInplaceMinMax(Buffer2DView<T, Target>& buf_in, T vmin, T vmax, T clamp_min = 0.0f, T clamp_max = 1.0f);

/**
 * Rescale element-wise and clamp.
 * out = in * alpha + beta
 */
template<typename T1, typename T2, typename Target>
void rescaleBuffer(const Buffer2DView<T1, Target>& buf_in, Buffer2DView<T2, Target>& buf_out, 
                   float alpha, float beta = 0.0f, float clamp_min = 0.0f, float clamp_max = 1.0f);

/**
 * Normalize buffer to 0..1 range (float only) for now.
 */
template<typename T, typename Target>
void normalizeBufferInplace(Buffer2DView<T, Target>& buf_in);

/**
 * Clamp Buffer
 */
template<typename T, typename Target>
void clampBuffer(Buffer1DView<T, Target>& buf_io, T a, T b);

/**
 * Clamp Buffer
 */
template<typename T, typename Target>
void clampBuffer(Buffer2DView<T, Target>& buf_io, T a, T b);


/**
 * Find minimal value of a 1D buffer.
 */
template<typename T, typename Target>
T calcBufferMin(const Buffer1DView<T, Target>& buf_in);

/**
 * Find maximal value of a 1D buffer.
 */
template<typename T, typename Target>
T calcBufferMax(const Buffer1DView<T, Target>& buf_in);

/**
 * Find mean value of a 1D buffer.
 */
template<typename T, typename Target>
T calcBufferMean(const Buffer1DView<T, Target>& buf_in);

/**
 * Find minimal value of a 2D buffer.
 */
template<typename T, typename Target>
T calcBufferMin(const Buffer2DView<T, Target>& buf_in);

/**
 * Find maximal value of a 2D buffer.
 */
template<typename T, typename Target>
T calcBufferMax(const Buffer2DView<T, Target>& buf_in);

/**
 * Find mean value of a 2D buffer.
 */
template<typename T, typename Target>
T calcBufferMean(const Buffer2DView<T, Target>& buf_in);

/**
 * Downsample by half.
 */
template<typename T, typename Target>
void downsampleHalf(const Buffer2DView<T, Target>& buf_in, Buffer2DView<T, Target>& buf_out);

/**
 * Downsample by half (ignore invalid).
 */
template<typename T, typename Target>
void downsampleHalfNoInvalid(const Buffer2DView<T, Target>& buf_in, Buffer2DView<T, Target>& buf_out);

/**
 * Leave even rows and columns.
 */
template<typename T, typename Target>
void leaveQuarter(const Buffer2DView<T, Target>& buf_in, Buffer2DView<T, Target>& buf_out);

/**
 * Fills remaining pyramid levels with downsampleHalf.
 */
template<typename T, std::size_t Levels, typename Target>
static inline void fillPyramidBilinear(ImagePyramidView<T,Levels,Target>& pyr)
{
    for(std::size_t l = 1 ; l < Levels ; ++l) 
    {
        downsampleHalf(pyr[l-1],pyr[l]);
    }
}

/**
 * Fills remaining pyramid levels with leaveQuarter.
 */
template<typename T, std::size_t Levels, typename Target>
static inline void fillPyramidCrude(ImagePyramidView<T,Levels,Target>& pyr)
{
    for(std::size_t l = 1 ; l < Levels ; ++l) 
    {
        leaveQuarter(pyr[l-1],pyr[l]);
    }
}

/**
 * Join buffers.
 */
template<typename TCOMP, typename Target>
void join(const Buffer2DView<typename type_traits<TCOMP>::ChannelType, Target>& buf_in1, 
          const Buffer2DView<typename type_traits<TCOMP>::ChannelType, Target>& buf_in2, 
          Buffer2DView<TCOMP, Target>& buf_out);
template<typename TCOMP, typename Target>
void join(const Buffer2DView<typename type_traits<TCOMP>::ChannelType, Target>& buf_in1, 
          const Buffer2DView<typename type_traits<TCOMP>::ChannelType, Target>& buf_in2, 
          const Buffer2DView<typename type_traits<TCOMP>::ChannelType, Target>& buf_in3, 
          Buffer2DView<TCOMP, Target>& buf_out);
template<typename TCOMP, typename Target>
void join(const Buffer2DView<typename type_traits<TCOMP>::ChannelType, Target>& buf_in1, 
          const Buffer2DView<typename type_traits<TCOMP>::ChannelType, Target>& buf_in2, 
          const Buffer2DView<typename type_traits<TCOMP>::ChannelType, Target>& buf_in3, 
          const Buffer2DView<typename type_traits<TCOMP>::ChannelType, Target>& buf_in4, 
          Buffer2DView<TCOMP, Target>& buf_out);

/**
 * Split buffers.
 */
template<typename TCOMP, typename Target>
void split(const Buffer2DView<TCOMP, Target>& buf_in, Buffer2DView<typename type_traits<TCOMP>::ChannelType, Target>& buf_out1, 
           Buffer2DView<typename type_traits<TCOMP>::ChannelType, Target>& buf_out2);
template<typename TCOMP, typename Target>
void split(const Buffer2DView<TCOMP, Target>& buf_in, Buffer2DView<typename type_traits<TCOMP>::ChannelType, Target>& buf_out1, 
           Buffer2DView<typename type_traits<TCOMP>::ChannelType, Target>& buf_out2, 
           Buffer2DView<typename type_traits<TCOMP>::ChannelType, Target>& buf_out3);
template<typename TCOMP, typename Target>
void split(const Buffer2DView<TCOMP, Target>& buf_in, Buffer2DView<typename type_traits<TCOMP>::ChannelType, Target>& buf_out1, 
           Buffer2DView<typename type_traits<TCOMP>::ChannelType, Target>& buf_out2, 
           Buffer2DView<typename type_traits<TCOMP>::ChannelType, Target>& buf_out3, 
           Buffer2DView<typename type_traits<TCOMP>::ChannelType, Target>& buf_out4);

/**
 * fillBuffer
 */
template<typename T, typename Target>
void fillBuffer(Buffer1DView<T, Target>& buf_in, const typename type_traits<T>::ChannelType& v);

/**
 * fillBuffer
 */
template<typename T, typename Target>
void fillBuffer(Buffer2DView<T, Target>& buf_in, const typename type_traits<T>::ChannelType& v);

/**
 * Invert Buffer
 */
template<typename T, typename Target>
void invertBuffer(Buffer1DView<T, Target>& buf_io);

/**
 * Invert Buffer
 */
template<typename T, typename Target>
void invertBuffer(Buffer2DView<T, Target>& buf_io);

/**
 * Threshold Buffer
 */
template<typename T, typename Target>
void thresholdBuffer(const Buffer2DView<T, Target>& buf_in, Buffer2DView<T, Target>& buf_out, 
                     T thr, T val_below, T val_above);

/**
 * Threshold Buffer
 */
template<typename T, typename Target>
void thresholdBuffer(const Buffer2DView<T, Target>& buf_in, Buffer2DView<T, Target>& buf_out, 
                     T thr, T val_below, T val_above, T minval, T maxval, bool saturation = false);

/**
 * Flip X.
 */
template<typename T, typename Target>
void flipXBuffer(const Buffer2DView<T, Target>& buf_in, Buffer2DView<T, Target>& buf_out);

/**
 * Flip Y.
 */
template<typename T, typename Target>
void flipYBuffer(const Buffer2DView<T, Target>& buf_in, Buffer2DView<T, Target>& buf_out);

/**
 * Substract.
 */
template<typename T, typename Target>
void bufferSubstract(const Buffer2DView<T, Target>& buf_in1,
                     const Buffer2DView<T, Target>& buf_in2,
                     Buffer2DView<T, Target>& buf_out);

/**
 * Substract L1.
 */
template<typename T, typename Target>
void bufferSubstractL1(const Buffer2DView<T, Target>& buf_in1,
                       const Buffer2DView<T, Target>& buf_in2,
                       Buffer2DView<T, Target>& buf_out);

/**
 * Substract L2.
 */
template<typename T, typename Target>
void bufferSubstractL2(const Buffer2DView<T, Target>& buf_in1,
                       const Buffer2DView<T, Target>& buf_in2,
                       Buffer2DView<T, Target>& buf_out);

template<typename T, typename Target>
T bufferSum(const Buffer1DView<T, Target>& buf_in, const T& initial, unsigned int tpb = 32);

template<typename T, typename Target>
T bufferSum(const Buffer2DView<T, Target>& buf_in, const T& initial, unsigned int tpb = 32);

}

}


#endif // VISIONCORE_IMAGE_BUFFEROPS_HPP
