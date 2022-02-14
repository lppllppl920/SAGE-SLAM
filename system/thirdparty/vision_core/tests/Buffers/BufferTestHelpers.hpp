/**
 * ****************************************************************************
 * Copyright (c) 2017, Robert Lukierski.
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
 * Buffer Test Helpers.
 * ****************************************************************************
 */

#include <sophus/se3.hpp>

#include <VisionCore/Buffers/Buffer1D.hpp>
#include <VisionCore/Buffers/Buffer2D.hpp>
#include <VisionCore/Buffers/Buffer3D.hpp>

template<typename T>
struct BufferElementOps
{
    EIGEN_DEVICE_FUNC static inline void assign(T& val, std::size_t el, std::size_t max)
    {
        val = el;
    }
    
    EIGEN_DEVICE_FUNC static inline void check(const T& val, const T& gt, std::size_t el = 0)
    {
        ASSERT_EQ(val, gt) << "Wrong data at " << el;
    }
};

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct BufferElementOps<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
{
    typedef Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> EigenT;
    
    EIGEN_DEVICE_FUNC static inline void assign(EigenT& val, std::size_t el, std::size_t max)
    {
        for(std::size_t r = 0 ; r < _Rows ; ++r)
        {
            for(std::size_t c = 0 ; c < _Cols ; ++c)
            {
                val(r,c) = (_Scalar)(_Rows * _Cols * el);
            }
        }
    }
    
    EIGEN_DEVICE_FUNC static inline void check(const EigenT& val, const EigenT& gt, std::size_t el = 0)
    {
        for(std::size_t r = 0 ; r < _Rows ; ++r)
        {
            for(std::size_t c = 0 ; c < _Cols ; ++c)
            {
                ASSERT_EQ(val(r,c), (_Rows * _Cols * el)) << "Wrong data at " << el << " , " << r << " , " << c;
            }
        }
    }
};

template<typename _Scalar, int _Options>
struct BufferElementOps<Sophus::SE3<_Scalar, _Options>>
{
    typedef Sophus::SE3<_Scalar, _Options> SophusT;
    
    EIGEN_DEVICE_FUNC static inline void assign(SophusT& val, std::size_t el, std::size_t max)
    {
        
    }
    
    EIGEN_DEVICE_FUNC static inline void check(const SophusT& val, const SophusT& gt, std::size_t el = 0)
    {
      //ASSERT_EQ(val, gt) << "Wrong data at " << el;
    }
};

template<typename BufferElementT>
void LaunchKernel_WriteBuffer1D(const vc::Buffer1DView<BufferElementT, vc::TargetDeviceCUDA>& buffer_gpu, std::size_t BufferSize);
template<typename BufferElementT>
void LaunchKernel_WriteBuffer2D(const vc::Buffer2DView<BufferElementT, vc::TargetDeviceCUDA>& buffer_gpu, 
                                std::size_t BufferSizeX, std::size_t BufferSizeY);
