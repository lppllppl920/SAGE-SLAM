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
 * PLY file IO.
 * ****************************************************************************
 */

#ifndef VISIONCORE_IO_PLYMODEL_HPP
#define VISIONCORE_IO_PLYMODEL_HPP

#include <stdint.h>
#include <stddef.h>
#include <cstring>

#include <stdexcept>
#include <vector>
#include <string>
#include <memory>
#include <functional>

#include <VisionCore/Platform.hpp>

#include <Eigen/Dense>
#include <Eigen/StdVector>

namespace vc
{
    
struct ColorPoint
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    Eigen::Matrix<float,3,1> Position;
    Eigen::Matrix<float,3,1> Color;
};

struct NormalPoint
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    Eigen::Matrix<float,3,1> Position;
    Eigen::Matrix<float,3,1> Normal;
};

struct ColorNormalPoint
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    Eigen::Matrix<float,3,1> Position;
    Eigen::Matrix<float,3,1> Color;
    Eigen::Matrix<float,3,1> Normal;
};
    
struct Surfel
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    Eigen::Matrix<float,3,1> Position;
    Eigen::Matrix<float,3,1> Color;
    Eigen::Matrix<float,3,1> Normal;
    float                    Radius;
};

namespace io
{

template<typename T>
bool loadPLY(const std::string& plyfilename, std::vector<T, Eigen::aligned_allocator<T>>& vec, float scalePos = 1.0f, std::vector<std::vector<std::size_t>>* idx = nullptr);

template<typename T>
bool savePLY(const std::string& plyfilename, const std::vector<T,Eigen::aligned_allocator<T>>& vec, float scalePos = 1.0f, const std::vector<std::vector<std::size_t>>* idx = nullptr);

inline std::ostream& operator<<(std::ostream& os, const Surfel& s)
{
    os << "Surfel(p = " << s.Position(0) << " , " << s.Position(1) << " , " << s.Position(2) 
           << " | c = " << s.Color(0) << " , " << s.Color(1) << " , " << s.Color(2) 
           << " | n = " << s.Normal(0) << " , " << s.Normal(1) << " , " << s.Normal(2) 
           << " | r = " << s.Radius << ")";
    return os;
}
  
}
    
template<> struct type_traits<Surfel>
{
    typedef float ChannelType;
    static constexpr int ChannelCount = 10;
    static constexpr bool IsCUDAType = false;
    static constexpr bool IsEigenType = false;
};

template<> struct type_traits<ColorPoint>
{
    typedef float ChannelType;
    static constexpr int ChannelCount = 6;
    static constexpr bool IsCUDAType = false;
    static constexpr bool IsEigenType = false;
};

template<> struct type_traits<NormalPoint>
{
    typedef float ChannelType;
    static constexpr int ChannelCount = 6;
    static constexpr bool IsCUDAType = false;
    static constexpr bool IsEigenType = false;
};

template<> struct type_traits<ColorNormalPoint>
{
    typedef float ChannelType;
    static constexpr int ChannelCount = 9;
    static constexpr bool IsCUDAType = false;
    static constexpr bool IsEigenType = false;
};
    
}

#endif // VISIONCORE_IO_PLYMODEL_HPP
