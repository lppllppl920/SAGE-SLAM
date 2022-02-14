/**
 * ****************************************************************************
 * Copyright (c) 2018, Robert Lukierski.
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
 * Generic traits etc.
 * ****************************************************************************
 */

#ifndef VISIONCORE_TYPETRAITS_HPP
#define VISIONCORE_TYPETRAITS_HPP

// ---------------------------------------------------------------------------
// Eigen is mandatory
// ---------------------------------------------------------------------------
#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

// ---------------------------------------------------------------------------
// CUDA Macros and CUDA types if CUDA not available
// ---------------------------------------------------------------------------
#include <VisionCore/HelpersCUDA.hpp>

namespace vc
{
// ---------------------------------------------------------------------------
// Our internal type traits
// ---------------------------------------------------------------------------
    template<typename T> struct type_traits 
    { 
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<char>
    {
        typedef char ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<unsigned char>
    {
        typedef unsigned char ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<short>
    {
        typedef short ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<unsigned short>
    {
        typedef unsigned short ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<int>
    {
        typedef int ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<unsigned int>
    {
        typedef unsigned int ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<long>
    {
        typedef long ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<unsigned long>
    {
        typedef unsigned long ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<float>
    {
        typedef float ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<double>
    {
        typedef double ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<long double>
    {
        typedef long double ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<char1>
    {
        typedef char ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<uchar1>
    {
        typedef unsigned char ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<char2>
    {
        typedef char ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<uchar2>
    {
        typedef unsigned char ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<char3>
    {
        typedef char ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<uchar3>
    {
        typedef unsigned char ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<char4>
    {
        typedef char ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<uchar4>
    {
        typedef unsigned char ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<short1>
    {
        typedef short ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ushort1>
    {
        typedef unsigned short ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<short2>
    {
        typedef short ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ushort2>
    {
        typedef unsigned short ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<short3>
    {
        typedef short ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ushort3>
    {
        typedef unsigned short ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<short4>
    {
        typedef short ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ushort4>
    {
        typedef unsigned short ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<int1>
    {
        typedef int ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<uint1>
    {
        typedef unsigned int ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<int2>
    {
        typedef int ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<uint2>
    {
        typedef unsigned int ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<int3>
    {
        typedef int ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<uint3>
    {
        typedef unsigned int ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<int4>
    {
        typedef int ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<uint4>
    {
        typedef unsigned int ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<long1>
    {
        typedef long ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ulong1>
    {
        typedef unsigned long ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<long2>
    {
        typedef long ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ulong2>
    {
        typedef unsigned long ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<long3>
    {
        typedef long ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ulong3>
    {
        typedef unsigned long ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<long4>
    {
        typedef long ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ulong4>
    {
        typedef unsigned long ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<float1>
    {
        typedef float ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<float2>
    {
        typedef float ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<float3>
    {
        typedef float ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<float4>
    {
        typedef float ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<longlong1>
    {
        typedef long long ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ulonglong1>
    {
        typedef unsigned long long ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<longlong2>
    {
        typedef long long ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ulonglong2>
    {
        typedef unsigned long long ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<longlong3>
    {
        typedef long long ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ulonglong3>
    {
        typedef unsigned long long ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<longlong4>
    {
        typedef long long ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ulonglong4>
    {
        typedef unsigned long long ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<double1>
    {
        typedef double ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<double2>
    {
        typedef double ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<double3>
    {
        typedef double ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<double4>
    {
        typedef double ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    struct type_traits<Eigen::Array<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols>>
    {
        typedef _Scalar ChannelType;
        static constexpr int ChannelCount = _Rows * _Cols;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = true;
    };
    
    template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    struct type_traits<Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols>>
    {
        typedef _Scalar ChannelType;
        static constexpr int ChannelCount = _Rows * _Cols;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = true;
    };
}

#endif // VISIONCORE_TYPETRAITS_HPP
