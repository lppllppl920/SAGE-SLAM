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
 * Join/Split helpers.
 * ****************************************************************************
 */

#ifndef VISIONCORE_JOIN_SPLIT_HELPERS_HPP
#define VISIONCORE_JOIN_SPLIT_HELPERS_HPP

#include <VisionCore/Platform.hpp>

namespace internal
{
    
template<typename TCOMP, bool is_eig = vc::type_traits<TCOMP>::IsEigenType , 
                         bool is_cud = vc::type_traits<TCOMP>::IsCUDAType, 
                         int Dimension = vc::type_traits<TCOMP>::ChannelCount>
struct JoinSplitHelper
{
    typedef typename vc::type_traits<TCOMP>::ChannelType TSCALAR;
    
    EIGEN_DEVICE_FUNC static inline void join(const TSCALAR& v1, const TSCALAR& v2, TCOMP& out);
    EIGEN_DEVICE_FUNC static inline void join(const TSCALAR& v1, const TSCALAR& v2, const TSCALAR& v3, TCOMP& out);
    EIGEN_DEVICE_FUNC static inline void join(const TSCALAR& v1, const TSCALAR& v2, const TSCALAR& v3, const TSCALAR& v4, TCOMP& out);
    
    EIGEN_DEVICE_FUNC static inline void split(const TCOMP& input, TSCALAR& v1,TSCALAR& v2);
    EIGEN_DEVICE_FUNC static inline void split(const TCOMP& input, TSCALAR& v1,TSCALAR& v2, TSCALAR& v3);
    EIGEN_DEVICE_FUNC static inline void split(const TCOMP& input, TSCALAR& v1,TSCALAR& v2, TSCALAR& v3, TSCALAR& v4);
    
    EIGEN_DEVICE_FUNC static inline TCOMP invertedValue(const TCOMP& v);
};

template<typename TCOMP>
struct JoinSplitHelper<TCOMP,true,false,2> // for eigen2
{
    typedef typename vc::type_traits<TCOMP>::ChannelType TSCALAR;
    static constexpr int Dimension = vc::type_traits<TCOMP>::ChannelCount;
    
    static_assert(TCOMP::ColsAtCompileTime == 1, "Only vectors please");
    
    EIGEN_DEVICE_FUNC static inline void join(const TSCALAR& v1, const TSCALAR& v2, TCOMP& out)
    {
        out << v1 , v2;
    }
    
    EIGEN_DEVICE_FUNC static inline void split(const TCOMP& input, TSCALAR& v1,TSCALAR& v2)
    {
        v1 = input(0);
        v2 = input(1);
    }
    
    EIGEN_DEVICE_FUNC static inline TCOMP invertedValue(const TCOMP& v)
    {
        return (TSCALAR(1.0) / v.array()).matrix();
    }
};

template<typename TCOMP>
struct JoinSplitHelper<TCOMP,true,false,3> // for eigen3
{
    typedef typename vc::type_traits<TCOMP>::ChannelType TSCALAR;
    static constexpr int Dimension = vc::type_traits<TCOMP>::ChannelCount;
    
    static_assert(TCOMP::ColsAtCompileTime == 1, "Only vectors please");
    
    EIGEN_DEVICE_FUNC static inline void join(const TSCALAR& v1, const TSCALAR& v2, const TSCALAR& v3, TCOMP& out)
    {
        out << v1 , v2 , v3;
    }
    
    EIGEN_DEVICE_FUNC static inline void split(const TCOMP& input, TSCALAR& v1,TSCALAR& v2, TSCALAR& v3)
    {
        v1 = input(0);
        v2 = input(1);
        v3 = input(2);
    }
    
    EIGEN_DEVICE_FUNC static inline TCOMP invertedValue(const TCOMP& v)
    {
        return (TSCALAR(1.0) / v.array()).matrix();
    }
};

template<typename TCOMP>
struct JoinSplitHelper<TCOMP,true,false,4> // for eigen4
{
    typedef typename vc::type_traits<TCOMP>::ChannelType TSCALAR;
    static constexpr int Dimension = vc::type_traits<TCOMP>::ChannelCount;
    
    static_assert(TCOMP::ColsAtCompileTime == 1, "Only vectors please");
    
    EIGEN_DEVICE_FUNC static inline void join(const TSCALAR& v1, const TSCALAR& v2, const TSCALAR& v3, const TSCALAR& v4, TCOMP& out)
    {
        out << v1 , v2 , v3 , v4;
    }
    
    EIGEN_DEVICE_FUNC static inline void split(const TCOMP& input, TSCALAR& v1,TSCALAR& v2, TSCALAR& v3, TSCALAR& v4)
    {
        v1 = input(0);
        v2 = input(1);
        v3 = input(2);
        v4 = input(3);
    }
    
    EIGEN_DEVICE_FUNC static inline TCOMP invertedValue(const TCOMP& v)
    {
        return (TSCALAR(1.0) / v.array()).matrix();
    }
};

template<typename TCOMP>
struct JoinSplitHelper<TCOMP,false,true,2> // for CUDA-2 types
{
    typedef typename vc::type_traits<TCOMP>::ChannelType TSCALAR;
    static constexpr int Dimension = vc::type_traits<TCOMP>::ChannelCount;
    
    EIGEN_DEVICE_FUNC static inline void join(const TSCALAR& v1, const TSCALAR& v2, TCOMP& out)
    {
        out.x = v1; out.y = v2;
    }
    
    EIGEN_DEVICE_FUNC static inline void split(const TCOMP& input, TSCALAR& v1,TSCALAR& v2)
    {
        v1 = input.x; v2 = input.y;
    }
    
    EIGEN_DEVICE_FUNC static inline TCOMP invertedValue(const TCOMP& v)
    {
        TCOMP ret;
        ret.x = TSCALAR(1.0) / v.x;
        ret.y = TSCALAR(1.0) / v.y;
        return ret;
    }
};
    
template<typename TCOMP>
struct JoinSplitHelper<TCOMP,false,true,3> // for CUDA-3 types
{
    typedef typename vc::type_traits<TCOMP>::ChannelType TSCALAR;
    static constexpr int Dimension = vc::type_traits<TCOMP>::ChannelCount;
    
    EIGEN_DEVICE_FUNC static inline void join(const TSCALAR& v1, const TSCALAR& v2, const TSCALAR& v3, TCOMP& out)
    {
        out.x = v1; out.y = v2; out.z = v3;
    }
    
    EIGEN_DEVICE_FUNC static inline void split(const TCOMP& input, TSCALAR& v1,TSCALAR& v2, TSCALAR& v3)
    {
        v1 = input.x; v2 = input.y; v3 = input.z;
    }
    
    EIGEN_DEVICE_FUNC static inline TCOMP invertedValue(const TCOMP& v)
    {
        TCOMP ret;
        ret.x = TSCALAR(1.0) / v.x;
        ret.y = TSCALAR(1.0) / v.y;
        ret.z = TSCALAR(1.0) / v.z;
        return ret;
    }
};

template<typename TCOMP>
struct JoinSplitHelper<TCOMP,false,true,4> // for CUDA-4 types
{
    typedef typename vc::type_traits<TCOMP>::ChannelType TSCALAR;
    static constexpr int Dimension = vc::type_traits<TCOMP>::ChannelCount;
    
    EIGEN_DEVICE_FUNC static inline void join(const TSCALAR& v1, const TSCALAR& v2, const TSCALAR& v3, const TSCALAR& v4, TCOMP& out)
    {
        out.x = v1; out.y = v2; out.z = v3; out.w = v4;
    }

    EIGEN_DEVICE_FUNC static inline void split(const TCOMP& input, TSCALAR& v1,TSCALAR& v2, TSCALAR& v3, TSCALAR& v4)
    {
        v1 = input.x; v2 = input.y; v3 = input.z; v4 = input.w;
    }
    
    EIGEN_DEVICE_FUNC static inline TCOMP invertedValue(const TCOMP& v)
    {
        TCOMP ret;
        ret.x = TSCALAR(1.0) / v.x;
        ret.y = TSCALAR(1.0) / v.y;
        ret.z = TSCALAR(1.0) / v.z;
        ret.w = TSCALAR(1.0) / v.w;
        return ret;
    }
};

// undefined for other types
template<typename TCOMP, int Whatever>
struct JoinSplitHelper<TCOMP,false,false,Whatever> 
{ 
    typedef typename vc::type_traits<TCOMP>::ChannelType TSCALAR;
    
    EIGEN_DEVICE_FUNC static inline TCOMP invertedValue(const TCOMP& v)
    {
        return TCOMP(1.0) / v;
    }
};
    
}

#endif // VISIONCORE_JOIN_SPLIT_HELPERS_HPP
