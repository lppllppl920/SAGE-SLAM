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
 * Host/Device Hamming Distance calculations.
 * ****************************************************************************
 */

#ifndef VISIONCORE_MATH_HAMMINGDISTANCE_HPP
#define VISIONCORE_MATH_HAMMINGDISTANCE_HPP

#include <VisionCore/Platform.hpp>

namespace vc
{

namespace math
{
    
namespace internal
{
    template<typename T>
    struct PopCount
    {
#ifdef VISIONCORE_CUDA_KERNEL_SPACE
        EIGEN_DEVICE_FUNC static inline unsigned run(const T& arg)
        {
            return __popc(arg);
        }
#endif // VISIONCORE_CUDA_KERNEL_SPACE
    };
    
#ifndef VISIONCORE_CUDA_KERNEL_SPACE
    template<> struct PopCount<unsigned int>
    {
        static inline unsigned run(const unsigned int& arg) { return __builtin_popcount(arg); }
    };
    
    template<> struct PopCount<unsigned long>
    {
        static inline unsigned run(const unsigned long& arg) { return __builtin_popcountl(arg); }
    };
    
    template<> struct PopCount<unsigned long long>
    {
        static inline unsigned run(const unsigned long long& arg) { return __builtin_popcountll(arg); }
    };
#endif // VISIONCORE_CUDA_KERNEL_SPACE    
}

template<typename T>
EIGEN_DEVICE_FUNC static inline unsigned popcnt(const T& arg) { return internal::PopCount<T>::run(arg); }

EIGEN_DEVICE_FUNC static inline unsigned hammingDistance(unsigned int p, unsigned int q)
{
    return popcnt(p^q);
}

EIGEN_DEVICE_FUNC static inline unsigned hammingDistance(const uint2 p, const uint2 q)
{
    return popcnt(p.x^q.x) + popcnt(p.y^q.y);
}

EIGEN_DEVICE_FUNC static inline unsigned hammingDistance(const uint3 p, const uint3 q)
{
    return popcnt(p.x^q.x) + popcnt(p.y^q.y) + popcnt(p.z^q.z);
}

EIGEN_DEVICE_FUNC static inline unsigned hammingDistance(const uint4 p, const uint4 q)
{
    return popcnt(p.x^q.x) + popcnt(p.y^q.y) + popcnt(p.z^q.z) + popcnt(p.w^q.w);
}

EIGEN_DEVICE_FUNC static inline unsigned hammingDistance(unsigned long p, unsigned long q)
{
    return popcnt(p^q);
}

EIGEN_DEVICE_FUNC static inline unsigned hammingDistance(const ulong2 p, const ulong2 q)
{
    return popcnt(p.x^q.x) + popcnt(p.y^q.y);
}

EIGEN_DEVICE_FUNC static inline unsigned hammingDistance(const ulong3 p, const ulong3 q)
{
    return popcnt(p.x^q.x) + popcnt(p.y^q.y) + popcnt(p.z^q.z);
}

EIGEN_DEVICE_FUNC static inline unsigned hammingDistance(const ulong4 p, const ulong4 q)
{
    return popcnt(p.x^q.x) + popcnt(p.y^q.y) + popcnt(p.z^q.z) + popcnt(p.w^q.w);
}    

EIGEN_DEVICE_FUNC static inline unsigned hammingDistance(unsigned long long p, unsigned long long q)
{
    return popcnt(p^q);
}

EIGEN_DEVICE_FUNC static inline unsigned hammingDistance(const ulonglong2 p, const ulonglong2 q)
{
    return popcnt(p.x^q.x) + popcnt(p.y^q.y);
}

EIGEN_DEVICE_FUNC static inline unsigned hammingDistance(const ulonglong3 p, const ulonglong3 q)
{
    return popcnt(p.x^q.x) + popcnt(p.y^q.y) + popcnt(p.z^q.z);
}

EIGEN_DEVICE_FUNC static inline unsigned hammingDistance(const ulonglong4 p, const ulonglong4 q)
{
    return popcnt(p.x^q.x) + popcnt(p.y^q.y) + popcnt(p.z^q.z) + popcnt(p.w^q.w);
}

}

}

#endif // VISIONCORE_MATH_HAMMINGDISTANCE_HPP
