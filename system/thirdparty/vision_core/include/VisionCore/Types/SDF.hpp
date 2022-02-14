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
 * Sign-Distance Cell Element.
 * ****************************************************************************
 */

#ifndef VISIONCORE_TYPES_SDF_HPP
#define VISIONCORE_TYPES_SDF_HPP

#include <VisionCore/Platform.hpp>

namespace vc
{

namespace types
{
    
template<typename T = float>    
struct VISIONCORE_ALIGN(8) SDF
{
    EIGEN_DEVICE_FUNC inline SDF() {}
    
    EIGEN_DEVICE_FUNC inline SDF(T v) : Value(v), Weight(1) {}
    
    EIGEN_DEVICE_FUNC inline SDF(T v, T w) : Value(v), Weight(w) {}
    
    EIGEN_DEVICE_FUNC inline operator T() const 
    {
        return Value;
    }
    
    EIGEN_DEVICE_FUNC inline void clamp(T minval, T maxval) 
    {
        Value = ::clamp(Value, minval, maxval);
    }
    
    EIGEN_DEVICE_FUNC inline void limitWeight(T max_weight) 
    {
        Weight = fminf(Weight, max_weight);
    }
    
    EIGEN_DEVICE_FUNC inline void operator+=(const SDF<T>& rhs)
    {
        if(rhs.Weight > T(0.0)) 
        {
            Value = (Weight * Value + rhs.Weight * rhs.Value);
            Weight += rhs.Weight;
            Value /= Weight;
        }
    }
    
    T Value;
    T Weight;
};

template<typename T = float>    
EIGEN_DEVICE_FUNC inline SDF<T> operator+(const SDF<T>& lhs, const SDF<T>& rhs)
{
    SDF<T> res = lhs;
    res += rhs;
    return res;
}
    
}

}

#endif // VISIONCORE_TYPES_SDF_HPP
