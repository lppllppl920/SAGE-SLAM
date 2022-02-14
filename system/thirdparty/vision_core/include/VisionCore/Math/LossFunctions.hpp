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
 * Loss functions and M-estimators.
 * ****************************************************************************
 */

#ifndef VISIONCORE_MATH_LOSS_FUNCTIONS_HPP
#define VISIONCORE_MATH_LOSS_FUNCTIONS_HPP

#include <VisionCore/Platform.hpp>

namespace vc
{
    
namespace math
{
    
template<typename T>
EIGEN_DEVICE_FUNC static inline T lossHuber(T x, T delta)
{
    using Eigen::numext::abs;
    
    const T aa = abs(x);
    
    if(aa <= delta)
    {
        return T(0.5) * x * x;
    }
    else
    {
        return delta * ( aa - delta * T(0.5));
    }
}

template<typename T>
EIGEN_DEVICE_FUNC static inline T lossCauchy(T x, T c)
{
    const T term = x/c;
    return (c*c*T(0.5)) * log(T(1.0) + term * term);
}

template<typename T>
EIGEN_DEVICE_FUNC static inline T lossTurkey(T x, T c)
{
    using Eigen::numext::abs;
  
    const T aa = abs(x);
    
    if(aa <= c)
    {
        const T term = T(1.0) - ((x / c) * (x / c));
        return ((c*c)/T(6.0)) * (T(1.0) - term * term * term);
    }
    else
    {
        return (c*c) / T(6.0);
    }
}

template<typename T>
EIGEN_DEVICE_FUNC static inline T lossGermanMcClure(T x)
{
    return (x*x / T(2.0)) / (T(1.0) + x*x);
}

template<typename T>
EIGEN_DEVICE_FUNC static inline T lossWelsch(T x, T c)
{
    using Eigen::numext::exp;
    const T term = x/c;
    return (c*c / T(2.0)) * ( T(1.0) - exp(-(term * term)) );
}

template<typename T>
EIGEN_DEVICE_FUNC static inline T lossL1(T x)
{
    using Eigen::numext::abs;
    return abs(x);
}

template<typename T>
EIGEN_DEVICE_FUNC static inline T lossL2(T x)
{
    return x*x;
}
    
}

}

#endif // VISIONCORE_MATH_LOSS_FUNCTIONS_HPP
