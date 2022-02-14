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
 * Polar/Spherical/Cartesian conversions.
 * ****************************************************************************
 */

#ifndef VISIONCORE_POLAR_SPHERICAL_HPP
#define VISIONCORE_POLAR_SPHERICAL_HPP

#include <VisionCore/Platform.hpp>

namespace vc
{
    
namespace math
{

template<typename T>
EIGEN_DEVICE_FUNC inline void cartesianToPolar(const T& x, const T& y, T& r, T& theta)
{
    using Eigen::numext::sqrt;
    using Eigen::numext::atan2;
    
    r = sqrt(x * x + y * y);
    theta = atan2(y, x);
}

template<typename T>
EIGEN_DEVICE_FUNC inline Eigen::Matrix<T,2,1> cartesianToPolar(const Eigen::Matrix<T,2,1>& cart)
{
    using Eigen::numext::sqrt;
    using Eigen::numext::atan2;
    
    Eigen::Matrix<T,2,1> polar;
    polar(0) = sqrt(cart(0) * cart(0) + cart(1) * cart(1));
    polar(1) = atan2(cart(1), cart(0));
    return polar;
}

template<typename T>
EIGEN_DEVICE_FUNC inline void cartesianToSpherical(const T& x, const T& y, const T& z, T& rho, T& psi, T& theta)
{
    using Eigen::numext::sqrt;
    using Eigen::numext::atan2;
    
    rho = sqrt(x * x + y * y + z * z);
    psi = atan2(sqrt(x*x + y*y), z);
    theta = atan2(y, x);
}

template<typename T>
EIGEN_DEVICE_FUNC inline Eigen::Matrix<T,3,1> cartesianToSpherical(const Eigen::Matrix<T,3,1>& cart)
{
    Eigen::Matrix<T,3,1> spherical;
    cartesianToSpherical<T>(cart(0),cart(1),cart(2),spherical(0),spherical(1),spherical(2));
    return spherical;
}

template<typename T>
EIGEN_DEVICE_FUNC inline void polarToCartesian(const T& r, const T& theta, T& x, T& y)
{
    using Eigen::numext::sin;
    using Eigen::numext::cos;
    
    x = r * cos(theta);
    y = r * sin(theta);
}

template<typename T>
EIGEN_DEVICE_FUNC inline Eigen::Matrix<T,2,1> polarToCartesian(const Eigen::Matrix<T,2,1>& polar)
{
    using Eigen::numext::sin;
    using Eigen::numext::cos;
    
    Eigen::Matrix<T,2,1> cart;
    cart(0) = polar(0) * cos(polar(1));
    cart(1) = polar(0) * sin(polar(1));
    return cart;
}

template<typename T>
EIGEN_DEVICE_FUNC inline void sphericalToCartesian(const T& rho, const T& psi, const T& theta, T& x, T& y, T& z)
{
    using Eigen::numext::sin;
    using Eigen::numext::cos;
    
    x = rho * sin(psi) * cos(theta);
    y = rho * sin(psi) * sin(theta);
    z = rho * cos(psi);
}

template<typename T>
EIGEN_DEVICE_FUNC inline Eigen::Matrix<T,3,1> sphericalToCartesian(const Eigen::Matrix<T,3,1>& spherical)
{
    Eigen::Matrix<T,3,1> cart;
    cart(0) = spherical(0) * sin(spherical(1)) * cos(spherical(2));
    cart(1) = spherical(0) * sin(spherical(1)) * sin(spherical(2));
    cart(2) = spherical(0) * cos(spherical(1));
    return cart;
}

}

}

#endif // VISIONCORE_MATH_POLAR_SPHERICAL_HPP
