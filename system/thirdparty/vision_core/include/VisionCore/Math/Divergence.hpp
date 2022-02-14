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
 * Divergence functions.
 * ****************************************************************************
 */

#ifndef VISIONCORE_MATH_DIVERGENCE_HPP
#define VISIONCORE_MATH_DIVERGENCE_HPP

#include <VisionCore/Platform.hpp>

#include <VisionCore/Buffers/Buffer2D.hpp>

namespace vc 
{
    
namespace math
{

template<typename T>
EIGEN_DEVICE_FUNC inline T projectUnitBall(T val, T maxrad = T(1.0))
{
    using Eigen::numext::maxi;
    using Eigen::numext::abs;
    return val / maxi(T(1.0), abs(val) / maxrad );
}

template<typename T, int Rows>
EIGEN_DEVICE_FUNC inline Eigen::Matrix<T,Rows,1> projectUnitBall(const Eigen::Matrix<T,Rows,1>& val, T maxrad = T(1.0))
{
    using Eigen::numext::maxi;
    using Eigen::numext::sqrt;
    return val / maxi(T(1.0), val.norm() / maxrad );
}

template<typename T, typename Target>
EIGEN_DEVICE_FUNC inline Eigen::Matrix<T,2,1> gradUFwd(const Buffer2DView<T, Target>& imgu, T u, size_t x, size_t y)
{
    Eigen::Matrix<T,2,1> du(T(0.0), T(0.0));
    if(x < imgu.width() - 1) du(0) = imgu(x+1,y) - u;
    if(y < imgu.height() - 1) du(1) = imgu(x,y+1) - u;
    return du;
}

template<typename T, typename Target>
EIGEN_DEVICE_FUNC inline T divA(const Buffer2DView<Eigen::Matrix<T,2,1>, Target>& A, int x, int y)
{
    const Eigen::Matrix<T,2,1>& p = A(x,y);
    T divA = p(0) + p(1);
    if(x > 0) divA -= A(x - 1, y)(0);
    if(y > 0) divA -= A(x, y - 1)(1);
    return divA;
}

template<typename T, typename Target>
EIGEN_DEVICE_FUNC inline Eigen::Matrix<T,4,1> TGVEpsilon(const Buffer2DView<Eigen::Matrix<T,2,1>, Target>& imgA, size_t x, size_t y)
{
    const Eigen::Matrix<T,2,1>& A = imgA(x,y);

    T dy_v0 = 0;
    T dx_v0 = 0;
    T dx_v1 = 0;
    T dy_v1 = 0;

    if(x < imgA.width() - 1) 
    {
        const Eigen::Matrix<T,2,1>& Apx = imgA(x + 1, y);
        dx_v0 = Apx(0) - A(0);
        dx_v1 = Apx(1) - A(1);
    }

    if(y < imgA.height() - 1) 
    {
        const Eigen::Matrix<T,2,1>& Apy = imgA(x, y + 1);
        dy_v0 = Apy(0) - A(0);
        dy_v1 = Apy(1) - A(1);
    }

    return Eigen::Matrix<T,4,1>(dx_v0, dy_v1, (dy_v0+dx_v1)/T(2.0), (dy_v0+dx_v1)/T(2.0) );
}

template<typename T, typename Target>
EIGEN_DEVICE_FUNC inline Eigen::Matrix<T,2,1> TGVDivA(const Buffer2DView<Eigen::Matrix<T,4,1>, Target>& A, int x, int y)
{
    const Eigen::Matrix<T,4,1>& p = A(x,y);
    Eigen::Matrix<T,2,1> divA(p(0) + p(2), p(2) + p(1));

    if(0 < x)
    {
        divA(0) -= A(x - 1, y)(0);
        divA(1) -= A(x - 1, y)(2);
    }

    if(0 < y)
    {
        divA(0) -= A(x, y - 1)(2);
        divA(1) -= A(x, y - 1)(1);
    }

    return divA;
}

}

}

#endif // VISIONCORE_MATH_DIVERGENCE_HPP
