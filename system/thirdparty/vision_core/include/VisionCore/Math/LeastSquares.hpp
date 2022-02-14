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
 * Simple Least-Squares Solver.
 * ****************************************************************************
 */

#ifndef VISIONCORE_MATH_LEAST_SQUARES_HPP
#define VISIONCORE_MATH_LEAST_SQUARES_HPP

#include <VisionCore/Platform.hpp>
#include <Eigen/Cholesky>

namespace vc
{
    
namespace math
{

// TODO eigenify
    
template<typename T, int dim>
class LSQ
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    typedef T Scalar;
    static constexpr int Dimension = dim;
    typedef Eigen::Matrix<Scalar,Dimension,Dimension> MatrixType;
    typedef Eigen::Matrix<Scalar,Dimension,1> VectorType;
    typedef Eigen::AutoDiffScalar<Eigen::Matrix<Scalar,Dimension,1>> JetType;
    
    EIGEN_DEVICE_FUNC inline LSQ(const std::size_t nc = 0)
    {
        reset();
        NumConstraints = nc;
    }
    
    EIGEN_DEVICE_FUNC inline void reset()
    {
        A.setZero();
        B.setZero();
        Error = Scalar(0.0);
        NumConstraints = 0;
    }
    
    EIGEN_DEVICE_FUNC inline void update(const VectorType& J, const Scalar& res, const Scalar& weight = Scalar(1.0))
    {
        A.noalias() += J * J.transpose() * weight;
        B.noalias() -= J * (res * weight);
        Error += res * res * weight;
        NumConstraints += 1;
    }
    
    EIGEN_DEVICE_FUNC inline void update(const JetType& j, const Scalar& weight = Scalar(1.0))
    {
        update(j.deriviatives(), j.value(), weight);
    }
    
    EIGEN_DEVICE_FUNC inline void finishAndDivide()
    {
        A /= (Scalar)NumConstraints;
        B /= (Scalar)NumConstraints;
        Error /= (Scalar)NumConstraints;
    }
    
    EIGEN_DEVICE_FUNC inline VectorType solve()
    {
        return A.ldlt().solve(-B);
    }
    
    MatrixType A;
    VectorType B;
    Scalar Error;
    std::size_t NumConstraints;
};

}

}

#endif // VISIONCORE_MATH_LEAST_SQUARES_HPP
