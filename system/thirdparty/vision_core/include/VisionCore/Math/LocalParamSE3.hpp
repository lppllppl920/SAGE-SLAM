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
 * SE3 Local Parametrization for Ceres-Solver.
 * ****************************************************************************
 */


#ifndef VISIONCORE_MATH_LOCAL_PARAM_SE3_HPP
#define VISIONCORE_MATH_LOCAL_PARAM_SE3_HPP

#include <ceres/local_parameterization.h>
#include <ceres/autodiff_local_parameterization.h>
#include <sophus/se3.hpp>

/// @note: From Steven Lovegrove
/// @todo: update for new Sophus

namespace vc
{
    
namespace math
{

class LocalParameterizationSE3 : public ceres::LocalParameterization 
{
public:
    virtual ~LocalParameterizationSE3() {}
    
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const
    {
        const Eigen::Map<const Sophus::SE3d> T(x);
        const Eigen::Map<const Eigen::Matrix<double,6,1> > dx(delta);
        Eigen::Map<Sophus::SE3d> Tdx(x_plus_delta);
        Tdx = T * Sophus::SE3d::exp(dx);
        return true;
    }
    
    virtual bool ComputeJacobian(const double* x, double* jacobian) const
    {
        // Largely zeroes.
        memset(jacobian,0, sizeof(double)*7*6);
        
        // Elements of quaternion
        const double q1	= x[0];
        const double q2	= x[1];
        const double q3	= x[2];
        const double q0	= x[3];
        
        // Common terms
        const double half_q0 = 0.5*q0;
        const double half_q1 = 0.5*q1;
        const double half_q2 = 0.5*q2;
        const double half_q3 = 0.5*q3;
        
        const double q1_sq = q1*q1;
        const double q2_sq = q2*q2;
        const double q3_sq = q3*q3;
        
        // d output_quaternion / d update
        jacobian[3]  =  half_q0;
        jacobian[4]  = -half_q3;
        jacobian[5]  =  half_q2;
        
        jacobian[9]  =  half_q3;
        jacobian[10] =  half_q0;
        jacobian[11] = -half_q1;
        
        jacobian[15] = -half_q2;
        jacobian[16] =  half_q1;
        jacobian[17] =  half_q0;
        
        jacobian[21] = -half_q1;
        jacobian[22] = -half_q2;
        jacobian[23] = -half_q3;    
        
        // d output_translation / d update
        jacobian[24]  = 1.0 - 2.0 * (q2_sq + q3_sq);
        jacobian[25]  = 2.0 * (q1*q2 - q0*q3);
        jacobian[26]  = 2.0 * (q1*q3 + q0*q2);
        
        jacobian[30]  = 2.0 * (q1*q2 + q0*q3);
        jacobian[31]  = 1.0 - 2.0 * (q1_sq + q3_sq);
        jacobian[32]  = 2.0 * (q2*q3 - q0*q1);
        
        jacobian[36] = 2.0 * (q1*q3 - q0*q2);
        jacobian[37] = 2.0 * (q2*q3 + q0*q1);
        jacobian[38] = 1.0 - 2.0 * (q1_sq + q2_sq);
        
        return true;
    }
    
    virtual int GlobalSize() const { return 7; }
    virtual int LocalSize() const { return 6; }
};

struct AutoDiffLocalParameterizationSE3 
{
    static constexpr int GlobalSize = 7; 
    static constexpr int LocalSize = 6; 
    
    template<typename T>
    EIGEN_DEVICE_FUNC bool operator()(const T* x, const T* delta, T* x_plus_delta) const 
    {
        const Eigen::Map<const Sophus::SE3<T>> pose(x);
        const Eigen::Map<const Eigen::Matrix<T,LocalSize,1> > dx(delta);
        Eigen::Map<Sophus::SE3<T>> Tdx(x_plus_delta);
        Tdx = pose * Sophus::SE3<T>::exp(dx);
        return true;
    }
};

typedef ceres::AutoDiffLocalParameterization<AutoDiffLocalParameterizationSE3, 
                                             AutoDiffLocalParameterizationSE3::GlobalSize, 
                                             AutoDiffLocalParameterizationSE3::LocalSize> ADLocalParameterizationSE3T;

}

}

#endif // VISIONCORE_MATH_LOCAL_PARAM_SE3_HPP
