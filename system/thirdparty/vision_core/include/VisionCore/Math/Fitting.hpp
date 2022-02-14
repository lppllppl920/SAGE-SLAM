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
 * Fitting things to data.
 * ****************************************************************************
 */

#ifndef VISIONCORE_MATH_FITTING_HPP
#define VISIONCORE_MATH_FITTING_HPP

#include <VisionCore/Platform.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <Eigen/LU>

#include <VisionCore/Math/Statistics.hpp>
#include <VisionCore/Types/Hypersphere.hpp>

namespace vc
{
    
namespace math
{

/**
 * Fitting plane from 3D points.
 */
template<typename T>
class PlaneFitting
{
public:
    typedef MultivariateStats<Eigen::Matrix<T,3,1>> StatsT;
    typedef typename StatsT::VectorType VectorT;
    typedef typename StatsT::CovarianceType CovarianceMatrixT;
    typedef Eigen::Hyperplane<T,3> PlaneT;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    PlaneFitting() { reset(); }
    
    inline void reset() { stats.reset(); }
    inline std::size_t count() const { return stats.count(); }
    
    inline void operator()(const VectorT& x)
    {
        stats(x);
    }

    bool getPlane(PlaneT& p, T& curvature) const
    {
        return getPlane(stats, p, curvature);
    }

private:
    bool getPlane(const StatsT& ss, PlaneT& p, T& curvature) const
    {
        if(ss.count() < 3)
        {
            return false;
        }
        
        const VectorT mean_point = ss.mean();
        const CovarianceMatrixT cm = ss.covariance();
        
        getPlane(cm, mean_point, p, curvature);
        
        return true;
    }
    
    void getPlane(const CovarianceMatrixT& cm, const VectorT& mean_point, PlaneT& p, T& curvature) const
    {
        using Eigen::numext::abs;

        Eigen::SelfAdjointEigenSolver<CovarianceMatrixT> es(cm);
        
        const T eigen_value = es.eigenvalues()(0);
        const VectorT eigen_vector = es.eigenvectors().col(0); 
        
        p.normal() = eigen_vector;
        
        T eig_sum = cm.coeff(0) + cm.coeff(4) + cm.coeff(8);
        if(eig_sum != T(0.0))
        {
            curvature = abs(eigen_value / eig_sum);
        }
        else
        {
            curvature = T(0.0);
        }
        
        // Hessian form (D = nc . p_plane (centroid here) + p)
        p.offset() = T(1.0) * eigen_vector.dot(mean_point); 
    }
    
    math::MultivariateStats<VectorT> stats;
};

/**
 * Circle from 3 points.
 */
template<typename T>
EIGEN_DEVICE_FUNC static inline types::CircleT<T> circleFrom3Points(const Eigen::Matrix<T,2,1>& p1, const Eigen::Matrix<T,2,1>& p2, const Eigen::Matrix<T,2,1>& p3)
{
    T ma = (p2(1) - p1(1))/(p2(0) - p1(0));
    T mb = (p3(1) - p2(1))/(p3(0) - p2(0));
    
    T cx = (ma*mb * (p1(1) - p3(1)) + mb * (p1(0) + p2(0)) - ma * (p2(0) + p3(0)))/(2 * (mb - ma));
    T cy = (mb*p3(1)+(mb-ma)*p2(1)-ma*p1(1)+p3(0)-p1(0))/(2*(mb-ma));
    
    types::CircleT<T> ret;
    
    ret.coeff() << cx , cy;
    ret.radius() = ret(p1);
    
    return ret;
}

// ****************** AFFINE FROM 3D POINT CORRESPONDANCES ***********

/**
 * Establish transform from 3D corresponding points.
 */
template<typename T>
class TransformationFrom3DPointPairs
{
public:
    typedef Eigen::Matrix<T,3,1> VectorT;
    typedef Eigen::Matrix<T,3,3> MatrixT;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    inline void reset() 
    { 
        sample_count = 0;
        accumulated_weight_ = 0.0;
        mean1.fill(0);
        mean2.fill(0);
        covariance.fill(0);
    }
    
    inline std::size_t count() const { return sample_count; }
    
    inline void operator()(const VectorT& p1, const VectorT& p2, T weight = T(1.0))
    {
        if(weight == T(0.0))
            return;
   
        ++sample_count;
        accumulated_weight_ += weight;
        T alpha = weight/accumulated_weight_;
        
        VectorT diff1 = p1 - mean1, diff2 = p2 - mean2;
        covariance = (T(1.0) - alpha)*(covariance + alpha * (diff2 * diff1.transpose()));
        
        mean1 += alpha * (diff1);
        mean2 += alpha * (diff2);
    }
    
    inline Eigen::Affine3f getTransformation ()
    {
        Eigen::JacobiSVD<MatrixT> svd (covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const MatrixT& u = svd.matrixU(),& v = svd.matrixV();
        
        MatrixT s;
        s.setIdentity();
        if(u.determinant()*v.determinant() < T(0.0f))
        {
            s(2,2) = T(-1.0);
        }
        
        MatrixT r = u * s * v.transpose();
        VectorT t = mean2 - r*mean1;
        
        Eigen::Transform<T,3,Eigen::Affine> ret(Eigen::Matrix<T,4,4>::Zero());
        ret.block<3,3>(0,0) = r;
        ret.block<3,1>(0,3) = t;
        
        return ret;
    }
private:
    std::size_t sample_count;
    T accumulated_weight_;
    VectorT mean1, mean2;
    MatrixT covariance;
};
   
}

}

#endif // VISIONCORE_MATH_FITTING_HPP
