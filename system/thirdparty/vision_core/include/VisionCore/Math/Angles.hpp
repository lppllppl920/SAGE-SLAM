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
 * Useful things to operate on angular quantities.
 * ****************************************************************************
 */

#ifndef VISIONCORE_MATH_ANGLES_HPP
#define VISIONCORE_MATH_ANGLES_HPP

#include <VisionCore/Platform.hpp>
#include <VisionCore/Math/Statistics.hpp>
#include <sophus/so3.hpp>

namespace vc
{
    
namespace math
{
 
template<typename Derived>
static EIGEN_DEVICE_FUNC inline Eigen::Matrix<typename Sophus::SO3Base<Derived>::Scalar, 3, 1> toEulerAngles(const Sophus::SO3Base<Derived>& m)
{
    using Eigen::numext::atan2;
    using Eigen::numext::sqrt; 
    
    Eigen::Matrix<typename Sophus::SO3Base<Derived>::Scalar, 3, 1> ret;
    
    const typename Sophus::SO3Base<Derived>::Transformation& rm = m.matrix();
    
    ret(0) = atan2(rm(2,1), rm(2,2)); // theta_x
    ret(1) = atan2(-rm(2,0), sqrt(rm(2,1) * rm(2,1) + rm(2,2) * rm(2,2))); // theta_y
    ret(2) = atan2(rm(1,0), rm(0,0)); // theta_z
    
    return ret;
}

template<typename T>
static EIGEN_DEVICE_FUNC inline Sophus::SO3<T> rotationBetweenTwoVectors(const Eigen::Matrix<T,3,1>& v1, 
                                                                         const Eigen::Matrix<T,3,1>& v2)
{
    const Eigen::Matrix<T,3,1> cab = v1.cross(v2);
    const typename Sophus::SO3<T>::Transformation hcab = Sophus::SO3<T>::hat(cab);
    return Sophus::SO3<T>(Sophus::SO3<T>::Transformation::Identity() + hcab + (hcab * hcab) * 
                          ( (1.0f - v1.dot(v2))/(cab.squaredNorm()) ));
}
    
/**
 * Constrain angle to 0..2pi.
 */    
template<typename T>
static EIGEN_DEVICE_FUNC inline T constrainAngle(T x)
{
    using Eigen::numext::fmod;
    
    x = fmod(x, T(2.0*M_PI) );
    
    if (x < T(0.0))
    {
        x += T(2.0*M_PI);
    }
    
    return x;
}

/**
 * Substract angles and keep the range in 0..2pi.
 */
template<typename T>
static EIGEN_DEVICE_FUNC inline T angleDiff(T a,T b)
{
    using Eigen::numext::fmod;
    
    T dif = fmod(b - a + T(M_PI), T(2.0*M_PI));
    
    if (dif < T(0.0))
    {
        dif += T(2.0*M_PI);
    }
    
    return dif - T(M_PI);
}

/**
 * Calculate mid angle and keep the range in 0..2pi.
 */
template<typename T>
static EIGEN_DEVICE_FUNC inline T bisectAngle(T a,T b)
{
    return constrainAngle<T>(a + angleDiff<T>(a,b) * T(0.5));
}

/**
 * Circular running mean, variation, std. deviation, count and min,max.
 * @note: Check variance/stddev for correctness.
 */
template<typename T>
class CircularMean
{
    typedef RunningStats<Eigen::Matrix<T,2,1>> PointStatsT;
public:    
    typedef T Scalar;
    typedef typename PointStatsT::VectorType PointT;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    EIGEN_DEVICE_FUNC inline CircularMean()
    {
        reset();
    }
    
    EIGEN_DEVICE_FUNC inline void reset()
    {
        stats.reset();
    }
    
    EIGEN_DEVICE_FUNC inline void operator()(const T& x)
    {
        PointT pt;
        math::polarToCartesian(T(1.0), x, pt(0), pt(1));
        stats(pt);
    }
    
    EIGEN_DEVICE_FUNC inline std::size_t count() const  { return stats.count(); }
    EIGEN_DEVICE_FUNC inline T mean() const { return getAngle(stats.mean()); }
    EIGEN_DEVICE_FUNC inline T variance() const { return getAngle(stats.variance()); }
    EIGEN_DEVICE_FUNC inline T stddev() const  { return getAngle(stats.stddev()); }
    EIGEN_DEVICE_FUNC inline T min_value() const { return getAngle(stats.min_value()); }
    EIGEN_DEVICE_FUNC inline T max_value() const { return getAngle(stats.max_value()); }
private:
    EIGEN_DEVICE_FUNC inline T getAngle(const PointT& pt) const 
    { 
        T r, theta;
        math::cartesianToPolar(pt(0),pt(1),r,theta);
        return theta;
    }
    PointStatsT stats;
};

/** 
 * This function computes the great-circle distance of two points on earth
 * using the Haversine formula, assuming spherical shape of the planet. A
 * well-known numerical issue with the formula is reduced accuracy in the
 * case of near antipodal points.
 * 
 * lat1, lon1: latitude and longitude of first point, radians.
 * lat2, lon2: latitude and longitude of second point, radians.
 * radius: radius of the earth in user-defined units, e.g. 6378.2 km.
 * returns: distance of the two points, in the same units as radius.
 * Reference: http://en.wikipedia.org/wiki/Great-circle_distance
 */
template<typename T>
static EIGEN_DEVICE_FUNC inline T greatCircleDistance(T lat1, T lon1, T lat2, T lon2, T radius)
{
    using Eigen::numext::cos;
    using Eigen::numext::sin;
    using Eigen::numext::asin;
    using Eigen::numext::mini;
    using Eigen::numext::sqrt;
    
    T dlat, dlon, c1, c2, d1, d2, a, c, t;
    
    c1 = cos(lat1);
    c2 = cos(lat2);
    dlat = lat2 - lat1;
    dlon = lon2 - lon1;
    d1 = sin(dlat/T(2.0));
    d2 = sin(dlon/T(2.0));
    t = d2 * d2 * c1 * c2;
    a = d1 * d1 + t;
    c = T(2.0) * asin(mini(T(1.0),sqrt(a)));
    
    return radius * c;
}

}
    
}

#endif // VISIONCORE_MATH_ANGLES_HPP
