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
 * Statistical Analysis. 
 * ****************************************************************************
 */

#ifndef VISIONCORE_MATH_STATISTICS_HPP
#define VISIONCORE_MATH_STATISTICS_HPP

#include <VisionCore/Platform.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <VisionCore/Buffers/Buffer1D.hpp>
#include <VisionCore/Buffers/Buffer2D.hpp>
#include <VisionCore/Types/Polynomial.hpp>
#include <VisionCore/Types/Gaussian.hpp>
#include <VisionCore/Math/PolarSpherical.hpp>

/**
 * @note:
 * http://rebcabin.github.io/blog/2013/01/22/covariance-matrices/
 */

namespace vc
{

namespace math
{
    
/**
 * Running mean, variation, std. deviation, count and min,max.
 */
template<typename T>
class RunningStats
{
public:    
    typedef T Scalar;
    static constexpr int Rows = 1;
    static constexpr int Cols = 1;
    static constexpr int Dimension = Rows * Cols;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    EIGEN_DEVICE_FUNC inline RunningStats()
    {
        reset();
    }
    
    EIGEN_DEVICE_FUNC inline void reset()
    {
        cnt = 0;
        minV = std::numeric_limits<T>::max();
        maxV = -std::numeric_limits<T>::max();
        sumX = sumXX = T(0.0);
    }
    
    EIGEN_DEVICE_FUNC inline void operator()(const T& x)
    {
        using Eigen::numext::mini;
        using Eigen::numext::maxi;
      
        // process
        cnt++;
        
        sumX += x;
        sumXX += (x * x);
        
        maxV = max(maxV, x); // NOTE what about weight?
        minV = min(minV, x);
    }
    
    EIGEN_DEVICE_FUNC inline std::size_t count() const 
    { 
        return cnt; 
    }
    
    EIGEN_DEVICE_FUNC inline T mean() const
    {
        if(cnt > 0)
        {
            return sumX / T(cnt);
        }
        else
        {
            return T(0.0);
        }
    }
    
    EIGEN_DEVICE_FUNC inline T variance() const
    {        
        if(cnt > 0)
        {
            const T bessel_correction = T(cnt) / T(cnt - 1);
            const T cm = mean();
            const T term1 = sumXX / T(cnt);
            
            return bessel_correction * (term1 - (cm*cm));
        }
        else
        {
            return T(0.0);
        }
    }
        
    EIGEN_DEVICE_FUNC inline T stddev() const 
    {
        using Eigen::numext::sqrt;
        return sqrt(variance());
    }
    
    EIGEN_DEVICE_FUNC inline T running_sum() const { return sumX; }
    
    EIGEN_DEVICE_FUNC inline T min_value() const { return minV; }
    EIGEN_DEVICE_FUNC inline T max_value() const { return maxV; }
    
    EIGEN_DEVICE_FUNC inline types::Gaussian<T> gaussian() const 
    { 
        return types::Gaussian<T>(mean(), variance()); 
    }
private:
    std::size_t cnt;
    T sumX, sumXX;
    T minV, maxV;
};

/**
 * Specialization for Eigen.
 */
template<typename _Scalar, int _Rows, int _Cols, int _Options>
class RunningStats<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options>>
{
public:    
    typedef Eigen::Matrix<_Scalar, _Rows, _Cols, _Options> VectorType;
    typedef _Scalar Scalar;
    static constexpr int Rows = _Rows;
    static constexpr int Cols = _Cols;
    static constexpr int Dimension = Rows * Cols;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    EIGEN_DEVICE_FUNC inline RunningStats()
    {
        reset();
    }
    
    EIGEN_DEVICE_FUNC inline void reset()
    {
        cnt = 0;
        minV = VectorType::Constant(std::numeric_limits<Scalar>::max());
        maxV = VectorType::Constant(-std::numeric_limits<Scalar>::max());
        sumX = VectorType::Zero();
        sumXX = VectorType::Zero();
    }
    
    EIGEN_DEVICE_FUNC inline void operator()(const VectorType& x)
    {
        // process
        cnt++;
        
        sumX += x;
        sumXX += VectorType(x.array() * x.array());
        
        maxV = maxV.array().cwiseMax(x.array());
        minV = minV.array().cwiseMin(x.array());
    }
    
    EIGEN_DEVICE_FUNC inline std::size_t count() const 
    { 
        return cnt; 
    }
    
    EIGEN_DEVICE_FUNC inline VectorType mean() const
    {
        if(cnt > 0)
        {
            return sumX / Scalar(cnt);
        }
        else
        {
            return VectorType::Zero();
        }
    }
    
    EIGEN_DEVICE_FUNC inline VectorType variance() const
    {
        if(cnt > 0)
        {
            const Scalar bessel_correction = Scalar(cnt) / Scalar(cnt - 1);
            const VectorType cm = mean();
            const VectorType term1 = sumXX / Scalar(cnt);
            
            return bessel_correction * (term1 - VectorType(cm.array() * cm.array()));
        }
        else
        {
            return VectorType::Zero();
        }
    }
    
    EIGEN_DEVICE_FUNC inline VectorType stddev() const
    {
        return sqrt(variance().array());
    }
    
    EIGEN_DEVICE_FUNC inline VectorType running_sum() const { return sumX; }
    
    EIGEN_DEVICE_FUNC inline VectorType min_value() const { return minV; }
    EIGEN_DEVICE_FUNC inline VectorType max_value() const { return maxV; }
    
    EIGEN_DEVICE_FUNC inline types::Gaussian<VectorType> gaussian() const 
    { 
        return types::Gaussian<VectorType>(mean(), variance()); 
    }
private:
    std::size_t cnt;
    VectorType sumX, sumXX, minV, maxV;
};

// ******************** RUNNING MULTIVARIATE *******************************

/**
 * Multivariate analysis.
 */
template<typename T>
class MultivariateStats;

template<typename _Scalar, int Rows, int Options>
class MultivariateStats<Eigen::Matrix<_Scalar, Rows, 1, Options>>
{
public:    
    typedef Eigen::Matrix<_Scalar, Rows, 1, Options> VectorType;
    typedef Eigen::Matrix<_Scalar, Rows, Rows, Options> CovarianceType;
    typedef _Scalar Scalar;
    static constexpr std::size_t Dimensions = Rows;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    EIGEN_DEVICE_FUNC inline MultivariateStats()
    {
        reset();
    }
    
    EIGEN_DEVICE_FUNC inline void reset()
    {
        cnt = 0;
        minV = VectorType::Constant(std::numeric_limits<Scalar>::max());
        maxV = VectorType::Constant(-std::numeric_limits<Scalar>::max());
        sumX = VectorType::Zero();
        sumXX = CovarianceType::Zero();
    }
    
    EIGEN_DEVICE_FUNC inline void operator()(const VectorType& x)
    {
        // process
        cnt++;
        
        sumX += x;
        
        sumXX += (x * x.transpose());
        
        maxV = maxV.array().cwiseMax(x.array());
        minV = minV.array().cwiseMin(x.array());
    }
    
    EIGEN_DEVICE_FUNC inline std::size_t count() const 
    { 
        return cnt; 
    }
    
    EIGEN_DEVICE_FUNC inline VectorType mean() const
    {
        if(cnt > 0)
        {
            return sumX / Scalar(cnt);
        }
        else
        {
            return VectorType::Zero();
        }
    }
    
    EIGEN_DEVICE_FUNC inline CovarianceType covariance() const
    {
        if(cnt > 0)
        {
            const Scalar bessel_correction = Scalar(cnt) / Scalar(cnt - 1);
            const VectorType m = mean();
            return bessel_correction * ( (sumXX / Scalar(cnt)) - (m * m.transpose()) );
        }
        else
        {
            return CovarianceType::Zero();
        }
    }
    
    EIGEN_DEVICE_FUNC inline VectorType running_sum() const { return sumX; }
    
    EIGEN_DEVICE_FUNC inline VectorType min_value() const { return minV; }
    EIGEN_DEVICE_FUNC inline VectorType max_value() const { return maxV; }
    
    EIGEN_DEVICE_FUNC inline types::MultivariateGaussian<Scalar,Rows> gaussian() const 
    { 
        return types::MultivariateGaussian<Scalar,Rows>(mean(), covariance()); 
    }
private:
    std::size_t cnt;
    VectorType sumX, minV, maxV;
    CovarianceType sumXX;
};

// ******************** ALLAN VARIANCE *******************************
// From: http://www.leapsecond.com/tools/adev_lib.c

template<typename InputIterator>
typename std::iterator_traits<InputIterator>::value_type allanDeviation(std::size_t tau, 
                                                                        InputIterator itbegin, 
                                                                        InputIterator itend, 
                                                                        bool isoverlapping = false, 
                                                                        std::size_t* terms = nullptr)
{
    typedef typename std::iterator_traits<InputIterator>::value_type T;
    using namespace std;
    
    std::size_t n = 0;
    T sum = T(0.0);
    const std::size_t stride = isoverlapping ? 1 : tau;
    
    for ( ; (itbegin + 2 * tau) < itend ; itbegin += stride) 
    {
        const T v = *(itbegin + 2 * tau) - T(2.0) * *(itbegin + tau) + *itbegin;
        sum += v * v;
        n += 1;
    }
    
    sum /= T(2.0);
    
    if(terms != nullptr) { *terms = n; }
    
    if (n < 3) { return T(0.0); }
    
    return sqrt(sum / n) / tau;
}

template<typename InputIterator>
typename std::iterator_traits<InputIterator>::value_type hadamardDeviation(std::size_t tau, 
                                                                           InputIterator itbegin, 
                                                                           InputIterator itend, 
                                                                           bool isoverlapping = false, 
                                                                           std::size_t* terms = nullptr)
{
    typedef typename std::iterator_traits<InputIterator>::value_type T;
    using namespace std;
    
    std::size_t n = 0;
    T sum = T(0.0);
    const std::size_t stride = isoverlapping ? 1 : tau;
    
    for ( ; (itbegin + 3 * tau) < itend; itbegin += stride) 
    {
        const T v = *(itbegin + 3 * tau) - T(3.0) * *(itbegin + 2 * tau) + T(3.0) * *(itbegin + tau) - *itbegin;
        sum += v * v;
        n += 1;
    }
    
    sum /= T(6.0);
    
    if(terms != nullptr) { *terms = n; }
    
    if (n < 3) { return T(0.0); }
    
    return sqrt(sum / n) / tau;
}

template<typename InputIterator>
typename std::iterator_traits<InputIterator>::value_type modifiedAllanDeviation(std::size_t tau, 
                                                                                InputIterator itbegin, 
                                                                                InputIterator itend, 
                                                                                std::size_t* terms = nullptr)
{
    typedef typename std::iterator_traits<InputIterator>::value_type T;
    using namespace std;
    
    std::size_t n = 0;
    T sum = T(0.0), v = T(0.0);
    
    InputIterator orgbegin = itbegin;
    
    for ( ; (itbegin + 2 * tau) < itend && (std::size_t)std::distance(orgbegin, itbegin) < tau; ++itbegin) 
    {
        v += *(itbegin + 2 * tau) - T(2.0) * *(itbegin + tau) + *itbegin;
    }
    
    sum += v * v;
    n += 1;
    
    itbegin = orgbegin;
    
    for ( ; (itbegin + 3 * tau) < itend; ++itbegin) 
    {
        v += *(itbegin + 3 * tau) - T(3.0) * *(itbegin + 2 * tau) + T(3.0) * *(itbegin + tau) - *itbegin;
        sum += v * v;
        n += 1;
    }
    
    sum /= T(2.0) * T(tau) * T(tau);
    
    if(terms != nullptr) { *terms = n; }
    if(n < 3) { return T(0.0); }
    
    return sqrt(sum / n) / tau;
}

template<typename InputIterator>
typename std::iterator_traits<InputIterator>::value_type timeDeviation(std::size_t tau, 
                                                                       InputIterator itbegin, 
                                                                       InputIterator itend, 
                                                                       std::size_t* terms = nullptr)
{
    typedef typename std::iterator_traits<InputIterator>::value_type T;
    using namespace std;
    return T(tau) * modifiedAllanDeviation(tau, itbegin, itend, terms) / sqrt(3.0);
}

// ******************** PEARSON PRODUCT MOMENT *******************************

/**
 * Calculate r_xy from data.
 */
template<typename T>
class PearsonProductMoment;

template<typename Derived, typename BaseType>
class PearsonProductMomentBase
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    EIGEN_DEVICE_FUNC void reset()
    {
        stats_x.reset();
        stats_x2.reset();
        stats_y.reset();
        stats_y2.reset();
        stats_xy.reset();
    }
    
    EIGEN_DEVICE_FUNC const RunningStats<BaseType>& statsX() const { return stats_x; }
    EIGEN_DEVICE_FUNC const RunningStats<BaseType>& statsY() const { return stats_y; }
    EIGEN_DEVICE_FUNC const RunningStats<BaseType>& statsX2() const { return stats_x2; }
    EIGEN_DEVICE_FUNC const RunningStats<BaseType>& statsY2() const { return stats_y2; }
    EIGEN_DEVICE_FUNC const RunningStats<BaseType>& statsXY() const { return stats_xy; }
protected:
    EIGEN_DEVICE_FUNC RunningStats<BaseType>& statsX() { return stats_x; }
    EIGEN_DEVICE_FUNC RunningStats<BaseType>& statsY() { return stats_y; }
    EIGEN_DEVICE_FUNC RunningStats<BaseType>& statsX2() { return stats_x2; }
    EIGEN_DEVICE_FUNC RunningStats<BaseType>& statsY2() { return stats_y2; }
    EIGEN_DEVICE_FUNC RunningStats<BaseType>& statsXY() { return stats_xy; }
    
private:
    RunningStats<BaseType> stats_x;
    RunningStats<BaseType> stats_y;
    RunningStats<BaseType> stats_xy;
    RunningStats<BaseType> stats_x2;
    RunningStats<BaseType> stats_y2;
};

template<typename T>
class PearsonProductMoment : public PearsonProductMomentBase<PearsonProductMoment<T>, T>
{
    typedef PearsonProductMomentBase<PearsonProductMoment<T>, T> Base;
public:    
    typedef T BaseType;
    typedef types::Polynomial<BaseType,1> PolynomialT;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    EIGEN_DEVICE_FUNC inline PearsonProductMoment()
    {
        Base::reset();
    }
        
    EIGEN_DEVICE_FUNC void operator()(const T& x, const T& y)
    {
        Base::statsX()(x);
        Base::statsX2()(x*x);
        Base::statsY()(y);
        Base::statsY2()(y*y);
        Base::statsXY()(x*y);
    }
    
    EIGEN_DEVICE_FUNC BaseType result() const
    {
        using Eigen::numext::sqrt;
        return (Base::statsXY().mean() - (Base::statsX().mean() * Base::statsY().mean())) / 
                sqrt( ( Base::statsX2().mean() - ( Base::statsX().mean() * Base::statsX().mean() ) ) * 
                      ( Base::statsY2().mean() - ( Base::statsY().mean() * Base::statsY().mean() ) ) );
    }
    
    EIGEN_DEVICE_FUNC inline BaseType coefficientOfDetermination() const
    {
        BaseType r = result();
        return r*r;
    }
    
    EIGEN_DEVICE_FUNC inline PolynomialT lineEquation() const
    {
        PolynomialT ret;
        BaseType b = result() * ( Base::statsY().stddev() / Base::statsX().stddev() );
        BaseType a = Base::statsY().mean() - b * Base::statsX().mean();
        ret.coeff() << a , b;
        return ret;
    }
};

template<typename _Scalar, int _Rows, int _Cols>
class PearsonProductMoment<Eigen::Matrix<_Scalar, _Rows, _Cols> > : 
  public PearsonProductMomentBase<PearsonProductMoment<Eigen::Matrix<_Scalar, _Rows, _Cols> > , Eigen::Matrix<_Scalar, _Rows, _Cols>>
{
    typedef PearsonProductMomentBase<PearsonProductMoment<Eigen::Matrix<_Scalar, _Rows, _Cols> > , Eigen::Matrix<_Scalar, _Rows, _Cols>> Base;
public:
    typedef Eigen::Matrix<_Scalar, _Rows, _Cols> BaseType;    
    typedef types::Polynomial<BaseType,1> PolynomialT;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    EIGEN_DEVICE_FUNC inline PearsonProductMoment()
    {
        Base::reset();
    }
    
    EIGEN_DEVICE_FUNC void operator()(const BaseType& x, const BaseType& y)
    {
        Base::statsX()(x);
        Base::statsX2()(x.array() * x.array());
        Base::statsY()(y);
        Base::statsY2()(y.array() * y.array());
        Base::statsXY()(x.array() * y.array());
    }
    
    EIGEN_DEVICE_FUNC BaseType result() const
    {
        using Eigen::numext::sqrt;
        
        return (Base::statsXY().mean().array() - (Base::statsX().mean().array() * Base::statsY().mean().array())) / 
                sqrt( ( Base::statsX2().mean().array() - ( Base::statsX().mean().array() * Base::statsX().mean().array() ) ) * 
                      ( Base::statsY2().mean().array() - ( Base::statsY().mean().array() * Base::statsY().mean().array() ) ) );
    }
    
    EIGEN_DEVICE_FUNC inline BaseType coefficientOfDetermination() const
    {
        BaseType r = result();
        return r.array() * r.array();
    }
    
    EIGEN_DEVICE_FUNC inline PolynomialT lineEquation()
    {
        PolynomialT ret;
        BaseType b = result().array() * ( Base::statsY().stddev().array() / Base::statsX().stddev().array() );
        BaseType a = Base::statsY().mean().array() - b.array() * Base::statsX().mean().array();
        ret.coeff() << a , b;
        return ret;
    }
};
    
}

}

#endif // VISIONCORE_MATH_STATISTICS_HPP
