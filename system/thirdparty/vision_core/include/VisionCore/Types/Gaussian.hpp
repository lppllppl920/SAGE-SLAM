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
 * Gaussian distribution.
 * ****************************************************************************
 */
#ifndef VISIONCORE_TYPES_GAUSSIAN_HPP
#define VISIONCORE_TYPES_GAUSSIAN_HPP

#include <VisionCore/Platform.hpp>

#include <Eigen/Dense>

namespace vc
{
namespace types
{
    template<typename T> class Gaussian;
    template<typename _Scalar, int _Dimension = 1> class MultivariateGaussian;
}
}

namespace Eigen 
{
    namespace internal 
    {
        // Gaussian of scalars or Eigens
        template<typename T>
        struct traits< vc::types::Gaussian<T> > 
        {
            static constexpr int Rows = 1;
            static constexpr int Cols = 1;
            static constexpr int Dimension = Rows * Cols;
            typedef T Scalar;
            typedef Scalar MeanType;
        };
        
        template<typename _Scalar, int _Rows, int _Cols>
        struct traits<vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols> > > 
        {
            static constexpr int Rows = _Rows;
            static constexpr int Cols = _Cols;
            static constexpr int Dimension = Rows * Cols;
            typedef _Scalar Scalar;
            typedef Matrix<Scalar,_Rows,_Cols> MeanType;
        };
        
        template<typename _Scalar, int _Rows, int _Cols, int _Options>
        struct traits<Map<vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols> >, _Options> > : 
            traits<vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols> > >
        {
            static constexpr int Rows = _Rows;
            static constexpr int Cols = _Cols;
            static constexpr int Dimension = Rows * Cols;
            typedef _Scalar Scalar;
            typedef Map<Matrix<Scalar,_Rows,_Cols>, _Options> MeanType;
        };
        
        template<typename _Scalar, int _Rows, int _Cols, int _Options>
        struct traits<Map<const vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols> >, _Options> > : 
            traits<const vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols> > > 
        {
            static constexpr int Rows = _Rows;
            static constexpr int Cols = _Cols;
            static constexpr int Dimension = Rows * Cols;
            typedef _Scalar Scalar;
            typedef Map<const Matrix<Scalar,_Rows,_Cols>, _Options> MeanType;
        };
        
        // Multivariate
        template<typename _Scalar, int _Dimension>
        struct traits<vc::types::MultivariateGaussian<_Scalar,_Dimension> > 
        {
            static constexpr int Dimension = _Dimension;
            typedef _Scalar Scalar;
            typedef Matrix<Scalar,_Dimension,1> MeanType;
            typedef Matrix<Scalar,_Dimension,_Dimension> CovarianceType;
        };
        
        template<typename _Scalar, int _Dimension, int _Options>
        struct traits<Map<vc::types::MultivariateGaussian<_Scalar,_Dimension>, _Options> > : 
            traits<vc::types::MultivariateGaussian<_Scalar, _Dimension> > 
        {
            static constexpr int Dimension = _Dimension;
            typedef _Scalar Scalar;
            typedef Map<Matrix<Scalar,_Dimension,1>, _Options> MeanType;
            typedef Map<Matrix<Scalar,_Dimension,_Dimension>, _Options> CovarianceType;
        };
        
        template<typename _Scalar, int _Dimension, int _Options>
        struct traits<Map<const vc::types::MultivariateGaussian<_Scalar,_Dimension>, _Options> > : 
            traits<const vc::types::MultivariateGaussian<_Scalar, _Dimension> > 
        {
            static constexpr int Dimension = _Dimension;
            typedef _Scalar Scalar;
            typedef Map<const Matrix<Scalar,_Dimension,1>, _Options> MeanType;
            typedef Map<const Matrix<Scalar,_Dimension,_Dimension>, _Options> CovarianceType;
        };
        
    }
}

namespace vc
{
    
namespace types
{

/**
 * Type agnostic Gaussian operations.
 */
template<typename Derived>
class GaussianBase
{
public:
    static constexpr int Dimension = Eigen::internal::traits<Derived>::Dimension;
    static constexpr int Rows = Eigen::internal::traits<Derived>::Rows;
    static constexpr int Cols = Eigen::internal::traits<Derived>::Cols;
    typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Derived>::MeanType MeanType;
    typedef const typename Eigen::internal::traits<Derived>::MeanType ConstMeanType;
    
    EIGEN_DEVICE_FUNC MeanType& mean() 
    {
        return static_cast<Derived*>(this)->mean_nonconst();
    }
    
    EIGEN_DEVICE_FUNC const MeanType& mean() const
    {
        return static_cast<const Derived*>(this)->mean_const();
    }
    
    EIGEN_DEVICE_FUNC MeanType& variance() 
    {
        return static_cast<Derived*>(this)->variance_nonconst();
    }
    
    EIGEN_DEVICE_FUNC const MeanType& variance() const
    {
        return static_cast<const Derived*>(this)->variance_const();
    }
    
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC inline GaussianBase<Derived>& operator=(const GaussianBase<OtherDerived>& other)
    {
        mean() = other.mean();
        variance() = other.variance();
        return *this;
    }
    
#if 0
    EIGEN_DEVICE_FUNC inline GaussianBase<Derived> operator+(const GaussianBase<Derived>& other) const
    {
    // TBD
    }
    
    EIGEN_DEVICE_FUNC inline GaussianBase<Derived> operator-(const GaussianBase<Derived>& other) const
    {
    // TBD
    }
    
    template<typename OtherScalar>
    EIGEN_DEVICE_FUNC inline GaussianBase<Derived> operator*(const OtherScalar& other) const
    {
    // TBD
    }
    
    template<typename OtherScalar>
    EIGEN_DEVICE_FUNC inline GaussianBase<Derived> operator/(const OtherScalar& other) const
    {
    // TBD
    }
#endif
    
#ifdef VISIONCORE_ENABLE_CEREAL
    template<typename Archive>
    void load(Archive & archive, std::uint32_t const version)
    {
        archive(cereal::make_nvp("Mean", mean()));
        archive(cereal::make_nvp("Variance", variance()));
    }
    
    template<typename Archive>
    void save(Archive & archive, std::uint32_t const version) const
    {
        archive(cereal::make_nvp("Mean", mean()));
        archive(cereal::make_nvp("Variance", variance()));
    }
#endif // VISIONCORE_ENABLE_CEREAL 
};

/**
 * Eigen operations for Gaussian.
 */
template<typename Derived, int _Dimension>
class GaussianDispatchingBase : public GaussianBase<Derived>
{
    typedef GaussianBase<Derived> Base;
    friend class GaussianBase<Derived>;
public:
    static constexpr int Dimension = Eigen::internal::traits<Derived>::Dimension;
    static constexpr int Rows = Eigen::internal::traits<Derived>::Rows;
    static constexpr int Cols = Eigen::internal::traits<Derived>::Cols;
    typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Derived>::MeanType MeanType;
    typedef const typename Eigen::internal::traits<Derived>::MeanType ConstMeanType;
    
    using Base::operator=;
    
    template<typename _NewScalarType>
    EIGEN_DEVICE_FUNC inline Gaussian<Eigen::Matrix<_NewScalarType, Rows, Cols> >  cast() const 
    {
        return Gaussian<Eigen::Matrix<_NewScalarType, Rows, Cols>>(Base::mean().template cast<_NewScalarType>(), 
                                                                   Base::variance().template cast<_NewScalarType>());
    }
    
    EIGEN_DEVICE_FUNC inline MeanType evaluate(const MeanType& x)
    {
        using Eigen::numext::sqrt;
        using Eigen::numext::exp;
        using Eigen::numext::pow;
        
        const MeanType inv_sqrt_2pi_s2 = Scalar(1.0) / sqrt(Scalar(2.0 * M_PI) * Base::variance().array());
        const MeanType a = (x - Base::mean());
        return inv_sqrt_2pi_s2.array() * exp(-(pow(a.array(), 2))/(Scalar(2.0) * Base::variance().array()));
    }
    
    EIGEN_DEVICE_FUNC inline void setZero()
    {
        Base::mean() = MeanType::Zero();
        Base::variance() = MeanType::Zero();
    }
    
    EIGEN_DEVICE_FUNC MeanType stddev() const
    {
        return sqrt(Base::variance().array());
    }
};

/**
 * Scalar operations for Gaussian.
 */
template<typename Derived>
class GaussianDispatchingBase<Derived, 1> : public GaussianBase<Derived>
{
    typedef GaussianBase<Derived> Base;
    friend class GaussianBase<Derived>;
public:    
    static constexpr int Dimension = Eigen::internal::traits<Derived>::Dimension;
    static constexpr int Rows = Eigen::internal::traits<Derived>::Rows;
    static constexpr int Cols = Eigen::internal::traits<Derived>::Cols;
    typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Derived>::MeanType MeanType;
    typedef const typename Eigen::internal::traits<Derived>::MeanType ConstMeanType;
    
    using Base::operator=;
    
    template<typename NewScalarType>
    EIGEN_DEVICE_FUNC inline Gaussian<NewScalarType> cast() const 
    {
        return Gaussian<NewScalarType>((NewScalarType)Base::mean(), (NewScalarType)Base::variance());
    }
    
    EIGEN_DEVICE_FUNC inline MeanType evaluate(const MeanType& x)
    {
        using Eigen::numext::sqrt;
        using Eigen::numext::exp;
        
        const Scalar inv_sqrt_2pi_s2 = Scalar(1.0) / sqrt(Scalar(2.0 * M_PI) * Base::variance());
        const Scalar a = (x - Base::mean());
        return inv_sqrt_2pi_s2 * exp(-(a * a)/(Scalar(2.0) * Base::variance()));
    }
    
    EIGEN_DEVICE_FUNC inline void setZero()
    {
        Base::mean() = Scalar(0.0);
        Base::variance() = Scalar(0.0);
    }
    
    EIGEN_DEVICE_FUNC MeanType stddev() const
    {
        using Eigen::numext::sqrt;
        return sqrt(Base::variance());
    }
};

/**
 * Multivariate Gaussian Base.
 */
template<typename Derived>
class MultivariateGaussianBase
{
public:
    static constexpr int Dimension = Eigen::internal::traits<Derived>::Dimension;
    typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Derived>::MeanType MeanType;
    typedef const typename Eigen::internal::traits<Derived>::MeanType ConstMeanType;
    typedef typename Eigen::internal::traits<Derived>::CovarianceType CovarianceType;
    typedef const typename Eigen::internal::traits<Derived>::CovarianceType ConstCovarianceType;
    
    template<typename NewScalarType>
    EIGEN_DEVICE_FUNC inline MultivariateGaussian<NewScalarType,Dimension> cast() const 
    {
        return MultivariateGaussian<NewScalarType,Dimension>(mean().template cast<NewScalarType>(), 
                                                             covariance().template cast<NewScalarType>());
    }
        
    EIGEN_DEVICE_FUNC MeanType& mean() 
    {
        return static_cast<Derived*>(this)->mean_nonconst();
    }
    
    EIGEN_DEVICE_FUNC const MeanType& mean() const
    {
        return static_cast<Derived*>(this)->mean_const();
    }
        
    EIGEN_DEVICE_FUNC CovarianceType& covariance() 
    {
        return static_cast<Derived*>(this)->covariance_nonconst();
    }
    
    EIGEN_DEVICE_FUNC const CovarianceType& covariance() const
    {
        return static_cast<Derived*>(this)->covariance_const();
    }
      
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC inline MultivariateGaussianBase<Derived>& operator=(const MultivariateGaussianBase<OtherDerived>& other)
    {
        mean() = other.mean();
        covariance() = other.covariance();
        return *this;
    }
    
#if 0
EIGEN_DEVICE_FUNC inline MultivariateGaussianBase<Derived> operator+(const MultivariateGaussianBase<Derived>& other) const
    {
        // TBD
    }
    
    EIGEN_DEVICE_FUNC inline MultivariateGaussianBase<Derived> operator-(const MultivariateGaussianBase<Derived>& other) const
    {
        // TBD
    }
    
    template<typename OtherScalar>
    EIGEN_DEVICE_FUNC inline MultivariateGaussianBase<Derived> operator*(const OtherScalar& other) const
    {
        // TBD
    }
    
    template<typename OtherScalar>
    EIGEN_DEVICE_FUNC inline MultivariateGaussianBase<Derived> operator/(const OtherScalar& other) const
    {
        // TBD
    }
#endif
    
    EIGEN_DEVICE_FUNC inline Scalar evaluate(const MeanType& x)
    {
        using Eigen::numext::sqrt;
        using Eigen::numext::pow;
        using Eigen::numext::exp;
        
        auto term1 = x - mean();
        auto det = covariance().determinant();
        auto part1 = (Scalar(1.0) / ( sqrt(pow(Scalar(2.0 * M_PI), Scalar(Dimension)) * det) ) );
        auto part2 = exp(Scalar(-0.5) * term1.transpose() * covariance().inverse() * term1);
        
        return part1 * part2;
    }
    
    EIGEN_DEVICE_FUNC inline void setZero()
    {
        mean().setZero();
        covariance().setZero();
    }
    
#ifdef VISIONCORE_ENABLE_CEREAL
    template<typename Archive>
    void load(Archive & archive, std::uint32_t const version)
    {
        archive(cereal::make_nvp("Mean", mean()));
        archive(cereal::make_nvp("Covariance", covariance()));
    }
    
    template<typename Archive>
    void save(Archive & archive, std::uint32_t const version) const
    {
        archive(cereal::make_nvp("Mean", mean()));
        archive(cereal::make_nvp("Covariance", covariance()));
    }
#endif // VISIONCORE_ENABLE_CEREAL    
};

/**
 * Gaussian distribution.
 */
template<typename T>
class Gaussian : public GaussianDispatchingBase<Gaussian<T>, Eigen::internal::traits<Gaussian<T> >::Dimension>
{
    typedef GaussianDispatchingBase<Gaussian<T>, Eigen::internal::traits<Gaussian<T> >::Dimension> Base;
public:
    static constexpr int Dimension = Eigen::internal::traits<Gaussian>::Dimension;
    static constexpr int Rows = Eigen::internal::traits<Gaussian>::Rows;
    static constexpr int Cols = Eigen::internal::traits<Gaussian>::Cols;
    typedef typename Eigen::internal::traits<Gaussian>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Gaussian>::MeanType MeanType;
    typedef const typename Eigen::internal::traits<Gaussian>::MeanType ConstMeanType;
    
    friend class GaussianDispatchingBase<Gaussian<T>, Eigen::internal::traits<Gaussian<T> >::Dimension>;
    friend class GaussianBase<Gaussian<T> >;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    using Base::operator=;
#if 0
    using Base::operator*;
    using Base::operator+;
    using Base::operator-;
    using Base::operator/;
#endif
    
    EIGEN_DEVICE_FUNC inline Gaussian()
    {
    }
    
    template<typename OtherDerived> 
    EIGEN_DEVICE_FUNC inline Gaussian(const GaussianDispatchingBase<GaussianBase<OtherDerived>, 
                                      Eigen::internal::traits<Gaussian>::Dimension>& other)
        : mean_(other.mean()), variance_(other.variance())
    {
    }
    
    EIGEN_DEVICE_FUNC inline Gaussian(ConstMeanType& mean, ConstMeanType& variance)
        : mean_(mean), variance_(variance)
    {
    }
    
    EIGEN_DEVICE_FUNC inline ~Gaussian()
    {
    }
    
protected:
    EIGEN_DEVICE_FUNC inline ConstMeanType& mean_const() const { return mean_; }
    EIGEN_DEVICE_FUNC inline MeanType& mean_nonconst() { return mean_; }
    
    EIGEN_DEVICE_FUNC inline ConstMeanType& variance_const() const { return variance_; }
    EIGEN_DEVICE_FUNC inline MeanType& variance_nonconst() { return variance_; }
    
    MeanType mean_;
    MeanType variance_;
};

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const Gaussian<T>& p)
{
    os << "N(" << p.mean() << "," << p.stddev() <<  ")";
    return os;
}


/**
 * Multivariate Gaussian distribution.
 */
template<typename _Scalar, int _Dimension>
class MultivariateGaussian : public MultivariateGaussianBase<MultivariateGaussian<_Scalar,_Dimension>>
{
    typedef MultivariateGaussianBase<MultivariateGaussian<_Scalar,_Dimension>> Base;
public:
    static constexpr int Dimension = Eigen::internal::traits<MultivariateGaussian>::Dimension;
    typedef typename Eigen::internal::traits<MultivariateGaussian>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<MultivariateGaussian>::MeanType MeanType;
    typedef const typename Eigen::internal::traits<MultivariateGaussian>::MeanType ConstMeanType;
    typedef typename Eigen::internal::traits<MultivariateGaussian>::CovarianceType CovarianceType;
    typedef const typename Eigen::internal::traits<MultivariateGaussian>::CovarianceType ConstCovarianceType;
    
    friend class vc::types::MultivariateGaussianBase<MultivariateGaussian<_Scalar,_Dimension>>;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    using Base::operator=;
#if 0
    using Base::operator*;
    using Base::operator+;
    using Base::operator-;
    using Base::operator/;
#endif
    
    EIGEN_DEVICE_FUNC inline MultivariateGaussian()
    {
    }
    
    template<typename OtherDerived> 
    EIGEN_DEVICE_FUNC inline MultivariateGaussian(const MultivariateGaussianBase<OtherDerived>& other)
        : mean_(other.mean()), covariance_(other.covariance())
    {
    }
    
    EIGEN_DEVICE_FUNC inline MultivariateGaussian(ConstMeanType& mean, ConstCovarianceType& covariance)
        : mean_(mean), covariance_(covariance)
    {
    }
    
    EIGEN_DEVICE_FUNC inline ~MultivariateGaussian()
    {
    }
                    
protected:
    EIGEN_DEVICE_FUNC inline ConstMeanType& mean_const() const { return mean_; }
    EIGEN_DEVICE_FUNC inline MeanType& mean_nonconst() { return mean_; }
    
    EIGEN_DEVICE_FUNC inline ConstCovarianceType& covariance_const() const { return covariance_; }
    EIGEN_DEVICE_FUNC inline CovarianceType& covariance_nonconst() { return covariance_; }
    
    MeanType mean_;
    CovarianceType covariance_;
};

template<typename _Scalar, int _Dimension>
inline std::ostream& operator<<(std::ostream& os, const MultivariateGaussian<_Scalar,_Dimension>& p)
{
    os << "MultivariateGaussian(" << p.mean() << "," << p.covariance() <<  ")";
    return os;
}

}
    
}

namespace Eigen 
{
/**
 * Specialisation of Eigen::Map for Gaussian.
 */
template<typename _Scalar, int _Rows, int _Cols, int _Options>
class Map<vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols>>, _Options>
    : public vc::types::GaussianDispatchingBase<
        Map<vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols>>, _Options>, 
            Eigen::internal::traits<Map<vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols>>, _Options>>::Dimension> 
{
    typedef vc::types::GaussianDispatchingBase<
        Map<vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols>>, _Options>, 
            Eigen::internal::traits<Map<vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols>>, _Options>>::Dimension> Base;
public:
    static constexpr int Dimension = Eigen::internal::traits<Map>::Dimension;
    typedef typename Eigen::internal::traits<Map>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Map>::MeanType MeanType;
    typedef const typename Eigen::internal::traits<Map>::MeanType ConstMeanType;
    
    friend class vc::types::GaussianDispatchingBase<
                Map<vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols>>, _Options>,
                Eigen::internal::traits<Map<vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols>>, _Options>>::Dimension>;
    friend class vc::types::GaussianBase<Map<vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols>>, _Options> >;
    
    EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
#if 0
    using Base::operator*;
    using Base::operator+;
    using Base::operator-;
    using Base::operator/;
#endif
    
    EIGEN_DEVICE_FUNC inline Map(Scalar* coeffs) : mean_(coeffs), variance_(coeffs + Dimension)
    {
    }
    
protected:
    EIGEN_DEVICE_FUNC inline ConstMeanType& mean_const() const { return mean_; }
    EIGEN_DEVICE_FUNC inline MeanType& mean_nonconst() { return mean_; }
    
    EIGEN_DEVICE_FUNC inline ConstMeanType& variance_const() const { return variance_; }
    EIGEN_DEVICE_FUNC inline MeanType& variance_nonconst() { return variance_; }
    
    MeanType mean_;
    MeanType variance_;
};

/**
 * Specialisation of Eigen::Map for const Gaussian.
 */
template<typename _Scalar, int _Rows, int _Cols, int _Options>
class Map<const vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols>>, _Options>
    : public vc::types::GaussianDispatchingBase<
        Map<const vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols>>, _Options>, 
          Eigen::internal::traits<const vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols>>>::Dimension> 
{
    typedef vc::types::GaussianDispatchingBase<
      Map<const vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols>>, _Options>, 
      Eigen::internal::traits<const vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols>>>::Dimension>  Base;
public:
    static constexpr int Dimension = Eigen::internal::traits<Map>::Dimension;
    typedef typename Eigen::internal::traits<Map>::Scalar Scalar;    
    typedef const typename Eigen::internal::traits<Map>::MeanType ConstMeanType;
    
    friend class vc::types::GaussianDispatchingBase<
      Map<const vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols>>, _Options>, 
      Eigen::internal::traits<const vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols>>>::Dimension>;
    friend class vc::types::GaussianBase<Map<const vc::types::Gaussian<Matrix<_Scalar, _Rows, _Cols>>, _Options> >;
    
    EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
#if 0
    using Base::operator*;
    using Base::operator+;
    using Base::operator-;
    using Base::operator/;
#endif
    
    EIGEN_DEVICE_FUNC inline Map(const Scalar* coeffs) : mean_(coeffs), variance_(coeffs + Dimension)
    {
    }
    
protected:
    EIGEN_DEVICE_FUNC inline ConstMeanType& mean_const() const { return mean_; }    
    EIGEN_DEVICE_FUNC inline ConstMeanType& variance_const() const { return variance_; }
    
    ConstMeanType mean_;
    ConstMeanType variance_;
};
    
/**
 * Specialisation of Eigen::Map for Multivariate Gaussian.
 */
template<typename _Scalar, int _Dimension, int _Options>
class Map<vc::types::MultivariateGaussian<_Scalar,_Dimension>, _Options>
  : public vc::types::MultivariateGaussianBase<
    Map<vc::types::MultivariateGaussian<_Scalar,_Dimension>, _Options> > 
{
    typedef vc::types::MultivariateGaussianBase<Map<vc::types::MultivariateGaussian<_Scalar,_Dimension>, _Options>> Base;
    
public:
    static constexpr int Dimension = Eigen::internal::traits<Map>::Dimension;
    typedef typename Eigen::internal::traits<Map>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Map>::MeanType MeanType;
    typedef const typename Eigen::internal::traits<Map>::MeanType ConstMeanType;
    typedef typename Eigen::internal::traits<Map>::CovarianceType CovarianceType;
    typedef const typename Eigen::internal::traits<Map>::CovarianceType ConstCovarianceType;
    
    friend class vc::types::MultivariateGaussianBase<Map<vc::types::MultivariateGaussian<_Scalar,_Dimension>, _Options> >;
    
    EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
#if 0
    using Base::operator*;
    using Base::operator+;
    using Base::operator-;
    using Base::operator/;
#endif
    
    EIGEN_DEVICE_FUNC inline Map(Scalar* coeffs) : mean_(coeffs), covariance_(coeffs + _Dimension)
    {
    }
    
protected:
    EIGEN_DEVICE_FUNC inline ConstMeanType& mean_const() const { return mean_; }
    EIGEN_DEVICE_FUNC inline MeanType& mean_nonconst() { return mean_; }
    
    EIGEN_DEVICE_FUNC inline ConstCovarianceType& covariance_const() const { return covariance_; }
    EIGEN_DEVICE_FUNC inline CovarianceType& covariance_nonconst() { return covariance_; }
    
    MeanType mean_;
    CovarianceType covariance_;
};

/**
 * Specialisation of Eigen::Map for const Multivariate Gaussian.
 */
template<typename _Scalar, int _Dimension, int _Options>
class Map<const vc::types::MultivariateGaussian<_Scalar,_Dimension>, _Options>
  : public vc::types::MultivariateGaussianBase<
    Map<const vc::types::MultivariateGaussian<_Scalar,_Dimension>, _Options> > 
{
    typedef vc::types::MultivariateGaussianBase<Map<const vc::types::MultivariateGaussian<_Scalar,_Dimension>, _Options> > Base;
    
public:
    static constexpr int Dimension = Eigen::internal::traits<Map>::Dimension;
    typedef typename Eigen::internal::traits<Map>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Map>::MeanType MeanType;
    typedef const typename Eigen::internal::traits<Map>::MeanType ConstMeanType;
    typedef typename Eigen::internal::traits<Map>::CovarianceType CovarianceType;
    typedef const typename Eigen::internal::traits<Map>::CovarianceType ConstCovarianceType;
    
    friend class vc::types::MultivariateGaussianBase<Map<const vc::types::MultivariateGaussian<_Scalar,_Dimension>, _Options> >;
    
    EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
#if 0
    using Base::operator*;
    using Base::operator+;
    using Base::operator-;
    using Base::operator/;
#endif
    
    EIGEN_DEVICE_FUNC inline Map(const Scalar* coeffs) : mean_(coeffs), covariance_(coeffs + _Dimension)
    {
    }
    
protected:
    EIGEN_DEVICE_FUNC inline ConstMeanType& mean_const() const { return mean_; }    
    EIGEN_DEVICE_FUNC inline ConstCovarianceType& covariance_const() const { return covariance_; }
    
    MeanType mean_;
    CovarianceType covariance_;
};

}

#endif // VISIONCORE_TYPES_GAUSSIAN_HPP
