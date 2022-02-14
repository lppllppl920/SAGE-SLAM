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
 * Extension of Eigen's Polynomials.
 * ****************************************************************************
 */

#ifndef VISIONCORE_TYPES_POLYNOMIAL_HPP
#define VISIONCORE_TYPES_POLYNOMIAL_HPP

#include <VisionCore/Platform.hpp>

#include <unsupported/Eigen/Polynomials>

namespace vc
{
namespace types
{
template<typename _Scalar, int _Rank = 0> class Polynomial;

template <typename T> using LineT = Polynomial<T,1>;
template <typename T> using QuadraticT = Polynomial<T,2>;
template <typename T> using CubicT = Polynomial<T,3>;
template <typename T> using QuarticT = Polynomial<T,4>;
template <typename T> using QuinticT = Polynomial<T,5>;
}
}

namespace Eigen 
{
    namespace internal 
    {
        template<typename _Scalar, int _Rank>
        struct traits<vc::types::Polynomial<_Scalar,_Rank> > 
        {
            static constexpr int Rank = _Rank;
            typedef _Scalar Scalar;
            typedef Matrix<Scalar,_Rank+1,1> CoeffType;
        };
        
        template<typename _Scalar, int _Rank, int _Options>
        struct traits<Map<vc::types::Polynomial<_Scalar,_Rank>, _Options> > : traits<vc::types::Polynomial<_Scalar, _Rank> > 
        {
            static constexpr int Rank = _Rank;
            typedef _Scalar Scalar;
            typedef Map<Matrix<Scalar,_Rank+1,1>, _Options> CoeffType;
        };
        
        template<typename _Scalar, int _Rank, int _Options>
        struct traits<Map<const vc::types::Polynomial<_Scalar,_Rank>, _Options> > : traits<const vc::types::Polynomial<_Scalar, _Rank> > 
        {
            static constexpr int Rank = _Rank;
            typedef _Scalar Scalar;
            typedef Map<const Matrix<Scalar,_Rank+1,1>, _Options> CoeffType;
        };
        
    }
}

namespace vc
{

namespace types
{
    
template<typename Derived>
class PolynomialBase
{
public:
    static constexpr int Rank = Eigen::internal::traits<Derived>::Rank;
    typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Derived>::CoeffType CoeffType;
    
    static inline constexpr int size() { return Rank; }
    
    template<typename NewScalarType>
    EIGEN_DEVICE_FUNC inline Polynomial<NewScalarType,Rank> cast() const 
    {
        return Polynomial<NewScalarType,Rank>(coeff().template cast<NewScalarType>());
    }
        
    EIGEN_DEVICE_FUNC CoeffType& coeff() 
    {
        return static_cast<Derived*>(this)->coeff_nonconst();
    }
    
    EIGEN_DEVICE_FUNC const CoeffType& coeff() const
    {
        return static_cast<const Derived*>(this)->coeff_const();
    }
    
    EIGEN_DEVICE_FUNC inline Scalar& operator[](std::size_t ix)
    {
        return coeff()(ix);
    }
    
    EIGEN_DEVICE_FUNC inline const Scalar& operator[](std::size_t ix) const
    {
        return coeff()(ix);
    }
        
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC inline PolynomialBase<Derived>& operator=(const PolynomialBase<OtherDerived>& other)
    {
        coeff() = other.coeff();
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline Polynomial<Scalar,Rank> operator+(const PolynomialBase<Derived>& other) const
    {
        Polynomial<Scalar,Rank> result(*this);
        result += other;
        return result;
    }
    
    EIGEN_DEVICE_FUNC inline Polynomial<Scalar,Rank> operator-(const PolynomialBase<Derived>& other) const
    {
        Polynomial<Scalar,Rank> result(*this);
        result -= other;
        return result;
    }
    
    EIGEN_DEVICE_FUNC inline PolynomialBase<Derived>& operator+=(const PolynomialBase<Derived>& other) 
    {
        coeff() += other.coeff();
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline PolynomialBase<Derived>& operator-=(const PolynomialBase<Derived>& other) 
    {
        coeff() -= other.coeff();
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline Polynomial<Scalar,Rank+Rank> operator*(const PolynomialBase<Derived>& other) const
    {
        Polynomial<Scalar,Rank+Rank> result;
        result.setZero();
        
        for(int i = 0 ; i <= Rank ; ++i)
        {
            for(int j = 0 ; j <= Rank ; ++j)
            {
                result.coeff()(i+j) += coeff()(i) * other.coeff()(j);
            }
        }
        
        return result;
    }
    
    EIGEN_DEVICE_FUNC inline Polynomial<Scalar,Rank> operator*(const Scalar& other) const
    {
        Polynomial<Scalar,Rank> result(*this);
        result *= other;
        return result;
    }
    
    EIGEN_DEVICE_FUNC inline PolynomialBase<Derived>& operator*=(const Scalar& other) 
    {
        coeff() *= other;
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline Scalar cauchy_max_bound()
    {
        return Eigen::cauchy_max_bound(*this);
    }
    
    EIGEN_DEVICE_FUNC inline Scalar cauchy_min_bound()
    {
        return Eigen::cauchy_min_bound(*this);
    }
    
#if 0
    template<typename OtherScalar>
    EIGEN_DEVICE_FUNC inline PolynomialBase<Derived> operator/(const OtherScalar& other) const
    {
        // TBD
    }
#endif

    EIGEN_DEVICE_FUNC inline Scalar operator()(const Scalar& x) const
    {
        return evaluate(x);
    }
    
    EIGEN_DEVICE_FUNC inline Scalar evaluate(const Scalar& x) const
    {
        using Eigen::numext::pow;
        
        Scalar sum = coeff()(0);

        for(std::size_t i = 1 ; i <= Rank ; ++i)
        {
            sum += ( coeff()(i) * pow(x, Scalar(i)) );
        }
        
        return sum;
    }
    
    EIGEN_DEVICE_FUNC inline Scalar evaluateHorner(const Scalar& x) const
    {
        return Eigen::poly_eval(*this, x);
    }
    
    inline typename Eigen::PolynomialSolver<Scalar, Rank>::RootsType solve()
    {
        Eigen::PolynomialSolver<Scalar, Rank> solver;
        solver.compute(coeff());
        return solver.roots();
    }
    
    EIGEN_DEVICE_FUNC inline void setZero()
    {
        coeff().setZero();
    }
    
#ifdef VISIONCORE_ENABLE_CEREAL
    template<typename Archive>
    void load(Archive & archive, std::uint32_t const version)
    {
        archive(cereal::make_nvp("Coefficients", coeff()));
    }
    
    template<typename Archive>
    void save(Archive & archive, std::uint32_t const version) const
    {
        archive(cereal::make_nvp("Coefficients", coeff()));
    }
#endif // VISIONCORE_ENABLE_CEREAL    
};

/**
 * Generic Polynomial.
 */
template<typename _Scalar, int _Rank>
class Polynomial : public PolynomialBase<Polynomial<_Scalar,_Rank>> 
{
    typedef PolynomialBase<Polynomial<_Scalar,_Rank>> Base;
public:
    static constexpr int Rank = Eigen::internal::traits<Polynomial>::Rank;
    typedef typename Eigen::internal::traits<Polynomial>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Polynomial>::CoeffType CoeffType;
    
    friend class vc::types::PolynomialBase<Polynomial<_Scalar,_Rank>>;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    //EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Polynomial)
    using Base::operator=;   
    using Base::operator+;
    using Base::operator-;
    using Base::operator+=;
    using Base::operator-=;
    using Base::operator*;
    using Base::operator*=;
#if 0
    using Base::operator/;
#endif
    
    EIGEN_DEVICE_FUNC inline Polynomial()
    {
    }
    
    EIGEN_DEVICE_FUNC inline Polynomial(const Polynomial<_Scalar,_Rank>& other) : coeff_(other.coeff())
    {
    }
    
    template<typename OtherDerived> 
    EIGEN_DEVICE_FUNC inline Polynomial(const PolynomialBase<OtherDerived>& other) : coeff_(other.coeff())
    {
    }
    
    EIGEN_DEVICE_FUNC inline Polynomial(const CoeffType& coeff) : coeff_(coeff)
    {
    }
    
    EIGEN_DEVICE_FUNC inline ~Polynomial()
    {
    }
                    
protected:
    EIGEN_DEVICE_FUNC inline const CoeffType& coeff_const() const { return coeff_; }
    EIGEN_DEVICE_FUNC inline CoeffType& coeff_nonconst() { return coeff_; }
    
    CoeffType coeff_;
};

template<typename _Scalar, int _Rank>
inline std::ostream& operator<<(std::ostream& os, const Polynomial<_Scalar,_Rank>& p)
{
    for(int r = _Rank ; r >= 0 ; r--)
    {
        if(p[r] > _Scalar(0.0))
        {
            if(r != _Rank)
            {
                os << " + ";
            }
            
            os << p[r];
        }
        else
        {
            os << " - " << -p[r];
        }
        
        if(r > 0)
        {
            if(r != 1)
            {
                os << "x^" << r;
            }
            else
            {
                os << "x";
            }
        }
    }
    return os;
}

}
    
}

namespace Eigen 
{
/**
 * Specialisation of Eigen::Map for Polynomial.
 */
template<typename _Scalar, int _Rank, int _Options>
class Map<vc::types::Polynomial<_Scalar,_Rank>, _Options> : 
    public vc::types::PolynomialBase<Map<vc::types::Polynomial<_Scalar,_Rank>, _Options> > 
{
    typedef vc::types::PolynomialBase<Map<vc::types::Polynomial<_Scalar,_Rank>, _Options> > Base;
    
public:
    static constexpr int Rank = Eigen::internal::traits<Map>::Rank;
    typedef typename Eigen::internal::traits<Map>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Map>::CoeffType CoeffType;
    
    friend class vc::types::PolynomialBase<Map<vc::types::Polynomial<_Scalar,_Rank>, _Options> >;
    
    EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
    
    using Base::operator+;
    using Base::operator-;
    using Base::operator+=;
    using Base::operator-=;
    using Base::operator*;
    using Base::operator*=;
#if 0
    using Base::operator/;
#endif
    
    EIGEN_DEVICE_FUNC inline Map(Scalar* coeffs) : coeff_(coeffs)
    {
    }
    
protected:
    EIGEN_DEVICE_FUNC inline const CoeffType& coeff_const() const { return coeff_; }
    EIGEN_DEVICE_FUNC inline CoeffType& coeff_nonconst() { return coeff_; }
    
    CoeffType coeff_;
};

/**
 * Specialisation of Eigen::Map for const Polynomial.
 */
template<typename _Scalar, int _Rank, int _Options>
class Map<const vc::types::Polynomial<_Scalar,_Rank>, _Options> : 
    public vc::types::PolynomialBase<Map<const vc::types::Polynomial<_Scalar,_Rank>, _Options> > 
{
    typedef vc::types::PolynomialBase<Map<const vc::types::Polynomial<_Scalar,_Rank>, _Options> > Base;
    
public:
    static constexpr int Rank = Eigen::internal::traits<Map>::Rank;
    typedef typename Eigen::internal::traits<Map>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Map>::CoeffType CoeffType;
        
    friend class vc::types::PolynomialBase<Map<const vc::types::Polynomial<_Scalar,_Rank>, _Options> >;
    
    EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
    
    using Base::operator+;
    using Base::operator-;
    using Base::operator+=;
    using Base::operator-=;
    using Base::operator*;
    using Base::operator*=;
#if 0
    using Base::operator/;
#endif
    
    EIGEN_DEVICE_FUNC inline Map(const Scalar* coeffs) : coeff_(coeffs)
    {
    }
    
protected:
    EIGEN_DEVICE_FUNC inline const CoeffType& coeff_const() const { return coeff_; }    
    
    const CoeffType coeff_;
};

}

#endif // VISIONCORE_POLYNOMIAL_HPP
