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
 * N-Sphere.
 * ****************************************************************************
 */

#ifndef VISIONCORE_TYPES_HYPERSPHERE_HPP
#define VISIONCORE_TYPES_HYPERSPHERE_HPP

#include <VisionCore/Platform.hpp>

#include <Eigen/Dense>

// https://en.wikipedia.org/wiki/N-sphere

namespace vc
{
    
namespace types
{
    
template<typename _Scalar, int _Dimension = 0> class Hypersphere;

template <typename T> using CircleT = Hypersphere<T,1>;
template <typename T> using SphereT = Hypersphere<T,2>;

namespace internal
{

template<typename T, int Dimension>
struct helper_surface_area;

template<typename T, int Dimension>
struct helper_volume
{
    EIGEN_DEVICE_FUNC static inline T calc(T radius)
    {
        return (radius / T(Dimension)) * helper_surface_area<T,Dimension-1>::calc(radius);
    }
};

template<typename T, int Dimension>
struct helper_surface_area
{
    EIGEN_DEVICE_FUNC static inline T calc(T radius)
    {
        return T(2.0 * M_PI) * radius * helper_volume<T,Dimension-1>::calc(radius);
    }
};

template<typename T>
struct helper_surface_area<T,0>
{
    EIGEN_DEVICE_FUNC static inline T calc(T radius) { return T(2.0); }
};

template<typename T>
struct helper_volume<T,0>
{
    EIGEN_DEVICE_FUNC static inline T calc(T radius) { return T(1.0); }
};
    
}

}

}

namespace Eigen 
{
    namespace internal 
    {
        template<typename _Scalar, int _Dimension>
        struct traits<vc::types::Hypersphere<_Scalar,_Dimension> > 
        {
            static constexpr int Dimension = _Dimension;
            typedef _Scalar Scalar;
            typedef Matrix<Scalar,_Dimension+1,1> CoeffType;
        };
        
        template<typename _Scalar, int _Dimension, int _Options>
        struct traits<Map<vc::types::Hypersphere<_Scalar,_Dimension>, _Options> >
            : traits<vc::types::Hypersphere<_Scalar, _Dimension> > 
        {
            static constexpr int Dimension = _Dimension;
            typedef _Scalar Scalar;
            typedef Map<Matrix<Scalar,_Dimension+1,1>, _Options> CoeffType;
        };
        
        template<typename _Scalar, int _Dimension, int _Options>
        struct traits<Map<const vc::types::Hypersphere<_Scalar,_Dimension>, _Options> >
            : traits<const vc::types::Hypersphere<_Scalar, _Dimension> > 
        {
            static constexpr int Dimension = _Dimension;
            typedef _Scalar Scalar;
            typedef Map<const Matrix<Scalar,_Dimension+1,1>, _Options> CoeffType;
        };
        
    }
}

namespace vc
{
    
namespace types
{

template<typename Derived>
class HypersphereBase
{
public:
    static constexpr int Dimension = Eigen::internal::traits<Derived>::Dimension;
    typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Derived>::CoeffType CoeffType;
    typedef Eigen::Matrix<Scalar,Dimension+1,1> VectorT;
    
    static inline constexpr int dimension() { return Dimension; }
    
    template<typename NewScalarType>
    EIGEN_DEVICE_FUNC inline Hypersphere<NewScalarType,Dimension> cast() const 
    {
        return Hypersphere<NewScalarType,Dimension>(coeff().template cast<NewScalarType>(), 
                                                    (NewScalarType)radius());
    }
        
    EIGEN_DEVICE_FUNC CoeffType& coeff() 
    {
        return static_cast<Derived*>(this)->coeff_nonconst();
    }
    
    EIGEN_DEVICE_FUNC const CoeffType& coeff() const
    {
        return static_cast<const Derived*>(this)->coeff_const();
    }
    
    EIGEN_DEVICE_FUNC Scalar& radius() 
    {
        return static_cast<Derived*>(this)->radius_nonconst();
    }
    
    EIGEN_DEVICE_FUNC const Scalar& radius() const
    {
        return static_cast<const Derived*>(this)->radius_const();
    }
        
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC inline HypersphereBase<Derived>& operator=(const HypersphereBase<OtherDerived>& other)
    {
        coeff() = other.coeff();
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline Scalar volume()
    {
        using Eigen::numext::pow;
        return (pow(Scalar(M_PI), Scalar(Dimension+1)/Scalar(2.0)) / (tgamma(Scalar(Dimension+1)/Scalar(2.0) + Scalar(1.0)))) * pow(radius(), (Scalar)(Dimension+1) );
    }
    
    template<int OtherDimension>
    EIGEN_DEVICE_FUNC inline Scalar volume()
    {
        return internal::helper_volume<Scalar,OtherDimension+1>::calc(radius());
    }
    
    EIGEN_DEVICE_FUNC inline Scalar surfaceArea()
    {
        return internal::helper_surface_area<Scalar,Dimension>::calc(radius());
    }
    
    template<int OtherDimension>
    EIGEN_DEVICE_FUNC inline Scalar surfaceArea()
    {
        return internal::helper_surface_area<Scalar,OtherDimension>::calc(radius());
    }

    EIGEN_DEVICE_FUNC inline Scalar operator()(const VectorT& x) const
    {
        return evaluate(x);
    }
    
    EIGEN_DEVICE_FUNC inline Scalar evaluate(const VectorT& x) const
    {
        Scalar sum = Scalar(0.0);
        
        for(int i = 0 ; i < Dimension + 1 ; ++i)
        {
            const Scalar term = coeff()(i) - x(i);
            sum += (term * term);
        }
        
        return sqrt(sum);
    }
    
    EIGEN_DEVICE_FUNC inline bool isPointInside(const VectorT& x) const
    {
        return evaluate(x) < radius();
    }
    
    EIGEN_DEVICE_FUNC inline void setZero()
    {
        coeff().setZero();
        radius() = Scalar(0.0);
    }
    
#ifdef VISIONCORE_ENABLE_CEREAL
    template<typename Archive>
    void load(Archive & archive, std::uint32_t const version)
    {
        archive(cereal::make_nvp("Coefficients", coeff()));
        archive(cereal::make_nvp("Radius", radius()));
    }
    
    template<typename Archive>
    void save(Archive & archive, std::uint32_t const version) const
    {
        archive(cereal::make_nvp("Coefficients", coeff()));
        archive(cereal::make_nvp("Radius", radius()));
    }
#endif // VISIONCORE_ENABLE_CEREAL    
};

/**
 * Generic Hypersphere.
 */
template<typename _Scalar, int _Dimension>
class Hypersphere : public HypersphereBase<Hypersphere<_Scalar,_Dimension>> 
{
    typedef HypersphereBase<Hypersphere<_Scalar,_Dimension>> Base;
public:
    static constexpr int Dimension = Eigen::internal::traits<Hypersphere>::Dimension;
    typedef typename Eigen::internal::traits<Hypersphere>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Hypersphere>::CoeffType CoeffType;
    
    friend class vc::types::HypersphereBase<Hypersphere<_Scalar,_Dimension>>;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    //EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Hypersphere)
    using Base::operator=;   
    
    EIGEN_DEVICE_FUNC inline Hypersphere()
    {
    }
    
    EIGEN_DEVICE_FUNC inline Hypersphere(const Hypersphere<_Scalar,_Dimension>& other) : coeff_(other.coeff()), radius_(other.radius())
    {
    }
    
    template<typename OtherDerived> 
    EIGEN_DEVICE_FUNC inline Hypersphere(const HypersphereBase<OtherDerived>& other) : coeff_(other.coeff()), radius_(other.radius())
    {
    }
    
    EIGEN_DEVICE_FUNC inline Hypersphere(const CoeffType& coeff, const Scalar& r) : coeff_(coeff), radius_(r)
    {
    }
    
    EIGEN_DEVICE_FUNC inline ~Hypersphere()
    {
    }
                    
protected:
    EIGEN_DEVICE_FUNC inline const CoeffType& coeff_const() const { return coeff_; }
    EIGEN_DEVICE_FUNC inline CoeffType& coeff_nonconst() { return coeff_; }
    
    EIGEN_DEVICE_FUNC inline const Scalar& radius_const() const { return radius_; }
    EIGEN_DEVICE_FUNC inline Scalar& radius_nonconst() { return radius_; }
    
    CoeffType coeff_;
    Scalar radius_;
};

template<typename _Scalar, int _Dimension>
inline std::ostream& operator<<(std::ostream& os, const Hypersphere<_Scalar,_Dimension>& p)
{
    os << _Dimension << "-Sphere(r = " << p.radius() << ", " << p.coeff() << ")";
    return os;
}
    
}

}

namespace Eigen 
{
/**
 * Specialisation of Eigen::Map for Hypersphere.
 */
template<typename _Scalar, int _Dimension, int _Options>
class Map<vc::types::Hypersphere<_Scalar,_Dimension>, _Options>
    : public vc::types::HypersphereBase<Map<vc::types::Hypersphere<_Scalar,_Dimension>, _Options> > 
{
    typedef vc::types::HypersphereBase<Map<vc::types::Hypersphere<_Scalar,_Dimension>, _Options> > Base;
    
public:
    static constexpr int Dimension = Eigen::internal::traits<Map>::Dimension;
    typedef typename Eigen::internal::traits<Map>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Map>::CoeffType CoeffType;
    
    friend class vc::types::HypersphereBase<Map<vc::types::Hypersphere<_Scalar,_Dimension>, _Options> >;
    
    EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
    
    EIGEN_DEVICE_FUNC inline Map(Scalar* coeffs) : coeff_(coeffs), radius_(coeffs + Dimension + 1)
    {
    }
    
protected:
    EIGEN_DEVICE_FUNC inline const CoeffType& coeff_const() const { return coeff_; }
    EIGEN_DEVICE_FUNC inline CoeffType& coeff_nonconst() { return coeff_; }
    
    EIGEN_DEVICE_FUNC inline const Scalar& radius_const() const { return *radius_; }
    EIGEN_DEVICE_FUNC inline Scalar& radius_nonconst() { return *radius_; }
    
    CoeffType coeff_;
    Scalar* radius_;
};

/**
 * Specialisation of Eigen::Map for const Hypersphere.
 */
template<typename _Scalar, int _Dimension, int _Options>
class Map<const vc::types::Hypersphere<_Scalar,_Dimension>, _Options>
    : public vc::types::HypersphereBase<Map<const vc::types::Hypersphere<_Scalar,_Dimension>, _Options> > 
{
    typedef vc::types::HypersphereBase<Map<const vc::types::Hypersphere<_Scalar,_Dimension>, _Options> > Base;
    
public:
    static constexpr int Dimension = Eigen::internal::traits<Map>::Dimension;
    typedef typename Eigen::internal::traits<Map>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Map>::CoeffType CoeffType;
        
    friend class vc::types::HypersphereBase<Map<const vc::types::Hypersphere<_Scalar,_Dimension>, _Options> >;
    
    EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
    
    EIGEN_DEVICE_FUNC inline Map(const Scalar* coeffs) : coeff_(coeffs), radius_(coeffs + Dimension + 1)
    {
    }
    
protected:
    EIGEN_DEVICE_FUNC inline const CoeffType& coeff_const() const { return coeff_; }    
    EIGEN_DEVICE_FUNC inline const Scalar& radius_const() const { return *radius_; }
    
    const CoeffType coeff_;
    const Scalar* radius_;
};

}

#endif // VISIONCORE_TYPES_HYPERSPHERE_HPP
