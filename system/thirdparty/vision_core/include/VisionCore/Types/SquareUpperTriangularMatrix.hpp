/**
 * ****************************************************************************
 * Copyright (c) 2017, Robert Lukierski.
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
 * Square Upper Triangular Matrix - minimal storage.
 * ****************************************************************************
 */

#ifndef VISIONCORE_TYPES_SQUTM_HPP
#define VISIONCORE_TYPES_SQUTM_HPP

#include <VisionCore/Platform.hpp>

namespace vc
{
namespace types
{
template<typename _Scalar, int _Rows, int _Options = 0> class SquareUpperTriangularMatrix;
}
}

namespace Eigen 
{
    namespace internal 
    {
        template<typename _Scalar, int _Rows, int _Options>
        struct traits<vc::types::SquareUpperTriangularMatrix<_Scalar,_Rows,_Options> > 
        {
            static constexpr int Rows = _Rows;
            static constexpr int Options = _Options;
            typedef _Scalar Scalar;
            typedef Matrix<Scalar,(_Rows * _Rows - _Rows)/2 + _Rows,1, _Options> CoeffType;
        };
        
        template<typename _Scalar, int _Rows, int _Options>
        struct traits<Map<vc::types::SquareUpperTriangularMatrix<_Scalar,_Rows>, _Options> > :   
            traits<vc::types::SquareUpperTriangularMatrix<_Scalar, _Rows, _Options> > 
        {
            static constexpr int Rows = _Rows;
            static constexpr int Options = _Options;
            typedef _Scalar Scalar;
            typedef Map<Matrix<Scalar,(_Rows * _Rows - _Rows)/2 + _Rows,1>, _Options> CoeffType;
        };
        
        template<typename _Scalar, int _Rows, int _Options>
        struct traits<Map<const vc::types::SquareUpperTriangularMatrix<_Scalar,_Rows>, _Options> > : 
            traits<const vc::types::SquareUpperTriangularMatrix<_Scalar, _Rows, _Options> > 
        {
            static constexpr int Rank = _Rows;
            static constexpr int Options = _Options;
            typedef _Scalar Scalar;
            typedef Map<const Matrix<Scalar,(_Rows * _Rows - _Rows)/2 + _Rows,1>, _Options> CoeffType;
        };
        
    }
}

namespace vc
{

namespace types
{
  
namespace detail
{
  
template<int Rows, int Index, typename Derived, typename OtherDerived>
struct run_outer_product_impl
{
    EIGEN_DEVICE_FUNC static inline void run(Derived& coeff, const OtherDerived& vec)
    {
        static constexpr int LinearLinex = Rows * Index - ((Index - 1) * Index) / 2;
        
        coeff.template segment<Rows - Index>(LinearLinex) = vec(Index) * vec.template tail<Rows - Index>();
        run_outer_product_impl<Rows, Index - 1, Derived, OtherDerived>::run(coeff, vec);
    }
};

template<int Rows, typename Derived, typename OtherDerived>
struct run_outer_product_impl<Rows, 0, Derived, OtherDerived>
{
    EIGEN_DEVICE_FUNC static inline void run(Derived& coeff, const OtherDerived& vec)
    {
        coeff.template head<Rows>() = vec(0) * vec;
    }
};

template<int Rows, int Index, typename Derived, typename OtherDerived>
EIGEN_DEVICE_FUNC static inline void run_outer_product(Derived& coeff, 
                                                       const OtherDerived& vec)
{
    run_outer_product_impl<Rows, Index, Derived, OtherDerived>::run(coeff, vec);
}

}
    
template<typename Derived>
class SquareUpperTriangularMatrixBase
{
public:
    static constexpr int Rows = Eigen::internal::traits<Derived>::Rows;
    static constexpr int Options = Eigen::internal::traits<Derived>::Options;
    typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Derived>::CoeffType CoeffType;
    typedef Eigen::Matrix<Scalar, Rows, Rows, Options> DenseMatrixType;
    
    static inline constexpr int size() { return Rows; }
    
    EIGEN_DEVICE_FUNC static inline const SquareUpperTriangularMatrix<Scalar,Rows,Options> Zero()
    {
       return Constant(Scalar(0));
    }
    
    EIGEN_DEVICE_FUNC static inline const SquareUpperTriangularMatrix<Scalar,Rows,Options> Constant(const Scalar& value)
    {
        SquareUpperTriangularMatrix<Scalar,Rows,Options> ret;
        ret.coeff() = CoeffType::Constant(value);
        return ret;
    }
    
    EIGEN_DEVICE_FUNC DenseMatrixType toDenseMatrix() const
    {
        DenseMatrixType res;
        
        // TODO FIXME change to templates
        for(int ic = 0 ; ic < Rows ; ++ic)
        {
            for(int ir = 0 ; ir < Rows ; ++ir)
            {
                res(ir,ic) = coeff()(toLinearIndex(ir,ic));
            }
        }
        
        return res;
    }
    
    template<typename NewScalarType>
    EIGEN_DEVICE_FUNC inline SquareUpperTriangularMatrix<NewScalarType,Rows,Options> cast() const 
    {
        return SquareUpperTriangularMatrix<NewScalarType,Rows,Options>(coeff().template cast<NewScalarType>());
    }
        
    EIGEN_DEVICE_FUNC CoeffType& coeff() 
    {
        return static_cast<Derived*>(this)->coeff_nonconst();
    }
    
    EIGEN_DEVICE_FUNC const CoeffType& coeff() const
    {
        return static_cast<const Derived*>(this)->coeff_const();
    }
    
    EIGEN_DEVICE_FUNC inline Scalar& operator()(std::size_t ir, std::size_t ic)
    {
        return coeff()(toLinearIndex(ir,ic));
    }
    
    EIGEN_DEVICE_FUNC inline const Scalar& operator()(std::size_t ir, std::size_t ic) const
    {
        return coeff()(toLinearIndex(ir,ic));
    }
        
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC inline SquareUpperTriangularMatrixBase<Derived>& operator=(const SquareUpperTriangularMatrixBase<OtherDerived>& other)
    {
        coeff() = other.coeff();
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline SquareUpperTriangularMatrix<Scalar,Rows,Options> operator+(const SquareUpperTriangularMatrixBase<Derived>& other) const
    {
        SquareUpperTriangularMatrix<Scalar,Rows,Options> result(*this);
        result += other;
        return result;
    }
    
    EIGEN_DEVICE_FUNC inline SquareUpperTriangularMatrix<Scalar,Rows,Options> operator-(const SquareUpperTriangularMatrixBase<Derived>& other) const
    {
        SquareUpperTriangularMatrix<Scalar,Rows,Options> result(*this);
        result -= other;
        return result;
    }
    
    EIGEN_DEVICE_FUNC inline SquareUpperTriangularMatrixBase<Derived>& operator+=(const SquareUpperTriangularMatrixBase<Derived>& other) 
    {
        coeff() += other.coeff();
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline SquareUpperTriangularMatrixBase<Derived>& operator-=(const SquareUpperTriangularMatrixBase<Derived>& other) 
    {
        coeff() -= other.coeff();
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline SquareUpperTriangularMatrix<Scalar,Rows,Options> operator*(const Scalar& other) const
    {
        SquareUpperTriangularMatrix<Scalar,Rows,Options> result(*this);
        result *= other;
        return result;
    }
    
    EIGEN_DEVICE_FUNC inline SquareUpperTriangularMatrixBase<Derived>& operator*=(const Scalar& other) 
    {
        coeff() *= other;
        return *this;
    }
    
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC inline void fromOuterProduct(const OtherDerived& b)
    {
        EIGEN_STATIC_ASSERT_FIXED_SIZE(CoeffType);
        EIGEN_STATIC_ASSERT_FIXED_SIZE(OtherDerived);
        EIGEN_STATIC_ASSERT(OtherDerived::RowsAtCompileTime == Rows, YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES);
        EIGEN_STATIC_ASSERT(OtherDerived::ColsAtCompileTime == 1, YOU_PASSED_A_COLUMN_VECTOR_BUT_A_ROW_VECTOR_WAS_EXPECTED);
        
        detail::run_outer_product<Rows, Rows-1>(coeff(), b);
    }
    
    EIGEN_DEVICE_FUNC void fill(const Scalar& value) 
    { 
        setConstant(value); 
    }
    
    EIGEN_DEVICE_FUNC SquareUpperTriangularMatrixBase<Derived>& setConstant(const Scalar& value)
    { 
        return *this = SquareUpperTriangularMatrixBase<Derived>::Constant(value); 
    }

    EIGEN_DEVICE_FUNC SquareUpperTriangularMatrixBase<Derived>& setZero() 
    { 
        return setConstant(Scalar(0)); 
    }
    
    EIGEN_DEVICE_FUNC SquareUpperTriangularMatrixBase<Derived>& setOnes() 
    { 
        return setConstant(Scalar(1)); 
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
private:
    EIGEN_DEVICE_FUNC inline std::size_t toLinearIndex(std::size_t ir, std::size_t ic) const
    {
        if(ir > ic) // swap
        {
            const std::size_t tmp = ir;
            ir = ic;
            ic = tmp;
        }

        return Rows * ir - ((ir - 1) * ir) / 2 + (ic - ir);
    }
};

/**
 * Generic SquareUpperTriangularMatrix.
 */
template<typename _Scalar, int _Rows, int _Options>
class SquareUpperTriangularMatrix : public SquareUpperTriangularMatrixBase<SquareUpperTriangularMatrix<_Scalar,_Rows,_Options>> 
{
    typedef SquareUpperTriangularMatrixBase<SquareUpperTriangularMatrix<_Scalar,_Rows,_Options>> Base;
public:
    static constexpr int Rows = Eigen::internal::traits<SquareUpperTriangularMatrix>::Rows;
    static constexpr int Options = Eigen::internal::traits<SquareUpperTriangularMatrix>::Options;
    typedef typename Eigen::internal::traits<SquareUpperTriangularMatrix>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<SquareUpperTriangularMatrix>::CoeffType CoeffType;
    
    friend class vc::types::SquareUpperTriangularMatrixBase<SquareUpperTriangularMatrix<Scalar,Rows,Options>>;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    //EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(SquareUpperTriangularMatrix)
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
    
    EIGEN_DEVICE_FUNC inline SquareUpperTriangularMatrix()
    {
    }
    
    EIGEN_DEVICE_FUNC inline SquareUpperTriangularMatrix(const SquareUpperTriangularMatrix<Scalar,Rows,Options>& other) : coeff_(other.coeff())
    {
    }
    
    template<typename OtherDerived> 
    EIGEN_DEVICE_FUNC inline SquareUpperTriangularMatrix(const SquareUpperTriangularMatrixBase<OtherDerived>& other) : coeff_(other.coeff())
    {
    }
    
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC inline SquareUpperTriangularMatrix(const OtherDerived& b)
    {
        Base::fromOuterProduct(b);
    }
    
    EIGEN_DEVICE_FUNC inline ~SquareUpperTriangularMatrix()
    {
    }
                    
protected:
    EIGEN_DEVICE_FUNC inline const CoeffType& coeff_const() const { return coeff_; }
    EIGEN_DEVICE_FUNC inline CoeffType& coeff_nonconst() { return coeff_; }
    
    CoeffType coeff_;
};

template<typename _Scalar, int _Rows, int _Options = 0>
inline std::ostream& operator<<(std::ostream& os, const SquareUpperTriangularMatrix<_Scalar,_Rows,_Options>& p)
{
    // eigen is column major
    for(int c = 0 ; c < _Rows ; ++c)
    {
        for(int r = 0 ; r < _Rows ; ++r)
        {
            os << p(r,c) << " ";
        }
        os << std::endl;
    }
    
    return os;
}

}
    
}

namespace Eigen 
{
/**
 * Specialisation of Eigen::Map for SquareUpperTriangularMatrix.
 */
template<typename _Scalar, int _Rows, int _Options>
class Map<vc::types::SquareUpperTriangularMatrix<_Scalar,_Rows>, _Options> : 
  public vc::types::SquareUpperTriangularMatrixBase<Map<vc::types::SquareUpperTriangularMatrix<_Scalar,_Rows>, _Options> > 
{
  typedef vc::types::SquareUpperTriangularMatrixBase<Map<vc::types::SquareUpperTriangularMatrix<_Scalar,_Rows>, _Options> > Base;
    
public:
    static constexpr int Rows = Eigen::internal::traits<Map>::Rows;
    static constexpr int Options = Eigen::internal::traits<Map>::Options;
    typedef typename Eigen::internal::traits<Map>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Map>::CoeffType CoeffType;
    
    friend class vc::types::SquareUpperTriangularMatrixBase<Map<vc::types::SquareUpperTriangularMatrix<Scalar,Rows>, Options> >;
    
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
 * Specialisation of Eigen::Map for const SquareUpperTriangularMatrix.
 */
template<typename _Scalar, int _Rows, int _Options>
class Map<const vc::types::SquareUpperTriangularMatrix<_Scalar,_Rows>, _Options> : 
public vc::types::SquareUpperTriangularMatrixBase<Map<const vc::types::SquareUpperTriangularMatrix<_Scalar,_Rows>, _Options> > 
{
    typedef vc::types::SquareUpperTriangularMatrixBase<Map<const vc::types::SquareUpperTriangularMatrix<_Scalar,_Rows>, _Options> > Base;
    
public:
    static constexpr int Rows = Eigen::internal::traits<Map>::Rows;
    static constexpr int Options = Eigen::internal::traits<Map>::Options;
    typedef typename Eigen::internal::traits<Map>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Map>::CoeffType CoeffType;
        
    friend class vc::types::SquareUpperTriangularMatrixBase<Map<const vc::types::SquareUpperTriangularMatrix<Scalar,Rows>, Options> >;
    
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

#endif // VISIONCORE_TYPES_SQUTM_HPP
