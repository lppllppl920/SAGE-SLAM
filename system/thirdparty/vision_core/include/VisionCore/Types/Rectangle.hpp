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
 * Just a Rectangle.
 * ****************************************************************************
 */

#ifndef VISIONCORE_TYPES_RECTANGLE_BOX
#define VISIONCORE_TYPES_RECTANGLE_BOX

#include <VisionCore/Platform.hpp>

namespace vc
{
namespace types
{
    template<typename _Scalar> class Rectangle;
}
}

namespace Eigen 
{
    namespace internal 
    {
        template<typename _Scalar>
        struct traits<vc::types::Rectangle<_Scalar> > 
        {
            typedef _Scalar Scalar;
            typedef Matrix<Scalar,4,1> CoeffType;
        };
        
        template<typename _Scalar, int _Options>
        struct traits<Map<vc::types::Rectangle<_Scalar>, _Options> > : traits<vc::types::Rectangle<_Scalar> > 
        {
            typedef _Scalar Scalar;
            typedef Map<Matrix<Scalar,4,1>, _Options> CoeffType;
        };
        
        template<typename _Scalar, int _Options>
        struct traits<Map<const vc::types::Rectangle<_Scalar>, _Options> > : traits<const vc::types::Rectangle<_Scalar> > 
        {
            typedef _Scalar Scalar;
            typedef Map<const Matrix<Scalar,4,1>, _Options> CoeffType;
        };
    }
}

namespace vc
{

namespace types
{
    
template<typename Derived>
class RectangleBase
{
public:
    typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Derived>::CoeffType CoeffType;
    
    template<typename NewScalarType>
    EIGEN_DEVICE_FUNC inline Rectangle<NewScalarType> cast() const 
    {
        return Rectangle<NewScalarType>(coeff().template cast<NewScalarType>());
    }
    
    EIGEN_DEVICE_FUNC inline const Scalar& x1() const { return coeff()(0); }
    EIGEN_DEVICE_FUNC inline const Scalar& y1() const { return coeff()(1); }
    EIGEN_DEVICE_FUNC inline const Scalar& x2() const { return coeff()(2); }
    EIGEN_DEVICE_FUNC inline const Scalar& y2() const { return coeff()(3); }
    
    EIGEN_DEVICE_FUNC inline Scalar& x1() { return coeff()(0); }
    EIGEN_DEVICE_FUNC inline Scalar& y1() { return coeff()(1); }
    EIGEN_DEVICE_FUNC inline Scalar& x2() { return coeff()(2); }
    EIGEN_DEVICE_FUNC inline Scalar& y2() { return coeff()(3); }
    
    EIGEN_DEVICE_FUNC inline CoeffType& coeff() 
    {
        return static_cast<Derived*>(this)->coeff_nonconst();
    }
    
    EIGEN_DEVICE_FUNC inline const CoeffType& coeff() const
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
    
    EIGEN_DEVICE_FUNC inline Scalar width() const
    {
        return max(Scalar(0.0),x2() + Scalar(1.0) - x1());
    }
    
    EIGEN_DEVICE_FUNC inline Scalar height() const
    {
        return max(Scalar(0.0),y2() + Scalar(1.0) - y1());
    }
    
    EIGEN_DEVICE_FUNC inline Scalar area() const
    {
        return width() * height();
    }
    
    EIGEN_DEVICE_FUNC inline bool intersectsWith(const Rectangle<Scalar>& other) const
    {
        if(y2() < other.y1())   return false;
        if(y1() > other.y2())   return false;
        if(x2() < other.x1())   return false;
        if(x1() > other.x2())   return false;
        return true;
    }
    
    EIGEN_DEVICE_FUNC inline bool contains(const Rectangle<Scalar>& other) const
    {
        if(y1() >= other.y1()) return false;
        if(x1() >= other.x1()) return false;
        if(x2() <= other.x2()) return false;
        if(y2() <= other.y2()) return false;
        return true;
    }
    
    EIGEN_DEVICE_FUNC inline void insert(Scalar x, Scalar y)
    {
        x1() = min(x1(),x);
        x2() = max(x2(),x);
        y1() = min(y1(),y);
        y2() = max(y2(),y);
    }
    
    EIGEN_DEVICE_FUNC inline void insert(const Rectangle<Scalar>& d)
    {
        x1() = min(x1(),d.x1());
        x2() = max(x2(),d.x2());
        y1() = min(y1(),d.y1());
        y2() = max(y2(),d.y2());
    }
    
    EIGEN_DEVICE_FUNC inline Rectangle<Scalar> grow(Scalar r) const
    {
        Rectangle<Scalar> ret(*this);
        ret.x1() -= r;
        ret.x2() += r;
        ret.y1() -= r;
        ret.y2() += r;
        return ret;
    }
    
    EIGEN_DEVICE_FUNC inline Rectangle<Scalar> clamp(Scalar minx, Scalar miny, Scalar maxx, Scalar maxy) const
    {
        Rectangle<Scalar> ret(*this);
        ret.x1() = max(minx,ret.x1());
        ret.y1() = max(miny,ret.y1());
        ret.x2() = min(maxx,ret.x2());
        ret.y2() = min(maxy,ret.y2());
        return ret;
    }
    
    EIGEN_DEVICE_FUNC inline bool contains(Scalar x, Scalar y) const
    {
        return x1() <= x && x <= x2() && y1() <= y && y <= y2();
    }
    
    EIGEN_DEVICE_FUNC inline bool contains(const Eigen::Matrix<Scalar,2,1>& p) const
    {
        return contains(p[0],p[1]);
    }
    
    EIGEN_DEVICE_FUNC inline Eigen::Matrix<Scalar,2,1> center() const
    {
        return Eigen::Matrix<Scalar,2,1>((x2() + x1())/Scalar(2.0), (y2() + y1())/Scalar(2.0));
    }
    
#ifdef VISIONCORE_ENABLE_CEREAL
    template<typename Archive>
    void load(Archive & archive, std::uint32_t const version)
    {
        archive(cereal::make_nvp("X1", x1()));
        archive(cereal::make_nvp("Y1", y1()));
        archive(cereal::make_nvp("X2", x2()));
        archive(cereal::make_nvp("Y2", y2()));
    }
    
    template<typename Archive>
    void save(Archive & archive) const
    {
        archive(cereal::make_nvp("X1", x1()));
        archive(cereal::make_nvp("Y1", y1()));
        archive(cereal::make_nvp("X2", x2()));
        archive(cereal::make_nvp("Y2", y2()));
    }    
#endif // VISIONCORE_ENABLE_CEREAL    
};


template<typename _Scalar>    
class Rectangle : public RectangleBase<Rectangle<_Scalar>> 
{
    typedef RectangleBase<Rectangle<_Scalar>> Base;
public:
    typedef typename Eigen::internal::traits<Rectangle>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Rectangle>::CoeffType CoeffType;
    
    friend class vc::types::RectangleBase<Rectangle<_Scalar>>;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    EIGEN_DEVICE_FUNC inline Rectangle() 
    {
        coeff_nonconst() << std::numeric_limits<Scalar>::max(),
                            std::numeric_limits<Scalar>::max(),
                            std::numeric_limits<Scalar>::min(),
                            std::numeric_limits<Scalar>::min();
    }
    
    EIGEN_DEVICE_FUNC inline Rectangle(const CoeffType& ct) 
    {
        coeff_nonconst() = ct;
    }
    
    EIGEN_DEVICE_FUNC inline Rectangle(const Rectangle<Scalar>& other) : coeff_(other.coeff())
    {
    }
    
    template<typename OtherDerived> 
    EIGEN_DEVICE_FUNC inline Rectangle(const RectangleBase<OtherDerived>& other) : coeff_(other.coeff())
    {
    }
        
    EIGEN_DEVICE_FUNC inline Rectangle(Scalar vx1, Scalar vy1, Scalar vx2, Scalar vy2)
    {
        coeff_nonconst() << vx1 , vy1 , vx2 , vy2;
    }
    
protected:
    EIGEN_DEVICE_FUNC inline const CoeffType& coeff_const() const { return coeff_; }
    EIGEN_DEVICE_FUNC inline CoeffType& coeff_nonconst() { return coeff_; }
    
    CoeffType coeff_;
};

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const Rectangle<T>& p)
{
    os << "Rectangle(" << p.x1() << "," << p.y1() << " | " << p.x2() << "," << p.y2() <<  ")";
    return os;
}

}
    
}

namespace Eigen 
{
/**
 * Specialisation of Eigen::Map for Rectangle.
 */
template<typename _Scalar, int _Options>
class Map<vc::types::Rectangle<_Scalar>, _Options> : 
    public vc::types::RectangleBase<Map<vc::types::Rectangle<_Scalar>, _Options> > 
{
    typedef vc::types::RectangleBase<Map<vc::types::Rectangle<_Scalar>, _Options> > Base;
    
public:
    typedef typename Eigen::internal::traits<Map>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Map>::CoeffType CoeffType;
    
    friend class vc::types::RectangleBase<Map<vc::types::Rectangle<_Scalar>, _Options> >;
    
    EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
    
    EIGEN_DEVICE_FUNC inline Map(Scalar* coeffs) : coeff_(coeffs)
    {
    }
    
protected:
    EIGEN_DEVICE_FUNC inline const CoeffType& coeff_const() const { return coeff_; }
    EIGEN_DEVICE_FUNC inline CoeffType& coeff_nonconst() { return coeff_; }
    
    CoeffType coeff_;
};

/**
 * Specialisation of Eigen::Map for const Rectangle.
 */
template<typename _Scalar, int _Options>
class Map<const vc::types::Rectangle<_Scalar>, _Options> : 
    public vc::types::RectangleBase<Map<const vc::types::Rectangle<_Scalar>, _Options> > 
{
    typedef vc::types::RectangleBase<Map<const vc::types::Rectangle<_Scalar>, _Options> > Base;
    
public:
    typedef typename Eigen::internal::traits<Map>::Scalar Scalar;    
    typedef typename Eigen::internal::traits<Map>::CoeffType CoeffType;
    
    friend class vc::types::RectangleBase<Map<const vc::types::Rectangle<_Scalar>, _Options> >;
    
    EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
    
    EIGEN_DEVICE_FUNC inline Map(const Scalar* coeffs) : coeff_(coeffs)
    {
    }
    
protected:
    EIGEN_DEVICE_FUNC inline const CoeffType& coeff_const() const { return coeff_; }    
    
    const CoeffType coeff_;
};
    
}

#endif // VISIONCORE_TYPES_RECTANGLE_BOX
