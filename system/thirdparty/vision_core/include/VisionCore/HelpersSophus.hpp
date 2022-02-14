/**
 * 
 * Core Libraries.
 * Sophus Interpolations & other missing bits.
 * 
 * Copyright (c) Robert Lukierski 2016. All rights reserved.
 * Author: Robert Lukierski.
 * 
 */


#ifndef VISIONCORE_SOPHUS_MISSINGBITS_HPP
#define VISIONCORE_SOPHUS_MISSINGBITS_HPP

#include <VisionCore/Platform.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <sophus/so2.hpp>
#include <sophus/so3.hpp>
#include <sophus/se2.hpp>
#include <sophus/se3.hpp>
#include <sophus/rxso3.hpp>
#include <sophus/sim3.hpp>

/**
 * NOTE: Don't use <sophus/interpolate.hpp> for SE groups. It's wrong, see page 41:
 * https://www.cvl.isy.liu.se/education/graduate/geometry-for-computer-vision-2014/geometry2014/lecture7.pdf
 */

namespace Sophus
{

// SO2
template<typename T>
EIGEN_DEVICE_FUNC static inline Sophus::SO2<T> interpolateLinear(const Sophus::SO2<T>& t0, const Sophus::SO2<T>& t1, T ratio)
{
    return Sophus::SO2<T>(t0 * Sophus::SO2<T>::exp(ratio * ( t0.inverse() * t1 ).log() ));
}
    
// SE2
template<typename T>
EIGEN_DEVICE_FUNC static inline Sophus::SE2<T> interpolateLinear(const Sophus::SE2<T>& t0, const Sophus::SE2<T>& t1, T ratio)
{
    return Sophus::SE2<T>(t0.so2() * Sophus::SO2<T>::exp(ratio * ( t0.so2().inverse() * t1.so2() ).log() ), 
                          t0.translation() + ratio * (t1.translation() - t0.translation()));
}    

// SO3
template<typename T>
EIGEN_DEVICE_FUNC static inline Sophus::SO3<T> interpolateLinear(const Sophus::SO3<T>& t0, const Sophus::SO3<T>& t1, T ratio)
{
    return Sophus::SO3<T>(t0 * Sophus::SO3<T>::exp(ratio * ( t0.inverse() * t1 ).log() ));
}

// SE3
template<typename T>
EIGEN_DEVICE_FUNC static inline Sophus::SE3<T> interpolateLinear(const Sophus::SE3<T>& t0, const Sophus::SE3<T>& t1, T ratio)
{
    return Sophus::SE3<T>(t0.so3() * Sophus::SO3<T>::exp(ratio * ( t0.so3().inverse() * t1.so3() ).log() ), 
                          t0.translation() + ratio * (t1.translation() - t0.translation()));
}
    
template<typename T>
struct B4SplineGenerator 
{ 

    EIGEN_DEVICE_FUNC static inline void get(const T& u, T& b1, T& b2, T& b3)
    {
        const T u2 = u*u;
        const T u3 = u*u*u;
        
        b1 = (u3 - T(3.0) * u2 + T(3.0) * u + T(5.0)) / T(6.0);
        b2 = (T(-2.0) * u3 + T(3.0) * u2 + T(3.0) * u + T(1.0)) / T(6.0);
        b3 = u3 / T(6.0);
    }

#if 0
    EIGEN_DEVICE_FUNC static inline void get(const T& u, T& b1, T& b2, T& b3)
    {
        Eigen::Matrix<T,4,1> v(T(1.0),u,u*u,u*u*u);
        Eigen::Matrix<T,4,4> m;
        m << T(6.0) , T(0.0) , T(0.0) , T(0.0),
             T(5.0) , T(3.0) , T(-3.0), T(1.0),
             T(1.0) , T(3.0) , T(3.0), T(-2.0),
             T(0.0) , T(0.0) , T(0.0) , T(1.0);
        
        const Eigen::Matrix<T,4,1> uvec = (m * v) / T(6.0);
        b1 = uvec(1);
        b2 = uvec(2);
        b3 = uvec(3);
    }
#endif
};

// B4-Spline SO2
template<typename T>
EIGEN_DEVICE_FUNC static inline Sophus::SO2<T> interpolateB4Spline(const Sophus::SO2<T>& tm1, 
                                                                   const Sophus::SO2<T>& t0, 
                                                                   const Sophus::SO2<T>& t1, 
                                                                   const Sophus::SO2<T>& t2, 
                                                                   const T& u)
{
    T b1, b2, b3;
    
    B4SplineGenerator<T>::get(u,b1,b2,b3);
    
    return tm1 * Sophus::SO2<T>::exp(b1 * ( tm1.inverse() * t0 ).log() + 
                                          b2 * ( t0.inverse()  * t1 ).log() +
                                          b3 * ( t1.inverse()  * t2 ).log() );
}

// B4-Spline SE2
template<typename T>
EIGEN_DEVICE_FUNC static inline Sophus::SE2<T> interpolateB4Spline(const Sophus::SE2<T>& tm1, 
                                                                   const Sophus::SE2<T>& t0, 
                                                                   const Sophus::SE2<T>& t1, 
                                                                   const Sophus::SE2<T>& t2, 
                                                                   const T& u)
{
    T b1, b2, b3;
    
    B4SplineGenerator<T>::get(u,b1,b2,b3);
    
    const Sophus::SO2<T> rot_final = tm1.so2() * Sophus::SO2<T>::exp(b1 * ( tm1.so2().inverse() * t0.so2() ).log() + 
                                                                               b2 * ( t0.so2().inverse()  * t1.so2() ).log() +
                                                                               b3 * ( t1.so2().inverse()  * t2.so2() ).log() );
    
    const typename Sophus::SE2<T>::Point tr_final = tm1.translation() + (b1 * (t0.translation() - tm1.translation())) + 
                                                                             (b2 * (t1.translation() - t0.translation())) + 
                                                                             (b3 * (t2.translation() - t1.translation()));
    
    return Sophus::SE2<T>(rot_final, tr_final);
}

// B4-Spline SO3
template<typename T>
EIGEN_DEVICE_FUNC static inline Sophus::SO3<T> interpolateB4Spline(const Sophus::SO3<T>& tm1, 
                                                                   const Sophus::SO3<T>& t0, 
                                                                   const Sophus::SO3<T>& t1, 
                                                                   const Sophus::SO3<T>& t2, 
                                                                   const T& u)
{
    T b1, b2, b3;
    
    B4SplineGenerator<T>::get(u,b1,b2,b3);
    
    return tm1 * Sophus::SO3<T>::exp(b1 * ( tm1.inverse() * t0 ).log() + 
                                          b2 * ( t0.inverse()  * t1 ).log() +
                                          b3 * ( t1.inverse()  * t2 ).log() );
}

// B4-Spline SE3
template<typename T>
EIGEN_DEVICE_FUNC static inline Sophus::SE3<T> interpolateB4Spline(const Sophus::SE3<T>& tm1, 
                                                                   const Sophus::SE3<T>& t0, 
                                                                   const Sophus::SE3<T>& t1, 
                                                                   const Sophus::SE3<T>& t2, 
                                                                   const T& u)
{
    T b1, b2, b3;
    
    B4SplineGenerator<T>::get(u,b1,b2,b3);
    
    const Sophus::SO3<T> rot_final = tm1.so3() * Sophus::SO3<T>::exp(b1 * ( tm1.so3().inverse() * t0.so3() ).log() + 
                                                                               b2 * ( t0.so3().inverse()  * t1.so3() ).log() +
                                                                               b3 * ( t1.so3().inverse()  * t2.so3() ).log() );
    
    const typename Sophus::SE3<T>::Point tr_final = tm1.translation() + (b1 * (t0.translation() - tm1.translation())) + 
                                                                             (b2 * (t1.translation() - t0.translation())) + 
                                                                             (b3 * (t2.translation() - t1.translation()));
    
    return Sophus::SE3<T>(rot_final, tr_final);
}

// let's put ostreams here
template<typename Derived>
inline std::ostream& operator<<(std::ostream& os, const SO2Base<Derived>& p)
{
    os << "(" << p.log() << ")"; 
    return os;
}

template<typename Derived>
inline std::ostream& operator<<(std::ostream& os, const SO3Base<Derived>& p)
{
    os << "(" << p.unit_quaternion().x() << "," << p.unit_quaternion().y() << "," 
       << p.unit_quaternion().z() << "|" << p.unit_quaternion().w() << ")"; 
    return os;
}

template<typename Derived>
inline std::ostream& operator<<(std::ostream& os, const SE2Base<Derived>& p)
{
    os << "[t = " << p.translation()(0) << "," << p.translation()(1) << " | r = " << p.so2() << ")";
    return os;
}

template<typename Derived>
inline std::ostream& operator<<(std::ostream& os, const SE3Base<Derived>& p)
{
    os << "[t = " << p.translation()(0) << "," << p.translation()(1) << "," << p.translation()(2) << " | r = " << p.so3() << ")";
    return os;
}

#ifdef VISIONCORE_ENABLE_CEREAL

/**
 * SO2
 */    
template<typename Archive, typename Derived>
void load(Archive & archive, SO2Base<Derived>& m, std::uint32_t const version)
{
    typename SO2Base<Derived>::Point cplx;
    archive(cplx);
    m.setComplex(cplx);
}

template<typename Archive, typename Derived>
void save(Archive & archive, SO2Base<Derived> const & m, std::uint32_t const version)
{
    archive(m.unit_complex());
}

/**
 * SO3
 */    
template<typename Archive, typename Derived>
void load(Archive & archive, SO3Base<Derived>& m, std::uint32_t const version)
{
    Eigen::Quaternion<typename SO3Base<Derived>::Scalar> quaternion;
    archive(cereal::make_nvp("Quaternion", quaternion));
    m.setQuaternion(quaternion);
}

template<typename Archive, typename Derived>
void save(Archive & archive, SO3Base<Derived> const & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Quaternion", m.unit_quaternion()));
}

/**
 * SE2
 */    
template<typename Archive, typename Derived>
void load(Archive & archive, SE2Base<Derived>& m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Translation", m.translation()));
    archive(cereal::make_nvp("Rotation", m.so2()));
}

template<typename Archive, typename Derived>
void save(Archive & archive, SE2Base<Derived> const & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Translation", m.translation()));
    archive(cereal::make_nvp("Rotation", m.so2()));
}

/**
 * SE3
 */    
template<typename Archive, typename Derived>
void load(Archive & archive, SE3Base<Derived>& m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Translation", m.translation()));
    archive(cereal::make_nvp("Rotation", m.so3()));
}

template<typename Archive, typename Derived>
void save(Archive & archive, SE3Base<Derived> const & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Translation", m.translation()));
    archive(cereal::make_nvp("Rotation", m.so3()));
}

/**
 * RxSO3
 */    
template<typename Archive, typename Derived>
void load(Archive & archive, RxSO3Base<Derived>& m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Quaternion", m.quaternion()));
}

template<typename Archive, typename Derived>
void save(Archive & archive, RxSO3Base<Derived> const & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Quaternion", m.quaternion()));
}

/**
 * Sim3
 */    
template<typename Archive, typename Derived>
void load(Archive & archive, Sim3Base<Derived>& m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Translation", m.translation()));
    archive(cereal::make_nvp("Rotation", m.rxso3()));
}

template<typename Archive, typename Derived>
void save(Archive & archive, Sim3Base<Derived> const & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Translation", m.translation()));
    archive(cereal::make_nvp("Rotation", m.rxso3()));
}

#endif // VISIONCORE_ENABLE_CEREAL

/**
 * Stuff that Hauke removed.
 * 
 * \param alpha1 rotation around x-axis
 * \param alpha2 rotation around y-axis
 * \param alpha3 rotation around z-axis
 *
 * Since rotations in 3D do not commute, the order of the individual rotations
 * matter. Here, the following convention is used. We calculate a SO3 member
 * corresponding to the rotation matrix \f$ R \f$ such
 * that \f$ R=\exp\left(\begin{array}{c}\alpha_1\\ 0\\ 0\end{array}\right)
 *    \cdot   \exp\left(\begin{array}{c}0\\ \alpha_2\\ 0\end{array}\right)
 *    \cdot   \exp\left(\begin{array}{c}0\\ 0\\ \alpha_3\end{array}\right)\f$.
 */
template<typename T>
EIGEN_DEVICE_FUNC static inline SO3<T> fromEulerAngles(const T alpha1, const T alpha2, const T alpha3)
{
    typedef typename SO3<T>::Tangent Tangent;
    const static T zero = static_cast<T>(0);
    
    return SO3<T>((SO3<T>::exp(Tangent(alpha1, zero, zero)) *
                   SO3<T>::exp(Tangent(zero, alpha2, zero)) *
                   SO3<T>::exp(Tangent(zero, zero, alpha3)))
                 );
}

}

#endif // VISIONCORE_SOPHUS_MISSINGBITS_HPP
