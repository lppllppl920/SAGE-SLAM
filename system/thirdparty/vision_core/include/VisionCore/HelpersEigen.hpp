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
 * Bits and bobs missing from Eigen.
 * ****************************************************************************
 */

#ifndef VISIONCORE_EIGEN_MISSING_BITS_HPP
#define VISIONCORE_EIGEN_MISSING_BITS_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace Eigen
{

namespace numext
{

template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T atan2(const T& y, const T &x)
{
    EIGEN_USING_STD_MATH(atan2);
    return atan2(y,x);
}

#ifdef VISIONCORE_CUDA_COMPILER
template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float atan2(const float& y, const float &x) { return ::atan2f(y,x); }

template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double atan2(const double& y, const double &x) { return ::atan2(y,x); }
#endif // VISIONCORE_CUDA_COMPILER

}

}

#include <unsupported/Eigen/AutoDiff>

namespace Eigen
{

template<typename DerType>
inline const Eigen::AutoDiffScalar<EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(typename Eigen::internal::remove_all<DerType>::type,
                                                                          typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar, product)>
atan(const Eigen::AutoDiffScalar<DerType>& x)
{
    using namespace Eigen;
    EIGEN_UNUSED typedef typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar Scalar;
    using numext::atan;
    return Eigen::MakeAutoDiffScalar(atan(x.value()),x.derivatives() * ( Scalar(1) / (Scalar(1) + x.value() * x.value()) ));
}

// let's put ostreams here
template <typename _Scalar, int _AmbientDim, int _Options>
inline std::ostream& operator<<(std::ostream& os, const Hyperplane<_Scalar,_AmbientDim,_Options>& p)
{
    os << "Hyperplane(" << p.normal() << " , " << p.offset() << ")";
    return os;
}

#ifdef VISIONCORE_ENABLE_CEREAL
/**
 * Matrix
 */
template<typename Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void load(Archive & archive, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & m, std::uint32_t const version)
{
    for(int r = 0 ; r < m.rows() ; ++r)
    {
        for(int c = 0 ; c < m.cols() ; ++c)
        {
            archive(m(r,c));
        }
    }
}

template<typename Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void save(Archive & archive, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> const & m, std::uint32_t const version)
{
    for(int r = 0 ; r < m.rows() ; ++r)
    {
        for(int c = 0 ; c < m.cols() ; ++c)
        {
            archive(m(r,c));
        }
    }
}

/**
 * Quaternion
 */
template<typename Archive, typename _Scalar, int _Options>
void load(Archive & archive, Eigen::Quaternion<_Scalar, _Options> & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("X", m.x()));
    archive(cereal::make_nvp("Y", m.y()));
    archive(cereal::make_nvp("Z", m.z()));
    archive(cereal::make_nvp("W", m.w()));
}

template<typename Archive, typename _Scalar, int _Options>
void save(Archive & archive, Eigen::Quaternion<_Scalar, _Options> const & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("X", m.x()));
    archive(cereal::make_nvp("Y", m.y()));
    archive(cereal::make_nvp("Z", m.z()));
    archive(cereal::make_nvp("W", m.w()));
}

/**
 * AngleAxis
 */
template<typename Archive, typename _Scalar>
void load(Archive & archive, Eigen::AngleAxis<_Scalar> & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Angle", m.angle()));
    archive(cereal::make_nvp("Axis", m.axis()));
}

template<typename Archive, typename _Scalar>
void save(Archive & archive, Eigen::AngleAxis<_Scalar> const & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Angle", m.angle()));
    archive(cereal::make_nvp("Axis", m.axis()));
}

/**
 * Rotation2D
 */
template<typename Archive, typename _Scalar>
void load(Archive & archive, Eigen::Rotation2D<_Scalar> & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Angle", m.angle()));
}

template<typename Archive, typename _Scalar>
void save(Archive & archive, Eigen::Rotation2D<_Scalar> const & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Angle", m.angle()));
}

/**
 * AutoDiffScalar
 */
template<typename Archive, typename ADT>
void load(Archive & archive, Eigen::AutoDiffScalar<ADT> & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Value", m.value()));
    archive(cereal::make_nvp("Deriviatives", m.derivatives()));
}

template<typename Archive, typename ADT>
void save(Archive & archive, Eigen::AutoDiffScalar<ADT> const & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Value", m.value()));
    archive(cereal::make_nvp("Deriviatives", m.derivatives()));
}

#endif // VISIONCORE_ENABLE_CEREAL

}

#endif // VISIONCORE_EIGEN_MISSING_BITS_HPP
