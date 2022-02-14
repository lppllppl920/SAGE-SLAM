/**
 * ****************************************************************************
 * Copyright (c) 2018, Robert Lukierski.
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
 * Helpers for magic Auto-Diff types.
 * ****************************************************************************
 */

#ifndef VISIONCORE_HELPERSAUTODIFF_HPP
#define VISIONCORE_HELPERSAUTODIFF_HPP

namespace vc
{

// A traits class to make it easier to work with mixed auto / numeric diff.
template<typename T>
struct ADTraits 
{
    typedef T Scalar;
    static constexpr std::size_t DerDimension = 0;
    EIGEN_DEVICE_FUNC inline static constexpr bool isScalar() { return true; }
    EIGEN_DEVICE_FUNC inline static Scalar getScalar(const Scalar& t)  { return t; }
    EIGEN_DEVICE_FUNC static void setScalar(const Scalar& scalar, Scalar* t)  { *t = scalar; }
    EIGEN_DEVICE_FUNC inline static Scalar getDerivative(const Scalar& t, std::size_t n) { return 0.0f; }
};

template<typename ADT>
struct ADTraits<Eigen::AutoDiffScalar<ADT>> 
{
    typedef typename Eigen::AutoDiffScalar<ADT>::Scalar Scalar;
    static constexpr std::size_t DerDimension = Eigen::AutoDiffScalar<ADT>::DerType::RowsAtCompileTime;
    EIGEN_DEVICE_FUNC inline static constexpr bool isScalar() { return false; }
    EIGEN_DEVICE_FUNC inline static Scalar getScalar(const Eigen::AutoDiffScalar<ADT>& t) { return t.value(); }
    EIGEN_DEVICE_FUNC static void setScalar(const Scalar& scalar, Eigen::AutoDiffScalar<ADT>* t)  { t->value() = scalar; }
    EIGEN_DEVICE_FUNC inline static Scalar getDerivative(const Eigen::AutoDiffScalar<ADT>& t, std::size_t n) { return t.derivatives()(n); }
};

#ifdef VISIONCORE_HAVE_CERES
template<typename T, int N>
struct ADTraits<ceres::Jet<T, N> > 
{
    typedef T Scalar;
    static constexpr std::size_t DerDimension = N;
    EIGEN_DEVICE_FUNC inline static constexpr bool isScalar() { return false; }
    EIGEN_DEVICE_FUNC inline static Scalar getScalar(const ceres::Jet<T, N>& t) { return t.a; }
    EIGEN_DEVICE_FUNC static void setScalar(const Scalar& scalar, ceres::Jet<T, N>* t)  { t->a = scalar; }
    EIGEN_DEVICE_FUNC inline static Scalar getDerivative(const ceres::Jet<T,N>& t, std::size_t n) { return t.v(n); }
};
#endif // VISIONCORE_HAVE_CERES

// Chain rule
template<typename FunctionType, int kNumArgs, typename ArgumentType>
struct Chain 
{
    EIGEN_DEVICE_FUNC inline static ArgumentType Rule(const FunctionType &f, const FunctionType dfdx[kNumArgs], 
                                                      const ArgumentType x[kNumArgs]) 
    {
        // In the default case of scalars, there's nothing to do since there are no
        // derivatives to propagate.
        (void) dfdx;  // Ignored.
        (void) x;  // Ignored.
        return f;
    }
};

template<typename FunctionType, int kNumArgs, typename ADT>
struct Chain<FunctionType, kNumArgs, Eigen::AutoDiffScalar<ADT> > 
{
    EIGEN_DEVICE_FUNC inline static Eigen::AutoDiffScalar<ADT> Rule(const FunctionType &f, const FunctionType dfdx[kNumArgs], const Eigen::AutoDiffScalar<ADT> x[kNumArgs]) 
    {
        // x is itself a function of another variable ("z"); what this function
        // needs to return is "f", but with the derivative with respect to z
        // attached to the jet. So combine the derivative part of x's jets to form
        // a Jacobian matrix between x and z (i.e. dx/dz).
        Eigen::Matrix<typename Eigen::AutoDiffScalar<ADT>::Scalar, kNumArgs, ADT::RowsAtCompileTime> dxdz;
        for (int i = 0; i < kNumArgs; ++i) 
        {
            dxdz.row(i) = x[i].derivatives().transpose();
        }
        
        // Map the input gradient dfdx into an Eigen row vector.
        Eigen::Map<const Eigen::Matrix<FunctionType, 1, kNumArgs> >
        vector_dfdx(dfdx, 1, kNumArgs);
        
        // Now apply the chain rule to obtain df/dz. Combine the derivative with
        // the scalar part to obtain f with full derivative information.
        Eigen::AutoDiffScalar<ADT> jet_f;
        jet_f.value() = f;
        jet_f.derivatives() = vector_dfdx.template cast<typename Eigen::AutoDiffScalar<ADT>::Scalar>() * dxdz;  // Also known as dfdz.
        return jet_f;
    }
};
#ifdef VISIONCORE_HAVE_CERES
template<typename FunctionType, int kNumArgs, typename T, int N>
struct Chain<FunctionType, kNumArgs, ceres::Jet<T, N> > 
{
    EIGEN_DEVICE_FUNC inline static ceres::Jet<T, N> Rule(const FunctionType &f, const FunctionType dfdx[kNumArgs], 
                                                          const ceres::Jet<T, N> x[kNumArgs]) 
    {
        // x is itself a function of another variable ("z"); what this function
        // needs to return is "f", but with the derivative with respect to z
        // attached to the jet. So combine the derivative part of x's jets to form
        // a Jacobian matrix between x and z (i.e. dx/dz).
        Eigen::Matrix<T, kNumArgs, N> dxdz;
        for (int i = 0; i < kNumArgs; ++i) 
        {
            dxdz.row(i) = x[i].v.transpose();
        }
        
        // Map the input gradient dfdx into an Eigen row vector.
        Eigen::Map<const Eigen::Matrix<FunctionType, 1, kNumArgs> >
        vector_dfdx(dfdx, 1, kNumArgs);
        
        // Now apply the chain rule to obtain df/dz. Combine the derivative with
        // the scalar part to obtain f with full derivative information.
        ceres::Jet<T, N> jet_f;
        jet_f.a = f;
        jet_f.v = vector_dfdx.template cast<T>() * dxdz;  // Also known as dfdz.
        return jet_f;
    }
};
#endif // VISIONCORE_HAVE_CERES
}

#endif // VISIONCORE_HELPERSAUTODIFF_HPP
