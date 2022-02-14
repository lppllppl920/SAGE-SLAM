/**
 * ****************************************************************************
 * Copyright (c) 2016, Robert Lukierski.
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
 * Fourier Transform.
 * ****************************************************************************
 */

#ifndef VISIONCORE_MATH_FOURIER_HPP
#define VISIONCORE_MATH_FOURIER_HPP

#include <memory>
#include <complex>
#include <type_traits>

#include <VisionCore/Buffers/Buffer1D.hpp>
#include <VisionCore/Buffers/Buffer2D.hpp>
#include <VisionCore/Buffers/Image2D.hpp>

namespace vc
{

namespace math
{

namespace internal
{
    template<typename T>
    struct FFTTypeTraits { };

    template<>
    struct FFTTypeTraits<float>
    {
        typedef float BaseType;
        static constexpr bool IsComplex = false;
        static constexpr unsigned int Fields = 1;
        static constexpr bool IsEigen = false;
    };

    template<>
    struct FFTTypeTraits<double>
    {
        typedef float BaseType;
        static constexpr bool IsComplex = false;
        static constexpr unsigned int Fields = 1;
        static constexpr bool IsEigen = false;
    };

    template<typename T_REAL>
    struct FFTTypeTraits<Eigen::Matrix<T_REAL,2,1>>
    {
        typedef T_REAL BaseType;
        static constexpr bool IsComplex = true;
        static constexpr unsigned int Fields = 2;
        static constexpr bool IsEigen = true;
    };

    template<typename T_REAL>
    struct FFTTypeTraits<std::complex<T_REAL>>
    {
        typedef T_REAL BaseType;
        static constexpr bool IsComplex = true;
        static constexpr unsigned int Fields = 2;
        static constexpr bool IsEigen = false;
    };

    template<typename T_COMPLEX, typename T_REAL>
    struct complexOps { };

    // interfacing with Eigen::Vector2 (used as complex)
    template<typename T_REAL>
    struct complexOps<Eigen::Matrix<T_REAL,2,1>,T_REAL>
    {
        typedef Eigen::Matrix<T_REAL,2,1> ComplexT;

        EIGEN_DEVICE_FUNC static inline T_REAL getReal(const ComplexT& cpx)
        {
            return cpx(0);
        }

        EIGEN_DEVICE_FUNC static inline T_REAL getImag(const ComplexT& cpx)
        {
            return cpx(1);
        }

        EIGEN_DEVICE_FUNC static inline ComplexT makeComplex(const T_REAL& re, const T_REAL& im)
        {
            return ComplexT(re, im);
        }

        EIGEN_DEVICE_FUNC static inline ComplexT conjugate(const ComplexT& cpx)
        {
            return ComplexT(cpx(0), -cpx(1));
        }

        EIGEN_DEVICE_FUNC static inline ComplexT multiply(const ComplexT& cpx1, const ComplexT& cpx2)
        {
            return ComplexT(cpx1(0) * cpx2(0) - cpx1(1) * cpx2(1), cpx1(1) * cpx2(0) + cpx1(0) * cpx2(1));
        }

        EIGEN_DEVICE_FUNC static inline ComplexT multiply(const ComplexT& cpx1, const T_REAL& scalar)
        {
            return multiply(cpx1, ComplexT(scalar, 0.0f));
        }

        EIGEN_DEVICE_FUNC static inline ComplexT divide(const ComplexT& cpx1, const ComplexT& cpx2)
        {
            return ComplexT( (cpx1(0) * cpx2(0) + cpx1(1) * cpx2(1)) / (cpx2(0) * cpx2(0) + cpx2(1) * cpx2(1)) , (cpx1(1) * cpx2(0) - cpx1(0) * cpx2(1)) / (cpx2(0) * cpx2(0) + cpx2(1) * cpx2(1)) );
        }

        EIGEN_DEVICE_FUNC static inline ComplexT divide(const ComplexT& cpx1, const T_REAL& scalar)
        {
            return divide(cpx1, ComplexT(scalar, 0.0f));
        }

        EIGEN_DEVICE_FUNC static inline T_REAL norm(const ComplexT& cpx)
        {
            return sqrt(cpx(0) * cpx(0) + cpx(1) * cpx(1));
        }
    };

    // interfacing with std::complex
    template<typename T_REAL>
    struct complexOps<std::complex<T_REAL>,T_REAL>
    {
        typedef std::complex<T_REAL> ComplexT;

        EIGEN_DEVICE_FUNC static inline T_REAL getReal(const ComplexT& cpx)
        {
            return cpx.real();
        }

        EIGEN_DEVICE_FUNC static inline T_REAL getImag(const ComplexT& cpx)
        {
            return cpx.imag();
        }

        EIGEN_DEVICE_FUNC static inline ComplexT makeComplex(const T_REAL& re, const T_REAL& im)
        {
            return ComplexT(re, im);
        }

        EIGEN_DEVICE_FUNC static inline ComplexT conjugate(const ComplexT& cpx)
        {
            return ComplexT(cpx.real(),-cpx.imag());
        }

        EIGEN_DEVICE_FUNC static inline ComplexT multiply(const ComplexT& cpx1, const ComplexT& cpx2)
        {
            //return cpx1 * cpx2;
            return ComplexT(cpx1.real() * cpx2.real() - cpx1.imag() * cpx2.imag(), cpx1.real() * cpx2.imag() + cpx1.imag() * cpx2.real());
        }
        EIGEN_DEVICE_FUNC static inline ComplexT multiply(const ComplexT& cpx1, const T_REAL& scalar)
        {
            return multiply(cpx1, ComplexT(scalar));
        }

        EIGEN_DEVICE_FUNC static inline ComplexT divide(const ComplexT& cpx1, const ComplexT& cpx2)
        {
            //return cpx1 / cpx2;
            return ComplexT( (cpx1.real() * cpx2.real() + cpx1.imag() * cpx2.imag()) / (cpx2.real() * cpx2.real() + cpx2.imag() * cpx2.imag()) , (cpx1.imag() * cpx2.real() - cpx1.real() * cpx2.imag()) / (cpx2.real() * cpx2.real() + cpx2.imag() * cpx2.imag()));
        }

        EIGEN_DEVICE_FUNC static inline ComplexT divide(const ComplexT& cpx1, const T_REAL& scalar)
        {
            return divide(cpx1, ComplexT(scalar));
        }

        EIGEN_DEVICE_FUNC static inline T_REAL norm(const ComplexT& cpx)
        {
            return sqrt(cpx.real() * cpx.real() + cpx.imag() * cpx.imag());
        }
    };

    template<typename T_COMPLEX>
    EIGEN_DEVICE_FUNC static inline T_COMPLEX crossPowerSpectrum(const T_COMPLEX& v1, const T_COMPLEX& v2)
    {
        typedef typename math::internal::FFTTypeTraits<T_COMPLEX>::BaseType T_REAL;

        const T_COMPLEX conj = math::internal::complexOps<T_COMPLEX, T_REAL>::conjugate(v2);
        const T_COMPLEX num = math::internal::complexOps<T_COMPLEX, T_REAL>::multiply(v1, conj);
        const T_REAL denom = math::internal::complexOps<T_COMPLEX, T_REAL>::norm(math::internal::complexOps<T_COMPLEX, T_REAL>::multiply(v1, v2));
        return math::internal::complexOps<T_COMPLEX, T_REAL>::divide(num, denom);
    }
}

/**
 * Versatile 1D/2D Fast Fourier Transform.
 *
 * Performed transforms:
 *
 * R2C: // forward
 *  float -> float2 (GPU)
 *  float -> Eigen::Vector2f (GPU, CPU)
 *  double -> Eigen::Vector2d (CPU)
 *  float -> std::complex<float> (GPU, CPU)
 *  double -> std::complex<float> (CPU)
 *
 * C2R: // inverse
 *  float2 -> float (GPU)
 *  Eigen::Vector2f -> float (GPU, CPU)
 *  Eigen::Vector2d -> double (CPU)
 *  std::complex<float> -> float (GPU, CPU)
 *  std::complex<double> -> double (CPU)
 *
 * C2C: // forward & inverse
 * float2 (GPU)
 * Eigen::Vector2f, std::complex<float> (GPU, CPU)
 * Eigen::Vector2d, std::complex<double> (CPU)
 *
 */

struct PersistentFFT
{
    virtual ~PersistentFFT() { }
    virtual void execute() = 0;
};

template<typename T_INPUT, typename T_OUTPUT, typename Target>
void fft(int npoint, const Buffer1DView<T_INPUT, Target>& buf_in,
         Buffer1DView<T_OUTPUT, Target>& buf_out, bool forward = true);

template<typename T_INPUT, typename T_OUTPUT, typename Target>
void fft(const Buffer2DView<T_INPUT, Target>& buf_in,
         Buffer2DView<T_OUTPUT, Target>& buf_out, bool forward = true);

template<typename T_INPUT, typename T_OUTPUT, typename Target>
std::unique_ptr<PersistentFFT> makeFFT(int npoint, const Buffer1DView<T_INPUT, Target>& buf_in,
                                       Buffer1DView<T_OUTPUT, Target>& buf_out, bool forward = true);

template<typename T_INPUT, typename T_OUTPUT, typename Target>
std::unique_ptr<PersistentFFT> makeFFT(const Buffer2DView<T_INPUT, Target>& buf_in,
                                       Buffer2DView<T_OUTPUT, Target>& buf_out, bool forward = true);

template<typename T_COMPLEX, typename Target>
void splitComplex(const Buffer1DView<T_COMPLEX, Target>& buf_in,
                  Buffer1DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_real,
                  Buffer1DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_imag);

template<typename T_COMPLEX, typename Target>
void splitComplex(const Buffer2DView<T_COMPLEX, Target>& buf_in,
                  Buffer2DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_real,
                  Buffer2DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_imag);

template<typename T_COMPLEX, typename Target>
void joinComplex(const Buffer1DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_real,
                 const Buffer1DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_imag,
                 Buffer1DView<T_COMPLEX, Target>& buf_out);

template<typename T_COMPLEX, typename Target>
void joinComplex(const Buffer2DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_real,
                 const Buffer2DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_imag,
                 Buffer2DView<T_COMPLEX, Target>& buf_out);

template<typename T_COMPLEX, typename Target>
void magnitude(const Buffer1DView<T_COMPLEX, Target>& buf_in,
               Buffer1DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_out);

template<typename T_COMPLEX, typename Target>
void magnitude(const Buffer2DView<T_COMPLEX, Target>& buf_in,
               Buffer2DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_out);

template<typename T_COMPLEX, typename Target>
void phase(const Buffer1DView<T_COMPLEX, Target>& buf_in,
           Buffer1DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_out);

template<typename T_COMPLEX, typename Target>
void phase(const Buffer2DView<T_COMPLEX, Target>& buf_in,
           Buffer2DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_out);

template<typename T_COMPLEX, typename Target>
void convertToComplex(const Buffer1DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_in,
                      Buffer1DView<T_COMPLEX, Target>& buf_out);

template<typename T_COMPLEX, typename Target>
void convertToComplex(const Buffer2DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_in,
                      Buffer2DView<T_COMPLEX, Target>& buf_out);

template<typename T_COMPLEX, typename Target>
void calculateCrossPowerSpectrum(const Buffer1DView<T_COMPLEX, Target>& buf_fft1,
                                 const Buffer1DView<T_COMPLEX, Target>& buf_fft2,
                                 Buffer1DView<T_COMPLEX, Target>& buf_fft_out);

template<typename T_COMPLEX, typename Target>
void calculateCrossPowerSpectrum(const Buffer2DView<T_COMPLEX, Target>& buf_fft1,
                                 const Buffer2DView<T_COMPLEX, Target>& buf_fft2,
                                 Buffer2DView<T_COMPLEX, Target>& buf_fft_out);

}

}


#endif // VISIONCORE_MATH_FOURIER_HPP
