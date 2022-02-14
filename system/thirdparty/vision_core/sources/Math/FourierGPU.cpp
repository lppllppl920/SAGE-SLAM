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

#include <VisionCore/Math/Fourier.hpp>

#include <VisionCore/LaunchUtils.hpp>
#include <VisionCore/CUDAException.hpp>

#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>

#include <cufft.h>

template<>
struct vc::math::internal::FFTTypeTraits<cufftComplex>
{
    typedef float BaseType;
    static constexpr bool IsComplex = true;
    static constexpr unsigned int Fields = 2;
    static constexpr bool IsEigen = false;
};

template<typename T_INPUT, typename T_OUTPUT>
struct FFTTransformType { };

// R2C

template<>
struct FFTTransformType<float, cufftComplex>
{
    static constexpr bool ForwardFFT = true;
    static constexpr cufftType CUDAType = CUFFT_R2C;
    static constexpr char const* Name = "R2C";

    typedef cufftReal       InputVarType;
    typedef cufftComplex    OutputVarType;
};

template<>
struct FFTTransformType<float, Eigen::Vector2f>
{
    static constexpr bool ForwardFFT = true;
    static constexpr cufftType CUDAType = CUFFT_R2C;
    static constexpr char const* Name = "R2C";

    typedef cufftReal       InputVarType;
    typedef cufftComplex    OutputVarType;
};

template<>
struct FFTTransformType<float, std::complex<float>>
{
    static constexpr bool ForwardFFT = true;
    static constexpr cufftType CUDAType = CUFFT_R2C;
    static constexpr char const* Name = "R2C";

    typedef cufftReal       InputVarType;
    typedef cufftComplex    OutputVarType;
};

// C2R

template<>
struct FFTTransformType<cufftComplex, float>
{
    static constexpr bool ForwardFFT = false;
    static constexpr cufftType CUDAType = CUFFT_C2R;
    static constexpr char const* Name = "C2R";

    typedef cufftComplex    InputVarType;
    typedef cufftReal   OutputVarType;
};

template<>
struct FFTTransformType<Eigen::Vector2f, float>
{
    static constexpr bool ForwardFFT = false;
    static constexpr cufftType CUDAType = CUFFT_C2R;
    static constexpr char const* Name = "C2R";

    typedef cufftComplex    InputVarType;
    typedef cufftReal       OutputVarType;
};


template<>
struct FFTTransformType<std::complex<float>, float>
{
    static constexpr bool ForwardFFT = false;
    static constexpr cufftType CUDAType = CUFFT_C2R;
    static constexpr char const* Name = "C2R";

    typedef cufftComplex    InputVarType;
    typedef cufftReal       OutputVarType;
};

// C2C

template<>
struct FFTTransformType<cufftComplex, cufftComplex>
{
    static constexpr bool ForwardFFT = true;
    static constexpr cufftType CUDAType = CUFFT_C2C;
    static constexpr char const* Name = "C2C";

    typedef cufftComplex    InputVarType;
    typedef cufftComplex    OutputVarType;
};

template<>
struct FFTTransformType<Eigen::Vector2f, Eigen::Vector2f>
{
    static constexpr bool ForwardFFT = true;
    static constexpr cufftType CUDAType = CUFFT_C2C;
    static constexpr char const* Name = "C2C";

    typedef cufftComplex    InputVarType;
    typedef cufftComplex    OutputVarType;
};

template<>
struct FFTTransformType<std::complex<float>, std::complex<float>>
{
    static constexpr bool ForwardFFT = true;
    static constexpr cufftType CUDAType = CUFFT_C2C;
    static constexpr char const* Name = "C2C";

    typedef cufftComplex    InputVarType;
    typedef cufftComplex    OutputVarType;
};

template<>
struct FFTTransformType<cufftComplex, Eigen::Vector2f>
{
    static constexpr bool ForwardFFT = true;
    static constexpr cufftType CUDAType = CUFFT_C2C;
    static constexpr char const* Name = "C2C";

    typedef cufftComplex    InputVarType;
    typedef cufftComplex    OutputVarType;
};

template<>
struct FFTTransformType<cufftComplex, std::complex<float>>
{
    static constexpr bool ForwardFFT = true;
    static constexpr cufftType CUDAType = CUFFT_C2C;
    static constexpr char const* Name = "C2C";

    typedef cufftComplex    InputVarType;
    typedef cufftComplex    OutputVarType;
};

template<>
struct FFTTransformType<Eigen::Vector2f, cufftComplex>
{
    static constexpr bool ForwardFFT = true;
    static constexpr cufftType CUDAType = CUFFT_C2C;
    static constexpr char const* Name = "C2C";

    typedef cufftComplex    InputVarType;
    typedef cufftComplex    OutputVarType;
};

template<>
struct FFTTransformType<std::complex<float>, cufftComplex>
{
    static constexpr bool ForwardFFT = true;
    static constexpr cufftType CUDAType = CUFFT_C2C;
    static constexpr char const* Name = "C2C";

    typedef cufftComplex    InputVarType;
    typedef cufftComplex    OutputVarType;
};

template<>
struct FFTTransformType<std::complex<float>, Eigen::Vector2f>
{
    static constexpr bool ForwardFFT = true;
    static constexpr cufftType CUDAType = CUFFT_C2C;
    static constexpr char const* Name = "C2C";

    typedef cufftComplex    InputVarType;
    typedef cufftComplex    OutputVarType;
};

template<>
struct FFTTransformType<Eigen::Vector2f, std::complex<float>>
{
    static constexpr bool ForwardFFT = true;
    static constexpr cufftType CUDAType = CUFFT_C2C;
    static constexpr char const* Name = "C2C";

    typedef cufftComplex    InputVarType;
    typedef cufftComplex    OutputVarType;
};

// -------------------------------------------

template<cufftType v>
struct ExecutionHelper { };

template<>
struct ExecutionHelper<CUFFT_R2C>
{
    static inline cufftResult exec(cufftHandle plan, cufftReal *idata, cufftComplex *odata, int dir = CUFFT_FORWARD)
    {
        return cufftExecR2C(plan, idata, odata);
    }
};

template<>
struct ExecutionHelper<CUFFT_C2R>
{
    static inline cufftResult exec(cufftHandle plan, cufftComplex *idata, cufftReal *odata, int dir = CUFFT_INVERSE)
    {
        return cufftExecC2R(plan, idata, odata);
    }
};

template<>
struct ExecutionHelper<CUFFT_C2C>
{
    static inline cufftResult exec(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int dir = CUFFT_FORWARD)
    {
        return cufftExecC2C(plan, idata, odata, dir);
    }
};

template<typename T_INPUT, typename T_OUTPUT>
struct ProperExecution
{
    static cufftResult exec(cufftHandle plan, T_INPUT* buf_in, T_OUTPUT* buf_out, bool fwd)
    {
        typedef typename FFTTransformType<T_INPUT,T_OUTPUT>::InputVarType InputVarType;
        typedef typename FFTTransformType<T_INPUT,T_OUTPUT>::OutputVarType OutputVarType;

        return ExecutionHelper<FFTTransformType<T_INPUT,T_OUTPUT>::CUDAType>::exec(plan,
                                                                                   reinterpret_cast<InputVarType*>(buf_in),
                                                                                   reinterpret_cast<OutputVarType*>(buf_out),
                                                                                   fwd == true ? CUFFT_FORWARD : CUFFT_INVERSE);
    }
};

// -------------------------------------------

template<typename T_INPUT, typename T_OUTPUT, typename Target>
struct CUDAPersistentFFT1D : public vc::math::PersistentFFT
{
    CUDAPersistentFFT1D(const CUDAPersistentFFT1D&) = delete; // no copies
    CUDAPersistentFFT1D& operator=(const CUDAPersistentFFT1D& other) = delete; // no copies
    CUDAPersistentFFT1D() = delete;

    CUDAPersistentFFT1D(CUDAPersistentFFT1D&& other) noexcept = delete; /* : p(std::move(other.p))
    {

    }*/

    CUDAPersistentFFT1D& operator=(CUDAPersistentFFT1D&& other) = delete;/*
    {
        p = std::move(other.p); return *this;
    }*/

    CUDAPersistentFFT1D(const vc::Buffer1DView<T_INPUT, Target >& bin,
                        vc::Buffer1DView<T_OUTPUT, Target >& bout, cufftHandle _p, bool f)
        : buf_in(bin), buf_out(bout), p(_p), forward(f)
    {

    }

    ~CUDAPersistentFFT1D()
    {
        cufftDestroy(p);
    }

    virtual void execute()
    {
        cufftResult res = ProperExecution<T_INPUT, T_OUTPUT>::exec(p, const_cast<T_INPUT*>(buf_in.ptr()), buf_out.ptr(), forward);
        if(res != CUFFT_SUCCESS) { throw std::runtime_error("Exec Error"); }

        // wait for it
        const cudaError err = cudaDeviceSynchronize();
        if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }
    }

    const vc::Buffer1DView<T_INPUT, Target >& buf_in;
    vc::Buffer1DView<T_OUTPUT, Target >& buf_out;
    cufftHandle p;
    bool forward;
};

template<typename T_INPUT, typename T_OUTPUT, typename Target>
struct CUDAPersistentFFT2D : public vc::math::PersistentFFT
{
    CUDAPersistentFFT2D(const CUDAPersistentFFT2D&) = delete; // no copies
    CUDAPersistentFFT2D& operator=(const CUDAPersistentFFT2D& other) = delete; // no copies
    CUDAPersistentFFT2D() = delete;

    CUDAPersistentFFT2D(CUDAPersistentFFT2D&& other) noexcept = delete;
    /*: p(std::move(other.p))
    {

    }*/

    CUDAPersistentFFT2D& operator=(CUDAPersistentFFT2D&& other) = delete; /*
    {
        p = std::move(other.p); return *this;
    }*/

    CUDAPersistentFFT2D(const vc::Buffer2DView<T_INPUT, Target >& bin,
                        vc::Buffer2DView<T_OUTPUT, Target >& bout, cufftHandle _p, bool f)
    : buf_in(bin), buf_out(bout), p(_p), forward(f)
    {

    }

    ~CUDAPersistentFFT2D()
    {
        cufftDestroy(p);
    }

    virtual void execute()
    {
        cufftResult res = ProperExecution<T_INPUT, T_OUTPUT>::exec(p, const_cast<T_INPUT*>(buf_in.ptr()), buf_out.ptr(), forward);
        if(res != CUFFT_SUCCESS) { throw std::runtime_error("Exec Error"); }

        // wait for it
        const cudaError err = cudaDeviceSynchronize();
        if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }
    }

    const vc::Buffer2DView<T_INPUT, Target >& buf_in;
    vc::Buffer2DView<T_OUTPUT, Target >& buf_out;
    cufftHandle p;
    bool forward;
};

// -------------------------------------------

template<typename T_INPUT, typename T_OUTPUT, typename Target>
void vc::math::fft(int npoint, const vc::Buffer1DView<T_INPUT, Target >& buf_in,
                   vc::Buffer1DView<T_OUTPUT, Target >& buf_out, bool forward)
{
    cufftHandle plan;

    cufftResult res = cufftPlan1d(&plan, npoint, FFTTransformType<T_INPUT, T_OUTPUT>::CUDAType, std::max(buf_in.size(), buf_out.size()) / npoint);
    if(res != CUFFT_SUCCESS) { throw std::runtime_error("Plan Error"); }

    if(FFTTransformType<T_INPUT, T_OUTPUT>::CUDAType == CUFFT_C2R) { forward = false; }

    res = ProperExecution<T_INPUT, T_OUTPUT>::exec(plan, const_cast<T_INPUT*>(buf_in.ptr()), buf_out.ptr(), forward);
    if(res != CUFFT_SUCCESS) { throw std::runtime_error("Exec Error"); }

    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }

    cufftDestroy(plan);
}

template<typename T_INPUT, typename T_OUTPUT, typename Target>
void vc::math::fft(const vc::Buffer2DView<T_INPUT, Target>& buf_in,
                   vc::Buffer2DView<T_OUTPUT, Target>& buf_out, bool forward)
{
    cufftHandle plan;

    int rank = 2; // 2D fft
    int n[] = {(int)(std::max(buf_in.height(), buf_out.height())), (int)(std::max(buf_in.width(), buf_out.width()))};    // Size of the Fourier transform
    int istride = 1, ostride = 1; // Stride lengths
    int idist = 1, odist = 1;     // Distance between batches
    int inembed[] = {(int)buf_in.height(), (int)(buf_in.pitch() / sizeof(T_INPUT))}; // Input size with pitch
    int onembed[] = {(int)buf_out.height(), (int)(buf_out.pitch() / sizeof(T_OUTPUT))}; // Output size with pitch
    int batch = 1;
    cufftResult res = cufftPlanMany(&plan, rank, n,
                                    inembed, istride, idist,
                                    onembed, ostride, odist,
                                    FFTTransformType<T_INPUT, T_OUTPUT>::CUDAType, batch);
    if(res != CUFFT_SUCCESS) { throw std::runtime_error("Plan Error"); }

    if(FFTTransformType<T_INPUT, T_OUTPUT>::CUDAType == CUFFT_C2R) { forward = false; }

    res = ProperExecution<T_INPUT, T_OUTPUT>::exec(plan, const_cast<T_INPUT*>(buf_in.ptr()), buf_out.ptr(), forward);
    if(res != CUFFT_SUCCESS) { throw std::runtime_error("Exec Error"); }

    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw vc::CUDAException(err, "Error launching the kernel"); }

    cufftDestroy(plan);
}

template<typename T_INPUT, typename T_OUTPUT, typename Target>
std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT(int npoint, const vc::Buffer1DView<T_INPUT, Target >& buf_in,
                                                           vc::Buffer1DView<T_OUTPUT, Target >& buf_out, bool forward)
{
    cufftHandle plan;

    cufftResult res = cufftPlan1d(&plan, npoint, FFTTransformType<T_INPUT, T_OUTPUT>::CUDAType, std::max(buf_in.size(), buf_out.size()) / npoint);
    if(res != CUFFT_SUCCESS) { throw std::runtime_error("Plan Error"); }

    return std::unique_ptr<vc::math::PersistentFFT>(new CUDAPersistentFFT1D<T_INPUT,T_OUTPUT,Target>(buf_in, buf_out, plan, forward));
}

template<typename T_INPUT, typename T_OUTPUT, typename Target>
std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT(const vc::Buffer2DView<T_INPUT, Target>& buf_in,
                                                           vc::Buffer2DView<T_OUTPUT, Target>& buf_out, bool forward)
{
    cufftHandle plan;

    int rank = 2; // 2D fft
    int n[] = {(int)(std::max(buf_in.height(), buf_out.height())), (int)(std::max(buf_in.width(), buf_out.width()))};    // Size of the Fourier transform
    int istride = 1, ostride = 1; // Stride lengths
    int idist = 1, odist = 1;     // Distance between batches
    int inembed[] = {(int)buf_in.height(), (int)(buf_in.pitch() / sizeof(T_INPUT))}; // Input size with pitch
    int onembed[] = {(int)buf_out.height(), (int)(buf_out.pitch() / sizeof(T_OUTPUT))}; // Output size with pitch
    int batch = 1;
    cufftResult res = cufftPlanMany(&plan, rank, n,
                                    inembed, istride, idist,
                                    onembed, ostride, odist,
                                    FFTTransformType<T_INPUT, T_OUTPUT>::CUDAType, batch);
    if(res != CUFFT_SUCCESS) { throw std::runtime_error("Plan Error"); }

    if(FFTTransformType<T_INPUT, T_OUTPUT>::CUDAType == CUFFT_C2R) { forward = false; }

    return std::unique_ptr<vc::math::PersistentFFT>(new CUDAPersistentFFT2D<T_INPUT,T_OUTPUT,Target>(buf_in, buf_out, plan, forward));
}

namespace vc
{
namespace math
{
namespace internal
{

// interfacing with cufftComplex (e.g. float2)
template<typename T_REAL>
struct complexOps<cufftComplex,T_REAL>
{
    EIGEN_DEVICE_FUNC static inline T_REAL getReal(const cufftComplex& cpx)
    {
        return cpx.x;
    }

    EIGEN_DEVICE_FUNC static inline T_REAL getImag(const cufftComplex& cpx)
    {
        return cpx.y;
    }

    EIGEN_DEVICE_FUNC static inline cufftComplex makeComplex(const T_REAL& re, const T_REAL& im)
    {
        return make_cuComplex(re, im);
    }

    EIGEN_DEVICE_FUNC static inline cufftComplex conjugate(const cufftComplex& cpx)
    {
        return make_cuComplex(cpx.x, -cpx.y);
    }

    EIGEN_DEVICE_FUNC static inline cufftComplex multiply(const cufftComplex& cpx1, const cufftComplex& cpx2)
    {
        return make_cuComplex(cpx1.x * cpx2.x - cpx1.y * cpx2.y, cpx1.y * cpx2.x + cpx1.x * cpx2.y);
    }

    EIGEN_DEVICE_FUNC static inline cufftComplex multiply(const cufftComplex& cpx1, const T_REAL& scalar)
    {
        return multiply(cpx1, make_cuComplex(scalar, 0.0f));
    }

    EIGEN_DEVICE_FUNC static inline cufftComplex divide(const cufftComplex& cpx1, const cufftComplex& cpx2)
    {
        return make_cuComplex( (cpx1.x * cpx2.x + cpx1.y * cpx2.y) / (cpx2.x * cpx2.x + cpx2.y * cpx2.y) , (cpx1.y * cpx2.x - cpx1.x * cpx2.y) / (cpx2.x * cpx2.x + cpx2.y * cpx2.y) );
    }

    EIGEN_DEVICE_FUNC static inline cufftComplex divide(const cufftComplex& cpx1, const T_REAL& scalar)
    {
        return divide(cpx1, make_cuComplex(scalar, 0.0f));
    }

    EIGEN_DEVICE_FUNC static inline T_REAL norm(const cufftComplex& cpx)
    {
        return sqrt(cpx.x * cpx.x + cpx.y * cpx.y);
    }
};

}
}
}

template<typename T_COMPLEX, typename T_REAL, typename Target>
__global__ void Kernel_splitComplex1D(const vc::Buffer1DView< T_COMPLEX, Target > buf_in,
                                      vc::Buffer1DView< T_REAL, Target > buf_real,
                                      vc::Buffer1DView< T_REAL, Target > buf_imag)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;

    if(buf_in.inBounds(x)) // is valid
    {
        const T_COMPLEX& complex = buf_in(x);
        buf_real(x) = vc::math::internal::complexOps<T_COMPLEX,T_REAL>::getReal(complex);
        buf_imag(x) = vc::math::internal::complexOps<T_COMPLEX,T_REAL>::getImag(complex);
    }
}

template<typename T_COMPLEX, typename Target>
void vc::math::splitComplex(const vc::Buffer1DView< T_COMPLEX, Target >& buf_in,
                              vc::Buffer1DView< typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target >& buf_real,
                              vc::Buffer1DView< typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target >& buf_imag)
{
    dim3 gridDim, blockDim;

    vc::InitDimFromBuffer(blockDim, gridDim, buf_in);

    // run kernel
    Kernel_splitComplex1D<T_COMPLEX, typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target><<<gridDim,blockDim>>>(buf_in, buf_real, buf_imag);

    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T_COMPLEX, typename T_REAL, typename Target>
__global__ void Kernel_splitComplex2D(const vc::Buffer2DView< T_COMPLEX, Target > buf_in,
                                    vc::Buffer2DView< T_REAL, Target > buf_real,
                                    vc::Buffer2DView< T_REAL, Target > buf_imag)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;

    if(buf_in.inBounds(x,y)) // is valid
    {
        const T_COMPLEX& complex = buf_in(x,y);
        buf_real(x,y) = vc::math::internal::complexOps<T_COMPLEX,T_REAL>::getReal(complex);
        buf_imag(x,y) = vc::math::internal::complexOps<T_COMPLEX,T_REAL>::getImag(complex);
    }
}

template<typename T_COMPLEX, typename Target>
void vc::math::splitComplex(const vc::Buffer2DView< T_COMPLEX, Target >& buf_in,
                              vc::Buffer2DView< typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target >& buf_real,
                           vc::Buffer2DView< typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target >& buf_imag)
{
    dim3 gridDim, blockDim;

    vc::InitDimFromBufferOver(blockDim, gridDim, buf_in);

    // run kernel
    Kernel_splitComplex2D<T_COMPLEX, typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target><<<gridDim,blockDim>>>(buf_in, buf_real, buf_imag);

    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T_COMPLEX, typename T_REAL, typename Target>
__global__ void Kernel_joinComplex1D(const vc::Buffer1DView< T_REAL, Target > buf_real,
                                     const vc::Buffer1DView< T_REAL, Target > buf_imag,
                                     vc::Buffer1DView< T_COMPLEX, Target > buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;

    if(buf_out.inBounds(x)) // is valid
    {
        buf_out(x) = vc::math::internal::complexOps<T_COMPLEX,T_REAL>::makeComplex(buf_real(x), buf_imag(x));
    }
}

template<typename T_COMPLEX, typename Target>
void vc::math::joinComplex(const vc::Buffer1DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target >& buf_real,
                             const vc::Buffer1DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target >& buf_imag,
                             vc::Buffer1DView<T_COMPLEX, Target >& buf_out)
{
    dim3 gridDim, blockDim;

    vc::InitDimFromBuffer(blockDim, gridDim, buf_out);

    // run kernel
    Kernel_joinComplex1D<T_COMPLEX, typename internal::FFTTypeTraits<T_COMPLEX>::BaseType,Target><<<gridDim,blockDim>>>(buf_real, buf_imag, buf_out);

    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T_COMPLEX, typename T_REAL, typename Target>
__global__ void Kernel_joinComplex2D(const vc::Buffer2DView< T_REAL, Target > buf_real,
                                     const vc::Buffer2DView< T_REAL, Target > buf_imag,
                                     vc::Buffer2DView< T_COMPLEX, Target > buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;

    if(buf_out.inBounds(x,y)) // is valid
    {
        buf_out(x,y) = vc::math::internal::complexOps<T_COMPLEX,T_REAL>::makeComplex(buf_real(x,y), buf_imag(x,y));
    }
}

template<typename T_COMPLEX, typename Target>
void vc::math::joinComplex(const vc::Buffer2DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target >& buf_real,
                             const vc::Buffer2DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target >& buf_imag,
                             vc::Buffer2DView<T_COMPLEX, Target >& buf_out)
{
    dim3 gridDim, blockDim;

    vc::InitDimFromBufferOver(blockDim, gridDim, buf_out);

    // run kernel
    Kernel_joinComplex2D<T_COMPLEX, typename internal::FFTTypeTraits<T_COMPLEX>::BaseType,Target><<<gridDim,blockDim>>>(buf_real, buf_imag, buf_out);

    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T_COMPLEX, typename T_REAL, typename Target>
__global__ void Kernel_magnitude1D(const vc::Buffer1DView<T_COMPLEX, Target> buf_in,
                                   vc::Buffer1DView<T_REAL, Target> buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;

    if(buf_in.inBounds(x)) // is valid
    {
        const T_COMPLEX& complex = buf_in(x);
        buf_out(x) = sqrtf(vc::math::internal::complexOps<T_COMPLEX,T_REAL>::getReal(complex) * vc::math::internal::complexOps<T_COMPLEX,T_REAL>::getReal(complex)
                         + vc::math::internal::complexOps<T_COMPLEX,T_REAL>::getImag(complex) * vc::math::internal::complexOps<T_COMPLEX,T_REAL>::getImag(complex));
    }
}

template<typename T_COMPLEX, typename Target>
void vc::math::magnitude(const vc::Buffer1DView<T_COMPLEX, Target>& buf_in,
                           vc::Buffer1DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_out)
{
    dim3 gridDim, blockDim;

    vc::InitDimFromBuffer(blockDim, gridDim, buf_out);

    // run kernel
    Kernel_magnitude1D<T_COMPLEX, typename internal::FFTTypeTraits<T_COMPLEX>::BaseType,Target><<<gridDim,blockDim>>>(buf_in, buf_out);

    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T_COMPLEX, typename T_REAL, typename Target>
__global__ void Kernel_magnitude2D(const vc::Buffer2DView<T_COMPLEX, Target> buf_in,
                                   vc::Buffer2DView<T_REAL, Target> buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;

    if(buf_in.inBounds(x,y)) // is valid
    {
        const T_COMPLEX& complex = buf_in(x,y);
        buf_out(x,y) = sqrtf(vc::math::internal::complexOps<T_COMPLEX,T_REAL>::getReal(complex) * vc::math::internal::complexOps<T_COMPLEX,T_REAL>::getReal(complex) +
                             vc::math::internal::complexOps<T_COMPLEX,T_REAL>::getImag(complex) * vc::math::internal::complexOps<T_COMPLEX,T_REAL>::getImag(complex));
    }
}

template<typename T_COMPLEX, typename Target>
void vc::math::magnitude(const vc::Buffer2DView<T_COMPLEX, Target>& buf_in,
                           vc::Buffer2DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_out)
{
    dim3 gridDim, blockDim;

    vc::InitDimFromBufferOver(blockDim, gridDim, buf_out);

    // run kernel
    Kernel_magnitude2D<T_COMPLEX, typename internal::FFTTypeTraits<T_COMPLEX>::BaseType,Target><<<gridDim,blockDim>>>(buf_in, buf_out);

    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T_COMPLEX, typename T_REAL, typename Target>
__global__ void Kernel_phase1D(const vc::Buffer1DView< T_COMPLEX, Target > buf_in,
                               vc::Buffer1DView< T_REAL, Target > buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;

    if(buf_in.inBounds(x)) // is valid
    {
        const T_COMPLEX& complex = buf_in(x);
        buf_out(x) = atan2(vc::math::internal::complexOps<T_COMPLEX,T_REAL>::getImag(complex), vc::math::internal::complexOps<T_COMPLEX,T_REAL>::getReal(complex));
    }
}

template<typename T_COMPLEX, typename Target>
void vc::math::phase(const vc::Buffer1DView<T_COMPLEX, Target>& buf_in,
                       vc::Buffer1DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_out)
{
    dim3 gridDim, blockDim;

    vc::InitDimFromBuffer(blockDim, gridDim, buf_out);

    // run kernel
    Kernel_phase1D<T_COMPLEX, typename internal::FFTTypeTraits<T_COMPLEX>::BaseType,Target><<<gridDim,blockDim>>>(buf_in, buf_out);

    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T_COMPLEX, typename T_REAL, typename Target>
__global__ void Kernel_phase2D(const vc::Buffer2DView< T_COMPLEX, Target > buf_in,
                               vc::Buffer2DView< T_REAL, Target > buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;

    if(buf_in.inBounds(x,y)) // is valid
    {
        const T_COMPLEX& complex = buf_in(x,y);
        buf_out(x,y) = atan2(vc::math::internal::complexOps<T_COMPLEX,T_REAL>::getImag(complex), vc::math::internal::complexOps<T_COMPLEX,T_REAL>::getReal(complex));
    }
}

template<typename T_COMPLEX, typename Target>
void vc::math::phase(const vc::Buffer2DView<T_COMPLEX, Target>& buf_in,
                       vc::Buffer2DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_out)
{
    dim3 gridDim, blockDim;

    vc::InitDimFromBufferOver(blockDim, gridDim, buf_out);

    // run kernel
    Kernel_phase2D<T_COMPLEX, typename internal::FFTTypeTraits<T_COMPLEX>::BaseType,Target><<<gridDim,blockDim>>>(buf_in, buf_out);

    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T_COMPLEX, typename T_REAL, typename Target>
__global__ void Kernel_convertToComplex1D(const vc::Buffer1DView< T_REAL, Target > buf_in,
                                          vc::Buffer1DView< T_COMPLEX, Target > buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;

    if(buf_in.inBounds(x)) // is valid
    {
        buf_out(x) = vc::math::internal::complexOps<T_COMPLEX,T_REAL>::makeComplex(buf_in(x), T_REAL(0.0f));
    }
}

template<typename T_COMPLEX, typename Target>
void vc::math::convertToComplex(const vc::Buffer1DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_in,
                                  vc::Buffer1DView<T_COMPLEX, Target>& buf_out)
{
    dim3 gridDim, blockDim;

    vc::InitDimFromBuffer(blockDim, gridDim, buf_out);

    // run kernel
    Kernel_convertToComplex1D<T_COMPLEX, typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target><<<gridDim,blockDim>>>(buf_in, buf_out);

    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T_COMPLEX, typename T_REAL, typename Target>
__global__ void Kernel_convertToComplex2D(const vc::Buffer2DView< T_REAL, Target > buf_in,
                                          vc::Buffer2DView< T_COMPLEX, Target > buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;

    if(buf_in.inBounds(x,y)) // is valid
    {
        buf_out(x,y) = vc::math::internal::complexOps<T_COMPLEX,T_REAL>::makeComplex(buf_in(x,y), T_REAL(0.0f));
    }
}

template<typename T_COMPLEX, typename Target>
void vc::math::convertToComplex(const vc::Buffer2DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_in,
                                  vc::Buffer2DView<T_COMPLEX, Target>& buf_out)
{
    dim3 gridDim, blockDim;

    vc::InitDimFromBuffer(blockDim, gridDim, buf_out);

    // run kernel
    Kernel_convertToComplex2D<T_COMPLEX, typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target><<<gridDim,blockDim>>>(buf_in, buf_out);

    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T_COMPLEX, typename Target>
__global__ void Kernel_calculateCrossPowerSpectrum1D(const vc::Buffer1DView<T_COMPLEX, Target> buf_fft1,
                                                     const vc::Buffer1DView<T_COMPLEX, Target> buf_fft2,
                                                     vc::Buffer1DView<T_COMPLEX, Target> buf_fft_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;

    if(buf_fft_out.inBounds(x)) // is valid
    {
        buf_fft_out(x) = vc::math::internal::crossPowerSpectrum<T_COMPLEX>(buf_fft1(x),buf_fft2(x));
    }
}

template<typename T_COMPLEX, typename Target>
void vc::math::calculateCrossPowerSpectrum(const vc::Buffer1DView<T_COMPLEX, Target>& buf_fft1,
                                              const vc::Buffer1DView<T_COMPLEX, Target>& buf_fft2,
                                              vc::Buffer1DView<T_COMPLEX, Target>& buf_fft_out)
{
    dim3 gridDim, blockDim;

    vc::InitDimFromBuffer(blockDim, gridDim, buf_fft_out);

    // run kernel
    Kernel_calculateCrossPowerSpectrum1D<T_COMPLEX,Target><<<gridDim,blockDim>>>(buf_fft1, buf_fft2, buf_fft_out);

    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T_COMPLEX, typename Target>
__global__ void Kernel_calculateCrossPowerSpectrum2D(const vc::Buffer2DView<T_COMPLEX, Target> buf_fft1,
                                                     const vc::Buffer2DView<T_COMPLEX, Target> buf_fft2,
                                                     vc::Buffer2DView<T_COMPLEX, Target> buf_fft_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;

    if(buf_fft_out.inBounds(x,y)) // is valid
    {
        buf_fft_out(x,y) = vc::math::internal::crossPowerSpectrum<T_COMPLEX>(buf_fft1(x,y),buf_fft2(x,y));
    }
}

template<typename T_COMPLEX, typename Target>
void vc::math::calculateCrossPowerSpectrum(const vc::Buffer2DView<T_COMPLEX, Target>& buf_fft1,
                                          const vc::Buffer2DView<T_COMPLEX, Target>& buf_fft2,
                                          vc::Buffer2DView<T_COMPLEX, Target>& buf_fft_out)
{
    dim3 gridDim, blockDim;

    vc::InitDimFromBufferOver(blockDim, gridDim, buf_fft_out);

    // run kernel
    Kernel_calculateCrossPowerSpectrum2D<T_COMPLEX,Target><<<gridDim,blockDim>>>(buf_fft1, buf_fft2, buf_fft_out);

    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
}

// R2C - 1D
template void vc::math::fft<float, cufftComplex>(int npoint, const vc::Buffer1DView<float, vc::TargetDeviceCUDA >& buf_in,
                                                 vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA >& buf_out, bool forward);
template void vc::math::fft<float, Eigen::Vector2f>(int npoint, const vc::Buffer1DView<float, vc::TargetDeviceCUDA >& buf_in,
                                                    vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_out, bool forward);
template void vc::math::fft<float, std::complex<float>>(int npoint, const vc::Buffer1DView<float, vc::TargetDeviceCUDA >& buf_in,
                                                        vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_out, bool forward);

template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<float, cufftComplex>(int npoint, const vc::Buffer1DView<float, vc::TargetDeviceCUDA >& buf_in,
                                                                                         vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA >& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<float, Eigen::Vector2f>(int npoint, const vc::Buffer1DView<float, vc::TargetDeviceCUDA >& buf_in,
                                                                                            vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<float, std::complex<float>>(int npoint, const vc::Buffer1DView<float, vc::TargetDeviceCUDA >& buf_in,
                                                                                                vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_out, bool forward);

// R2C - 2D
template void vc::math::fft<float, cufftComplex>(const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_in,
                                                         vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_out, bool forward);
template void vc::math::fft<float, Eigen::Vector2f>(const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_in,
                                                            vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_out, bool forward);
template void vc::math::fft<float, std::complex<float>>(const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_in,
                                                                vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_out, bool forward);

template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<float, cufftComplex>(const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_in,
                                                                                             vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<float, Eigen::Vector2f>(const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_in,
                                                                                                vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<float, std::complex<float>>(const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_in,
                                                                                                    vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_out, bool forward);

// C2R - 1D
template void vc::math::fft<cufftComplex, float>(int npoint, const vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA >& buf_in,
                                                 vc::Buffer1DView<float, vc::TargetDeviceCUDA >& buf_out, bool forward);
template void vc::math::fft<Eigen::Vector2f, float>(int npoint, const vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_in,
                                                    vc::Buffer1DView<float, vc::TargetDeviceCUDA >& buf_out, bool forward);
template void vc::math::fft<std::complex<float>, float>(int npoint, const vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_in,
                                                        vc::Buffer1DView<float, vc::TargetDeviceCUDA >& buf_out, bool forward);

template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<cufftComplex, float>(int npoint, const vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA >& buf_in,
                                                                                         vc::Buffer1DView<float, vc::TargetDeviceCUDA >& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<Eigen::Vector2f, float>(int npoint, const vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_in,
                                                                                            vc::Buffer1DView<float, vc::TargetDeviceCUDA >& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<std::complex<float>, float>(int npoint, const vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_in,
                                                                                                vc::Buffer1DView<float, vc::TargetDeviceCUDA >& buf_out, bool forward);

// C2R - 2D
template void vc::math::fft<cufftComplex, float>(const vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_in,
                                                         vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_out, bool forward);
template void vc::math::fft<Eigen::Vector2f, float>(const vc::Buffer2DView<Eigen::Vector2f,
                                                            vc::TargetDeviceCUDA>& buf_in, vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_out, bool forward);
template void vc::math::fft<std::complex<float>, float>(const vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_in,
                                                          vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_out, bool forward);

template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<cufftComplex, float>(const vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_in,
                                                                                             vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<Eigen::Vector2f, float>(const vc::Buffer2DView<Eigen::Vector2f,
                                                                                                vc::TargetDeviceCUDA>& buf_in, vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<std::complex<float>, float>(const vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_in,
                                                                                                    vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_out, bool forward);

// C2C - 1D
template void vc::math::fft<cufftComplex, cufftComplex>(int npoint, const vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA >& buf_in,
                                                        vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA >& buf_out, bool forward);
template void vc::math::fft<cufftComplex, Eigen::Vector2f>(int npoint, const vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA >& buf_in,
                                                           vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_out, bool forward);
template void vc::math::fft<cufftComplex, std::complex<float>>(int npoint, const vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA >& buf_in,
                                                               vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_out, bool forward);
template void vc::math::fft<Eigen::Vector2f, Eigen::Vector2f>(int npoint, const vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_in,
                                                              vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_out, bool forward);
template void vc::math::fft<Eigen::Vector2f, cufftComplex>(int npoint, const vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_in,
                                                           vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA >& buf_out, bool forward);
template void vc::math::fft<Eigen::Vector2f, std::complex<float>>(int npoint, const vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_in,
                                                                  vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_out, bool forward);
template void vc::math::fft<std::complex<float>, std::complex<float>>(int npoint, const vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_in,
                                                                      vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_out, bool forward);
template void vc::math::fft<std::complex<float>, Eigen::Vector2f>(int npoint, const vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_in,
                                                                  vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_out, bool forward);
template void vc::math::fft<std::complex<float>, cufftComplex>(int npoint, const vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_in,
                                                             vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA >& buf_out, bool forward);

template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<cufftComplex, cufftComplex>(int npoint, const vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA >& buf_in,
                                                                                                vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA >& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<cufftComplex, Eigen::Vector2f>(int npoint, const vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA >& buf_in,
                                                                                                   vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<cufftComplex, std::complex<float>>(int npoint, const vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA >& buf_in,
                                                                                                       vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<Eigen::Vector2f, Eigen::Vector2f>(int npoint, const vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_in,
                                                                                                      vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<Eigen::Vector2f, cufftComplex>(int npoint, const vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_in,
                                                                                                   vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA >& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<Eigen::Vector2f, std::complex<float>>(int npoint, const vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_in,
                                                                                                          vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<std::complex<float>, std::complex<float>>(int npoint, const vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_in,
                                                                                                              vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<std::complex<float>, Eigen::Vector2f>(int npoint, const vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_in,
                                                                                                          vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<std::complex<float>, cufftComplex>(int npoint, const vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_in,
                                                                                                       vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA >& buf_out, bool forward);

// C2C - 2D
template void vc::math::fft<cufftComplex, cufftComplex>(const vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_in,
                                                                vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_out, bool forward);
template void vc::math::fft<cufftComplex, Eigen::Vector2f>(const vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_in,
                                                                   vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_out, bool forward);
template void vc::math::fft<cufftComplex, std::complex<float>>(const vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_in,
                                                                       vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_out, bool forward);

template void vc::math::fft<Eigen::Vector2f, Eigen::Vector2f>(const vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_in,
                                                                      vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_out, bool forward);
template void vc::math::fft<Eigen::Vector2f, cufftComplex>(const vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_in,
                                                                   vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_out, bool forward);
template void vc::math::fft<Eigen::Vector2f, std::complex<float>>(const vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_in,
                                                                          vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_out, bool forward);

template void vc::math::fft<std::complex<float>, std::complex<float>>(const vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_in,
                                                                       vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_out, bool forward);
template void vc::math::fft<std::complex<float>, cufftComplex>(const vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_in,
                                                                       vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_out, bool forward);
template void vc::math::fft<std::complex<float>, Eigen::Vector2f>(const vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_in,
                                                                   vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_out, bool forward);

template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<cufftComplex, cufftComplex>(const vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_in,
                                                                                                    vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<cufftComplex, Eigen::Vector2f>(const vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_in,
                                                                                                       vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<cufftComplex, std::complex<float>>(const vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_in,
                                                                                                           vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_out, bool forward);

template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<Eigen::Vector2f, Eigen::Vector2f>(const vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_in,
                                                                                                          vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<Eigen::Vector2f, cufftComplex>(const vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_in,
                                                                                                       vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<Eigen::Vector2f, std::complex<float>>(const vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_in,
                                                                                                              vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_out, bool forward);

template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<std::complex<float>, std::complex<float>>(const vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_in,
                                                                                                                  vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<std::complex<float>, cufftComplex>(const vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_in,
                                                                                                           vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_out, bool forward);
template std::unique_ptr<vc::math::PersistentFFT> vc::math::makeFFT<std::complex<float>, Eigen::Vector2f>(const vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_in,
                                                                                                              vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_out, bool forward);

// splitter - 1D
template void vc::math::splitComplex<cufftComplex>(const vc::Buffer1DView< cufftComplex, vc::TargetDeviceCUDA >& buf_in,
                                                     vc::Buffer1DView< float, vc::TargetDeviceCUDA >& buf_real,
                                                     vc::Buffer1DView< float, vc::TargetDeviceCUDA >& buf_imag);
template void vc::math::splitComplex<Eigen::Vector2f>(const vc::Buffer1DView< Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_in,
                                                        vc::Buffer1DView< float, vc::TargetDeviceCUDA >& buf_real,
                                                        vc::Buffer1DView< float, vc::TargetDeviceCUDA >& buf_imag);
template void vc::math::splitComplex<std::complex<float>>(const vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_in,
                                                            vc::Buffer1DView< float, vc::TargetDeviceCUDA >& buf_real,
                                                            vc::Buffer1DView< float, vc::TargetDeviceCUDA >& buf_imag);

// splitter - 2D
template void vc::math::splitComplex<cufftComplex>(const vc::Buffer2DView< cufftComplex, vc::TargetDeviceCUDA >& buf_in,
                                                  vc::Buffer2DView< float, vc::TargetDeviceCUDA >& buf_real,
                                                  vc::Buffer2DView< float, vc::TargetDeviceCUDA >& buf_imag);
template void vc::math::splitComplex<Eigen::Vector2f>(const vc::Buffer2DView< Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_in,
                                                     vc::Buffer2DView< float, vc::TargetDeviceCUDA >& buf_real,
                                                     vc::Buffer2DView< float, vc::TargetDeviceCUDA >& buf_imag);
template void vc::math::splitComplex<std::complex<float>>(const vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_in,
                                                            vc::Buffer2DView< float, vc::TargetDeviceCUDA >& buf_real,
                                                            vc::Buffer2DView< float, vc::TargetDeviceCUDA >& buf_imag);

// joiner - 1D
template void vc::math::joinComplex<cufftComplex>(const vc::Buffer1DView<float, vc::TargetDeviceCUDA >& buf_real,
                                                    const vc::Buffer1DView<float, vc::TargetDeviceCUDA >& buf_imag,
                                                    vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA >& buf_out);
template void vc::math::joinComplex<Eigen::Vector2f>(const vc::Buffer1DView<float, vc::TargetDeviceCUDA >& buf_real,
                                                       const vc::Buffer1DView<float, vc::TargetDeviceCUDA >& buf_imag,
                                                       vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_out);
template void vc::math::joinComplex<std::complex<float>>(const vc::Buffer1DView<float, vc::TargetDeviceCUDA >& buf_real,
                                                           const vc::Buffer1DView<float, vc::TargetDeviceCUDA >& buf_imag,
                                                           vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_out);

// joiner - 2D
template void vc::math::joinComplex<cufftComplex>(const vc::Buffer2DView<float, vc::TargetDeviceCUDA >& buf_real,
                                                    const vc::Buffer2DView<float, vc::TargetDeviceCUDA >& buf_imag,
                                                    vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA >& buf_out);
template void vc::math::joinComplex<Eigen::Vector2f>(const vc::Buffer2DView<float, vc::TargetDeviceCUDA >& buf_real,
                                                    const vc::Buffer2DView<float, vc::TargetDeviceCUDA >& buf_imag,
                                                    vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA >& buf_out);
template void vc::math::joinComplex<std::complex<float>>(const vc::Buffer2DView<float, vc::TargetDeviceCUDA >& buf_real,
                                                           const vc::Buffer2DView<float, vc::TargetDeviceCUDA >& buf_imag,
                                                           vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA >& buf_out);

// magnitude - 1D
template void vc::math::magnitude<cufftComplex>(const vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA>& buf_in,
                                                  vc::Buffer1DView<float, vc::TargetDeviceCUDA>& buf_out);
template void vc::math::magnitude<Eigen::Vector2f>(const vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_in,
                                                     vc::Buffer1DView<float, vc::TargetDeviceCUDA>& buf_out);
template void vc::math::magnitude<std::complex<float>>(const vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_in,
                                                         vc::Buffer1DView<float, vc::TargetDeviceCUDA>& buf_out);

// magnitude - 2D
template void vc::math::magnitude<cufftComplex>(const vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_in,
                                                  vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_out);
template void vc::math::magnitude<Eigen::Vector2f>(const vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_in,
                                                     vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_out);
template void vc::math::magnitude<std::complex<float>>(const vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_in,
                                                         vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_out);

// phase - 1D
template void vc::math::phase<cufftComplex>(const vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA>& buf_in,
                                              vc::Buffer1DView<float, vc::TargetDeviceCUDA>& buf_out);
template void vc::math::phase<Eigen::Vector2f>(const vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_in,
                                                 vc::Buffer1DView<float, vc::TargetDeviceCUDA>& buf_out);
template void vc::math::phase<std::complex<float>>(const vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_in,
                                                     vc::Buffer1DView<float, vc::TargetDeviceCUDA>& buf_out);

// phase - 2D
template void vc::math::phase<cufftComplex>(const vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_in,
                                              vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_out);
template void vc::math::phase<Eigen::Vector2f>(const vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_in,
                                                 vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_out);
template void vc::math::phase<std::complex<float>>(const vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_in,
                                                     vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_out);

// convert - 1D
template void vc::math::convertToComplex<cufftComplex>(const vc::Buffer1DView<float, vc::TargetDeviceCUDA>& buf_in,
                                                         vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA>& buf_out);
template void vc::math::convertToComplex<Eigen::Vector2f>(const vc::Buffer1DView<float, vc::TargetDeviceCUDA>& buf_in,
                                                            vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_out);
template void vc::math::convertToComplex<std::complex<float>>(const vc::Buffer1DView<float, vc::TargetDeviceCUDA>& buf_in,
                                                                vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_out);

// convert - 2D
template void vc::math::convertToComplex<cufftComplex>(const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_in,
                                                         vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_out);
template void vc::math::convertToComplex<Eigen::Vector2f>(const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_in,
                                                            vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_out);
template void vc::math::convertToComplex<std::complex<float>>(const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf_in,
                                                                vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_out);

// cross power spectrum - 1D
template void vc::math::calculateCrossPowerSpectrum<cufftComplex>(const vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA>& buf_fft1,
                                                                     const vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA>& buf_fft2,
                                                                     vc::Buffer1DView<cufftComplex, vc::TargetDeviceCUDA>& buf_fft_out);
template void vc::math::calculateCrossPowerSpectrum<Eigen::Vector2f>(const vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_fft1,
                                                                        const vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_fft2,
                                                                        vc::Buffer1DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_fft_out);
template void vc::math::calculateCrossPowerSpectrum<std::complex<float>>(const vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_fft1,
                                                                           const vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_fft2,
                                                                           vc::Buffer1DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_fft_out);

// cross power spectrum - 2D
template void vc::math::calculateCrossPowerSpectrum<cufftComplex>(const vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_fft1,
                                                                 const vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_fft2,
                                                                 vc::Buffer2DView<cufftComplex, vc::TargetDeviceCUDA>& buf_fft_out);
template void vc::math::calculateCrossPowerSpectrum<Eigen::Vector2f>(const vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_fft1,
                                                                    const vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_fft2,
                                                                    vc::Buffer2DView<Eigen::Vector2f, vc::TargetDeviceCUDA>& buf_fft_out);
template void vc::math::calculateCrossPowerSpectrum<std::complex<float>>(const vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_fft1,
                                                                           const vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_fft2,
                                                                           vc::Buffer2DView<std::complex<float>, vc::TargetDeviceCUDA>& buf_fft_out);
