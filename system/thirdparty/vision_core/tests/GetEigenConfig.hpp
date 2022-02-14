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
 */

#ifndef GET_EIGEN_CONFIG_HPP
#define GET_EIGEN_CONFIG_HPP

#include <VisionCore/Platform.hpp>
#include <Eigen/Core>

#ifdef CORE_TESTS_HAVE_SOPHUS
#include <sophus/se3.hpp>
#endif // CORE_TESTS_HAVE_SOPHUS

static constexpr std::size_t MaxEigenConfigurationCount = 32;

inline EIGEN_DEVICE_FUNC void getEigenConfiguration(float* ptr, const Sophus::SE3f& v)
{
    ptr[0]  = EIGEN_WORLD_VERSION;
    ptr[1]  = EIGEN_MAJOR_VERSION;
    ptr[2]  = EIGEN_MINOR_VERSION;
    ptr[3]  = EIGEN_COMP_GNUC;
    ptr[4]  = EIGEN_COMP_CLANG;
    ptr[5]  = EIGEN_COMP_LLVM;
    ptr[6]  = EIGEN_ARCH_x86_64;
    ptr[7]  = EIGEN_ARCH_i386;
    ptr[8]  = EIGEN_ARCH_ARM;
    ptr[9]  = EIGEN_ARCH_ARM64;

#ifdef EIGEN_HAS_CONSTEXPR
    ptr[10] = EIGEN_HAS_CONSTEXPR;
#else // EIGEN_HAS_CONSTEXPR
    ptr[10] = 666.0f;
#endif // EIGEN_HAS_CONSTEXPR

#ifdef EIGEN_IDEAL_MAX_ALIGN_BYTES
    ptr[11] = EIGEN_IDEAL_MAX_ALIGN_BYTES;
#else // EIGEN_IDEAL_MAX_ALIGN_BYTES
    ptr[11] = 666.0f;
#endif // EIGEN_IDEAL_MAX_ALIGN_BYTES

#ifdef EIGEN_MAX_STATIC_ALIGN_BYTES
    ptr[12] = EIGEN_MAX_STATIC_ALIGN_BYTES;
#else // EIGEN_MAX_STATIC_ALIGN_BYTES
    ptr[12] = 666.0f;
#endif // EIGEN_MAX_STATIC_ALIGN_BYTES

#ifdef EIGEN_DONT_ALIGN
    ptr[13] = 1.0f;
#else // EIGEN_DONT_ALIGN
    ptr[13] = 0.0f;
#endif // EIGEN_DONT_ALIGN

#ifdef EIGEN_DONT_ALIGN_STATICALLY
    ptr[14] = 1.0f;
#else // EIGEN_DONT_ALIGN_STATICALLY
    ptr[14] = 0.0f;
#endif // EIGEN_DONT_ALIGN_STATICALLY

#ifdef EIGEN_MAX_ALIGN_BYTES
    ptr[15] = EIGEN_MAX_ALIGN_BYTES;
#else // EIGEN_MAX_ALIGN_BYTES
    ptr[15] = 666.0f;
#endif // EIGEN_MAX_ALIGN_BYTES

#ifdef EIGEN_UNALIGNED_VECTORIZE
    ptr[16] = EIGEN_UNALIGNED_VECTORIZE;
#else // EIGEN_UNALIGNED_VECTORIZE
    ptr[16] = 666.0f;
#endif // EIGEN_UNALIGNED_VECTORIZE

    ptr[17] = sizeof(Eigen::Vector2f);
    ptr[18] = sizeof(Eigen::Vector3f);
    ptr[19] = sizeof(Eigen::Vector4f);
    
#ifdef CORE_TESTS_HAVE_SOPHUS
    ptr[20] = sizeof(Sophus::SE3f);
#else // CORE_TESTS_HAVE_SOPHUS
    ptr[21] = 666.0f;
#endif // CORE_TESTS_HAVE_SOPHUS

#ifdef EIGEN_DONT_VECTORIZE
    ptr[21] = 1.0f;
#else // EIGEN_DONT_VECTORIZE
    ptr[21] = 0.0f;
#endif // EIGEN_DONT_VECTORIZE
    
#ifdef EIGEN_VECTORIZE_CUDA
    ptr[22] = 1.0f;
#else // EIGEN_VECTORIZE_CUDA
    ptr[22] = 0.0f;
#endif // EIGEN_VECTORIZE_CUDA
    
#ifndef __CUDACC__
    ptr[23] = 0.0f;
#else // __CUDACC__
  #ifndef __CUDA_ARCH__
    ptr[23] = 1.0f;
  #else //__CUDA_ARCH__
    ptr[23] = 2.0f;
  #endif // __CUDA_ARCH__
#endif // __CUDACC__
#ifdef EIGEN_CUDA_MAX_ALIGN_BYTES
    ptr[24] = EIGEN_CUDA_MAX_ALIGN_BYTES;
#else // EIGEN_CUDA_MAX_ALIGN_BYTES
    ptr[24] = -1.0f;
#endif // EIGEN_CUDA_MAX_ALIGN_BYTES
    
    ptr[25] = v.so3().unit_quaternion().x();
    ptr[26] = v.so3().unit_quaternion().y();
    ptr[27] = v.so3().unit_quaternion().z();
    ptr[28] = v.so3().unit_quaternion().w();
    
    ptr[29] = v.translation()(0);
    ptr[30] = v.translation()(1);
    ptr[31] = v.translation()(2);
}

#endif // GET_EIGEN_CONFIG_HPP
