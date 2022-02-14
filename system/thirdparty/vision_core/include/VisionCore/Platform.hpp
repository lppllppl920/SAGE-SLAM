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
 * Generic macros, functions, traits etc.
 * ****************************************************************************
 */

#ifndef VISIONCORE_PLATFORM_HPP
#define VISIONCORE_PLATFORM_HPP

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <limits>
#include <numeric>
#include <iosfwd>

#if defined(__CUDACC__) // NVCC
    #define VISIONCORE_ALIGN(n) __align__(n)
    #define VISIONCORE_ALIGN_PACK(n) __align__(n) // ??
#elif defined(__GNUC__) // GCC
    #define VISIONCORE_ALIGN(n) __attribute__((aligned(n)))
    #define VISIONCORE_ALIGN_PACK(n) __attribute__((packed,aligned(n)))
#elif defined(_MSC_VER) // MSVC
    #define VISIONCORE_ALIGN(n) __declspec(align(n))
#else
    #error "Please provide a definition for VISIONCORE_ALIGN macro for your host compiler!"
#endif

#ifdef __clang__
    #define VISIONCORE_COMPILER_CLANG
#endif // __clang__

// ---------------------------------------------------------------------------
// Eigen is mandatory
// ---------------------------------------------------------------------------
#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

// ---------------------------------------------------------------------------
// CUDA
// ---------------------------------------------------------------------------
#ifdef VISIONCORE_HAVE_CUDA
#include <cuda_runtime.h>
#include <VisionCore/CUDAException.hpp>

#ifdef __CUDA_ARCH__
    #define VISIONCORE_CUDA_KERNEL_SPACE
#endif // __CUDA_ARCH__

#ifdef __CUDACC__
    #define VISIONCORE_CUDA_COMPILER
#endif // __CUDACC__

#ifndef EIGEN_DEVICE_FUNC
    #define EIGEN_DEVICE_FUNC __host__ __device__
#endif // EIGEN_DEVICE_FUNC

#ifndef EIGEN_PURE_DEVICE_FUNC
    #define EIGEN_PURE_DEVICE_FUNC __device__
#endif // EIGEN_PURE_DEVICE_FUNC

#ifndef EIGEN_PURE_HOST_FUNC
    #define EIGEN_PURE_HOST_FUNC __host__
#endif // EIGEN_PURE_HOST_FUNC

#else // !VISIONCORE_HAVE_CUDA

#ifndef EIGEN_DEVICE_FUNC
    #define EIGEN_DEVICE_FUNC
#endif // EIGEN_DEVICE_FUNC

#ifndef EIGEN_PURE_DEVICE_FUNC
    #define EIGEN_PURE_DEVICE_FUNC
#endif // EIGEN_PURE_DEVICE_FUNC

#ifndef EIGEN_PURE_HOST_FUNC
    #define EIGEN_PURE_HOST_FUNC 
#endif // EIGEN_PURE_HOST_FUNC

#endif // VISIONCORE_HAVE_CUDA

// ---------------------------------------------------------------------------
// OpenCL
// ---------------------------------------------------------------------------
#ifdef VISIONCORE_HAVE_OPENCL
#include <CL/cl2.hpp>
#endif // VISIONCORE_HAVE_OPENCL

// ---------------------------------------------------------------------------
// Ceres-Solver
// ---------------------------------------------------------------------------
#ifdef VISIONCORE_HAVE_CERES
#include <ceres/jet.h>
#endif // VISIONCORE_HAVE_CERES

// ---------------------------------------------------------------------------
// Cereal - C++ Serializer
// ---------------------------------------------------------------------------
#if defined(VISIONCORE_HAVE_CEREAL) && !defined(__CUDACC__)
#define VISIONCORE_ENABLE_CEREAL
#include <cereal/cereal.hpp> // CUDA doesn't like cereal
#endif // VISIONCORE_ENABLE_CEREAL

// ---------------------------------------------------------------------------
// CUDA Macros and CUDA types if CUDA not available
// ---------------------------------------------------------------------------
#include <VisionCore/HelpersCUDA.hpp>

// ---------------------------------------------------------------------------
// Our internal type traits
// ---------------------------------------------------------------------------
#include <VisionCore/TypeTraits.hpp>

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------
#include <VisionCore/HelpersMisc.hpp>

// ---------------------------------------------------------------------------
// Our additions to Eigen & Sophus
// ---------------------------------------------------------------------------
#include <VisionCore/HelpersEigen.hpp>
#include <VisionCore/HelpersSophus.hpp>

// ---------------------------------------------------------------------------
// Random Helpers
// ---------------------------------------------------------------------------
#include <VisionCore/HelpersAutoDiff.hpp>

#endif // VISIONCORE_PLATFORM_HPP
