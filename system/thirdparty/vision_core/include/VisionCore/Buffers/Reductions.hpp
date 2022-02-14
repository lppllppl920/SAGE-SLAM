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
 * Helper for CUDA reductions.
 * ****************************************************************************
 */

#ifndef VISIONCORE_REDUCTIONS_HPP
#define VISIONCORE_REDUCTIONS_HPP

#ifdef VISIONCORE_HAVE_CUDA

#include <VisionCore/Platform.hpp>
#include <VisionCore/LaunchUtils.hpp>
#include <VisionCore/CUDAGenerics.hpp>

#include <VisionCore/Buffers/Buffer1D.hpp>

#ifdef VISIONCORE_CUDA_COMPILER
namespace vc
{

namespace internal
{

template<typename T>
inline EIGEN_PURE_DEVICE_FUNC void warpReduceSum(T& val)
{
    for(unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val += vc::shfl_down(val, offset);
    }
}

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline EIGEN_PURE_DEVICE_FUNC void warpReduceSum(
    Eigen::Ref<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> val)
{
    for(unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        #pragma unroll
        for(int c = 0; c < _Cols ; c++) // NOTE: assuming column major
        {
            #pragma unroll
            for(int r = 0; r < _Rows ; r++)
            {
                val(r,c) += vc::shfl_down(val(r,c), offset);
            }
        }
    }
}

template<typename T, typename WarpReduceFunT>
inline EIGEN_PURE_DEVICE_FUNC void blockReduce(T* val, WarpReduceFunT wrf, const T& zero)
{
    //typedef T VectorArrayT[SharedCount]; 
    //FIXED_SIZE_SHARED_VAR(sharedMem, VectorArrayT); // Shared mem for partial sums
    SharedMemory<T> sharedMem;
        
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wid = threadIdx.x / warpSize;
    
    wrf(*val); // Each warp performs partial reduction
    
    if(lane == 0)  { sharedMem[wid] = *val; } // Write reduced value to shared memory
    
    __syncthreads(); // Wait for all partial reductions
    
    //read from shared memory only if that warp existed
    *val = (threadIdx.x < blockDim.x / warpSize) ? sharedMem[lane] : zero;
    
    // Final reduce within first warp
    if(wid == 0) { wrf(*val); }
}
   
}

template<typename ReductionBodyT>
inline EIGEN_PURE_DEVICE_FUNC void runReductions(unsigned int Nblocks, ReductionBodyT rb)
{
    for(unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < Nblocks; i += blockDim.x * gridDim.x)
    {
        rb(i);
    }
}

template<typename T, typename WarpReduceFunT>
inline EIGEN_PURE_DEVICE_FUNC void finalizeReduction(T* block_scratch, T* curr_sum, WarpReduceFunT wrf, const T& zero)
{
    internal::blockReduce(curr_sum, wrf, zero);
    
    if(threadIdx.x == 0)
    {
        block_scratch[blockIdx.x] = *curr_sum;
    }
}

/*
{
  const std::size_t threads = 512;
  const std::size_t blocks = std::min((N + threads - 1) / threads, 1024);

  kernel<<<blocks, threads>>>(in, out, N);
  kernel<<<1, 1024>>>(out, out, blocks);
}
*/
    
}
#endif // VISIONCORE_CUDA_COMPILER
#endif // VISIONCORE_HAVE_CUDA

#endif // VISIONCORE_REDUCTIONS_HPP
