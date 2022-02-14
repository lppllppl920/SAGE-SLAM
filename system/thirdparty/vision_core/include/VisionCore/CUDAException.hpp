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
 * CUDA Exception.
 * ****************************************************************************
 */

#ifndef VISIONCORE_CUDAEXCEPTION_HPP
#define VISIONCORE_CUDAEXCEPTION_HPP

#ifdef VISIONCORE_HAVE_CUDA

// system
#include <stdexcept>
#include <string>
#include <sstream>

// CUDA
#include <cuda_runtime.h>

namespace vc
{

class CUDAException : public std::exception
{
public:
    CUDAException(cudaError err = cudaSuccess, const std::string& what = "") : mWhat(what), mErr(err) 
    { 
        std::stringstream ss;
        
        ss << "CUDAException: " << mWhat << std::endl;
        
        if(mErr != cudaSuccess) 
        {
            ss << "cudaError code: " << cudaGetErrorString(mErr);
            ss << " (" << mErr << ")" << std::endl;
        }
        
        mWhat = ss.str();
    }
    
    virtual ~CUDAException() throw() {}
    
    virtual const char* what() const throw() 
    {
        return mWhat.c_str();
    }
    
    cudaError getError() const { return mErr; }
private:
    std::string mWhat;
    cudaError mErr;
};

}

#endif // VISIONCORE_HAVE_CUDA

#endif // VISIONCORE_CUDAEXCEPTION_HPP
