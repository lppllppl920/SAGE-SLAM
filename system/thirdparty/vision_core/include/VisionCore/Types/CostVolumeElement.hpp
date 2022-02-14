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
 * Cost Volume Element.
 * ****************************************************************************
 */

#ifndef VISIONCORE_TYPES_COSTVOLUMEELEMENT_HPP
#define VISIONCORE_TYPES_COSTVOLUMEELEMENT_HPP

#include <VisionCore/Platform.hpp>

namespace vc
{
    
namespace types
{

template<typename T>    
struct VISIONCORE_ALIGN(8) CostVolumeElement
{
    EIGEN_DEVICE_FUNC inline CostVolumeElement() { }

    EIGEN_DEVICE_FUNC inline CostVolumeElement(T v)
    {
        Cost = v;
        Count = 1;
    }

    EIGEN_DEVICE_FUNC inline operator T() 
    {
        return get();
    }

    EIGEN_DEVICE_FUNC inline T get() const 
    {
        return Count > 0 ? Cost / (T)Count : __FLT_MAX__;
    }

    EIGEN_DEVICE_FUNC inline CostVolumeElement& operator=(const CostVolumeElement& other)
    {
        Cost = other.Cost;
        Count = other.Count;
        
        return *this;
    }

    EIGEN_DEVICE_FUNC inline CostVolumeElement& operator=(T v)
    {
        Cost = v;
        Count = 1;
        return *this;
    }

    EIGEN_DEVICE_FUNC inline CostVolumeElement& operator=(int v)
    {
        Cost = v;
        Count = 1;
        return *this;
    }

    unsigned int Count;
    T Cost;
};
    
}

}

#endif // VISIONCORE_TYPES_COSTVOLUMEELEMENT_HPP
