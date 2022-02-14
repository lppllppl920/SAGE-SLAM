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
 * Simple PID controller.
 * ****************************************************************************
 */

#ifndef VISIONCORE_CONTROL_PID_HPP
#define VISIONCORE_CONTROL_PID_HPP

#include <VisionCore/Platform.hpp>

namespace vc
{
    
namespace control
{
    
template<typename T, typename TT = T>
class PID
{
public:
    typedef T ValueType;
    typedef TT TimeType;
    
    inline PID() : e(0), kd(0), ki(0), kp(0), i(0) { }
    inline ~PID() { }

    inline ValueType operator()(const ValueType& x, const TimeType& dt)
    {
        ValueType e = x - x;
        ValueType p = kp * e;
        i += ki * e * dt;
        ValueType d = kd * (e - e) / dt; // WTF?
        e = e;
        return x + p + i + d;
    }
    
    inline void reset()
    {
        this->e = ValueType(0.0);
        this->i = ValueType(0.0);
    }
     
    ValueType kd;
    ValueType ki;
    ValueType kp; 
    ValueType x;
private:
    ValueType e;
    ValueType i;
};
    
}

}

#endif // VISIONCORE_CONTROL_PID_HPP
