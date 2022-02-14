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
 * Liang-Barsky Intersection Test.
 * ****************************************************************************
 */

#ifndef VISIONCORE_MATH_LIANG_BARSKY_HPP
#define VISIONCORE_MATH_LIANG_BARSKY_HPP

#include <VisionCore/Platform.hpp>


namespace vc
{
    
namespace math
{
    
template<typename T>
class LiangBarsky
{
public:
    EIGEN_DEVICE_FUNC LiangBarsky() { }
    
    EIGEN_DEVICE_FUNC LiangBarsky(T x1, T y1, T x2, T y2) : clipping_x1(x1), clipping_y1(y1), clipping_x2(x2), clipping_y2(y2)
    {
        
    }
    
    EIGEN_DEVICE_FUNC void setBoundingBox(T x1, T y1, T x2, T y2)
    {
        clipping_x1 = x1;
        clipping_y1 = y1;
        clipping_x2 = x2;
        clipping_y2 = y2;
    }
    
    EIGEN_DEVICE_FUNC bool intersects(T x1, T y1, T x2, T y2)
    {
        T dx, dy, tE, tL;
        
        dx = x2 - x1;
        dy = y2 - y1;
        
        if (is_zero(dx) && is_zero(dy) && point_inside(x1, y1)) { return true; }
        
        tE = 0;
        tL = 1;
        
        if (clipT(clipping_x1 - (T) x1,  dx, &tE, &tL) &&
            clipT((T) x1 - clipping_x2, -dx, &tE, &tL) &&
            clipT(clipping_y1 - (T) y1,  dy, &tE, &tL) &&
            clipT((T) y1 - clipping_y2, -dy, &tE, &tL)) 
        {
            if (tL < T(1.0)) 
            {
                x2 = (T) x1 + tL * dx;
                y2 = (T) y1 + tL * dy;
            }
            
            if (tE > 0) 
            {
                x1 += tE * dx;
                y1 += tE * dy;
            }
            
            return true;
        }
        
        return false;
    }
private:
    EIGEN_DEVICE_FUNC inline bool is_zero(T v) { return (v > -std::numeric_limits<T>::min() && v < std::numeric_limits<T>::min()); }
    EIGEN_DEVICE_FUNC inline bool point_inside(T x, T y) { return (x >= clipping_x1 && x <= clipping_x2 && y >= clipping_y1 && y <= clipping_y2); }
    EIGEN_DEVICE_FUNC inline int clipT(T num, T denom, T *tE, T *tL)
    {
        T t;
        
        if (is_zero(denom)) return (num <= T(0.0));
        
        t = num / denom;
        
        if (denom > T(0.0)) 
        {    
            if (t > *tL) return T(0.0);
            if (t > *tE) *tE = t;
            
        } 
        else 
        {
            if (t < *tE) return T(0.0);
            if (t < *tL) *tL = t;
        }
        
        return T(1.0);
    }
    
    T clipping_x1, clipping_y1, clipping_x2, clipping_y2;
};

}

}

#endif // VISIONCORE_MATH_LIANG_BARSKY_HPP
