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
 * Image Filters.
 * ****************************************************************************
 */

#include <VisionCore/Image/Filters.hpp>

#include <VisionCore/LaunchUtils.hpp>

template<typename T, typename Target>
void vc::image::bilateral(const vc::Buffer2DView<T,Target>& img_in, vc::Buffer2DView<T,Target>& img_out, 
                          const T& gs, const T& gr, std::size_t dim)
{
    vc::launchParallelFor(img_in.width(), img_in.height(), [&](std::size_t x, std::size_t y)
    {
        const T& p = img_in(x,y);
        T sum = T(0.0);
        T sumw = T(0.0);
        
        for(int r = -(int)dim; r <= (int)dim; ++r ) 
        {
            for(int c = -(int)dim; c <= (int)dim; ++c ) 
            {
                const T& q = img_in.getWithClampedRange(x+c, y+r);
                const T sd2 = r*r + c*c;
                const T id = p-q;
                const T id2 = id*id;
                const T sw = exp(-(sd2) / (T(2.0) * gs * gs));
                const T iw = exp(-(id2) / (T(2.0) * gr * gr));
                const T w = sw*iw;
                sumw += w;
                sum += w * q;
            }
        }
        
        img_out(x,y) = (T)(sum / sumw);
    });
}

template<typename T, typename Target>
void vc::image::bilateral(const vc::Buffer2DView<T,Target>& img_in, vc::Buffer2DView<T,Target>& img_out, 
                          const T& gs, const T& gr, const T& minval, std::size_t dim)
{
    vc::launchParallelFor(img_in.width(), img_in.height(), [&](std::size_t x, std::size_t y)
    {
            const T& p = img_in(x,y);
            T sum = T(0.0);
            T sumw = T(0.0);
            
            if( p >= minval) {
                for(int r = -(int)dim; r <= (int)dim; ++r ) 
                {
                    for(int c = -(int)dim; c <= (int)dim; ++c ) 
                    {
                        const T& q = img_in.getWithClampedRange(x+c, y+r);
                        if(q >= minval) 
                        {
                            const T sd2 = r*r + c*c;
                            const T id = p-q;
                            const T id2 = id*id;
                            const T sw = exp(-(sd2) / (T(2.0) * gs * gs));
                            const T iw = exp(-(id2) / (T(2.0) * gr * gr));
                            const T w = sw*iw;
                            sumw += w;
                            sum += w * q;
                        }
                    }
                }
            }
            
            img_out(x,y) = (T)(sum / sumw);
    });
}

#define GEN_IMPL(OUR_TYPE) \
template void vc::image::bilateral<OUR_TYPE,vc::TargetHost>(const vc::Buffer2DView<OUR_TYPE,vc::TargetHost>& img_in, vc::Buffer2DView<OUR_TYPE,vc::TargetHost>& img_out, const OUR_TYPE& gs, const OUR_TYPE& gr, std::size_t dim); \
template void vc::image::bilateral<OUR_TYPE,vc::TargetHost>(const vc::Buffer2DView<OUR_TYPE,vc::TargetHost>& img_in, vc::Buffer2DView<OUR_TYPE,vc::TargetHost>& img_out, const OUR_TYPE& gs, const OUR_TYPE& gr, const OUR_TYPE& minval, std::size_t dim);

GEN_IMPL(float)
