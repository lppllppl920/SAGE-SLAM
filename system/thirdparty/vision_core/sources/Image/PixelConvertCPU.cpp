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
 * Pixel Conversion Functions.
 * ****************************************************************************
 */

#include <VisionCore/Image/PixelConvert.hpp>

#include <VisionCore/LaunchUtils.hpp>

template<typename T_IN, typename T_OUT, typename Target>
void vc::image::convertBuffer(const vc::Buffer2DView<T_IN, Target>& buf_in, vc::Buffer2DView<T_OUT, Target>& buf_out)
{
    vc::launchParallelFor(buf_in.width(), buf_in.height(), [&](std::size_t x, std::size_t y)
    {
        if(buf_in.inBounds(x,y) && buf_out.inBounds(x,y)) // is valid
        {
            buf_out(x,y) = vc::image::convertPixel<T_OUT, T_IN>(buf_in(x,y));
        }
    });
}

// all conversions
template void vc::image::convertBuffer<uint8_t, uchar3>(const vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_in, vc::Buffer2DView<uchar3, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<uint8_t, uchar4>(const vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_in, vc::Buffer2DView<uchar4, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<uint8_t, float>(const vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_in, vc::Buffer2DView<float, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<uint8_t, float3>(const vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_in, vc::Buffer2DView<float3, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<uint8_t, float4>(const vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_in, vc::Buffer2DView<float4, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<uint8_t, Eigen::Vector3f>(const vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_in, vc::Buffer2DView<Eigen::Vector3f, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<uint8_t, Eigen::Vector4f>(const vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_in, vc::Buffer2DView<Eigen::Vector4f, vc::TargetHost>& buf_out);
//template void vc::image::convertBuffer<uint8_t, uint16_t>(const vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_in, vc::Buffer2DView<uint16_t, vc::TargetHost>& buf_out);

template void vc::image::convertBuffer<float, uint8_t>(const vc::Buffer2DView<float, vc::TargetHost>& buf_in, vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<float, uchar3>(const vc::Buffer2DView<float, vc::TargetHost>& buf_in, vc::Buffer2DView<uchar3, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<float, uchar4>(const vc::Buffer2DView<float, vc::TargetHost>& buf_in, vc::Buffer2DView<uchar4, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<float, float3>(const vc::Buffer2DView<float, vc::TargetHost>& buf_in, vc::Buffer2DView<float3, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<float, float4>(const vc::Buffer2DView<float, vc::TargetHost>& buf_in, vc::Buffer2DView<float4, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<float, Eigen::Vector3f>(const vc::Buffer2DView<float, vc::TargetHost>& buf_in, vc::Buffer2DView<Eigen::Vector3f, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<float, Eigen::Vector4f>(const vc::Buffer2DView<float, vc::TargetHost>& buf_in, vc::Buffer2DView<Eigen::Vector4f, vc::TargetHost>& buf_out);
//template void vc::image::convertBuffer<float, uint16_t>(const vc::Buffer2DView<float, vc::TargetHost>& buf_in, vc::Buffer2DView<uint16_t, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<float, double>(const vc::Buffer2DView<float, vc::TargetHost>& buf_in, vc::Buffer2DView<double, vc::TargetHost>& buf_out);

template void vc::image::convertBuffer<double, uint8_t>(const vc::Buffer2DView<double, vc::TargetHost>& buf_in, vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<double, uchar3>(const vc::Buffer2DView<double, vc::TargetHost>& buf_in, vc::Buffer2DView<uchar3, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<double, uchar4>(const vc::Buffer2DView<double, vc::TargetHost>& buf_in, vc::Buffer2DView<uchar4, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<double, float3>(const vc::Buffer2DView<double, vc::TargetHost>& buf_in, vc::Buffer2DView<float3, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<double, float4>(const vc::Buffer2DView<double, vc::TargetHost>& buf_in, vc::Buffer2DView<float4, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<double, Eigen::Vector3f>(const vc::Buffer2DView<double, vc::TargetHost>& buf_in, vc::Buffer2DView<Eigen::Vector3f, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<double, Eigen::Vector4f>(const vc::Buffer2DView<double, vc::TargetHost>& buf_in, vc::Buffer2DView<Eigen::Vector4f, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<double, float>(const vc::Buffer2DView<double, vc::TargetHost>& buf_in, vc::Buffer2DView<float, vc::TargetHost>& buf_out);

template void vc::image::convertBuffer<uchar3, uint8_t>(const vc::Buffer2DView<uchar3, vc::TargetHost>& buf_in, vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<uchar3, uchar4>(const vc::Buffer2DView<uchar3, vc::TargetHost>& buf_in, vc::Buffer2DView<uchar4, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<uchar3, float>(const vc::Buffer2DView<uchar3, vc::TargetHost>& buf_in, vc::Buffer2DView<float, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<uchar3, float3>(const vc::Buffer2DView<uchar3, vc::TargetHost>& buf_in, vc::Buffer2DView<float3, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<uchar3, float4>(const vc::Buffer2DView<uchar3, vc::TargetHost>& buf_in, vc::Buffer2DView<float4, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<uchar3, Eigen::Vector3f>(const vc::Buffer2DView<uchar3, vc::TargetHost>& buf_in, vc::Buffer2DView<Eigen::Vector3f, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<uchar3, Eigen::Vector4f>(const vc::Buffer2DView<uchar3, vc::TargetHost>& buf_in, vc::Buffer2DView<Eigen::Vector4f, vc::TargetHost>& buf_out);
//template void vc::image::convertBuffer<uchar3, uint16_t>(const vc::Buffer2DView<uchar3, vc::TargetHost>& buf_in, vc::Buffer2DView<uint16_t, vc::TargetHost>& buf_out);

template void vc::image::convertBuffer<uchar4, uint8_t>(const vc::Buffer2DView<uchar4, vc::TargetHost>& buf_in, vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<uchar4, uchar3>(const vc::Buffer2DView<uchar4, vc::TargetHost>& buf_in, vc::Buffer2DView<uchar3, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<uchar4, float>(const vc::Buffer2DView<uchar4, vc::TargetHost>& buf_in, vc::Buffer2DView<float, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<uchar4, float3>(const vc::Buffer2DView<uchar4, vc::TargetHost>& buf_in, vc::Buffer2DView<float3, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<uchar4, float4>(const vc::Buffer2DView<uchar4, vc::TargetHost>& buf_in, vc::Buffer2DView<float4, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<uchar4, Eigen::Vector3f>(const vc::Buffer2DView<uchar4, vc::TargetHost>& buf_in, vc::Buffer2DView<Eigen::Vector3f, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<uchar4, Eigen::Vector4f>(const vc::Buffer2DView<uchar4, vc::TargetHost>& buf_in, vc::Buffer2DView<Eigen::Vector4f, vc::TargetHost>& buf_out);
//template void vc::image::convertBuffer<uchar4, uint16_t>(vc::Buffer2DView<uchar4, vc::TargetHost>& buf_in, vc::Buffer2DView<uint16_t, vc::TargetHost>& buf_out);

template void vc::image::convertBuffer<float3, uint8_t>(const vc::Buffer2DView<float3, vc::TargetHost>& buf_in, vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<float3, uchar3>(const vc::Buffer2DView<float3, vc::TargetHost>& buf_in, vc::Buffer2DView<uchar3, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<float3, uchar4>(const vc::Buffer2DView<float3, vc::TargetHost>& buf_in, vc::Buffer2DView<uchar4, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<float3, float>(const vc::Buffer2DView<float3, vc::TargetHost>& buf_in, vc::Buffer2DView<float, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<float3, float4>(const vc::Buffer2DView<float3, vc::TargetHost>& buf_in, vc::Buffer2DView<float4, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<float3, Eigen::Vector3f>(const vc::Buffer2DView<float3, vc::TargetHost>& buf_in, vc::Buffer2DView<Eigen::Vector3f, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<float3, Eigen::Vector4f>(const vc::Buffer2DView<float3, vc::TargetHost>& buf_in, vc::Buffer2DView<Eigen::Vector4f, vc::TargetHost>& buf_out);
//template void vc::image::convertBuffer<float3, uint16_t>(const vc::Buffer2DView<float3, vc::TargetHost>& buf_in, vc::Buffer2DView<uint16_t, vc::TargetHost>& buf_out);

template void vc::image::convertBuffer<float4, uint8_t>(const vc::Buffer2DView<float4, vc::TargetHost>& buf_in, vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<float4, uchar3>(const vc::Buffer2DView<float4, vc::TargetHost>& buf_in, vc::Buffer2DView<uchar3, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<float4, uchar4>(const vc::Buffer2DView<float4, vc::TargetHost>& buf_in, vc::Buffer2DView<uchar4, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<float4, float>(const vc::Buffer2DView<float4, vc::TargetHost>& buf_in, vc::Buffer2DView<float, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<float4, float3>(const vc::Buffer2DView<float4, vc::TargetHost>& buf_in, vc::Buffer2DView<float3, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<float4, Eigen::Vector3f>(const vc::Buffer2DView<float4, vc::TargetHost>& buf_in, vc::Buffer2DView<Eigen::Vector3f, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<float4, Eigen::Vector4f>(const vc::Buffer2DView<float4, vc::TargetHost>& buf_in, vc::Buffer2DView<Eigen::Vector4f, vc::TargetHost>& buf_out);
//template void vc::image::convertBuffer<float4, uint16_t>(const vc::Buffer2DView<float4, vc::TargetHost>& buf_in, vc::Buffer2DView<uint16_t, vc::TargetHost>& buf_out);

template void vc::image::convertBuffer<Eigen::Vector3f, uint8_t>(const vc::Buffer2DView<Eigen::Vector3f, vc::TargetHost>& buf_in, vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<Eigen::Vector3f, uchar3>(const vc::Buffer2DView<Eigen::Vector3f, vc::TargetHost>& buf_in, vc::Buffer2DView<uchar3, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<Eigen::Vector3f, uchar4>(const vc::Buffer2DView<Eigen::Vector3f, vc::TargetHost>& buf_in, vc::Buffer2DView<uchar4, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<Eigen::Vector3f, float>(const vc::Buffer2DView<Eigen::Vector3f, vc::TargetHost>& buf_in, vc::Buffer2DView<float, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<Eigen::Vector3f, float3>(const vc::Buffer2DView<Eigen::Vector3f, vc::TargetHost>& buf_in, vc::Buffer2DView<float3, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<Eigen::Vector3f, float4>(const vc::Buffer2DView<Eigen::Vector3f, vc::TargetHost>& buf_in, vc::Buffer2DView<float4, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<Eigen::Vector3f, Eigen::Vector4f>(const vc::Buffer2DView<Eigen::Vector3f, vc::TargetHost>& buf_in, vc::Buffer2DView<Eigen::Vector4f, vc::TargetHost>& buf_out);
//template void vc::image::convertBuffer<Eigen::Vector3f, uint16_t>(const vc::Buffer2DView<Eigen::Vector3f, vc::TargetHost>& buf_in, vc::Buffer2DView<uint16_t, vc::TargetHost>& buf_out);

template void vc::image::convertBuffer<Eigen::Vector4f, uint8_t>(const vc::Buffer2DView<Eigen::Vector4f, vc::TargetHost>& buf_in, vc::Buffer2DView<uint8_t, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<Eigen::Vector4f, uchar3>(const vc::Buffer2DView<Eigen::Vector4f, vc::TargetHost>& buf_in, vc::Buffer2DView<uchar3, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<Eigen::Vector4f, uchar4>(const vc::Buffer2DView<Eigen::Vector4f, vc::TargetHost>& buf_in, vc::Buffer2DView<uchar4, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<Eigen::Vector4f, float>(const vc::Buffer2DView<Eigen::Vector4f, vc::TargetHost>& buf_in, vc::Buffer2DView<float, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<Eigen::Vector4f, float3>(const vc::Buffer2DView<Eigen::Vector4f, vc::TargetHost>& buf_in, vc::Buffer2DView<float3, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<Eigen::Vector4f, float4>(const vc::Buffer2DView<Eigen::Vector4f, vc::TargetHost>& buf_in, vc::Buffer2DView<float4, vc::TargetHost>& buf_out);
template void vc::image::convertBuffer<Eigen::Vector4f, Eigen::Vector3f>(const vc::Buffer2DView<Eigen::Vector4f, vc::TargetHost>& buf_in, vc::Buffer2DView<Eigen::Vector3f, vc::TargetHost>& buf_out);
//template void vc::image::convertBuffer<Eigen::Vector4f, uint16_t>(const vc::Buffer2DView<Eigen::Vector4f, vc::TargetHost>& buf_in, vc::Buffer2DView<uint16_t, vc::TargetHost>& buf_out);

// special
template void vc::image::convertBuffer<uint16_t, float>(const vc::Buffer2DView<uint16_t, vc::TargetHost>& buf_in, vc::Buffer2DView<float, vc::TargetHost>& buf_out);
