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
 * Pixel Conversion Functions.
 * ****************************************************************************
 */

#ifndef VISIONCORE_IMAGE_CONVERT_HPP
#define VISIONCORE_IMAGE_CONVERT_HPP

#include <VisionCore/Platform.hpp>

#include <VisionCore/Buffers/Buffer2D.hpp>

namespace vc
{
    
namespace image
{

template<typename To, typename Ti> EIGEN_DEVICE_FUNC inline To convertPixel(Ti p) { return p; }

// from uint8_t
template<> EIGEN_DEVICE_FUNC inline uchar3 convertPixel(uint8_t p) { return make_uchar3(p,p,p); }
template<> EIGEN_DEVICE_FUNC inline uchar4 convertPixel(uint8_t p) { return make_uchar4(p,p,p,255); }
template<> EIGEN_DEVICE_FUNC inline float convertPixel(uint8_t p) { return p / 255.0f; }
template<> EIGEN_DEVICE_FUNC inline float3 convertPixel(uint8_t p) { return make_float3(p / 255.0f, p / 255.0f, p / 255.0f); }
template<> EIGEN_DEVICE_FUNC inline float4 convertPixel(uint8_t p) { return make_float4(p / 255.0f, p / 255.0f, p / 255.0f, 1.0f); }
template<> EIGEN_DEVICE_FUNC inline Eigen::Vector3f convertPixel(uint8_t p) { return Eigen::Vector3f(p / 255.0f, p / 255.0f, p / 255.0f); }
template<> EIGEN_DEVICE_FUNC inline Eigen::Vector4f convertPixel(uint8_t p) { return Eigen::Vector4f(p / 255.0f, p / 255.0f, p / 255.0f, 1.0f); }

// from float
template<> EIGEN_DEVICE_FUNC inline uint8_t convertPixel(float p) { return p * 255.0f; }
template<> EIGEN_DEVICE_FUNC inline uchar3 convertPixel(float p) { return make_uchar3(p*255.0f,p*255.0f,p*255.0f); }
template<> EIGEN_DEVICE_FUNC inline uchar4 convertPixel(float p) { return make_uchar4(p*255.0f,p*255.0f,p*255.0f,255.0f); }
template<> EIGEN_DEVICE_FUNC inline float3 convertPixel(float p) { return make_float3(p,p,p); }
template<> EIGEN_DEVICE_FUNC inline float4 convertPixel(float p) { return make_float4(p,p,p,1.0f); }
template<> EIGEN_DEVICE_FUNC inline Eigen::Vector3f convertPixel(float p) { return Eigen::Vector3f(p,p,p); }
template<> EIGEN_DEVICE_FUNC inline Eigen::Vector4f convertPixel(float p) { return Eigen::Vector4f(p,p,p,1.0f); }
template<> EIGEN_DEVICE_FUNC inline double convertPixel(float p) { return p; }

// from double
template<> EIGEN_DEVICE_FUNC inline uint8_t convertPixel(double p) { return p * 255.0; }
template<> EIGEN_DEVICE_FUNC inline uchar3 convertPixel(double p) { return make_uchar3(p*255.0,p*255.0f,p*255.0); }
template<> EIGEN_DEVICE_FUNC inline uchar4 convertPixel(double p) { return make_uchar4(p*255.0,p*255.0f,p*255.0,255.0); }
template<> EIGEN_DEVICE_FUNC inline float3 convertPixel(double p) { return make_float3(p,p,p); }
template<> EIGEN_DEVICE_FUNC inline float4 convertPixel(double p) { return make_float4(p,p,p,1.0); }
template<> EIGEN_DEVICE_FUNC inline Eigen::Vector3f convertPixel(double p) { return Eigen::Vector3f(p,p,p); }
template<> EIGEN_DEVICE_FUNC inline Eigen::Vector4f convertPixel(double p) { return Eigen::Vector4f(p,p,p,1.0); }
template<> EIGEN_DEVICE_FUNC inline float convertPixel(double p) { return p; }

// from uchar3
template<> EIGEN_DEVICE_FUNC inline uint8_t convertPixel(uchar3 p) { const unsigned sum = p.x + p.y + p.z; return sum / 3; }
template<> EIGEN_DEVICE_FUNC inline uchar4 convertPixel(uchar3 p) { return make_uchar4(p.x, p.y, p.z, 255); }
template<> EIGEN_DEVICE_FUNC inline float convertPixel(uchar3 p) { return (float)(p.x + p.y + p.z) / (3.0f * 255.0f); }
template<> EIGEN_DEVICE_FUNC inline float3 convertPixel(uchar3 p) { return make_float3(p.x / 255.0f, p.y / 255.0f, p.z / 255.0f); }
template<> EIGEN_DEVICE_FUNC inline float4 convertPixel(uchar3 p) { return make_float4(p.x / 255.0f, p.y / 255.0f, p.z / 255.0f, 1.0f); }
template<> EIGEN_DEVICE_FUNC inline Eigen::Vector3f convertPixel(uchar3 p) { return Eigen::Vector3f(p.x / 255.0f, p.y / 255.0f, p.z / 255.0f); }
template<> EIGEN_DEVICE_FUNC inline Eigen::Vector4f convertPixel(uchar3 p) { return Eigen::Vector4f(p.x / 255.0f, p.y / 255.0f, p.z / 255.0f, 1.0f); }

// from uchar4
template<> EIGEN_DEVICE_FUNC inline uint8_t convertPixel(uchar4 p) { const unsigned sum = p.x + p.y + p.z; return sum / 3; } // NOTE no alpha
template<> EIGEN_DEVICE_FUNC inline uchar3 convertPixel(uchar4 p) { return make_uchar3(p.x,p.y,p.z); } // NOTE no alpha
template<> EIGEN_DEVICE_FUNC inline float convertPixel(uchar4 p) { return (p.x + p.y + p.z) / (3.0f * 255.0f); } // NOTE no alpha
template<> EIGEN_DEVICE_FUNC inline float3 convertPixel(uchar4 p) { return make_float3(p.x / 255.0f,p.y / 255.0f,p.z / 255.0f); } // NOTE no alpha
template<> EIGEN_DEVICE_FUNC inline float4 convertPixel(uchar4 p) { return make_float4(p.x / 255.0f,p.y / 255.0f,p.z / 255.0f, p.w / 255.0f); }
template<> EIGEN_DEVICE_FUNC inline Eigen::Vector3f convertPixel(uchar4 p) { return Eigen::Vector3f(p.x / 255.0f, p.y / 255.0f, p.z / 255.0f); } // NOTE no alpha
template<> EIGEN_DEVICE_FUNC inline Eigen::Vector4f convertPixel(uchar4 p) { return Eigen::Vector4f(p.x / 255.0f, p.y / 255.0f, p.z / 255.0f, p.w / 255.0f); }

// from float3
template<> EIGEN_DEVICE_FUNC inline uint8_t convertPixel(float3 p) { return ((p.x + p.y + p.z) / 3.0f) * 255; }
template<> EIGEN_DEVICE_FUNC inline uchar3 convertPixel(float3 p) { return make_uchar3(p.x * 255, p.y * 255, p.z * 255); }
template<> EIGEN_DEVICE_FUNC inline uchar4 convertPixel(float3 p) { return make_uchar4(p.x * 255, p.y * 255, p.z * 255, 255); }
template<> EIGEN_DEVICE_FUNC inline float convertPixel(float3 p) { return (p.x + p.y + p.z) / 3.0f; }
template<> EIGEN_DEVICE_FUNC inline float4 convertPixel(float3 p) { return make_float4(p.x, p.y, p.z, 1.0f); }
template<> EIGEN_DEVICE_FUNC inline Eigen::Vector3f convertPixel(float3 p) { return Eigen::Vector3f(p.x, p.y, p.z); }
template<> EIGEN_DEVICE_FUNC inline Eigen::Vector4f convertPixel(float3 p) { return Eigen::Vector4f(p.x, p.y, p.z, 1.0f); }

// from float4
template<> EIGEN_DEVICE_FUNC inline uint8_t convertPixel(float4 p) { return ((p.x + p.y + p.z) / 3.0f) * 255; } // NOTE no alpha
template<> EIGEN_DEVICE_FUNC inline uchar3 convertPixel(float4 p) { return make_uchar3(p.x * 255, p.y * 255, p.z * 255); } // NOTE no alpha
template<> EIGEN_DEVICE_FUNC inline uchar4 convertPixel(float4 p) { return make_uchar4(p.x * 255, p.y * 255, p.z * 255, p.w * 255); }
template<> EIGEN_DEVICE_FUNC inline float convertPixel(float4 p) { return (p.x + p.y + p.z) / 3.0f; } // NOTE no alpha
template<> EIGEN_DEVICE_FUNC inline float3 convertPixel(float4 p) { return make_float3(p.x, p.y, p.z); } // NOTE no alpha
template<> EIGEN_DEVICE_FUNC inline Eigen::Vector3f convertPixel(float4 p) { return Eigen::Vector3f(p.x, p.y, p.z); } // NOTE no alpha
template<> EIGEN_DEVICE_FUNC inline Eigen::Vector4f convertPixel(float4 p) { return Eigen::Vector4f(p.x, p.y, p.z, p.w); }

// from Eigen::Vector3f
template<> EIGEN_DEVICE_FUNC inline uint8_t convertPixel(Eigen::Vector3f p) { return ((p(0) + p(1) + p(2)) / 3.0f) * 255; }
template<> EIGEN_DEVICE_FUNC inline uchar3 convertPixel(Eigen::Vector3f p) { return make_uchar3(p(0) * 255, p(1) * 255, p(2) * 255); }
template<> EIGEN_DEVICE_FUNC inline uchar4 convertPixel(Eigen::Vector3f p) { return make_uchar4(p(0) * 255, p(1) * 255, p(2) * 255, 255); }
template<> EIGEN_DEVICE_FUNC inline float convertPixel(Eigen::Vector3f p) { return (p(0) + p(1) + p(2)) / 3.0f; }
template<> EIGEN_DEVICE_FUNC inline float3 convertPixel(Eigen::Vector3f p) { return make_float3(p(0), p(1), p(2)); }
template<> EIGEN_DEVICE_FUNC inline float4 convertPixel(Eigen::Vector3f p) { return make_float4(p(0), p(1), p(2), 1.0f); }
template<> EIGEN_DEVICE_FUNC inline Eigen::Vector4f convertPixel(Eigen::Vector3f p) { return Eigen::Vector4f(p(0), p(1), p(2), 1.0f); }

// from Eigen::Vector4f
template<> EIGEN_DEVICE_FUNC inline uint8_t convertPixel(Eigen::Vector4f p) { return ((p(0) + p(1) + p(2)) / 3.0f) * 255; } // NOTE no alpha
template<> EIGEN_DEVICE_FUNC inline uchar3 convertPixel(Eigen::Vector4f p) { return make_uchar3(p(0) * 255, p(1) * 255, p(2) * 255); } // NOTE no alpha
template<> EIGEN_DEVICE_FUNC inline uchar4 convertPixel(Eigen::Vector4f p) { return make_uchar4(p(0) * 255, p(1) * 255, p(2) * 255, p(3) * 255); }
template<> EIGEN_DEVICE_FUNC inline float convertPixel(Eigen::Vector4f p) { return (p(0) + p(1) + p(2)) / 3.0f; }
template<> EIGEN_DEVICE_FUNC inline float3 convertPixel(Eigen::Vector4f p) { return make_float3(p(0), p(1), p(2)); }
template<> EIGEN_DEVICE_FUNC inline float4 convertPixel(Eigen::Vector4f p) { return make_float4(p(0), p(1), p(2), p(3)); }
template<> EIGEN_DEVICE_FUNC inline Eigen::Vector3f convertPixel(Eigen::Vector4f p) { return Eigen::Vector3f(p(0), p(1), p(2)); }

/**
 * Convert 2D buffer between element types.
 */
template<typename T_IN, typename T_OUT, typename Target>
void convertBuffer(const Buffer2DView<T_IN, Target>& buf_in, Buffer2DView<T_OUT, Target>& buf_out);

}

}

#endif // VISIONCORE_IMAGE_CONVERT_HPP
