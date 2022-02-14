/**
 * ****************************************************************************
 * Copyright (c) 2018, Robert Lukierski.
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
 * Utility functions
 * ****************************************************************************
 */

#ifndef VISIONCORE_HELPERMISC_HPP
#define VISIONCORE_HELPERMISC_HPP

namespace vc
{
    namespace internal
    {
        template<typename T, std::float_round_style rs>
        struct RoundingDispatcher;
        
#ifndef VISIONCORE_CUDA_KERNEL_SPACE  
        template<typename T>
        struct RoundingDispatcher<T,std::float_round_style::round_toward_neg_infinity>
        {
            static inline T run(const T& v) { std::fesetround(FE_DOWNWARD); return std::round(v); }
        };
        
        template<typename T>
        struct RoundingDispatcher<T,std::float_round_style::round_to_nearest>
        {
            static inline T run(const T& v) { std::fesetround(FE_TONEAREST); return std::round(v); }
        };
        
        template<typename T>
        struct RoundingDispatcher<T,std::float_round_style::round_toward_zero>
        {
            static inline T run(const T& v) { std::fesetround(FE_TOWARDZERO); return std::round(v); }
        };
        
        template<typename T>
        struct RoundingDispatcher<T,std::float_round_style::round_toward_infinity>
        {
            static inline T run(const T& v) { std::fesetround(FE_UPWARD); return std::round(v); }
        };        
#else // !VISIONCORE_CUDA_KERNEL_SPACE
        template<>
        struct RoundingDispatcher<float,std::float_round_style::round_toward_neg_infinity>
        {
            EIGEN_DEVICE_FUNC static inline float run(const float& v) { return __float2int_rd(v); }
        };
        
        template<>
        struct RoundingDispatcher<float,std::float_round_style::round_to_nearest>
        {
            EIGEN_DEVICE_FUNC static inline float run(const float& v) { return __float2int_rn(v); }
        };
        
        template<>
        struct RoundingDispatcher<float,std::float_round_style::round_toward_zero>
        {
            EIGEN_DEVICE_FUNC static inline float run(const float& v) { return __float2int_ru(v); }
        };
        
        template<>
        struct RoundingDispatcher<float,std::float_round_style::round_toward_infinity>
        {
            EIGEN_DEVICE_FUNC static inline float run(const float& v) { return __float2int_rz(v); }
        };
#endif // VISIONCORE_CUDA_KERNEL_SPACE
    
        template<typename T>
        EIGEN_DEVICE_FUNC static inline bool isfinite_wrapper(const T& v)
        {
#ifndef VISIONCORE_CUDA_KERNEL_SPACE
            using std::isfinite;
#endif // VISIONCORE_CUDA_KERNEL_SPACE
            return isfinite(v);
        }
        
        template<typename T, int CC>
        struct cuda_type_funcs;
        
        template<typename TT> struct cuda_type_funcs<TT,1> 
        { 
            EIGEN_DEVICE_FUNC static inline void fill(TT& out, typename type_traits<TT>::ChannelType val) { out.x = val; } 
            EIGEN_DEVICE_FUNC static inline bool isvalid(const TT& val) { return isfinite_wrapper(val.x); }
            static inline void toStream(std::ostream& os, const TT& val) { os << "(" << val.x << ")"; }
        };
        
        template<typename TT> struct cuda_type_funcs<TT,2> 
        { 
            EIGEN_DEVICE_FUNC static inline void fill(TT& out, typename type_traits<TT>::ChannelType val) 
            { out.x = val; out.y = val; } 
            
            EIGEN_DEVICE_FUNC static inline bool isvalid(const TT& val) 
            { return isfinite_wrapper(val.x) && isfinite_wrapper(val.y); }
            
            static inline void toStream(std::ostream& os, const TT& val) 
            { os << "(" << val.x << "," << val.y << ")"; }
        };
        
        template<typename TT> struct cuda_type_funcs<TT,3> 
        { 
            EIGEN_DEVICE_FUNC static inline void fill(TT& out, typename type_traits<TT>::ChannelType val) 
            { out.x = val; out.y = val; out.z = val; } 
            
            EIGEN_DEVICE_FUNC static inline bool isvalid(const TT& val) 
            { return isfinite_wrapper(val.x) && isfinite_wrapper(val.y) && isfinite_wrapper(val.z); }
            
            static inline void toStream(std::ostream& os, const TT& val) 
            { os << "(" << val.x << "," << val.y << "," << val.z << ")"; }
        };
        
        template<typename TT> struct cuda_type_funcs<TT,4> 
        { 
            EIGEN_DEVICE_FUNC static inline void fill(TT& out, typename type_traits<TT>::ChannelType val) 
            { out.x = val; out.y = val; out.z = val; out.w = val; } 
            
            EIGEN_DEVICE_FUNC static inline bool isvalid(const TT& val) 
            { return isfinite_wrapper(val.x) && isfinite_wrapper(val.y) && isfinite_wrapper(val.z) && isfinite_wrapper(val.w); }
            
            static inline void toStream(std::ostream& os, const TT& val) 
            { os << "(" << val.x << "," << val.y << "," << val.z << "," << val.w << ")"; }
        };
        
        template<typename T> struct eigen_isvalid;
        
        template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
        struct eigen_isvalid<Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols>>
        {
            typedef Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> MatrixT;
            
            EIGEN_DEVICE_FUNC static inline bool isvalid(const MatrixT& val)
            {
                for(int r = 0 ; r < _Rows ; ++r)
                {
                    for(int c = 0 ; c < _Cols ; ++c)
                    {
                        if(!isfinite_wrapper(val(r,c)))
                        {
                            return false;
                        }
                    }
                }
                
                return true;
            }
        };
        
        template<typename T, bool is_eig = type_traits<T>::IsEigenType , bool is_cud = type_traits<T>::IsCUDAType >
        struct type_dispatcher_helper;
        
        // for Eigen
        template<typename T>
        struct type_dispatcher_helper<T,true,false>
        {
            EIGEN_DEVICE_FUNC static inline T fill(typename type_traits<T>::ChannelType v)
            {
                return T::Constant(v);
            }
            
            EIGEN_DEVICE_FUNC static inline bool isvalid(const T& v)
            {
                return eigen_isvalid<T>::isvalid(v);
            }
        };
        
        template<typename T>
        struct type_dispatcher_helper<T,false,true>
        {
            EIGEN_DEVICE_FUNC static inline T fill(typename type_traits<T>::ChannelType v)
            {
                T ret;
                cuda_type_funcs<T,type_traits<T>::ChannelCount>::fill(ret, v);
                return ret;
            }
            
            EIGEN_DEVICE_FUNC static inline bool isvalid(const T& v)
            {
                return cuda_type_funcs<T,type_traits<T>::ChannelCount>::isvalid(v);
            }
        };
        
        template<>
        struct type_dispatcher_helper<float,false,false>
        {
            EIGEN_DEVICE_FUNC static inline float fill(float v)
            {
                return v;
            }
            
            EIGEN_DEVICE_FUNC static inline bool isvalid(const float& v)
            {
                return isfinite_wrapper(v);
            }
        };
        
        template<>
        struct type_dispatcher_helper<double,false,false>
        {
            EIGEN_DEVICE_FUNC static inline float fill(double v)
            {
                return v;
            }
            
            EIGEN_DEVICE_FUNC static inline bool isvalid(const double& v)
            {
                return isfinite_wrapper(v);
            }
        };
        
        template<typename T>
        struct type_dispatcher_helper<T,false,false>
        {
            EIGEN_DEVICE_FUNC static inline T fill(T v)
            {
                return v;
            }
            
            EIGEN_DEVICE_FUNC static inline bool isvalid(const T& v)
            {
                return true;
            }
        };
    }
    
    template<typename T>
    EIGEN_DEVICE_FUNC static inline T zero()
    {
        return internal::type_dispatcher_helper<T>::fill(0.0);
    }
    
    template<typename T>
    EIGEN_DEVICE_FUNC static inline T setAll(typename type_traits<T>::ChannelType sv)
    {
        return internal::type_dispatcher_helper<T>::fill(sv);
    }
    
    template<typename T>
    EIGEN_DEVICE_FUNC static inline T getInvalid()
    {
        typedef typename type_traits<T>::ChannelType ScalarT;
        return internal::type_dispatcher_helper<T>::fill(std::numeric_limits<ScalarT>::quiet_NaN());
    }
    
    template<typename T>
    EIGEN_DEVICE_FUNC static inline bool isvalid(T val)
    {
        return internal::type_dispatcher_helper<T>::isvalid(val);
    }
    
    template<typename T, std::float_round_style rs = std::float_round_style::round_to_nearest>
    EIGEN_DEVICE_FUNC inline T round(const T& v)
    {
        return internal::RoundingDispatcher<T,rs>::run(v);
    }
}

inline EIGEN_DEVICE_FUNC float lerp(unsigned char a, unsigned char b, float t)
{
    return (float)a + t*((float)b-(float)a);
}

inline EIGEN_DEVICE_FUNC float2 lerp(uchar2 a, uchar2 b, float t)
{
    return make_float2(
        a.x + t*(b.x-a.x),
        a.y + t*(b.y-a.y)
    );
}

inline EIGEN_DEVICE_FUNC float3 lerp(uchar3 a, uchar3 b, float t)
{
    return make_float3(
        a.x + t*(b.x-a.x),
        a.y + t*(b.y-a.y),
        a.z + t*(b.z-a.z)
    );
}

inline EIGEN_DEVICE_FUNC float4 lerp(uchar4 a, uchar4 b, float t)
{
    return make_float4(
        a.x + t*(b.x-a.x),
        a.y + t*(b.y-a.y),
        a.z + t*(b.z-a.z),
        a.w + t*(b.w-a.w)
    );
}

template <typename T, int Rows, int Cols>
inline EIGEN_DEVICE_FUNC Eigen::Matrix<T,Rows,Cols> 
lerp(const Eigen::Matrix<T,Rows,Cols>& a, const Eigen::Matrix<T,Rows,Cols>& b, T t)
{
    return a + t * (b - a);
}

#define GENERATE_CUDATYPE_OPERATOR(XXXXX)  \
inline std::ostream& operator<<(std::ostream& os, const XXXXX& p) \
{ \
    vc::internal::cuda_type_funcs<XXXXX,vc::type_traits<XXXXX>::ChannelCount>::toStream(os,p); \
    return os; \
}

GENERATE_CUDATYPE_OPERATOR(char1)
GENERATE_CUDATYPE_OPERATOR(uchar1)
GENERATE_CUDATYPE_OPERATOR(char2)
GENERATE_CUDATYPE_OPERATOR(uchar2)
GENERATE_CUDATYPE_OPERATOR(char3)
GENERATE_CUDATYPE_OPERATOR(uchar3)
GENERATE_CUDATYPE_OPERATOR(char4)
GENERATE_CUDATYPE_OPERATOR(uchar4)
GENERATE_CUDATYPE_OPERATOR(short1)
GENERATE_CUDATYPE_OPERATOR(ushort1)
GENERATE_CUDATYPE_OPERATOR(short2)
GENERATE_CUDATYPE_OPERATOR(ushort2)
GENERATE_CUDATYPE_OPERATOR(short3)
GENERATE_CUDATYPE_OPERATOR(ushort3)
GENERATE_CUDATYPE_OPERATOR(short4)
GENERATE_CUDATYPE_OPERATOR(ushort4)
GENERATE_CUDATYPE_OPERATOR(int1)
GENERATE_CUDATYPE_OPERATOR(uint1)
GENERATE_CUDATYPE_OPERATOR(int2)
GENERATE_CUDATYPE_OPERATOR(uint2)
GENERATE_CUDATYPE_OPERATOR(int3)
GENERATE_CUDATYPE_OPERATOR(uint3)
GENERATE_CUDATYPE_OPERATOR(int4)
GENERATE_CUDATYPE_OPERATOR(uint4)
GENERATE_CUDATYPE_OPERATOR(long1)
GENERATE_CUDATYPE_OPERATOR(ulong1)
GENERATE_CUDATYPE_OPERATOR(long2)
GENERATE_CUDATYPE_OPERATOR(ulong2)
GENERATE_CUDATYPE_OPERATOR(long3)
GENERATE_CUDATYPE_OPERATOR(ulong3)
GENERATE_CUDATYPE_OPERATOR(long4)
GENERATE_CUDATYPE_OPERATOR(ulong4)
GENERATE_CUDATYPE_OPERATOR(float1)
GENERATE_CUDATYPE_OPERATOR(float2)
GENERATE_CUDATYPE_OPERATOR(float3)
GENERATE_CUDATYPE_OPERATOR(float4)
GENERATE_CUDATYPE_OPERATOR(longlong1)
GENERATE_CUDATYPE_OPERATOR(ulonglong1)
GENERATE_CUDATYPE_OPERATOR(longlong2)
GENERATE_CUDATYPE_OPERATOR(ulonglong2)
GENERATE_CUDATYPE_OPERATOR(longlong3)
GENERATE_CUDATYPE_OPERATOR(ulonglong3)
GENERATE_CUDATYPE_OPERATOR(longlong4)
GENERATE_CUDATYPE_OPERATOR(ulonglong4)
GENERATE_CUDATYPE_OPERATOR(double1)
GENERATE_CUDATYPE_OPERATOR(double2)
GENERATE_CUDATYPE_OPERATOR(double3)
GENERATE_CUDATYPE_OPERATOR(double4)
#undef GENERATE_CUDATYPE_OPERATOR

#endif // VISIONCORE_HELPERMISC_HPP
