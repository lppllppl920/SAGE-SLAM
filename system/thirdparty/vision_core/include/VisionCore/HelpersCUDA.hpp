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
 * Helper CUDA macros, types and original types if CUDA not available.
 * ****************************************************************************
 */

#ifndef VISIONCORE_HELPERSCUDA_HPP
#define VISIONCORE_HELPERSCUDA_HPP

#include <cstdint>
#include <cfenv>

// CUDA
#ifdef VISIONCORE_HAVE_CUDA
   
    /**
     * This overcomes stupid warnings.
     */
    #define FIXED_SIZE_SHARED_VAR(NAME, OURTYPE) \
    static __shared__ unsigned char NAME##MEM[sizeof(OURTYPE)]; \
    OURTYPE (&NAME) = reinterpret_cast<OURTYPE(&)>(NAME##MEM)
    
namespace vc
{
    template<typename T>
    struct SharedMemory
    {
        EIGEN_PURE_DEVICE_FUNC inline const T* ptr() const 
        {
            extern __device__ void error(void);
            error();
            return NULL;
        }
    };

    #define SPECIALIZE(TYPE) \
    template<> struct SharedMemory<TYPE> { \
        EIGEN_PURE_DEVICE_FUNC inline TYPE* ptr() const { extern __shared__ TYPE smem_##TYPE[]; return smem_##TYPE;  } \
        EIGEN_PURE_DEVICE_FUNC inline TYPE* ptr(std::size_t idx) { return (ptr() + idx); } \
        EIGEN_PURE_DEVICE_FUNC inline const TYPE* ptr(std::size_t idx) const { return (ptr() + idx); } \
        EIGEN_PURE_DEVICE_FUNC inline TYPE& operator()(std::size_t s) { return *ptr(s); } \
        EIGEN_PURE_DEVICE_FUNC inline const TYPE& operator()(std::size_t s) const { return *ptr(s); } \
        EIGEN_PURE_DEVICE_FUNC inline TYPE& operator[](std::size_t ix) { return operator()(ix); } \
        EIGEN_PURE_DEVICE_FUNC inline const TYPE& operator[](std::size_t ix) const { return operator()(ix); } \
    };
    
    #define SPECIALIZE_EIGEN(NAME,TYPE,DIMX,DIMY) \
    template<> struct SharedMemory<Eigen::Matrix<TYPE,DIMX,DIMY>> { \
        EIGEN_PURE_DEVICE_FUNC inline Eigen::Matrix<TYPE,DIMX,DIMY>* ptr() const { extern __shared__ Eigen::Matrix<TYPE,DIMX,DIMY> smem_##NAME[]; return smem_##NAME;  } \
        EIGEN_PURE_DEVICE_FUNC inline Eigen::Matrix<TYPE,DIMX,DIMY>* ptr(std::size_t idx) { return (ptr() + idx); } \
        EIGEN_PURE_DEVICE_FUNC inline const Eigen::Matrix<TYPE,DIMX,DIMY>* ptr(std::size_t idx) const { return (ptr() + idx); } \
        EIGEN_PURE_DEVICE_FUNC inline Eigen::Matrix<TYPE,DIMX,DIMY>& operator()(std::size_t s) { return *ptr(s); } \
        EIGEN_PURE_DEVICE_FUNC inline const Eigen::Matrix<TYPE,DIMX,DIMY>& operator()(std::size_t s) const { return *ptr(s); } \
        EIGEN_PURE_DEVICE_FUNC inline Eigen::Matrix<TYPE,DIMX,DIMY>& operator[](std::size_t ix) { return operator()(ix); } \
        EIGEN_PURE_DEVICE_FUNC inline const Eigen::Matrix<TYPE,DIMX,DIMY>& operator[](std::size_t ix) const { return operator()(ix); } \
    };
    
    SPECIALIZE(uint8_t)
    SPECIALIZE(uint16_t)
    SPECIALIZE(uint32_t)
    SPECIALIZE(int8_t)
    SPECIALIZE(int16_t)
    SPECIALIZE(int32_t)
    SPECIALIZE(float)
    SPECIALIZE(double)
    SPECIALIZE(char1)
    SPECIALIZE(uchar1)
    SPECIALIZE(char2)
    SPECIALIZE(uchar2)
    SPECIALIZE(char3)
    SPECIALIZE(uchar3)
    SPECIALIZE(char4)
    SPECIALIZE(uchar4)
    SPECIALIZE(short1)
    SPECIALIZE(ushort1)
    SPECIALIZE(short2)
    SPECIALIZE(ushort2)
    SPECIALIZE(short3)
    SPECIALIZE(ushort3)
    SPECIALIZE(short4)
    SPECIALIZE(ushort4)
    SPECIALIZE(int1)
    SPECIALIZE(uint1)
    SPECIALIZE(int2)
    SPECIALIZE(uint2)
    SPECIALIZE(int3)
    SPECIALIZE(uint3)
    SPECIALIZE(int4)
    SPECIALIZE(uint4)
    SPECIALIZE(long1)
    SPECIALIZE(ulong1)
    SPECIALIZE(long2)
    SPECIALIZE(ulong2)
    SPECIALIZE(long3)
    SPECIALIZE(ulong3)
    SPECIALIZE(long4)
    SPECIALIZE(ulong4)
    SPECIALIZE(float1)
    SPECIALIZE(float2)
    SPECIALIZE(float3)
    SPECIALIZE(float4)
    SPECIALIZE(longlong1)
    SPECIALIZE(ulonglong1)
    SPECIALIZE(longlong2)
    SPECIALIZE(ulonglong2)
    SPECIALIZE(longlong3)
    SPECIALIZE(ulonglong3)
    SPECIALIZE(longlong4)
    SPECIALIZE(ulonglong4)
    SPECIALIZE_EIGEN(emf11,float,1,1)
    SPECIALIZE_EIGEN(emf21,float,2,1)
    SPECIALIZE_EIGEN(emf31,float,3,1)
    SPECIALIZE_EIGEN(emf41,float,4,1)
    SPECIALIZE_EIGEN(emf51,float,5,1)
    SPECIALIZE_EIGEN(emf61,float,6,1)
    SPECIALIZE_EIGEN(emf22,float,2,2)
    SPECIALIZE_EIGEN(emf33,float,3,3)
    SPECIALIZE_EIGEN(emf44,float,4,4)
    SPECIALIZE_EIGEN(emf55,float,5,5)
    SPECIALIZE_EIGEN(emf66,float,6,6)
    #undef SPECIALIZE
    #undef SPECIALIZE_EIGEN      
}

#else // !VISIONCORE_HAVE_CUDA
/**
  * We don't have CUDA available, but want the types.
  */

#include <cmath>

// redefine types
typedef struct 
{
    signed char x;
} char1;

typedef struct
{
    unsigned char x;
} uchar1;


typedef struct VISIONCORE_ALIGN_PACK(2)
{
    signed char x, y;
} char2;

typedef struct VISIONCORE_ALIGN_PACK(2)
{
    unsigned char x, y;
} uchar2;

typedef struct 
{
    signed char x, y, z;
} char3;

typedef struct
{
    unsigned char x, y, z;
} uchar3;

typedef struct VISIONCORE_ALIGN_PACK(4)
{
    signed char x, y, z, w;
} char4;

typedef struct VISIONCORE_ALIGN_PACK(4)
{
    unsigned char x, y, z, w;
} uchar4;

typedef struct 
{
    short x;
} short1;

typedef struct 
{
    unsigned short x;
} ushort1;

typedef struct VISIONCORE_ALIGN_PACK(4)
{
    short x, y;
} short2;

typedef struct VISIONCORE_ALIGN_PACK(4) 
{
    unsigned short x, y;
} ushort2;

typedef struct 
{
    short x, y, z;
} short3;

typedef struct 
{
    unsigned short x, y, z;
} ushort3;

typedef struct VISIONCORE_ALIGN_PACK(8)
{
    short x, y, z, w;
} short4;

typedef struct VISIONCORE_ALIGN_PACK(8)
{
    unsigned short x, y, z, w;
} ushort4;

typedef struct  
{
    int x;
} int1;

typedef struct  
{
    unsigned int x;
} uint1;

typedef struct VISIONCORE_ALIGN_PACK(8)
{
    int x, y;
} int2;

typedef struct VISIONCORE_ALIGN_PACK(8)
{
    unsigned int x, y;
} uint2;

typedef struct
{
    int x, y, z;
} int3;

typedef struct 
{
    unsigned int x, y, z;
} uint3;

typedef struct VISIONCORE_ALIGN_PACK(16)
{
    int x, y, z, w;
} int4;

typedef struct VISIONCORE_ALIGN_PACK(16)
{
    unsigned int x, y, z, w;
} uint4;

typedef struct 
{
    long int x;
} long1;

typedef struct 
{
    unsigned long x;
} ulong1;

typedef struct VISIONCORE_ALIGN_PACK((2*sizeof(long int)))
{
    long int x, y;
} long2;

typedef struct VISIONCORE_ALIGN_PACK((2*sizeof(unsigned long int))) 
{
    unsigned long int x, y;
} ulong2;

typedef struct 
{
    long int x, y, z;
} long3;

typedef struct
{
    unsigned long int x, y, z;
} ulong3;

typedef struct VISIONCORE_ALIGN_PACK(16)
{
    long int x, y, z, w;
} long4;

typedef struct VISIONCORE_ALIGN_PACK(16)
{
    unsigned long int x, y, z, w;
} ulong4;

typedef struct
{
    float x;
} float1;

typedef struct VISIONCORE_ALIGN_PACK(8)
{
    float x; float y; 
} float2;

typedef struct
{
    float x, y, z;
} float3;

typedef struct VISIONCORE_ALIGN_PACK(16)
{
    float x, y, z, w;
} float4;

typedef struct 
{
    long long int x;
} longlong1;

typedef struct 
{
    unsigned long long int x;
} ulonglong1;

typedef struct VISIONCORE_ALIGN_PACK(16)
{
    long long int x, y;
} longlong2;

typedef struct VISIONCORE_ALIGN_PACK(16)
{
    unsigned long long int x, y;
} ulonglong2;

typedef struct 
{
    long long int x, y, z;
} longlong3;

typedef struct 
{
    unsigned long long int x, y, z;
} ulonglong3;

typedef struct VISIONCORE_ALIGN_PACK(16)
{
    long long int x, y, z ,w;
} longlong4;

typedef struct VISIONCORE_ALIGN_PACK(16) 
{
    unsigned long long int x, y, z, w;
} ulonglong4;

typedef struct 
{
    double x;
} double1;

typedef struct VISIONCORE_ALIGN_PACK(16)
{
    double x, y;
} double2;

typedef struct 
{
    double x, y, z;
} double3;

typedef struct VISIONCORE_ALIGN_PACK(16)
{
    double x, y, z, w;
} double4;

struct dim3 
{
    unsigned int x, y, z;
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
};

static inline char1 make_char1(signed char x)
{
    char1 t; t.x = x; return t;
}

static inline uchar1 make_uchar1(unsigned char x)
{
    uchar1 t; t.x = x; return t;
}

static inline char2 make_char2(signed char x, signed char y)
{
    char2 t; t.x = x; t.y = y; return t;
}

static inline uchar2 make_uchar2(unsigned char x, unsigned char y)
{
    uchar2 t; t.x = x; t.y = y; return t;
}

static inline char3 make_char3(signed char x, signed char y, signed char z)
{
    char3 t; t.x = x; t.y = y; t.z = z; return t;
}

static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z)
{
    uchar3 t; t.x = x; t.y = y; t.z = z; return t;
}

static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w)
{
    char4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w)
{
    uchar4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static inline short1 make_short1(short x)
{
    short1 t; t.x = x; return t;
}

static inline ushort1 make_ushort1(unsigned short x)
{
    ushort1 t; t.x = x; return t;
}

static inline short2 make_short2(short x, short y)
{
    short2 t; t.x = x; t.y = y; return t;
}

static inline ushort2 make_ushort2(unsigned short x, unsigned short y)
{
    ushort2 t; t.x = x; t.y = y; return t;
}

static inline short3 make_short3(short x,short y, short z)
{ 
    short3 t; t.x = x; t.y = y; t.z = z; return t;
}

static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z)
{
    ushort3 t; t.x = x; t.y = y; t.z = z; return t;
}

static inline short4 make_short4(short x, short y, short z, short w)
{
    short4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w)
{
    ushort4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static inline int1 make_int1(int x)
{
    int1 t; t.x = x; return t;
}

static inline uint1 make_uint1(unsigned int x)
{
    uint1 t; t.x = x; return t;
}

static inline int2 make_int2(int x, int y)
{
    int2 t; t.x = x; t.y = y; return t;
}

static inline uint2 make_uint2(unsigned int x, unsigned int y)
{
    uint2 t; t.x = x; t.y = y; return t;
}

static inline int3 make_int3(int x, int y, int z)
{
    int3 t; t.x = x; t.y = y; t.z = z; return t;
}

static inline uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z)
{
    uint3 t; t.x = x; t.y = y; t.z = z; return t;
}

static inline int4 make_int4(int x, int y, int z, int w)
{
    int4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{
    uint4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static inline long1 make_long1(long int x)
{
    long1 t; t.x = x; return t;
}

static inline ulong1 make_ulong1(unsigned long int x)
{
    ulong1 t; t.x = x; return t;
}

static inline long2 make_long2(long int x, long int y)
{
    long2 t; t.x = x; t.y = y; return t;
}

static inline ulong2 make_ulong2(unsigned long int x, unsigned long int y)
{
    ulong2 t; t.x = x; t.y = y; return t;
}

static inline long3 make_long3(long int x, long int y, long int z)
{
    long3 t; t.x = x; t.y = y; t.z = z; return t;
}

static inline ulong3 make_ulong3(unsigned long int x, unsigned long int y, unsigned long int z)
{
    ulong3 t; t.x = x; t.y = y; t.z = z; return t;
}

static inline long4 make_long4(long int x, long int y, long int z, long int w)
{
    long4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static inline ulong4 make_ulong4(unsigned long int x, unsigned long int y, unsigned long int z, unsigned long int w)
{
    ulong4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static inline float1 make_float1(float x)
{
    float1 t; t.x = x; return t;
}

static inline float2 make_float2(float x, float y)
{
    float2 t; t.x = x; t.y = y; return t;
}

static inline float3 make_float3(float x, float y, float z)
{
    float3 t; t.x = x; t.y = y; t.z = z; return t;
}

static inline float4 make_float4(float x, float y, float z, float w)
{
    float4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static inline longlong1 make_longlong1(long long int x)
{
    longlong1 t; t.x = x; return t;
}

static inline ulonglong1 make_ulonglong1(unsigned long long int x)
{
    ulonglong1 t; t.x = x; return t;
}

static inline longlong2 make_longlong2(long long int x, long long int y)
{
    longlong2 t; t.x = x; t.y = y; return t;
}

static inline ulonglong2 make_ulonglong2(unsigned long long int x, unsigned long long int y)
{
    ulonglong2 t; t.x = x; t.y = y; return t;
}

static inline longlong3 make_longlong3(long long int x, long long int y, long long int z)
{
    longlong3 t; t.x = x; t.y = y; t.z = z; return t;
}

static inline ulonglong3 make_ulonglong3(unsigned long long int x, unsigned long long int y, unsigned long long int z)
{
    ulonglong3 t; t.x = x; t.y = y; t.z = z; return t;
}

static inline longlong4 make_longlong4(long long int x, long long int y, long long int z, long long int w)
{
    longlong4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static inline ulonglong4 make_ulonglong4(unsigned long long int x, unsigned long long int y, unsigned long long int z, unsigned long long int w)
{
    ulonglong4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static inline double1 make_double1(double x)
{
    double1 t; t.x = x; return t;
}

static inline double2 make_double2(double x, double y)
{
    double2 t; t.x = x; t.y = y; return t;
}

static inline double3 make_double3(double x, double y, double z)
{
    double3 t; t.x = x; t.y = y; t.z = z; return t;
}

static inline double4 make_double4(double x, double y, double z, double w)
{
    double4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

#endif // VISIONCORE_HAVE_CUDA

#ifndef VISIONCORE_CUDA_COMPILER
inline int max(int a, int b)
{
    return a > b ? a : b;
}

inline int min(int a, int b)
{
    return a < b ? a : b;
}

inline float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}
#endif // VISIONCORE_CUDA_COMPILER

// lerp
EIGEN_DEVICE_FUNC inline float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

// clamp
EIGEN_DEVICE_FUNC inline float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

// lerp
EIGEN_DEVICE_FUNC inline double lerp(double a, double b, double t)
{
    return a + t*(b-a);
}

// clamp
EIGEN_DEVICE_FUNC inline double clamp(double f, double a, double b)
{
    return fmaxf(a, fminf(f, b));
}

// negate
EIGEN_DEVICE_FUNC inline int2 operator-(int2 &a)
{
    return make_int2(-a.x, -a.y);
}

// addition
EIGEN_DEVICE_FUNC inline int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}

EIGEN_DEVICE_FUNC inline void operator+=(int2 &a, int2 b)
{
    a.x += b.x; a.y += b.y;
}

// subtract
EIGEN_DEVICE_FUNC inline int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}

EIGEN_DEVICE_FUNC inline void operator-=(int2 &a, int2 b)
{
    a.x -= b.x; a.y -= b.y;
}

// multiply
EIGEN_DEVICE_FUNC inline int2 operator*(int2 a, int2 b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}

EIGEN_DEVICE_FUNC inline int2 operator*(int2 a, int s)
{
    return make_int2(a.x * s, a.y * s);
}

EIGEN_DEVICE_FUNC inline int2 operator*(int s, int2 a)
{
    return make_int2(a.x * s, a.y * s);
}

EIGEN_DEVICE_FUNC inline void operator*=(int2 &a, int s)
{
    a.x *= s; a.y *= s;
}

// additional constructors
EIGEN_DEVICE_FUNC inline float2 make_float2(float s)
{
    return make_float2(s, s);
}

EIGEN_DEVICE_FUNC inline float2 make_float2(int2 a)
{
    return make_float2(float(a.x), float(a.y));
}

// negate
EIGEN_DEVICE_FUNC inline float2 operator-(float2 &a)
{
    return make_float2(-a.x, -a.y);
}

// addition
EIGEN_DEVICE_FUNC inline float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

EIGEN_DEVICE_FUNC inline void operator+=(float2 &a, float2 b)
{
    a.x += b.x; a.y += b.y;
}

// subtract
EIGEN_DEVICE_FUNC inline float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

EIGEN_DEVICE_FUNC inline void operator-=(float2 &a, float2 b)
{
    a.x -= b.x; a.y -= b.y;
}

// multiply
EIGEN_DEVICE_FUNC inline float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}

EIGEN_DEVICE_FUNC inline float2 operator*(float2 a, float s)
{
    return make_float2(a.x * s, a.y * s);
}

EIGEN_DEVICE_FUNC inline float2 operator*(float s, float2 a)
{
    return make_float2(a.x * s, a.y * s);
}

EIGEN_DEVICE_FUNC inline void operator*=(float2 &a, float s)
{
    a.x *= s; a.y *= s;
}

// divide
EIGEN_DEVICE_FUNC inline float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}

EIGEN_DEVICE_FUNC inline float2 operator/(float2 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}

EIGEN_DEVICE_FUNC inline float2 operator/(float s, float2 a)
{
    float inv = 1.0f / s;
    return a * inv;
}

EIGEN_DEVICE_FUNC inline void operator/=(float2 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
EIGEN_DEVICE_FUNC inline float2 lerp(float2 a, float2 b, float t)
{
    return a + t*(b-a);
}

// clamp
EIGEN_DEVICE_FUNC inline float2 clamp(float2 v, float a, float b)
{
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}

EIGEN_DEVICE_FUNC inline float2 clamp(float2 v, float2 a, float2 b)
{
    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}

// dot product
EIGEN_DEVICE_FUNC inline float dot(float2 a, float2 b)
{ 
    return a.x * b.x + a.y * b.y;
}

// length
EIGEN_DEVICE_FUNC inline float length(float2 v)
{
    return sqrtf(dot(v, v));
}

// normalize
EIGEN_DEVICE_FUNC inline float2 normalize(float2 v)
{
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

// floor
EIGEN_DEVICE_FUNC inline float2 floor(const float2 v)
{
    return make_float2(floor(v.x), floor(v.y));
}

// reflect
EIGEN_DEVICE_FUNC inline float2 reflect(float2 i, float2 n)
{
    return i - 2.0f * n * dot(n,i);
}

// additional constructors
EIGEN_DEVICE_FUNC inline float3 make_float3(float s)
{
    return make_float3(s, s, s);
}

EIGEN_DEVICE_FUNC inline float3 make_float3(float2 a)
{
    return make_float3(a.x, a.y, 0.0f);
}

EIGEN_DEVICE_FUNC inline float3 make_float3(float2 a, float s)
{
    return make_float3(a.x, a.y, s);
}

EIGEN_DEVICE_FUNC inline float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z);  // discards w
}

EIGEN_DEVICE_FUNC inline float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

// negate
EIGEN_DEVICE_FUNC inline float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

// min
EIGEN_DEVICE_FUNC inline float3 fminf(float3 a, float3 b)
{
    return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}

// max
EIGEN_DEVICE_FUNC inline float3 fmaxf(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}

// addition
EIGEN_DEVICE_FUNC inline float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

EIGEN_DEVICE_FUNC inline float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

EIGEN_DEVICE_FUNC inline void operator+=(float3 &a, float3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

EIGEN_DEVICE_FUNC inline uchar3 operator+(uchar3 a, uchar3 b)
{
    return make_uchar3(a.x + b.x, a.y + b.y, a.z + b.z);
}

EIGEN_DEVICE_FUNC inline uchar3 operator+(uchar3 a, uint8_t b)
{
    return make_uchar3(a.x + b, a.y + b, a.z + b);
}

EIGEN_DEVICE_FUNC inline void operator+=(uchar3 &a, uchar3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

EIGEN_DEVICE_FUNC inline uchar3 operator/(uchar3 a, uchar3 b)
{
    return make_uchar3(a.x / b.x, a.y / b.y, a.z / b.z);
}

EIGEN_DEVICE_FUNC inline uchar3 operator/(uchar3 a, uint8_t s)
{
    return make_uchar3(a.x / s, a.y / s, a.z / s);
}

EIGEN_DEVICE_FUNC inline uchar4 operator+(uchar4 a, uchar4 b)
{
    return make_uchar4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

EIGEN_DEVICE_FUNC inline uchar4 operator+(uchar4 a, uint8_t b)
{
    return make_uchar4(a.x + b, a.y + b, a.z + b, a.w + b);
}

EIGEN_DEVICE_FUNC inline void operator+=(uchar4 &a, uchar4 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

EIGEN_DEVICE_FUNC inline uchar4 operator/(uchar4 a, uchar4 b)
{
    return make_uchar4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

EIGEN_DEVICE_FUNC inline uchar4 operator/(uchar4 a, uint8_t s)
{
    return make_uchar4(a.x / s, a.y / s, a.z / s, a.w / s);
}

// subtract
EIGEN_DEVICE_FUNC inline float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

EIGEN_DEVICE_FUNC inline float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}

EIGEN_DEVICE_FUNC inline void operator-=(float3 &a, float3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

// multiply
EIGEN_DEVICE_FUNC inline float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

EIGEN_DEVICE_FUNC inline float3 operator*(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

EIGEN_DEVICE_FUNC inline float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

EIGEN_DEVICE_FUNC inline void operator*=(float3 &a, float s)
{
    a.x *= s; a.y *= s; a.z *= s;
}

// divide
EIGEN_DEVICE_FUNC inline float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

EIGEN_DEVICE_FUNC inline float3 operator/(float3 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}

EIGEN_DEVICE_FUNC inline float3 operator/(float s, float3 a)
{
    float inv = 1.0f / s;
    return a * inv;
}

EIGEN_DEVICE_FUNC inline void operator/=(float3 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
EIGEN_DEVICE_FUNC inline float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}

// clamp
EIGEN_DEVICE_FUNC inline  float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

EIGEN_DEVICE_FUNC inline float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

// dot product
EIGEN_DEVICE_FUNC inline float dot(float3 a, float3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// cross product
EIGEN_DEVICE_FUNC inline float3 cross(float3 a, float3 b)
{ 
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

// length
EIGEN_DEVICE_FUNC inline float length(float3 v)
{
    return sqrtf(dot(v, v));
}

// normalize
EIGEN_DEVICE_FUNC inline float3 normalize(float3 v)
{
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

// floor
EIGEN_DEVICE_FUNC inline float3 floor(const float3 v)
{
    return make_float3(floor(v.x), floor(v.y), floor(v.z));
}

// reflect
EIGEN_DEVICE_FUNC inline float3 reflect(float3 i, float3 n)
{
    return i - 2.0f * n * dot(n,i);
}

// additional constructors
EIGEN_DEVICE_FUNC inline float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}

EIGEN_DEVICE_FUNC inline float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}

EIGEN_DEVICE_FUNC inline float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}

EIGEN_DEVICE_FUNC inline float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

// negate
EIGEN_DEVICE_FUNC inline float4 operator-(float4 &a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

// min
EIGEN_DEVICE_FUNC inline float4 fminf(float4 a, float4 b)
{
    return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}

// max
EIGEN_DEVICE_FUNC inline float4 fmaxf(float4 a, float4 b)
{
    return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

// addition
EIGEN_DEVICE_FUNC inline float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

EIGEN_DEVICE_FUNC inline void operator+=(float4 &a, float4 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

// subtract
EIGEN_DEVICE_FUNC inline float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

EIGEN_DEVICE_FUNC inline void operator-=(float4 &a, float4 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

// multiply
EIGEN_DEVICE_FUNC inline float4 operator*(float4 a, float s)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

EIGEN_DEVICE_FUNC inline float4 operator*(float s, float4 a)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

EIGEN_DEVICE_FUNC inline void operator*=(float4 &a, float s)
{
    a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}

// divide
EIGEN_DEVICE_FUNC inline float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

EIGEN_DEVICE_FUNC inline float4 operator/(float4 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}

EIGEN_DEVICE_FUNC inline float4 operator/(float s, float4 a)
{
    float inv = 1.0f / s;
    return a * inv;
}

EIGEN_DEVICE_FUNC inline void operator/=(float4 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
EIGEN_DEVICE_FUNC inline float4 lerp(float4 a, float4 b, float t)
{
    return a + t*(b-a);
}

// clamp
EIGEN_DEVICE_FUNC inline float4 clamp(float4 v, float a, float b)
{
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

EIGEN_DEVICE_FUNC inline float4 clamp(float4 v, float4 a, float4 b)
{
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

// dot product
EIGEN_DEVICE_FUNC inline float dot(float4 a, float4 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// length
EIGEN_DEVICE_FUNC inline float length(float4 r)
{
    return sqrtf(dot(r, r));
}

// normalize
EIGEN_DEVICE_FUNC inline float4 normalize(float4 v)
{
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

// floor
EIGEN_DEVICE_FUNC inline float4 floor(const float4 v)
{
    return make_float4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
}

// additional constructors
EIGEN_DEVICE_FUNC inline int3 make_int3(int s)
{
    return make_int3(s, s, s);
}

EIGEN_DEVICE_FUNC inline int3 make_int3(float3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}

// negate
EIGEN_DEVICE_FUNC inline int3 operator-(int3 &a)
{
    return make_int3(-a.x, -a.y, -a.z);
}

// min
EIGEN_DEVICE_FUNC inline int3 min(int3 a, int3 b)
{
    return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

// max
EIGEN_DEVICE_FUNC inline int3 max(int3 a, int3 b)
{
    return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

// addition
EIGEN_DEVICE_FUNC inline int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

EIGEN_DEVICE_FUNC inline void operator+=(int3 &a, int3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

// subtract
EIGEN_DEVICE_FUNC inline int3 operator-(int3 a, int3 b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

EIGEN_DEVICE_FUNC inline void operator-=(int3 &a, int3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

// multiply
EIGEN_DEVICE_FUNC inline int3 operator*(int3 a, int3 b)
{
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}

EIGEN_DEVICE_FUNC inline int3 operator*(int3 a, int s)
{
    return make_int3(a.x * s, a.y * s, a.z * s);
}

EIGEN_DEVICE_FUNC inline int3 operator*(int s, int3 a)
{
    return make_int3(a.x * s, a.y * s, a.z * s);
}

EIGEN_DEVICE_FUNC inline void operator*=(int3 &a, int s)
{
    a.x *= s; a.y *= s; a.z *= s;
}

// divide
EIGEN_DEVICE_FUNC inline int3 operator/(int3 a, int3 b)
{
    return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}

EIGEN_DEVICE_FUNC inline int3 operator/(int3 a, int s)
{
    return make_int3(a.x / s, a.y / s, a.z / s);
}

EIGEN_DEVICE_FUNC inline int3 operator/(int s, int3 a)
{
    return make_int3(a.x / s, a.y / s, a.z / s);
}

EIGEN_DEVICE_FUNC inline void operator/=(int3 &a, int s)
{
    a.x /= s; a.y /= s; a.z /= s;
}

// clamp
EIGEN_DEVICE_FUNC inline int clamp(int f, int a, int b)
{
    return max(a, min(f, b));
}

EIGEN_DEVICE_FUNC inline int3 clamp(int3 v, int a, int b)
{
    return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

EIGEN_DEVICE_FUNC inline int3 clamp(int3 v, int3 a, int3 b)
{
    return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

// additional constructors
EIGEN_DEVICE_FUNC inline uint3 make_uint3(uint s)
{
    return make_uint3(s, s, s);
}

EIGEN_DEVICE_FUNC inline uint3 make_uint3(float3 a)
{
    return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

// min
EIGEN_DEVICE_FUNC inline uint3 min(uint3 a, uint3 b)
{
    return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

// max
EIGEN_DEVICE_FUNC inline uint3 max(uint3 a, uint3 b)
{
    return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

// addition
EIGEN_DEVICE_FUNC inline uint3 operator+(uint3 a, uint3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}

EIGEN_DEVICE_FUNC inline void operator+=(uint3 &a, uint3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

// subtract
EIGEN_DEVICE_FUNC inline uint3 operator-(uint3 a, uint3 b)
{
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}

EIGEN_DEVICE_FUNC inline void operator-=(uint3 &a, uint3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

// multiply
EIGEN_DEVICE_FUNC inline uint3 operator*(uint3 a, uint3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}

EIGEN_DEVICE_FUNC inline uint3 operator*(uint3 a, uint s)
{
    return make_uint3(a.x * s, a.y * s, a.z * s);
}

EIGEN_DEVICE_FUNC inline uint3 operator*(uint s, uint3 a)
{
    return make_uint3(a.x * s, a.y * s, a.z * s);
}

EIGEN_DEVICE_FUNC inline void operator*=(uint3 &a, uint s)
{
    a.x *= s; a.y *= s; a.z *= s;
}

// divide
EIGEN_DEVICE_FUNC inline uint3 operator/(uint3 a, uint3 b)
{
    return make_uint3(a.x / b.x, a.y / b.y, a.z / b.z);
}

EIGEN_DEVICE_FUNC inline uint3 operator/(uint3 a, uint s)
{
    return make_uint3(a.x / s, a.y / s, a.z / s);
}

EIGEN_DEVICE_FUNC inline uint3 operator/(uint s, uint3 a)
{
    return make_uint3(a.x / s, a.y / s, a.z / s);
}

EIGEN_DEVICE_FUNC inline void operator/=(uint3 &a, uint s)
{
    a.x /= s; a.y /= s; a.z /= s;
}

// clamp
EIGEN_DEVICE_FUNC inline uint clamp(uint f, uint a, uint b)
{
    return max(a, min(f, b));
}

EIGEN_DEVICE_FUNC inline uint3 clamp(uint3 v, uint a, uint b)
{
    return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

EIGEN_DEVICE_FUNC inline uint3 clamp(uint3 v, uint3 a, uint3 b)
{
    return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

#endif // VISIONCORE_HELPERSCUDA_HPP
