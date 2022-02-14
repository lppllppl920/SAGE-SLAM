/**
 * (c) Bryan Catanzaro
 * Adapted from: https://github.com/bryancatanzaro/generics
 */

#ifndef VISIONCORE_CUDA_GENERICS_HPP
#define VISIONCORE_CUDA_GENERICS_HPP

#include <cstdint>

#ifdef VISIONCORE_HAVE_CUDA
#ifdef VISIONCORE_CUDA_COMPILER

#include <cuda_runtime.h>
#include <cuda.h>

namespace vc
{
namespace detail 
{
  
__device__ __forceinline__ unsigned int ACTIVE_MASK()
{
#if CUDA_VERSION >= 9000
    return __activemask();
#else
    // will be ignored anyway
    return 0xffffffff;
#endif
}


__device__ __forceinline__ int WARP_BALLOT(int predicate, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __ballot_sync(mask, predicate);
#else
    return __ballot(predicate);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_sync(mask, value, srcLane, width);
#else
    return __shfl(value, srcLane, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_UP(T value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_up_sync(mask, value, delta, width);
#else
    return __shfl_up(value, delta, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_down_sync(mask, value, delta, width);
#else
    return __shfl_down(value, delta, width);
#endif
}

    
template<typename T, int m>
struct array 
{
    typedef T value_type;
    typedef T head_type;
    typedef array<T, m-1> tail_type;
    static const int size = m;
    head_type head;
    tail_type tail;
    
    __host__ __device__ __forceinline__ array(head_type h, const tail_type& t) : head(h), tail(t) {}
    __host__ __device__ __forceinline__ array() : head(), tail() {}
    __host__ __device__ __forceinline__ array(const array& other) : head(other.head), tail(other.tail) {}
    __host__ __device__ __forceinline__ array& operator=(const array& other) 
    {
        head = other.head;
        tail = other.tail;
        return *this;
    }
    __host__ __device__ __forceinline__ bool operator==(const array& other) const 
    {
        return (head == other.head) && (tail == other.tail);
    }
    __host__ __device__ __forceinline__ bool operator!=(const array& other) const 
    {
        return !operator==(other);
    }
};

template<typename T>
struct array<T, 1> 
{
    typedef T value_type;
    typedef T head_type;
    static const int size = 1;
    head_type head;
    
    __host__ __device__ __forceinline__ array(head_type h) : head(h){}
    __host__ __device__ __forceinline__ array() : head() {}
    __host__ __device__ __forceinline__ array(const array& other) : head(other.head) {}
    __host__ __device__ __forceinline__ array& operator=(const array& other) 
    {
        head = other.head;
        return *this;
    }
    
    __host__ __device__ __forceinline__ bool operator==(const array& other) const 
    {
        return (head == other.head);
    }
    
    __host__ __device__ __forceinline__ bool operator!=(const array& other) const 
    {
        return !operator==(other);
    }
};

template<typename T>
struct array<T, 0> {};

template<typename T, int m, int i>
struct get_impl 
{
    __host__ __device__ __forceinline__ static T& impl(array<T, m>& src) 
    {
        return get_impl<T, m-1, i-1>::impl(src.tail);
    }
    
    __host__ __device__ __forceinline__ static T impl(const array<T, m>& src) 
    {
        return get_impl<T, m-1, i-1>::impl(src.tail);
    }
};

template<typename T, int m>
struct get_impl<T, m, 0> 
{
    __host__ __device__ __forceinline__ static T& impl(array<T, m>& src) 
    {
        return src.head;
    }
    __host__ __device__ __forceinline__ static T impl(const array<T, m>& src) 
    {
        return src.head;
    }
};

template<int i, typename T, int m>
__host__ __device__ __forceinline__ T& get(array<T, m>& src) 
{
    return detail::get_impl<T, m, i>::impl(src);
}

template<int i, typename T, int m>
__host__ __device__ __forceinline__ T get(const array<T, m>& src) 
{
    return detail::get_impl<T, m, i>::impl(src);
}

template<typename T, int p>
struct size_multiple_power_of_two 
{
    static const bool value = (sizeof(T) & ((1 << p) - 1)) == 0;
};

template<typename T, bool use_int=size_multiple_power_of_two<T, 2>::value>
struct working_type 
{
    typedef char type;
};

template<typename T>
struct working_type<T, true> 
{
    typedef int type;
};


template<typename T, typename U>
struct aliased_size 
{
    static const int value = sizeof(T)/sizeof(U);
};

template<typename T>
struct working_array 
{
    typedef typename working_type<T>::type U;
    static const int r = aliased_size<T, U>::value;
    typedef array<U, r> type;
};

template<typename T,
typename U=typename working_type<T>::type, int r=aliased_size<T, U>::value>
struct dismember 
{
    typedef array<U, r> result_type;
    static const int idx = aliased_size<T, U>::value - r;
    
    __host__ __device__ __forceinline__ static result_type impl(const T& t) 
    {
        return result_type(((const U*)&t)[idx],dismember<T, U, r-1>::impl(t));
    }
};

template<typename T, typename U>
struct dismember<T, U, 1> 
{
    typedef array<U, 1> result_type;
    static const int idx = aliased_size<T, U>::value - 1;
    
    __host__ __device__ __forceinline__ static result_type impl(const T& t) 
    {
        return result_type(((const U*)&t)[idx]);
    }
};

template<typename U, typename T>
__host__ __device__ __forceinline__ array<U, detail::aliased_size<T, U>::value> lyse(const T& in) 
{
    return detail::dismember<T, U>::impl(in);
}


template<typename T,
typename U=typename working_type<T>::type, int r=aliased_size<T, U>::value>
struct remember 
{
    static const int idx = aliased_size<T, U>::value - r;
    
    __host__ __device__ __forceinline__ static void impl(const array<U, r>& d, T& t) 
    {
        ((U*)&t)[idx] = d.head;
        remember<T, U, r-1>::impl(d.tail, t);
    }
};

template<typename T, typename U>
struct remember<T, U, 1> 
{
    static const int idx = aliased_size<T, U>::value - 1;
    
    __host__ __device__ __forceinline__ static void impl(const array<U, 1>& d, const T& t) 
    {
        ((U*)&t)[idx] = d.head;
    }
};


template<typename T>
__host__ __device__ __forceinline__ T fuse(const typename working_array<T>::type& in) 
{
    T result;
    typedef typename working_type<T>::type U;
    remember<T, U>::impl(in, result);
    return result;
}

template<typename T,
typename U=typename working_type<T>::type,int r = aliased_size<T, U>::value>
struct load_storage 
{
    typedef array<U, r> result_type;
    static const int idx = aliased_size<T, U>::value - r;
    
    __device__ __forceinline__ static result_type impl(const T* ptr) 
    {
        return result_type(__ldg(((const U*)ptr) + idx), load_storage<T, U, r-1>::impl(ptr));
    }
};

template<typename T, typename U>
struct load_storage<T, U, 1> 
{
    typedef array<U, 1> result_type;
    
    static const int idx = aliased_size<T, U>::value - 1;
    
    __device__ __forceinline__ static result_type impl(const T* ptr) 
    {
        return result_type(__ldg(((const U*)ptr) + idx));
    }
};

template<int s>
struct shuffle 
{
    __device__ __forceinline__ static void impl(array<int, s>& d, const int& i) 
    {
        d.head = WARP_SHFL(d.head, i);
        shuffle<s-1>::impl(d.tail, i);
    }
};

template<>
struct shuffle<1> 
{
    __device__ __forceinline__ static void impl(array<int, 1>& d, const int& i) 
    {
        d.head = WARP_SHFL(d.head, i);
    }
};

template<int s>
struct shuffle_down 
{
    __device__ __forceinline__ static void impl(array<int, s>& d, const int& i) 
    {
        d.head = WARP_SHFL_DOWN(d.head, i);
        shuffle_down<s-1>::impl(d.tail, i);
    }
};

template<>
struct shuffle_down<1> 
{
    __device__ __forceinline__ static void impl(array<int, 1>& d, const int& i) 
    {
        d.head = WARP_SHFL_DOWN(d.head, i);
    }
};

template<int s>
struct shuffle_up 
{
    __device__ __forceinline__ static void impl(array<int, s>& d, const int& i) 
    {
        d.head = WARP_SHFL_UP(d.head, i);
        shuffle_up<s-1>::impl(d.tail, i);
    }
};

template<>
struct shuffle_up<1> 
{
    __device__ __forceinline__ static void impl(array<int, 1>& d, const int& i) 
    {
        d.head = WARP_SHFL_UP(d.head, i);
    }
};

template<int s>
struct shuffle_xor 
{
    __device__ __forceinline__ static void impl(array<int, s>& d, const int& i) 
    {
        d.head = WARP_SHFL_XOR(d.head, i);
        shuffle_xor<s-1>::impl(d.tail, i);
    }
};

template<>
struct shuffle_xor<1> 
{
    __device__ __forceinline__ static void impl(array<int, 1>& d, const int& i) 
    {
        d.head = WARP_SHFL_XOR(d.head, i);
    }
};
    
} //end namespace detail

#if __CUDA_ARCH__ >= 350
// Device has ldg
template<typename T>
__device__ __forceinline__ T ldg(const T* ptr) 
{
    typedef typename detail::working_array<T>::type aliased;
    aliased storage = detail::load_storage<T>::impl(ptr);
    return detail::fuse<T>(storage);
}
#else
//Device does not, fall back.
template<typename T>
__device__ __forceinline__ T ldg(const T* ptr) 
{
    return *ptr;
}
#endif // __CUDA_ARCH__ >= 350

template<typename T>
__device__ __forceinline__ T shfl(const T& t, const int& i) 
{
    //X If you get a compiler error on this line, it is because
    //X sizeof(T) is not divisible by 4, and so this type is not
    //X supported currently.
    THRUST_STATIC_ASSERT((detail::size_multiple_power_of_two<T, 2>::value));
    
    typedef typename detail::working_array<T>::type aliased;
    aliased lysed = detail::lyse<int>(t);
    detail::shuffle<aliased::size>::impl(lysed, i);
    return detail::fuse<T>(lysed);
}

template<typename T>
__device__ __forceinline__ T shfl_down(const T& t, const int& i) 
{
    //X If you get a compiler error on this line, it is because
    //X sizeof(T) is not divisible by 4, and so this type is not
    //X supported currently.
    THRUST_STATIC_ASSERT((detail::size_multiple_power_of_two<T, 2>::value));
    
    typedef typename detail::working_array<T>::type aliased;
    aliased lysed = detail::lyse<int>(t);
    detail::shuffle_down<aliased::size>::impl(lysed, i);
    return detail::fuse<T>(lysed);
}

template<typename T>
__device__ __forceinline__ T shfl_up(const T& t, const int& i) 
{
    //X If you get a compiler error on this line, it is because
    //X sizeof(T) is not divisible by 4, and so this type is not
    //X supported currently.
    THRUST_STATIC_ASSERT((detail::size_multiple_power_of_two<T, 2>::value));
    
    typedef typename detail::working_array<T>::type aliased;
    aliased lysed = detail::lyse<int>(t);
    detail::shuffle_up<aliased::size>::impl(lysed, i);
    return detail::fuse<T>(lysed);
}

template<typename T>
__device__ __forceinline__ T shfl_xor(const T& t, const int& i) 
{
    
    //X If you get a compiler error on this line, it is because
    //X sizeof(T) is not divisible by 4, and so this type is not
    //X supported currently.
    THRUST_STATIC_ASSERT((detail::size_multiple_power_of_two<T, 2>::value));
    
    typedef typename detail::working_array<T>::type aliased;
    aliased lysed = detail::lyse<int>(t);
    detail::shuffle_xor<aliased::size>::impl(lysed, i);
    return detail::fuse<T>(lysed);
}

}
#endif // VISIONCORE_CUDA_COMPILER
#endif // VISIONCORE_HAVE_CUDA

#endif // VISIONCORE_CUDA_GENERICS_HPP
