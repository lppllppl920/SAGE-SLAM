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
 * Simple generation of random numbers.
 * ****************************************************************************
 */

#ifndef VISIONCORE_MATH_RANDOM_HPP
#define VISIONCORE_MATH_RANDOM_HPP

#include <VisionCore/Platform.hpp>

#include <cstdlib>
#include <ctime>

#include <type_traits>
#include <VisionCore/Buffers/Buffer1D.hpp>
#include <VisionCore/Buffers/Buffer2D.hpp>
#include <VisionCore/Buffers/Image2D.hpp>
#include <VisionCore/Buffers/Buffer3D.hpp>

#include <VisionCore/Types/Gaussian.hpp>

typedef struct curandGenerator_st *curandGenerator_t;

namespace vc
{
    
namespace math
{

class Random
{
public:
    static inline void seed()
    {
        seed(std::time(NULL));
    }
    
    static inline void seed(int v)
    {
        std::srand(v);
    }
    
    template <class T>
    static inline T value()
    {
        return (T)std::rand()/(T)RAND_MAX;
    }
    
    template <class T>
    static inline T value(T min, T max)
    {
        double rratio = (double)rand() / ((double)RAND_MAX + 1.0);
        T d = max - min + T(1.0);
        return T( rratio * d) + min;
    }

    template <class T>
    static inline T gaussian(T mean, T sigma)
    {
        // Box-Muller transformation
        T x1, x2, w, y1;
        
        do {
            x1 = T(2.0) * value<T>() - T(1.0);
            x2 = T(2.0) * value<T>() - T(1.0);
            w = x1 * x1 + x2 * x2;
        } while ( w >= T(1.0) || w == T(0.0));
        
        w = std::sqrt( (T(-2.0) * std::log( w ) ) / w );
        y1 = x1 * w;
        
        return( mean + y1 * sigma );
    }
};

template<typename Target>
struct RandomGenerator
{
    RandomGenerator(uint64_t seed);
    ~RandomGenerator();
    
    curandGenerator_t handle;
};

template<typename T, typename Target>
void generateRandom(RandomGenerator<Target>& gen, Buffer1DView<T,Target>& bufout, 
                    const types::Gaussian<typename type_traits<T>::ChannelType>& gauss);

template<typename T, typename Target>
void generateRandom(RandomGenerator<Target>& gen, Buffer2DView<T,Target>& bufout, 
                    const types::Gaussian<typename type_traits<T>::ChannelType>& gauss);

template<typename T, typename Target>
void generateRandom(RandomGenerator<Target>& gen, Buffer1DView<T,Target>& bufout, 
                    const typename type_traits<T>::ChannelType& mean, const typename type_traits<T>::ChannelType& stddev);

template<typename T, typename Target>
void generateRandom(RandomGenerator<Target>& gen, Buffer2DView<T,Target>& bufout, 
                    const typename type_traits<T>::ChannelType& mean, const typename type_traits<T>::ChannelType& stddev);

}

}

#endif // VISIONCORE_MATH_RANDOM_HPP
