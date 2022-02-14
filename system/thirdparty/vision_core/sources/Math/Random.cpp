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

#include <VisionCore/Math/Random.hpp>

#include <curand.h>

template<typename Target>
struct TargetDispatcher { };

template<>
struct TargetDispatcher<vc::TargetDeviceCUDA>
{
    static inline curandGenerator_t create(curandRngType rngt)
    {
        curandGenerator_t ret;
        curandCreateGenerator(&ret, rngt);
        return ret;
    }
};

template<>
struct TargetDispatcher<vc::TargetHost>
{
    static inline curandGenerator_t create(curandRngType rngt)
    {
        curandGenerator_t ret;
        curandCreateGeneratorHost(&ret, rngt);
        return ret;
    }
};

template<typename Target>
vc::math::RandomGenerator<Target>::RandomGenerator(uint64_t seed)
{
    handle = TargetDispatcher<Target>::create(CURAND_RNG_PSEUDO_DEFAULT);
    
    curandStatus_t res = curandSetPseudoRandomGeneratorSeed(handle, seed);
    if(res != CURAND_STATUS_SUCCESS)
    {
        std::stringstream ss;
        ss << "curand error: " << res;
        throw std::runtime_error(ss.str());
    }
}

template<typename Target>
vc::math::RandomGenerator<Target>::~RandomGenerator()
{
    curandDestroyGenerator(handle);
}

template<typename T, typename Target>
void vc::math::generateRandom(RandomGenerator<Target>& gen, vc::Buffer1DView<T,Target>& bufout, const vc::types::Gaussian<typename vc::type_traits<T>::ChannelType>& gauss)
{
    curandStatus_t res = curandGenerateNormal(gen.handle, bufout.ptr(), bufout.size(), gauss.mean(), gauss.stddev());
    if(res != CURAND_STATUS_SUCCESS)
    {
        std::stringstream ss;
        ss << "curand error: " << res;
        throw std::runtime_error(ss.str());
    }
}

template<typename T, typename Target>
void vc::math::generateRandom(RandomGenerator<Target>& gen, vc::Buffer2DView<T,Target>& bufout, const vc::types::Gaussian<typename vc::type_traits<T>::ChannelType>& gauss)
{
    curandStatus_t res = curandGenerateNormal(gen.handle, bufout.ptr(), bufout.height() * bufout.pitch(), gauss.mean(), gauss.stddev());
    if(res != CURAND_STATUS_SUCCESS)
    {
        std::stringstream ss;
        ss << "curand error: " << res;
        throw std::runtime_error(ss.str());
    }
}

template<typename T, typename Target>
void vc::math::generateRandom(RandomGenerator<Target>& gen, vc::Buffer1DView<T,Target>& bufout, const typename vc::type_traits<T>::ChannelType& mean, const typename vc::type_traits<T>::ChannelType& stddev)
{
    curandStatus_t res = curandGenerateNormal(gen.handle, bufout.ptr(), bufout.size(), mean, stddev);
    if(res != CURAND_STATUS_SUCCESS)
    {
        std::stringstream ss;
        ss << "curand error: " << res;
        throw std::runtime_error(ss.str());
    }
}

template<typename T, typename Target>
void vc::math::generateRandom(RandomGenerator<Target>& gen, vc::Buffer2DView<T,Target>& bufout, const typename vc::type_traits<T>::ChannelType& mean, const typename vc::type_traits<T>::ChannelType& stddev)
{
    curandStatus_t res = curandGenerateNormal(gen.handle, bufout.ptr(), bufout.height() * bufout.pitch(), mean, stddev);
    if(res != CURAND_STATUS_SUCCESS)
    {
        std::stringstream ss;
        ss << "curand error: " << res;
        throw std::runtime_error(ss.str());
    }
}

#define GENERATE_CODE(TYPE) \
template void vc::math::generateRandom<TYPE,vc::TargetDeviceCUDA>(RandomGenerator<vc::TargetDeviceCUDA>& gen, vc::Buffer1DView<TYPE,vc::TargetDeviceCUDA>& bufout, const vc::types::Gaussian<typename vc::type_traits<TYPE>::ChannelType>& gauss); \
template void vc::math::generateRandom<TYPE,vc::TargetHost>(RandomGenerator<vc::TargetHost>& gen, vc::Buffer1DView<TYPE,vc::TargetHost>& bufout, const vc::types::Gaussian<typename vc::type_traits<TYPE>::ChannelType>& gauss); \
template void vc::math::generateRandom<TYPE,vc::TargetDeviceCUDA>(RandomGenerator<vc::TargetDeviceCUDA>& gen, vc::Buffer2DView<TYPE,vc::TargetDeviceCUDA>& bufout, const vc::types::Gaussian<typename vc::type_traits<TYPE>::ChannelType>& gauss); \
template void vc::math::generateRandom<TYPE,vc::TargetHost>(RandomGenerator<vc::TargetHost>& gen, vc::Buffer2DView<TYPE,vc::TargetHost>& bufout, const vc::types::Gaussian<typename vc::type_traits<TYPE>::ChannelType>& gauss); \
template void vc::math::generateRandom<TYPE,vc::TargetDeviceCUDA>(RandomGenerator<vc::TargetDeviceCUDA>& gen, vc::Buffer1DView<TYPE,vc::TargetDeviceCUDA>& bufout, const typename vc::type_traits<TYPE>::ChannelType& mean, const typename vc::type_traits<TYPE>::ChannelType& stddev); \
template void vc::math::generateRandom<TYPE,vc::TargetHost>(RandomGenerator<vc::TargetHost>& gen, vc::Buffer1DView<TYPE,vc::TargetHost>& bufout, const typename vc::type_traits<TYPE>::ChannelType& mean, const typename vc::type_traits<TYPE>::ChannelType& stddev); \
template void vc::math::generateRandom<TYPE,vc::TargetDeviceCUDA>(RandomGenerator<vc::TargetDeviceCUDA>& gen, vc::Buffer2DView<TYPE,vc::TargetDeviceCUDA>& bufout, const typename vc::type_traits<TYPE>::ChannelType& mean, const typename vc::type_traits<TYPE>::ChannelType& stddev); \
template void vc::math::generateRandom<TYPE,vc::TargetHost>(RandomGenerator<vc::TargetHost>& gen, vc::Buffer2DView<TYPE,vc::TargetHost>& bufout, const typename vc::type_traits<TYPE>::ChannelType& mean, const typename vc::type_traits<TYPE>::ChannelType& stddev);

GENERATE_CODE(float)

template struct vc::math::RandomGenerator<vc::TargetDeviceCUDA>;
template struct vc::math::RandomGenerator<vc::TargetHost>;
