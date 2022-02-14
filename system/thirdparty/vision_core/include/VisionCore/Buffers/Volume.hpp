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
 * Volume access to 3D Host/Device Buffer.
 * ****************************************************************************
 */

#ifndef VISIONCORE_VOLUME_HPP
#define VISIONCORE_VOLUME_HPP

#include <VisionCore/Buffers/Buffer3D.hpp>


// CUDA
#ifdef VISIONCORE_HAVE_CUDA
#include <thrust/device_vector.h>
#include <npp.h>
#endif // VISIONCORE_HAVE_CUDA

namespace vc
{

template<typename T, typename Target = TargetHost>
class VolumeView : public Buffer3DView<T,Target>
{
public:
    typedef Buffer3DView<T,Target> BaseType;
    
    using BaseType::get;
    using BaseType::width;
    using BaseType::height;
    using BaseType::depth;
    using BaseType::inBounds;
    
    EIGEN_DEVICE_FUNC inline VolumeView() : BaseType()
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline ~VolumeView()
    {
        
    }

    EIGEN_DEVICE_FUNC inline VolumeView(const VolumeView<T,Target>& img) : BaseType(img)
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline VolumeView(VolumeView<T,Target>&& img) : BaseType(std::move(img))
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline VolumeView<T,Target>& operator=(const VolumeView<T,Target>& img)
    {
        BaseType::operator=(img);
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline VolumeView<T,Target>& operator=(VolumeView<T,Target>&& img)
    {
        BaseType::operator=(std::move(img));
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline VolumeView(typename BaseType::TargetType::template PointerType<T> optr, size_t w, size_t h, size_t d) : BaseType(optr, w, h ,d)
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline VolumeView(typename BaseType::TargetType::template PointerType<T> optr, size_t w, size_t h, size_t d, size_t opitch) : BaseType(optr, w, h, d, opitch)
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline VolumeView(typename BaseType::TargetType::template PointerType<T> optr, size_t w, size_t h, size_t d, size_t opitch, size_t oimg_pitch) : BaseType(optr, w, h, d, opitch, oimg_pitch)
    {
        
    }

    EIGEN_DEVICE_FUNC inline const T getFractionalNearestNeighbour(float x, float y, float z) const
    {
        float3 pos = {x , y , z};
        return getFractionalNearestNeighbour(pos);
    }

    EIGEN_DEVICE_FUNC inline const T getFractionalNearestNeighbour(float3 pos) const
    {
        const float3 pf = pos * make_float3(width() - 1, height() - 1, depth() - 1);
        return get(pf.x+0.5, pf.y+0.5, pf.z+0.5);
    }
    
    template<typename TR>
    EIGEN_DEVICE_FUNC inline TR getFractionalTrilinear(float x, float y, float z) const
    {
        float3 pos = {x , y , z};
        return getFractionalTrilinear<TR>(pos);
    }

    template<typename TR>
    EIGEN_DEVICE_FUNC inline TR getFractionalTrilinear(float3 pos) const
    {
        const float3 pf = pos * make_float3(width()- 1.f, height() - 1.f, depth() - 1.f);

        const int ix = floorf(pf.x);
        const int iy = floorf(pf.y);
        const int iz = floorf(pf.z);
        const float fx = pf.x - ix;
        const float fy = pf.y - iy;
        const float fz = pf.z - iz;

        const TR v0 = get(ix,iy,iz);
        const TR vx = get(ix+1,iy,iz);
        const TR vy = get(ix,iy+1,iz);
        const TR vxy = get(ix+1,iy+1,iz);
        const TR vz = get(ix,iy,iz+1);
        const TR vxz = get(ix+1,iy,iz+1);
        const TR vyz = get(ix,iy+1,iz+1);
        const TR vxyz = get(ix+1,iy+1,iz+1);

        return lerp(
            lerp(lerp(v0,vx,fx),  lerp(vy,vxy,fx), fy),
            lerp(lerp(vz,vxz,fx), lerp(vyz,vxyz,fx), fy),
            fz
        );
    }
    
    template<typename TR>
    EIGEN_DEVICE_FUNC inline TR getFractionalTrilinearClamped(float x, float y, float z) const
    {
        float3 pos = {x , y , z};
        return getFractionalTrilinearClamped<TR>(pos);
    }

    template<typename TR>
    EIGEN_DEVICE_FUNC inline TR getFractionalTrilinearClamped(float3 pos) const
    {
        const float3 pf = pos * make_float3(width() - 1.f, height() - 1.f, depth() - 1.f);

        const int ix = fmaxf(fminf(width() - 2, floorf(pf.x) ), 0);
        const int iy = fmaxf(fminf(height() - 2, floorf(pf.y) ), 0);
        const int iz = fmaxf(fminf(depth() - 2, floorf(pf.z) ), 0);
        const float fx = pf.x - ix;
        const float fy = pf.y - iy;
        const float fz = pf.z - iz;

        const TR v0 = get(ix,iy,iz);
        const TR vx = get(ix+1,iy,iz);
        const TR vy = get(ix,iy+1,iz);
        const TR vxy = get(ix+1,iy+1,iz);
        const TR vz = get(ix,iy,iz+1);
        const TR vxz = get(ix+1,iy,iz+1);
        const TR vyz = get(ix,iy+1,iz+1);
        const TR vxyz = get(ix+1,iy+1,iz+1);

        return lerp(
            lerp(lerp(v0,vx,fx),  lerp(vy,vxy,fx), fy),
                    lerp(lerp(vz,vxz,fx), lerp(vyz,vxyz,fx), fy),
            fz
        );
    }

    EIGEN_DEVICE_FUNC inline float3 getBackwardDiffDxDyDz(int x, int y, int z) const
    {
        const float v0 = get(x, y, z);
        return make_float3(
            v0 - get(x-1, y, z),
            v0 - get(x, y-1, z),
            v0 - get(x, y, z-1)
        );
    }
    
    template<typename TR>
    EIGEN_DEVICE_FUNC inline Eigen::Matrix<TR,3,1> getFractionalBackwardDiffDxDyDz(float x, float y, float z) const
    {
        Eigen::Matrix<TR,3,1> pos(x,y,z);
        return getFractionalBackwardDiffDxDyDz(pos);
    }

    template<typename TR>
    EIGEN_DEVICE_FUNC inline Eigen::Matrix<TR,3,1> getFractionalBackwardDiffDxDyDz(Eigen::Matrix<TR,3,1> pos) const
    {
        const float3 pf = pos * make_float3(width() - 1.f, height() - 1.f, depth() - 1.f);

        const int ix = fmaxf(fminf(width() - 2, floorf(pf.x) ), 1);
        const int iy = fmaxf(fminf(height() - 2, floorf(pf.y) ), 1);
        const int iz = fmaxf(fminf(depth() - 2, floorf(pf.z) ), 1);
        const float fx = pf.x - ix;
        const float fy = pf.y - iy;
        const float fz = pf.z - iz;

        const float3 v0 = getBackwardDiffDxDyDz(ix,iy,iz);
        const float3 vx = getBackwardDiffDxDyDz(ix+1,iy,iz);
        const float3 vy = getBackwardDiffDxDyDz(ix,iy+1,iz);
        const float3 vxy = getBackwardDiffDxDyDz(ix+1,iy+1,iz);
        const float3 vz = getBackwardDiffDxDyDz(ix,iy,iz+1);
        const float3 vxz = getBackwardDiffDxDyDz(ix+1,iy,iz+1);
        const float3 vyz = getBackwardDiffDxDyDz(ix,iy+1,iz+1);
        const float3 vxyz = getBackwardDiffDxDyDz(ix+1,iy+1,iz+1);

        float3 ret = lerp(
            lerp(lerp(v0,vx,fx),  lerp(vy,vxy,fx), fy),
            lerp(lerp(vz,vxz,fx), lerp(vyz,vxyz,fx), fy),
            fz
        );
        
        return Eigen::Matrix<TR,3,1>(ret.x, ret.y, ret.z);
    }
    
    template<typename TJET>
    EIGEN_DEVICE_FUNC inline TJET getJet(Eigen::Matrix<TJET,3,1>& pt) const
    {
        return getJet<TJET>(pt(0), pt(1), pt(2));
    }
    
    template<typename TJET>
    EIGEN_DEVICE_FUNC inline TJET getJet(const TJET& x, const TJET& y, const TJET& z) const
    {
        typename ADTraits<TJET>::Scalar scalar_x = ADTraits<TJET>::GetScalar(x);
        typename ADTraits<TJET>::Scalar scalar_y = ADTraits<TJET>::GetScalar(y);
        typename ADTraits<TJET>::Scalar scalar_z = ADTraits<TJET>::GetScalar(z);
        
        typename ADTraits<TJET>::Scalar sample[4];
        if(ADTraits<TJET>::isScalar()) 
        {
            // For the scalar case, only sample the image.
            if(inBounds(scalar_x, scalar_y, scalar_z))
            {
                sample[0] = getFractionalTrilinear(scalar_x, scalar_y, scalar_z);
            }
            else
            {
                sample[0] = 0.0f;
            }
        }
        else 
        {
            Eigen::Map<Eigen::Matrix<typename ADTraits<TJET>::Scalar,1,3> > tmp(sample + 1,2);
            
            sample[0] = 0.0f;
            // For the derivative case, sample the gradient as well.
            if(inBounds(scalar_x, scalar_y, scalar_z))
            {
                sample[0] = getFractionalTrilinear(scalar_x, scalar_y, scalar_z); // pixel value
            }
            
            tmp << 0.0f , 0.0f , 0.0f;
            if(inBounds(scalar_x, scalar_y, scalar_z, 1.0f))
            {
                tmp = getFractionalBackwardDiffDxDyDz<typename ADTraits<TJET>::Scalar>(scalar_x, scalar_y, scalar_z);
            }   
        }
        
        TJET xyz[3] = { x, y , z};
        return Chain<typename ADTraits<TJET>::Scalar, 3, TJET>::Rule(sample[0], sample + 1, xyz);
    }
};

template<typename T, typename Target = TargetHost>
class VolumeManaged : public VolumeView<T,Target>
{
public:
    typedef VolumeView<T,Target> ViewT;
    
    VolumeManaged() = delete;
    
    inline VolumeManaged(unsigned int w, unsigned int h, unsigned int d) : ViewT()
    {
        ViewT::xsize = w;
        ViewT::ysize = h;
        ViewT::zsize = d;
        
        std::size_t line_pitch = 0;
        std::size_t plane_pitch = 0;
        typename Target::template PointerType<T> ptr = 0;
        
        Target::template AllocatePitchedMem<T>(&ptr, &line_pitch, &plane_pitch, ViewT::xsize, ViewT::ysize, ViewT::zsize);
        
        ViewT::memptr = ptr;
        ViewT::line_pitch = line_pitch;
        ViewT::plane_pitch = plane_pitch;
    }
    
    inline ~VolumeManaged()
    {
        Target::template DeallocatePitchedMem<T>(ViewT::memptr);
    }
    
    VolumeManaged(const VolumeManaged<T,Target>& img) = delete;
    
    inline VolumeManaged(VolumeManaged<T,Target>&& img) : ViewT(std::move(img))
    {
        
    }
    
    VolumeManaged<T,Target>& operator=(const VolumeManaged<T,Target>& img) = delete;
    
    inline VolumeManaged<T,Target>& operator=(VolumeManaged<T,Target>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

}

#endif // VISIONCORE_VOLUME_HPP
