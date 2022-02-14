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
 * Axis Aligned Bounding Box.
 * ****************************************************************************
 */

#ifndef VISIONCORE_TYPES_AXIS_ALIGNED_BOUNDING_BOX_HPP
#define VISIONCORE_TYPES_AXIS_ALIGNED_BOUNDING_BOX_HPP

#include <VisionCore/Platform.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace vc
{
    
namespace types
{

template<typename T>
class AxisAlignedBoundingBox;    

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const AxisAlignedBoundingBox<T>& p);
    
/**
 * Extending Eigen here.
 */
template<typename T>
class AxisAlignedBoundingBox : public Eigen::AlignedBox<T,3>
{
    typedef Eigen::AlignedBox<T,3> BaseT;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    EIGEN_DEVICE_FUNC inline AxisAlignedBoundingBox() : BaseT()
    {   
    }

    EIGEN_DEVICE_FUNC inline AxisAlignedBoundingBox(const AxisAlignedBoundingBox& bbox) : BaseT(bbox)
    {
    }

    template<typename OtherVectorType1, typename OtherVectorType2>
    EIGEN_DEVICE_FUNC inline AxisAlignedBoundingBox(const OtherVectorType1& _min, const OtherVectorType2& _max) : BaseT(_min, _max)
    {
    }

    template<typename Derived>
    EIGEN_DEVICE_FUNC inline AxisAlignedBoundingBox(const Eigen::MatrixBase<Derived>& p) : BaseT(p)
    {
    }
    
    EIGEN_DEVICE_FUNC inline ~AxisAlignedBoundingBox()
    {
    }
    
    EIGEN_DEVICE_FUNC inline AxisAlignedBoundingBox(Eigen::Matrix<T,4,4>& T_ba, const AxisAlignedBoundingBox& bb_a) : BaseT()
    {
        extend(T_ba, bb_a);
    }

    EIGEN_DEVICE_FUNC inline AxisAlignedBoundingBox(const Eigen::Matrix<T,4,4>& T_wc, T w, T h, T fu, T fv, T u0, T v0, T near, T far) : BaseT()
    {
        extendFrustum(T_wc, w, h, fu, fv, u0, v0, near, far);
    }
    
    using BaseT::extend;

    EIGEN_DEVICE_FUNC inline void extend(const Eigen::Matrix<T,4,4>& T_ba, const Eigen::Matrix<T,3,1> p_a)
    {
        const Eigen::Matrix<T,3,1> p_b = T_ba.template block<3,3>(0,0) * p_a + T_ba.template block<3,1>(0,3);
        BaseT::m_max = p_b.cwiseMax(BaseT::m_max);
        BaseT::m_min = p_b.cwiseMin(BaseT::m_min);
    }

    EIGEN_DEVICE_FUNC inline void extend(const Eigen::Matrix<T,4,4>& T_ba, const AxisAlignedBoundingBox& bb_a)
    {
        if(!bb_a.isEmpty()) 
        {
            extend(T_ba, Eigen::Matrix<T,3,1>(bb_a.m_min(0), bb_a.m_min(1), bb_a.m_min(2)) );
            extend(T_ba, Eigen::Matrix<T,3,1>(bb_a.m_min(0), bb_a.m_min(1), bb_a.m_max(2)) );
            extend(T_ba, Eigen::Matrix<T,3,1>(bb_a.m_min(0), bb_a.m_max(1), bb_a.m_min(2)) );
            extend(T_ba, Eigen::Matrix<T,3,1>(bb_a.m_min(0), bb_a.m_max(1), bb_a.m_max(2)) );
            extend(T_ba, Eigen::Matrix<T,3,1>(bb_a.m_max(0), bb_a.m_min(1), bb_a.m_min(2)) );
            extend(T_ba, Eigen::Matrix<T,3,1>(bb_a.m_max(0), bb_a.m_min(1), bb_a.m_max(2)) );
            extend(T_ba, Eigen::Matrix<T,3,1>(bb_a.m_max(0), bb_a.m_max(1), bb_a.m_min(2)) );
            extend(T_ba, Eigen::Matrix<T,3,1>(bb_a.m_max(0), bb_a.m_max(1), bb_a.m_max(2)) );
        }
    }

    EIGEN_DEVICE_FUNC inline void extendFrustum(const Eigen::Matrix<T,4,4>& T_wc, T w, T h, T fu, T fv, T u0, T v0, T near, T far) 
    {
        const Eigen::Matrix<T,3,1> c_w = T_wc.template block<3,1>(0,3);
        const Eigen::Matrix<T,3,1> ray_tl = T_wc.template block<3,3>(0,0) * Eigen::Matrix<T,3,1>((0-u0)/fu,(0-v0)/fv, 1);
        const Eigen::Matrix<T,3,1> ray_tr = T_wc.template block<3,3>(0,0) * Eigen::Matrix<T,3,1>((w-u0)/fu,(0-v0)/fv, 1);
        const Eigen::Matrix<T,3,1> ray_bl = T_wc.template block<3,3>(0,0) * Eigen::Matrix<T,3,1>((0-u0)/fu,(h-v0)/fv, 1);
        const Eigen::Matrix<T,3,1> ray_br = T_wc.template block<3,3>(0,0) * Eigen::Matrix<T,3,1>((w-u0)/fu,(h-v0)/fv, 1);

        extend(c_w + near*ray_tl);
        extend(c_w + near*ray_tr);
        extend(c_w + near*ray_bl);
        extend(c_w + near*ray_br);
        extend(c_w + far*ray_tl);
        extend(c_w + far*ray_tr);
        extend(c_w + far*ray_bl);
        extend(c_w + far*ray_br);
    }

    EIGEN_DEVICE_FUNC inline Eigen::Matrix<T,3,1> halfSizeFromOrigin() const
    {
        return BaseT::m_max.cwiseMax((Eigen::Matrix<T,3,1>)(T(-1.0) * BaseT::m_min) );
    }
    
    EIGEN_DEVICE_FUNC inline void scaleFromCenter(const Eigen::Matrix<T,3,1>& scale)
    {
        Eigen::Matrix<T,3,1> center = center();
        BaseT::m_min.array() = scale.array() * (BaseT::m_min - center).array() + center.array();
        BaseT::m_max.array() = scale.array() * (BaseT::m_max - center).array() + center.array();
    }
    
    template <typename T2>
    EIGEN_DEVICE_FUNC T2 rayIntersect(const Eigen::ParametrizedLine<T2,3>& ray) const
    {
        if( !BaseT::isEmpty() ) 
        {
            // http://www.cs.utah.edu/~awilliam/box/box.pdf
            const Eigen::Matrix<T2,3,1> tminbound = (BaseT::m_min.template cast<T2>() - ray.origin()).array() / ray.direction().array();
            const Eigen::Matrix<T2,3,1> tmaxbound = (BaseT::m_max.template cast<T2>() - ray.origin()).array() / ray.direction().array();
            const Eigen::Matrix<T2,3,1> tmin = tminbound.cwiseMin(tmaxbound);
            const Eigen::Matrix<T2,3,1> tmax = tminbound.cwiseMax(tmaxbound);
            const T2 max_tmin = tmin.maxCoeff();
            const T2 min_tmax = tmax.minCoeff();
            if(max_tmin <= min_tmax ) 
            {
                return max_tmin;
            }
        }
        
        return Eigen::NumTraits<T2>::highest();
    }
    
#ifdef VISIONCORE_ENABLE_CEREAL
    template<typename Archive>
    void load(Archive & archive, std::uint32_t const version)
    {
        archive(cereal::make_nvp("Min", BaseT::m_min));
        archive(cereal::make_nvp("Max", BaseT::m_max));
    }
    
    template<typename Archive>
    void save(Archive & archive, std::uint32_t const version) const
    {
        archive(cereal::make_nvp("Min", BaseT::m_min));
        archive(cereal::make_nvp("Max", BaseT::m_max));
    }
#endif // VISIONCORE_ENABLE_CEREAL 

private:
    friend std::ostream& operator<< <>(std::ostream& os, const AxisAlignedBoundingBox& p);
};

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const AxisAlignedBoundingBox<T>& p)
{
    os << "AABB(" << p.min() << "," << p.max() <<  ")";
    return os;
}

}

}

#endif // VISIONCORE_TYPES_AXIS_ALIGNED_BOUNDING_BOX_HPP
