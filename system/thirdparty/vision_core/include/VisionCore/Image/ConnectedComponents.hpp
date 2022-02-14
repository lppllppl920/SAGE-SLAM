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
 * Blob Detector.
 * ****************************************************************************
 */

#ifndef VISIONCORE_IMAGE_CONN_COMP_HPP
#define VISIONCORE_IMAGE_CONN_COMP_HPP

#include <type_traits>
#include <vector>
#include <list>
#include <map>

#include <VisionCore/Buffers/Buffer1D.hpp>
#include <VisionCore/Buffers/Buffer2D.hpp>
#include <VisionCore/Buffers/Image2D.hpp>
#include <VisionCore/Buffers/Buffer3D.hpp>
#include <VisionCore/Types/Rectangle.hpp>

namespace vc
{
    
namespace image
{

/**
 * Blob detector.
 * 
 * "A linear-time component-labeling algorithm using contour tracing technique".
 * Chang, Fu - Chen, Chun-Jen - Lu, Chi-Jen.
 */

/**
 * Blob label type.
 */
typedef int32_t BlobID;

/**
 * This describes single blob and it's parameters.
 */
template<typename T>
struct Blob
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    typedef std::list<Eigen::Matrix<T,2,1>, Eigen::aligned_allocator<Eigen::Matrix<T,2,1>>> Vector2ListT;
    
    Blob() : SumX(0.0), SumY(0.0), Area(0.0), Perimeter(0.0), Center(0.0,0.0), Compactness(0.0), Roundness(0.0)
    { 
        
    }
    
    /**
     * Sum of X coordinates.
     */
    T SumX;
    
    /**
     * Sum of Y coordinates.
     */
    T SumY;
    
    /**
     * Total (binary) area.
     */
    T Area;
    
    /**
     * Blob perimeter.
     */
    T Perimeter;
    
    /**
     * Centroid.
     */
    Eigen::Matrix<T,2,1> Center;
    
    /**
     * Compactness factor.
     */
    T Compactness;
    
    /**
     * Roundness factor.
     */
    T Roundness;
    
    /**
     * Bounding Box.
     */
    types::Rectangle<int> BoundingBox;
    
    /**
     * List of outer contour points.
     */
    Vector2ListT ContourOuter;
    
    /**
     * List of inner contour points.
     */
    Vector2ListT ContourInner;
};

/**
 * Map mapping blob label with blob structure.
 */
template <typename T> using BlobMapT = std::map<BlobID,Blob<T>,std::less<BlobID>,Eigen::aligned_allocator<std::pair<BlobID,Blob<T>>>>;

typedef Buffer2DView<BlobID,TargetHost> BlobImageT;
typedef Buffer2DManaged<BlobID,TargetHost> BlobManagedImageT;

template<typename T,typename T2>
BlobID blobDetector(Buffer2DView<T,TargetHost>& img_thr, 
                    BlobImageT& output, BlobMapT<T2>& bmap, 
                    T valid_val, bool do_contour = false);

template<typename T>
struct Conic
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    types::Rectangle<int> BoundingBox;
    
    // quadratic form: x'*C*x = 0 with x = (x1,x2,1) are points on the ellipse
    Eigen::Matrix<T,3,3> C;
    
    // l:=C*x is tangent line throught the point x. The dual of C is adj(C).
    // For lines through C it holds: l'*adj(C)*l = 0.
    // If C has full rank it holds up to scale: adj(C) = C^{-1}
    Eigen::Matrix<T,3,3> Dual;
    
    // Center (c1,c2)
    Eigen::Matrix<T,2,1> Center;
};

template<typename T>
void computeGradient(const Buffer2DView<T,TargetHost>& img_in, 
                     Buffer2DView<Eigen::Matrix<T,2,1>, TargetHost>& grad_img);
template<typename T>
Conic<T> estimateConic(const Buffer2DView<Eigen::Matrix<T,2,1>,TargetHost>& grad_img, 
                       const Blob<T>& component);
    
}

}


#endif //VISIONCORE_IMAGE_CONN_COMP_HPP
