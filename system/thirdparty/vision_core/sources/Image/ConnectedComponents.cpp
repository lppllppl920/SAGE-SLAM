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

#include <VisionCore/Image/ConnectedComponents.hpp>

#include <VisionCore/LaunchUtils.hpp>
#include <VisionCore/Image/ImagePatch.hpp>
#include <Eigen/SVD>

template<typename T>
void vc::image::computeGradient(const vc::Buffer2DView<T,vc::TargetHost>& img_in, vc::Buffer2DView<Eigen::Matrix<T,2,1>, vc::TargetHost>& grad_img)
{
    tbb::parallel_for(tbb::blocked_range2d<std::size_t>(1, img_in.height() - 1, 1, img_in.width() - 1), [&](const tbb::blocked_range2d<std::size_t>& r)
    {
        for(std::size_t y = r.rows().begin() ; y != r.rows().end() ; ++y )
        {
            for(std::size_t x = r.cols().begin() ; x != r.cols().end() ; ++x ) 
            {
                Eigen::Matrix<T,2,1>& grad = grad_img(x,y);
                grad(0) = img_in(x+1,y) - img_in(x-1,y); // dX
                grad(1) = img_in(x,y+1) - img_in(x,y-1); // dY
            }
        }
    });
}

template<typename T>
vc::image::Conic<T> vc::image::estimateConic(const vc::Buffer2DView<Eigen::Matrix<T,2,1>,vc::TargetHost>& grad_img, const vc::image::Blob<T>& component)
{
    vc::image::Conic<T> conic;
    
    const vc::types::Rectangle<int> region = component.BoundingBox;
    
    // Form system Ax = b to solve
    Eigen::Matrix<T,5,5> A = Eigen::Matrix<T,5,5>::Zero();
    Eigen::Matrix<T,5,1> b = Eigen::Matrix<T,5,1>::Zero();
    
    for(int v = region.y1() ; v <= region.y2() ; ++v)
    {
        const Eigen::Matrix<T,2,1>* dIv = grad_img.rowPtr(v);
        
        for(int u = region.x1() ; u <= region.x2() ; ++u)
        {
            // li = (ai,bi,ci)' = (I_ui,I_vi, -dI' x_i)'
            const Eigen::Matrix<T,3,1> d = Eigen::Matrix<T,3,1>(dIv[u](0), dIv[u](1), -(dIv[u](0) * u + dIv[u](1) * v) );
            const Eigen::Matrix<T,3,1> li = d;
            Eigen::Matrix<T,5,1> Ki;
            Ki << li(0) * li(0), li(0) * li(1), li(1) * li(1), li(0) * li(2), li(1) * li(2);
            A += Ki * Ki.transpose();
            b += -Ki * li(2) * li(2);
        }
    }
    
    const Eigen::Matrix<T,5,1> x = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);
    
    Eigen::Matrix<T,3,3> C_star_norm;
    C_star_norm <<  x(0)        , x(1)/T(2.0)   , x(3)/T(2.0),  
                    x(1)/T(2.0) , x(2)          , x(4)/T(2.0),  
                    x(3)/T(2.0) , x(4)/T(2.0)   ,   T(1.0);
    
    conic.C = C_star_norm.inverse();

    conic.BoundingBox = region;
    conic.Dual = conic.C.inverse();
    conic.Dual /= conic.Dual(2,2);
    conic.Center = Eigen::Matrix<T,2,1>(conic.Dual(0,2),conic.Dual(1,2));
    
    return conic;
}

template<typename T>
static inline int tracer(vc::ImagePatch<T,vc::TargetHost>& inpk, vc::ImagePatch<vc::image::BlobID,vc::TargetHost>& outpk, int start, T valid_val)
{
    int ret = 8;
    
    int nidx = 0;
    // FIXME const max_ways = 7
    for(int i = 0 ; i <= 7 ; ++i) // visit all around
    {
        nidx = (start + i) % 8;
        
        if(inpk(nidx) == valid_val) // black
        {
            if(ret == 8) // first found
            {
                ret = nidx;
            }
        }
        else // white
        {
            // mark it
            outpk(nidx) = -1;
        }
    }
    
    return ret; // new direction
}

template<typename T>
static inline int first_look(vc::ImagePatch<T,vc::TargetHost>& inpk, vc::ImagePatch<vc::image::BlobID,vc::TargetHost>& outpk, bool internal, T valid_val)
{
    if(internal == false)
    {
        // external contour start direction
        return tracer<T>(inpk, outpk, 7, valid_val);
    }
    else
    {
        // internal contour start direction
        return tracer<T>(inpk, outpk, 3, valid_val);
    }
}

template<typename T,typename T2>
static void contour_tracing(vc::Buffer2DView<T,vc::TargetHost>& input, vc::image::BlobImageT& output, int x, int y, bool internal, vc::image::BlobMapT<T2>& bmap, T valid_val, bool do_contour)
{
    vc::ImagePatch<T,vc::TargetHost> krn_input(input, x, y);
    vc::ImagePatch<vc::image::BlobID,vc::TargetHost> krn_output(output, x, y);
    
    vc::image::BlobID label = krn_output(0,0);
    
    // first step from the beginning of the contour
    int start = first_look<T>(krn_input, krn_output, internal, valid_val);
    
    if(internal == false)
    {
        bmap[label].Perimeter = 0.0;
        
        if(do_contour)
        {
            bmap[label].ContourOuter.clear();
            bmap[label].ContourOuter.push_back(Eigen::Matrix<T2,2,1>((T2)x,(T2)y));
        }
    }
    else
    {
        if(do_contour)
        {
            bmap[label].ContourInner.clear();
            bmap[label].ContourInner.push_back(Eigen::Matrix<T2,2,1>((T2)x,(T2)y));
        }
    }
    
    if(start == 8) // isolated point, done
    {
        return;
    }
    
    int curr = start;
    while(1)
    {
        // label the point
        vc::image::Blob<T2>& bb = bmap[label];
        bb.SumX += krn_output.getX();
        bb.SumY += krn_output.getY();
        bb.Area += 1.0;
        bb.BoundingBox.insert(krn_output.getX(), krn_output.getY());
        
        if(internal == false)
        {
            bb.Perimeter += 1.0;
            if(do_contour)
            {
                bmap[label].ContourOuter.push_back(Eigen::Matrix<T2,2,1>((T2)krn_output.getX(),(T2)krn_output.getY()));
            }
        }
        else
        {
            if(do_contour)
            {
                bmap[label].ContourInner.push_back(Eigen::Matrix<T2,2,1>((T2)krn_output.getX(),(T2)krn_output.getY()));
            }
        }
        
        if(krn_output(0,0) != label)
        {
            krn_output(0,0) = label;
        }
        
        // switch tracer to the next point
        krn_input.move(curr);
        krn_output.move(curr);
        
        // get next direction
        curr = tracer<T>(krn_input, krn_output, (curr + 5) % 8, valid_val);
        
        // check exit conditions
        // isolated point
        if(curr == 8) 
        {
            break;
        }
        // came back to the same place
        if((krn_input.getX() == x) && (krn_input.getY() == y) && (curr == start)) 
        {
            break;
        }
    }
}



template<typename T,typename T2>
vc::image::BlobID vc::image::blobDetector(vc::Buffer2DView<T,vc::TargetHost>& img_thr, vc::image::BlobImageT& output, BlobMapT<T2>& bmap, T valid_val, bool do_contour)
{
    BlobID cur_label = 1;
    
    // clear output
    for(std::size_t y = 1 ; y < output.height() ; ++y)
    {
        for(std::size_t x = 1 ; x < output.width() ; ++x)
        {
            output(x,y) = 0;
        }
    }
    bmap.clear();
    
    vc::ImagePatch<T,vc::TargetHost> krn_input(img_thr);
    vc::ImagePatch<BlobID,vc::TargetHost> krn_output(output);
    
    // go over the image, unfortunately not parallel (FIXME maybe?)
    for(int j = 0 ; j < (int)img_thr.height() ; ++j)
    {
        for(int i = 0 ; i < (int)img_thr.width() ; ++i)
        {
            // move kernels
            krn_input.set(i,j);
            krn_output.set(i,j);
            
            if(krn_input(0,0) == valid_val) // P is black
            {
                // step 1
                if((krn_input(0,-1) != valid_val) && (krn_output(0,0) == 0)) // pixel above is white and P is unlabelled
                {
                    // label and add to the map
                    krn_output(0,0) = cur_label;
                    
                    contour_tracing<T,T2>(img_thr, output, i, j, false, bmap, valid_val, do_contour); // trace external contour
                    
                    cur_label++;
                }
                else // step 2
                {
                    if((krn_input(0,1) != valid_val) && (krn_output(0,1) == 0)) // pixel below white & unmarked
                    {
                        if(krn_output(0,0) == 0) // is not labelled
                        {
                            krn_output(0,0) = krn_output(-1,0); // set the same label as the previous pixel
                        }
                        
                        contour_tracing<T,T2>(img_thr, output, i, j, true, bmap, valid_val, do_contour); // trace internal contour
                    }
                    else // step 3
                    {
                        if(krn_output(0,0) == 0) // not yet labelled
                        {
                            krn_output(0,0) = krn_output(-1,0); // set the same label as the previous pixel
                        }
                    }
                    
                    Blob<T2>& bb = bmap[krn_output(0,0)];
                    bb.SumX += krn_output.getX();
                    bb.SumY += krn_output.getY();
                    bb.Area += 1.0;
                    bb.BoundingBox.insert(krn_output.getX(), krn_output.getY());
                }
            }
        }
    }
    
    // calculate blob parameters
    for(typename BlobMapT<T2>::iterator it = bmap.begin() ; it != bmap.end() ; ++it)
    {
        it->second.Center << it->second.SumX / it->second.Area , it->second.SumY / it->second.Area;
        it->second.Compactness = (it->second.Perimeter * it->second.Perimeter) / (T(4.0 * M_PI) * it->second.Area);
        it->second.Roundness = 1.0 / it->second.Compactness;
    }
    
    return cur_label - 1; // number of blobs found
}

// instantiate
template vc::image::BlobID vc::image::blobDetector<uint8_t,float>(vc::Buffer2DView<uint8_t,vc::TargetHost>& img_thr, vc::image::BlobImageT& output, BlobMapT<float>& bmap, uint8_t valid_val, bool do_contour);
template vc::image::Conic<float> vc::image::estimateConic<float>(const vc::Buffer2DView<Eigen::Matrix<float,2,1>,vc::TargetHost>& grad_img, const vc::image::Blob<float>& component);
template void vc::image::computeGradient<float>(const vc::Buffer2DView<float,vc::TargetHost>& img_in, vc::Buffer2DView<Eigen::Matrix<float,2,1>, vc::TargetHost>& grad_img);
