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
 * Wrapper to access Images in patches.
 * ****************************************************************************
 */

#ifndef VISIONCORE_IMAGE_PATCH_HPP
#define VISIONCORE_IMAGE_PATCH_HPP

#include <VisionCore/Platform.hpp>

#include <VisionCore/Buffers/Buffer2D.hpp>

namespace vc
{

template<typename T, typename Target = TargetHost>
class ImagePatch
{
public:
    typedef T ValueType;
    typedef Target TargetType;
    
    EIGEN_DEVICE_FUNC inline ImagePatch() = delete;
    
    /**
     * Constructor.
     * 
     * @param img Input array.
     * @param x Kernel X centre position.
     * @param y Kernel Y centre position.
     * @param defret Default value returned for out of bounds access.
     */
    EIGEN_DEVICE_FUNC inline ImagePatch(Buffer2DView<T,Target>& img, 
                                        int x = 0, int y = 0, T defret = 0.0) : m_img(img), m_defret(defret)
    {
        setX(x);
        setY(y);
    }
    
    EIGEN_DEVICE_FUNC inline ~ImagePatch()
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline ImagePatch(const ImagePatch<T,Target>& rhs) 
        : m_img(rhs.m_img), m_x(rhs.m_x), m_y(rhs.m_y), m_defret(rhs.m_defret)
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline ImagePatch(ImagePatch<T,Target>&& rhs) 
        : m_img(std::move(rhs.m_img)), m_x(rhs.m_x), m_y(rhs.m_y), m_defret(rhs.m_defret)
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline ImagePatch<T,Target>& operator=(const ImagePatch<T,Target>& rhs)
    {        
        m_img = rhs.m_img;
        m_x = rhs.m_x;
        m_y = rhs.m_y;
        m_defret = rhs.m_defret;
        
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline ImagePatch<T,Target>& operator=(ImagePatch<T,Target>&& rhs)
    {
        m_img = std::move(rhs.m_img);
        m_x = rhs.m_x;
        m_y = rhs.m_y;
        m_defret = rhs.m_defret;
        
        return *this;
    }
    
    /**
     * Return current X position of the kernel centre.
     */
    EIGEN_DEVICE_FUNC inline int getX() const
    {
        return m_x;
    }
    
    /**
     * Return current Y position of the kernel centre.
     */
    EIGEN_DEVICE_FUNC inline int getY() const
    {
        return m_y;
    }
    
    /**
     * Set current X position of the kernel centre.
     * 
     * @param v Position.
     */
    EIGEN_DEVICE_FUNC inline void setX(int v)
    {
        if(v < 0)
        {
            v = 0;
        }
        
        if(v >= (int)m_img.width())
        {
            v = (int)m_img.width() - 1;
        }
        
        m_x = v;
    }
    
    /**
     * Set current Y position of the kernel centre.
     * 
     * @param v Position.
     */
    EIGEN_DEVICE_FUNC inline void setY(int v)
    {
        if(v < 0)
        {
            v = 0;
        }
        
        if(v >= (int)m_img.height())
        {
            v = (int)m_img.height() - 1;
        }
        
        m_y = v;
    }
    
    /**
     * Set current X and Y position of the kernel centre.
     * 
     * @param x X position.
     * @param y Y position.
     */
    EIGEN_DEVICE_FUNC inline void set(int x, int y)
    {
        setX(x);
        setY(y);
    }
    
    /**
     * Set current X and Y position of the kernel centre.
     * 
     * @param pos Linear position on the image.
     */
    EIGEN_DEVICE_FUNC inline void set(int pos)
    {
        setX(pos % m_img.width());
        setY(pos / m_img.width());
    }
    
    /**
     * Reset current kernel position.
     */
    EIGEN_DEVICE_FUNC inline void reset()
    {
        setX(0);
        setY(0);
    }
    
    /**
     * Move kernel to one of the 8 directions.
     * 
     * @param index Direction.
     */
    EIGEN_DEVICE_FUNC inline void move(int index)
    {
        switch(index)
        {
            case 0:
                m_x++;
                break;
            case 1:
                m_x++;
                m_y++;
                break;
            case 2:
                m_y++;
                break;
            case 3:
                m_x--;
                m_y++;
                break;
            case 4:
                m_x--;
                break;
            case 5:
                m_x--;
                m_y--;
                break;
            case 6:
                m_y--;
                break;
            case 7:
                m_x++;
                m_y--;
                break;
            default:
                // wrong index
                break;
        }
    }
    
    /**
     * Move relative.
     * 
     */
    EIGEN_DEVICE_FUNC inline void move(int dx, int dy)
    {
        set(getX()+dx, getY()+dy);
    }
    
    /**
     * Get the pixel value from a given direction.
     * 
     * @param index Direction.
     * @return Pixel value.
     */
    EIGEN_DEVICE_FUNC inline T operator()(int index) const
    {
        switch(index)
        {
            case 0:
                return operator()(1,0);
                break;
            case 1:
                return operator()(1,1);
                break;
            case 2:
                return operator()(0,1);
                break;
            case 3:
                return operator()(-1,1);
                break;
            case 4:
                return operator()(-1,0);
                break;
            case 5:
                return operator()(-1,-1);
                break;
            case 6:
                return operator()(0,-1);
                break;
            case 7:
                return operator()(1,-1);
                break;
            default:
                return operator()(0,0);
                break;
        }
    }
    
    /**
     * Get/Set the pixel value from a given direction.
     * 
     * @param index Direction.
     * @return Pixel value.
     */
    EIGEN_DEVICE_FUNC inline T& operator()(int index)
    {
        switch(index)
        {
            case 0:
                return operator()(1,0);
                break;
            case 1:
                return operator()(1,1);
                break;
            case 2:
                return operator()(0,1);
                break;
            case 3:
                return operator()(-1,1);
                break;
            case 4:
                return operator()(-1,0);
                break;
            case 5:
                return operator()(-1,-1);
                break;
            case 6:
                return operator()(0,-1);
                break;
            case 7:
                return operator()(1,-1);
                break;
            default:
                return operator()(0,0);
                break;
        }
    }
    
    /**
     * Get the pixel value from a pixel at @a ox , @a oy offset from kernel centre.
     * 
     * @param ox X offset.
     * @param oy Y offset.
     * @return Pixel value.
     */
    EIGEN_DEVICE_FUNC inline T operator()(int ox, int oy) const
    {
        // only when within bounds
        if(((m_x + ox) >= 0) && ((m_x + ox) < (int)m_img.width()) && ((m_y + oy) >= 0) && ((m_y + oy) < (int)m_img.height()))
        {
            return m_img((m_x + ox),(m_y + oy));
        }
        else
        {
            return m_defret;
        }
    }
    
    /**
     * Get/Set the pixel value from a pixel at @a ox , @a oy offset from kernel centre.
     * 
     * @param ox X offset.
     * @param oy Y offset.
     * @return Pixel value.
     */
    EIGEN_DEVICE_FUNC inline T& operator()(int ox, int oy)
    {
        // only when within bounds
        if(((m_x + ox) >= 0) && ((m_x + ox) < (int)m_img.width()) && ((m_y + oy) >= 0) && ((m_y + oy) < (int)m_img.height()))
        {
            return m_img((m_x + ox),(m_y + oy));
        }
        else
        {
            return m_defret;
        }
    }
private:
    Buffer2DView<T,Target>& m_img;
    int m_x;
    int m_y;
    T m_defret;
};

}

#endif // VISIONCORE_IMAGE_PATCH_HPP
