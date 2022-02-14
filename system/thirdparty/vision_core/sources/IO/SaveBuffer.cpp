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
 * Image IO.
 * ****************************************************************************
 */

#include <VisionCore/IO/ImageIO.hpp>

#include <VisionCore/Buffers/Buffer1D.hpp>
#include <VisionCore/Buffers/Buffer2D.hpp>
#include <VisionCore/Buffers/Image2D.hpp>

#include <VisionCore/IO/File.hpp>
#include <cstdio>
#include <cstdarg>

template<typename T>
struct BinaryBufferSavingProxy { };

template<typename T2>
struct BinaryBufferSavingProxy<vc::Buffer1DView<T2, vc::TargetHost>>
{
  static inline void save(const std::string& fn, const vc::Buffer1DView<T2, vc::TargetHost>& b) 
  { 
    vc::io::File outf(fn.c_str(),"wb");
    
    outf.write(b.ptr(), b.size(), sizeof(T2));
    
    outf.flush();
  }
};

template<typename T2>
struct BinaryBufferSavingProxy<vc::Buffer2DView<T2, vc::TargetHost>>
{
    static inline void save(const std::string& fn, const vc::Buffer2DView<T2, vc::TargetHost>& b) 
    { 
        vc::io::File outf(fn.c_str(),"wb");
        
        outf.write(b.ptr(), b.totalElements(), sizeof(T2));
        
        outf.flush();
    }
};

template<typename T2>
struct BinaryBufferSavingProxy<vc::Image2DView<T2, vc::TargetHost>>
{
    static inline void save(const std::string& fn, const vc::Image2DView<T2, vc::TargetHost>& b) 
    { 
        BinaryBufferSavingProxy<vc::Buffer2DView<T2,vc::TargetHost>>::save(fn,b);
    }
};

template<typename T>
struct TextWriteElement { };

template<>
struct TextWriteElement<uint8_t> 
{
    static inline void write(vc::io::File& of, const uint8_t& val)
    {
        of.printf("0x%02X ", (int)val);
    }
};

template<>
struct TextWriteElement<uchar3> 
{
  static inline void write(vc::io::File& of, const uchar3& val)
  {
    of.printf("0x%02X 0x%02X 0x%02X ", (int)val.x, (int)val.y, (int)val.z);
  }
};

template<>
struct TextWriteElement<uint16_t> 
{
    static inline void write(vc::io::File& of, const uint16_t& val)
    {
        of.printf("0x%04X ", (int)val);
    }
};

template<>
struct TextWriteElement<float> 
{
    static inline void write(vc::io::File& of, const float& val)
    {
        of.printf("%04.4g ", val);
    }
};

template<>
struct TextWriteElement<float3> 
{
  static inline void write(vc::io::File& of, const float3& val)
  {
    of.printf("%04.4g %04.4g %04.4g ", val.x, val.y, val.z);
  }
};

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct TextWriteElement<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> 
{
    static inline void write(vc::io::File& of, const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& val)
    {
        for(std::size_t ic = 0 ; ic < _Cols ; ++ic)
        {
            for(std::size_t ir = 0 ; ir < _Rows ; ++ir)
            {
                TextWriteElement<_Scalar>::write(of, val(ir,ic));
            }
        }
    }
};

template<typename T>
struct TextBufferSavingProxy { };

template<typename T2>
struct TextBufferSavingProxy<vc::Buffer1DView<T2, vc::TargetHost>>
{
    static inline void save(const std::string& fn, const vc::Buffer1DView<T2, vc::TargetHost>& b) 
    { 
        vc::io::File outf(fn.c_str(),"w");
        
        outf.printf("%04lu\n", b.size());

        for(std::size_t i = 0 ; i < b.size() ; ++i)
        {
            /*
            TextWriteElement<T2>::write(outf, b(i));
            
            if((i != 0) && (i % 10 == 0))
            {
                outf.putc('\n');
            }
            */
            
            outf.printf("%04lu\t", i);
            TextWriteElement<T2>::write(outf, b(i));
            outf.putc('\n');
        }
    }
};

template<typename T2>
struct TextBufferSavingProxy<vc::Buffer2DView<T2, vc::TargetHost>>
{
    static inline void save(const std::string& fn, const vc::Buffer2DView<T2, vc::TargetHost>& b) 
    { 
        vc::io::File outf(fn.c_str(),"w");
        
        outf.printf("%04lu x %04lu\n", b.width(), b.height());
        
        for(std::size_t y = 0 ; y < b.height() ; ++y)
        {
            for(std::size_t x = 0 ; x < b.width() ; ++x)
            {
                /*
                const std::size_t lin_index = y * b.width() + x;
                TextWriteElement<T2>::write(outf, b(x,y));
                
                if((lin_index != 0) && (lin_index % 10 == 0))
                {
                    outf.putc('\n');
                }
                */
                outf.printf("%04lu x %04lu\t", x, y);
                TextWriteElement<T2>::write(outf, b(x,y));
                outf.putc('\n');
            }
        }
        
        outf.flush();
    }
};

template<typename T2>
struct TextBufferSavingProxy<vc::Image2DView<T2, vc::TargetHost>>
{
    static inline void save(const std::string& fn, const vc::Image2DView<T2, vc::TargetHost>& b) 
    {
        TextBufferSavingProxy<vc::Buffer2DView<T2, vc::TargetHost>>::save(fn, b);
    }
};

template<typename T>
void vc::io::saveBufferAsText(const std::string& fn, const T& input)
{
    TextBufferSavingProxy<T>::save(fn, input);
}

template<typename T>
void vc::io::saveBufferAsBinary(const std::string& fn, const T& input)
{
    BinaryBufferSavingProxy<T>::save(fn, input);
}

#define INST1D_FOR_TYPE(T)\
template void vc::io::saveBufferAsText<vc::Buffer1DView<T, vc::TargetHost>>\
  (const std::string& fn, const vc::Buffer1DView<T, vc::TargetHost>& input);\
template void vc::io::saveBufferAsBinary<vc::Buffer1DView<T, vc::TargetHost>>\
  (const std::string& fn, const vc::Buffer1DView<T, vc::TargetHost>& input);
  
#define INST2D_FOR_TYPE(T)\
template void vc::io::saveBufferAsText<vc::Buffer2DView<T, vc::TargetHost>>\
  (const std::string& fn, const vc::Buffer2DView<T, vc::TargetHost>& input);\
template void vc::io::saveBufferAsBinary<vc::Buffer2DView<T, vc::TargetHost>>\
  (const std::string& fn, const vc::Buffer2DView<T, vc::TargetHost>& input);\
  template void vc::io::saveBufferAsText<vc::Image2DView<T, vc::TargetHost>>\
  (const std::string& fn, const vc::Image2DView<T, vc::TargetHost>& input);\
  template void vc::io::saveBufferAsBinary<vc::Image2DView<T, vc::TargetHost>>\
  (const std::string& fn, const vc::Image2DView<T, vc::TargetHost>& input);

INST1D_FOR_TYPE(uint8_t)
INST1D_FOR_TYPE(uchar3)
INST1D_FOR_TYPE(uint16_t)
INST1D_FOR_TYPE(float)
INST1D_FOR_TYPE(float3)
INST1D_FOR_TYPE(Eigen::Vector2f)
INST1D_FOR_TYPE(Eigen::Vector3f)
  
INST2D_FOR_TYPE(uint8_t)
INST2D_FOR_TYPE(uchar3)
INST2D_FOR_TYPE(uint16_t)
INST2D_FOR_TYPE(float)
INST2D_FOR_TYPE(float3)
INST2D_FOR_TYPE(Eigen::Vector2f)
INST2D_FOR_TYPE(Eigen::Vector3f)
