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
 * PLY file IO.
 * ****************************************************************************
 */

#include <VisionCore/IO/PLYModel.hpp>

#include <algorithm>
#include <fstream>
#include <boost/algorithm/string.hpp>

struct Field
{
    enum class FunctionType
    {
        PosX = 0, PosY, PosZ,
        NormalX, NormalY, NormalZ,
        ColorRed, ColorBlue, ColorGreen, ColorAlpha,
        Radius
    };
    
    bool Valid = false;
    FunctionType Function = FunctionType::PosX;
    std::size_t  Size = sizeof(float);
    
    void parseField(const std::vector<std::string>& strs)
    {
        if(strs.size() != 3) { return; }
        
        if(boost::iequals(strs[0], "property"))
        {
            Valid = true;
            if(boost::iequals(strs[1], "float")) { Size = sizeof(float); }
            else if(boost::iequals(strs[1], "uchar")) { Size = sizeof(uint8_t); }
            else { Valid = false; }
            
            if(Valid)
            {
                if(boost::iequals(strs[2], "x")) { Function = FunctionType::PosX; }
                else if(boost::iequals(strs[2], "y")) { Function = FunctionType::PosY; }
                else if(boost::iequals(strs[2], "z")) { Function = FunctionType::PosZ; }
                else if(boost::iequals(strs[2], "nx")) { Function = FunctionType::NormalX; }
                else if(boost::iequals(strs[2], "ny")) { Function = FunctionType::NormalY; }
                else if(boost::iequals(strs[2], "nz")) { Function = FunctionType::NormalZ; }
                else if(boost::iequals(strs[2], "red")) { Function = FunctionType::ColorRed; }
                else if(boost::iequals(strs[2], "green")) { Function = FunctionType::ColorGreen; }
                else if(boost::iequals(strs[2], "blue")) { Function = FunctionType::ColorBlue; }
                else if(boost::iequals(strs[2], "alpha")) { Function = FunctionType::ColorAlpha; }
                else if(boost::iequals(strs[2], "radius")) { Function = FunctionType::Radius; }
                else { Valid = false; }
            }
        }
    }
    
    template<typename T>
    void readField(std::ifstream& plyfile, T& var)
    {
        if(sizeof(T) != Size) { throw std::runtime_error("Wrong size"); }
        
        plyfile.read(reinterpret_cast<char*>(&var), Size);
    }
};

template<typename T>
struct ReadWriteHelper { };

// Vector3f
template<> struct ReadWriteHelper<Eigen::Vector3f>
{
    static inline bool readVertices(std::ifstream& plyfile, Field* fields, std::size_t field_cnt, Eigen::Vector3f& sout, float scalePos)
    {
        for(std::size_t j = 0 ; j < field_cnt ; ++j)
        {
            Field& f = fields[j];
            
            switch(f.Function)
            {
                case Field::FunctionType::PosX: f.readField(plyfile, sout(0)); break;
                case Field::FunctionType::PosY: f.readField(plyfile, sout(1)); break;
                case Field::FunctionType::PosZ: f.readField(plyfile, sout(2)); break;
                default: break;
            }
        }
        
        sout *= scalePos;
        
        return true;
    }
    
    static inline bool writeVertices(const Eigen::Vector3f& sout, std::ofstream& fpout, float scalePos)
    {
        const Eigen::Vector3f npos = sout * scalePos;
        
        // Position
        fpout.write(reinterpret_cast<const char*>(&npos(0)), sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&npos(1)), sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&npos(2)), sizeof(float));
        
        return true;
    }
    
    static inline bool writeHeader(std::ofstream& fs)
    {
        fs << "\nproperty float x\nproperty float y\nproperty float z"; 
        return true;
    }
};

// ColorPoint
template<> struct ReadWriteHelper<vc::ColorPoint>
{
    static inline bool readVertices(std::ifstream& plyfile, Field* fields, std::size_t field_cnt, vc::ColorPoint& sout, float scalePos)
    {
        uchar4 tmpcol;
        
        for(std::size_t j = 0 ; j < field_cnt ; ++j)
        {
            Field& f = fields[j];
            
            switch(f.Function)
            {
                case Field::FunctionType::PosX: f.readField(plyfile, sout.Position(0)); break;
                case Field::FunctionType::PosY: f.readField(plyfile, sout.Position(1)); break;
                case Field::FunctionType::PosZ: f.readField(plyfile, sout.Position(2)); break;
                case Field::FunctionType::ColorRed: f.readField(plyfile, tmpcol.x); break;
                case Field::FunctionType::ColorGreen: f.readField(plyfile, tmpcol.y); break;
                case Field::FunctionType::ColorBlue: f.readField(plyfile, tmpcol.z); break;
                case Field::FunctionType::ColorAlpha: f.readField(plyfile, tmpcol.w); break;
                default: break;
            }
        }
        
        sout.Color << float(tmpcol.x) / 255.0f , float(tmpcol.y) / 255.0f , float(tmpcol.z) / 255.0f;
        sout.Position *= scalePos;
        
        return true;
    }
    
    static inline bool writeVertices(const vc::ColorPoint& sout, std::ofstream& fpout, float scalePos)
    {
        const Eigen::Vector3f npos = sout.Position * scalePos;
        
        // Position
        fpout.write(reinterpret_cast<const char*>(&npos(0)), sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&npos(1)), sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&npos(2)), sizeof(float));
        
        // Color
        uchar3 rgb = make_uchar3(int(sout.Color(0) * 255.0f),int(sout.Color(1) * 255.0f),int(sout.Color(2) * 255.0f));
        fpout.write(reinterpret_cast<const char*>(&rgb), sizeof(uchar3));
        
        return true;
    }
    
    static inline bool writeHeader(std::ofstream& fs)
    {
        fs << "\nproperty float x\nproperty float y\nproperty float z"; 
        fs << "\nproperty uchar red\nproperty uchar green\nproperty uchar blue"; 
        return true;
    }
};

// NormalPoint
template<> struct ReadWriteHelper<vc::NormalPoint>
{
    static inline bool readVertices(std::ifstream& plyfile, Field* fields, std::size_t field_cnt, vc::NormalPoint& sout, float scalePos)
    {
        for(std::size_t j = 0 ; j < field_cnt ; ++j)
        {
            Field& f = fields[j];
            
            switch(f.Function)
            {
                case Field::FunctionType::PosX: f.readField(plyfile, sout.Position(0)); break;
                case Field::FunctionType::PosY: f.readField(plyfile, sout.Position(1)); break;
                case Field::FunctionType::PosZ: f.readField(plyfile, sout.Position(2)); break;
                case Field::FunctionType::NormalX: f.readField(plyfile, sout.Normal(0)); break;
                case Field::FunctionType::NormalY: f.readField(plyfile, sout.Normal(1)); break;
                case Field::FunctionType::NormalZ: f.readField(plyfile, sout.Normal(2)); break;
                default: break;
            }
        }
        
        sout.Position *= scalePos;
        
        return true;
    }
    
    static inline bool writeVertices(const vc::NormalPoint& sout, std::ofstream& fpout, float scalePos)
    {
        const Eigen::Vector3f npos = sout.Position * scalePos;
        
        // Position
        fpout.write(reinterpret_cast<const char*>(&npos(0)), sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&npos(1)), sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&npos(2)), sizeof(float));
        
        // Normal
        const Eigen::Vector3f invN = -sout.Normal;
        fpout.write(reinterpret_cast<const char*>(&invN(0)), sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&invN(1)), sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&invN(2)), sizeof(float));
        
        return true;
    }
    
    static inline bool writeHeader(std::ofstream& fs)
    {
        fs << "\nproperty float x\nproperty float y\nproperty float z"; 
        fs << "\nproperty float nx\nproperty float ny\nproperty float nz";
        return true;
    }
};

// ColorNormalPoint
template<> struct ReadWriteHelper<vc::ColorNormalPoint>
{
    static inline bool readVertices(std::ifstream& plyfile, Field* fields, std::size_t field_cnt, vc::ColorNormalPoint& sout, float scalePos)
    {
        uchar4 tmpcol;
        
        for(std::size_t j = 0 ; j < field_cnt ; ++j)
        {
            Field& f = fields[j];
            
            switch(f.Function)
            {
                case Field::FunctionType::PosX: f.readField(plyfile, sout.Position(0)); break;
                case Field::FunctionType::PosY: f.readField(plyfile, sout.Position(1)); break;
                case Field::FunctionType::PosZ: f.readField(plyfile, sout.Position(2)); break;
                case Field::FunctionType::NormalX: f.readField(plyfile, sout.Normal(0)); break;
                case Field::FunctionType::NormalY: f.readField(plyfile, sout.Normal(1)); break;
                case Field::FunctionType::NormalZ: f.readField(plyfile, sout.Normal(2)); break;
                case Field::FunctionType::ColorRed: f.readField(plyfile, tmpcol.x); break;
                case Field::FunctionType::ColorGreen: f.readField(plyfile, tmpcol.y); break;
                case Field::FunctionType::ColorBlue: f.readField(plyfile, tmpcol.z); break;
                case Field::FunctionType::ColorAlpha: f.readField(plyfile, tmpcol.w); break;
                default: break;
            }
        }
        
        sout.Color << float(tmpcol.x) / 255.0f , float(tmpcol.y) / 255.0f , float(tmpcol.z) / 255.0f;
        sout.Position *= scalePos;
        
        return true;
    }
    
    static inline bool writeVertices(const vc::ColorNormalPoint& sout, std::ofstream& fpout, float scalePos)
    {
        const Eigen::Vector3f npos = sout.Position * scalePos;
        
        // Position
        fpout.write(reinterpret_cast<const char*>(&npos(0)), sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&npos(1)), sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&npos(2)), sizeof(float));
        
        // Color
        uchar3 rgb = make_uchar3(int(sout.Color(0) * 255.0f),int(sout.Color(1) * 255.0f),int(sout.Color(2) * 255.0f));
        fpout.write(reinterpret_cast<const char*>(&rgb), sizeof(uchar3));
        
        // Normal
        const Eigen::Vector3f invN = -sout.Normal;
        fpout.write(reinterpret_cast<const char*>(&invN(0)), sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&invN(1)), sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&invN(2)), sizeof(float));
        
        return true;
    }
    
    static inline bool writeHeader(std::ofstream& fs)
    {
        fs << "\nproperty float x\nproperty float y\nproperty float z"; 
        fs << "\nproperty uchar red\nproperty uchar green\nproperty uchar blue"; 
        fs << "\nproperty float nx\nproperty float ny\nproperty float nz";
        return true;
    }
};

// Surfel
template<> struct ReadWriteHelper<vc::Surfel>
{
    static inline bool readVertices(std::ifstream& plyfile, Field* fields, std::size_t field_cnt, vc::Surfel& sout, float scalePos)
    {
        uchar4 tmpcol;
        
        for(std::size_t j = 0 ; j < field_cnt ; ++j)
        {
            Field& f = fields[j];
            
            switch(f.Function)
            {
                case Field::FunctionType::PosX: f.readField(plyfile, sout.Position(0)); break;
                case Field::FunctionType::PosY: f.readField(plyfile, sout.Position(1)); break;
                case Field::FunctionType::PosZ: f.readField(plyfile, sout.Position(2)); break;
                case Field::FunctionType::NormalX: f.readField(plyfile, sout.Normal(0)); break;
                case Field::FunctionType::NormalY: f.readField(plyfile, sout.Normal(1)); break;
                case Field::FunctionType::NormalZ: f.readField(plyfile, sout.Normal(2)); break;
                case Field::FunctionType::ColorRed: f.readField(plyfile, tmpcol.x); break;
                case Field::FunctionType::ColorGreen: f.readField(plyfile, tmpcol.y); break;
                case Field::FunctionType::ColorBlue: f.readField(plyfile, tmpcol.z); break;
                case Field::FunctionType::ColorAlpha: f.readField(plyfile, tmpcol.w); break;
                case Field::FunctionType::Radius: f.readField(plyfile, sout.Radius); break;
                default: break;
            }
        }
        
        sout.Color << float(tmpcol.x) / 255.0f , float(tmpcol.y) / 255.0f , float(tmpcol.z) / 255.0f;
        sout.Position *= scalePos;
        sout.Radius *= scalePos;
        
        return true;
    }
    
    static inline bool writeVertices(const vc::Surfel& sout, std::ofstream& fpout, float scalePos)
    {
        const Eigen::Vector3f npos = sout.Position * scalePos;
        
        // Position
        fpout.write(reinterpret_cast<const char*>(&npos(0)), sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&npos(1)), sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&npos(2)), sizeof(float));
        
        // Color
        uchar3 rgb = make_uchar3(int(sout.Color(0) * 255.0f),int(sout.Color(1) * 255.0f),int(sout.Color(2) * 255.0f));
        fpout.write(reinterpret_cast<const char*>(&rgb), sizeof(uchar3));
        
        // Normal
        const Eigen::Vector3f invN = -sout.Normal;
        fpout.write(reinterpret_cast<const char*>(&invN(0)), sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&invN(1)), sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&invN(2)), sizeof(float));
        
        // Radius
        fpout.write(reinterpret_cast<const char*>(&sout.Radius), sizeof(float));
        
        return true;
    }
    
    static inline bool writeHeader(std::ofstream& fs)
    {
        fs << "\nproperty float x\nproperty float y\nproperty float z"; 
        fs << "\nproperty uchar red\nproperty uchar green\nproperty uchar blue"; 
        fs << "\nproperty float nx\nproperty float ny\nproperty float nz";
        fs << "\nproperty float radius";
        return true;
    }
};

template<typename T>
bool vc::io::loadPLY(const std::string& plyfilename, std::vector<T,Eigen::aligned_allocator<T>>& vec, float scalePos, std::vector<std::vector<std::size_t>>* idx)
{
    std::ifstream plyfile(plyfilename.c_str(), std::ios::binary);
    if (!plyfile.is_open()) 
    {
        return false; // wrong file name
    }
    
    std::string line_read;
    std::getline(plyfile, line_read);
    if(!boost::iequals(line_read, "ply")) 
    {
        return false; // wrong file type
    }
    
    std::size_t number_vertices = 0;
    std::size_t number_faces = 0;
    vec.clear();
    if(idx != nullptr) { idx->clear(); }
    bool got_element_vertex = false;
    bool got_element_face = false;
    Field fields[12];
    std::size_t field_cnt = 0;
    std::size_t field_cnt_idx = 0;
    
    while(1) 
    {
        std::getline(plyfile,line_read);
        
        std::vector<std::string> strs;
        boost::split(strs, line_read, boost::is_any_of("\t "));
        
        if(boost::iequals(strs[0], "property"))
        {
            if(got_element_vertex)
            {
                fields[field_cnt].Valid = false;
                fields[field_cnt].parseField(strs);
                if(fields[field_cnt].Valid)
                {
                    field_cnt++;
                }
            }
            else if(got_element_face)
            {
                field_cnt_idx++;
            }
        }
        else if(boost::iequals(strs[0], "comment"))
        {
            continue;
        }
        else if(boost::iequals(strs[0], "element") && boost::iequals(strs[1], "vertex"))
        {
            if(strs.size() != 3) { throw std::runtime_error("Problem"); }
            
            number_vertices = atoi(strs[2].c_str());
            vec.reserve(number_vertices);
            got_element_vertex = true;
        }
        else if(boost::iequals(strs[0], "element") && boost::iequals(strs[1], "face"))
        {
            if(strs.size() != 3) { throw std::runtime_error("Problem"); }
            
            number_faces = atoi(strs[2].c_str());
            if(idx != nullptr)
            {
                idx->resize(number_faces);
            }
            got_element_face = true;
        }
        else if(boost::iequals(strs[0], "end_header"))
        {
            break;
        }
    }
    
    for(std::size_t i = 0 ; i < number_vertices ; ++i)
    {
        T sout;
        
        bool ok = ReadWriteHelper<T>::readVertices(plyfile, fields, field_cnt, sout, scalePos);
        if(!ok) { break; }
        
        vec.push_back(sout);
    }
    
    if(idx != nullptr)
    {
        for(std::size_t i = 0 ; i < number_faces ; ++i)
        {
            uint8_t count;
            plyfile.read(reinterpret_cast<char*>(&count), sizeof(uint8_t));
            
            for(std::size_t v = 0 ; v < (std::size_t)count ; ++v)
            {
                uint32_t index;
                plyfile.read(reinterpret_cast<char*>(&index), sizeof(uint32_t));
                idx->at(i).push_back(index);
            }
        }
    }
    
    plyfile.close();
    
    return true;
}

template<typename T>
bool vc::io::savePLY(const std::string& plyfilename, const std::vector<T,Eigen::aligned_allocator<T>>& vec, float scalePos, const std::vector<std::vector<std::size_t>>* idx)
{
    const std::size_t surfel_count = vec.size();
    
    std::ofstream fs(plyfilename.c_str());
    // Write header
    fs << "ply\nformat binary_little_endian 1.0";
    fs << "\nelement vertex "<< surfel_count;
    bool ok = ReadWriteHelper<T>::writeHeader(fs);
    if(!ok) { return false; }
    if(idx != nullptr)
    {
        fs << "\nelement face "<< idx->size();
        fs << "\nproperty list uchar int vertex_indices";
    }
    fs << "\nend_header\n";
    fs.flush();
    fs.close();
    
    // Open file in binary appendable and write out
    std::ofstream fpout(plyfilename.c_str(), std::ios::app | std::ios::binary);
    
    for(std::size_t i = 0 ; i < surfel_count ; ++i)
    {
        const T& sout = vec[i];
        bool ok = ReadWriteHelper<T>::writeVertices(sout, fpout, scalePos);
        if(!ok) { break; }
    }
    
    if(idx != nullptr)
    {
        for(std::size_t i = 0 ; i < idx->size() ; ++i)
        {
            const std::vector<std::size_t>& slist = idx->at(i);
            uint8_t count = (uint8_t)slist.size();
            fpout.write(reinterpret_cast<const char*>(&count), sizeof(uint8_t));
            for(const auto& index : slist)
            {
                uint32_t index32 = index;
                fpout.write(reinterpret_cast<const char*>(&index32), sizeof(uint32_t));
            }
        }
    }
    
    fpout.flush();
    fpout.close();
    
    return true;
}

#define INST_FOR_TYPE(TYPE) \
template bool vc::io::loadPLY<TYPE>(const std::string& plyfilename, std::vector<TYPE,Eigen::aligned_allocator<TYPE>>& vec, float scalePos, std::vector<std::vector<std::size_t>>* idx); \
template bool vc::io::savePLY(const std::string& plyfilename, const std::vector<TYPE,Eigen::aligned_allocator<TYPE>>& vec, float scalePos, const std::vector<std::vector<std::size_t>>* idx);

INST_FOR_TYPE(Eigen::Vector3f)
INST_FOR_TYPE(vc::ColorPoint)
INST_FOR_TYPE(vc::NormalPoint)
INST_FOR_TYPE(vc::ColorNormalPoint)
INST_FOR_TYPE(vc::Surfel)
