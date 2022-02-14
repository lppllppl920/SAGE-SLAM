/**
 * ****************************************************************************
 * Copyright (c) 2017, Robert Lukierski.
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
 * Shaders.
 * ****************************************************************************
 */

#ifndef VISIONCORE_WRAPGL_PROGRAM_IMPL_HPP
#define VISIONCORE_WRAPGL_PROGRAM_IMPL_HPP

#include <regex>
#include <experimental/filesystem>

inline vc::wrapgl::Program::Program() : progid(0), linked(false)
{
    create();
}

inline vc::wrapgl::Program::~Program()
{
    destroy();
}

inline vc::wrapgl::Program::CompileRetT vc::wrapgl::Program::addPreprocessedShader(Type type, const std::string& source)
{
    bool was_ok = false;
    std::string err_log;
    
    GLhandleARB shader = glCreateShader((GLenum)type);
    WRAPGL_CHECK_ERROR();
    
    const char* csrc = source.c_str();
    
    glShaderSource(shader, 1, &csrc, NULL);
    WRAPGL_CHECK_ERROR();
    
    glCompileShader(shader);
    WRAPGL_CHECK_ERROR();
    
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    WRAPGL_CHECK_ERROR();
    
    if(status != (GLint)GL_TRUE) 
    {
        constexpr std::size_t SHADER_LOG_MAX_LEN = 10240;
        err_log.resize(SHADER_LOG_MAX_LEN);
        GLsizei len;
        glGetShaderInfoLog(shader, SHADER_LOG_MAX_LEN, &len, const_cast<char*>(err_log.data()));
        WRAPGL_CHECK_ERROR();
        err_log.resize(len);
    }
    else
    {
        was_ok = true;
        glAttachShader(progid, shader);
        WRAPGL_CHECK_ERROR();
    }
    
    return std::make_pair(was_ok,err_log);
}

inline vc::wrapgl::Program::CompileRetT vc::wrapgl::Program::addShaderFromSourceCode(Type type, const std::string& source, 
                                                                                     const IncPathVecT& inc_path)
{
    std::stringstream buf_in(source);
    std::stringstream buf_out;
    std::string err;
    
    const bool ok = parseShader(buf_in, buf_out, inc_path, err);
    if(!ok)
    {
        return std::make_pair(ok,err);
    }
    
    return addPreprocessedShader(type, buf_out.str());
}

inline vc::wrapgl::Program::CompileRetT vc::wrapgl::Program::addShaderFromSourceFile(Type type, const std::string& fn, 
                                                                                     const IncPathVecT& inc_path)
{
    std::ifstream ifs(fn);
    if(ifs)
    {
        std::stringstream buf_in;
        std::stringstream buf_out;
        std::string err;

        buf_in << ifs.rdbuf();
        
        const bool ok = parseShader(buf_in, buf_out, inc_path, err);
        if(!ok)
        {
            return std::make_pair(ok,err);
        }
        
        return addPreprocessedShader(type, buf_out.str());
    }
    else
    {
        return std::make_pair(false,"No such file");
    }
}

inline bool vc::wrapgl::Program::parseShader(std::istream& buf_in, std::ostream& buf_out,
                                             const IncPathVecT& inc_path, std::string& errout)
{
    std::string line;
    
    std::regex inc_regex(R"(^\s*#include\s*["<](.*)[">])");

    while(!buf_in.eof()) 
    {
        std::getline(buf_in, line);        
        std::smatch match;

        if (std::regex_match(line, match, inc_regex)) 
        {
            if(match.size() == 2) 
            {
                const std::string inc_fn = match[1].str();
                
                bool found = false;
                
                for(const std::string& ip : inc_path)
                {
                    std::stringstream fn;
                    fn << ip << "/" << inc_fn;

                    std::ifstream ifs(fn.str().c_str());
                    if(ifs.good()) 
                    {
                        const bool ok = parseShader(ifs, buf_out, inc_path, errout);
                        found = true;
                        break;
                    }
                }
                
                if(!found)
                {
                    std::stringstream ss;
                    ss << "Include file " << inc_fn << " not found in the search path";
                    errout = ss.str();
                    return false;
                }
            }
        }
        else
        {
            // Output directly
            buf_out << line << std::endl;
        }
    }
    
    return true;
}

inline void vc::wrapgl::Program::removeAllShaders()
{
    if(progid != 0)
    {
        for(auto& sh : shaders)
        {
            glDetachShader(progid, sh);
            WRAPGL_CHECK_ERROR();
            glDeleteShader(sh);
            WRAPGL_CHECK_ERROR();
        }
        
        linked = false;
    }
}

inline vc::wrapgl::Program::CompileRetT vc::wrapgl::Program::link()
{
    bool was_ok = false;
    std::string err_log;
    
    glLinkProgram(progid);
    WRAPGL_CHECK_ERROR();
    
    GLint status;
    glGetProgramiv(progid, GL_LINK_STATUS, &status);
    WRAPGL_CHECK_ERROR();
    if(status != (GLint)GL_TRUE) 
    {
        constexpr std::size_t PROGRAM_LOG_MAX_LEN = 10240;
        err_log.resize(PROGRAM_LOG_MAX_LEN);
        GLsizei len;
        glGetProgramInfoLog(progid, PROGRAM_LOG_MAX_LEN, &len, const_cast<char*>(err_log.data()));
        WRAPGL_CHECK_ERROR();
        err_log.resize(len);
    }
    else
    {
        was_ok = true;
        linked = true;
    }
    
    return std::make_pair(was_ok,err_log);
}

inline bool vc::wrapgl::Program::isLinked() const 
{ 
    return linked; 
}

inline bool vc::wrapgl::Program::isValid() const 
{ 
    return progid != 0; 
}

inline vc::wrapgl::Program::CompileRetT vc::wrapgl::Program::validate()
{
    bool was_ok = false;
    std::string err_log;
  
    glValidateProgram(progid);
    WRAPGL_CHECK_ERROR();
    
    GLint status;
    glGetProgramiv(progid, GL_VALIDATE_STATUS, &status);
    WRAPGL_CHECK_ERROR();
    
    if(status != (GLint)GL_TRUE) 
    {
        constexpr std::size_t PROGRAM_LOG_MAX_LEN = 10240;
        err_log.resize(PROGRAM_LOG_MAX_LEN);
        GLsizei len;
        glGetProgramInfoLog(progid, PROGRAM_LOG_MAX_LEN, &len, const_cast<char*>(err_log.data()));
        WRAPGL_CHECK_ERROR();
        err_log.resize(len);
    }
    else
    {
        was_ok = true;
    }
    
    return std::make_pair(was_ok,err_log);
}
    
inline void vc::wrapgl::Program::bind() const
{
    glUseProgram(progid);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::unbind() const
{
    glUseProgram(0);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::create()
{
    if(isValid()) { destroy(); }
    
    progid = glCreateProgram();
    WRAPGL_CHECK_ERROR();
    linked = false;
}

inline void vc::wrapgl::Program::destroy()
{
    if(progid != 0) 
    {
        removeAllShaders();
        glDeleteProgram(progid);
        WRAPGL_CHECK_ERROR();
        progid = 0;
    }
}

inline void vc::wrapgl::Program::dispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z) const
{
    glDispatchCompute(num_groups_x, num_groups_y, num_groups_z);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::memoryBarrier(MemoryBarrierMask mbm)
{
    glMemoryBarrier(mbm);
    WRAPGL_CHECK_ERROR();
}

inline GLuint vc::wrapgl::Program::id() const 
{ 
    return progid; 
}

inline void vc::wrapgl::Program::bindAttributeLocation(const char* name, int location)
{
    glBindAttribLocation(progid, location, name);
    WRAPGL_CHECK_ERROR();
    linked = false;
}

inline GLint vc::wrapgl::Program::attributeLocation(const char* name) const 
{ 
    const GLint ret = glGetAttribLocation(progid, name); 
    WRAPGL_CHECK_ERROR();
    return ret;
}

inline void vc::wrapgl::Program::setAttributeValue(int location, GLfloat value)
{
    glVertexAttrib1f(location, value);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setAttributeValue(int location, GLfloat x, GLfloat y)
{
    glVertexAttrib2f(location, x, y);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setAttributeValue(int location, GLfloat x, GLfloat y, GLfloat z)
{
    glVertexAttrib3f(location, x, y, z);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setAttributeValue(int location, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
    glVertexAttrib4f(location, x, y, z, w);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setAttributeValue(int location, const Eigen::Matrix<float,2,1>& value) 
{ 
    setAttributeValue(location, value(0), value(1)); 
}

inline void vc::wrapgl::Program::setAttributeValue(int location, const Eigen::Matrix<float,3,1>& value) 
{ 
    setAttributeValue(location, value(0), value(1), value(2)); 
}

inline void vc::wrapgl::Program::setAttributeValue(int location, const Eigen::Matrix<float,4,1>& value) 
{ 
    setAttributeValue(location, value(0), value(1), value(2), value(3));
}

inline void vc::wrapgl::Program::setAttributeValue(int location, const float2& value) 
{ 
    setAttributeValue(location, value.x, value.y); 
}

inline void vc::wrapgl::Program::setAttributeValue(int location, const float3& value) 
{ 
    setAttributeValue(location, value.x, value.y, value.z); 
}

inline void vc::wrapgl::Program::setAttributeValue(int location, const float4& value) 
{ 
    setAttributeValue(location, value.x, value.y, value.z, value.w); 
}

inline void vc::wrapgl::Program::setAttributeValue(const char* name, GLfloat value) 
{ 
    setAttributeValue(attributeLocation(name), value); 
}

inline void vc::wrapgl::Program::setAttributeValue(const char* name, GLfloat x, GLfloat y) 
{ 
    setAttributeValue(attributeLocation(name), x, y); 
}

inline void vc::wrapgl::Program::setAttributeValue(const char* name, GLfloat x, GLfloat y, GLfloat z) 
{ 
    setAttributeValue(attributeLocation(name), x, y, z); 
}

inline void vc::wrapgl::Program::setAttributeValue(const char* name, GLfloat x, GLfloat y, GLfloat z, GLfloat w) 
{ 
    setAttributeValue(attributeLocation(name), x, y, z, w); 
}

inline void vc::wrapgl::Program::setAttributeValue(const char* name, const Eigen::Matrix<float,2,1>& value) 
{   
    setAttributeValue(attributeLocation(name), value); 
}

inline void vc::wrapgl::Program::setAttributeValue(const char* name, const Eigen::Matrix<float,3,1>& value) 
{ 
    setAttributeValue(attributeLocation(name), value); 
}

inline void vc::wrapgl::Program::setAttributeValue(const char* name, const Eigen::Matrix<float,4,1>& value) 
{ 
    setAttributeValue(attributeLocation(name), value); 
}

inline void vc::wrapgl::Program::setAttributeValue(const char* name, const float2& value) 
{ 
    setAttributeValue(attributeLocation(name), value); 
}

inline void vc::wrapgl::Program::setAttributeValue(const char* name, const float3& value) 
{ 
    setAttributeValue(attributeLocation(name), value); 
}

inline void vc::wrapgl::Program::setAttributeValue(const char* name, const float4& value) 
{ 
    setAttributeValue(attributeLocation(name), value); 
}

inline void vc::wrapgl::Program::setAttributeArray(int location, int tupleSize, GLenum type, 
                                                   bool normalize, uintptr_t offset, int stride)
{
    glVertexAttribPointer(location, tupleSize, type, normalize ? GL_TRUE : GL_FALSE, 
                          stride, reinterpret_cast<const GLvoid*>(offset));
    WRAPGL_CHECK_ERROR();
}

template<typename T>
inline void vc::wrapgl::Program::setAttributeArray(int location, bool normalize, uintptr_t offset, int stride)
{
    typedef typename vc::type_traits<T>::ChannelType cht;
    static constexpr int chcnt = vc::type_traits<T>::ChannelCount;
    static constexpr GLenum gltype = internal::GLTypeTraits<cht>::opengl_data_type;
    
    setAttributeArray(location, chcnt, gltype, normalize, stride, offset);
}

inline void vc::wrapgl::Program::setAttributeArray(const char* name, int tupleSize, GLenum type, 
                                           bool normalize, uintptr_t offset, int stride)
{
    setAttributeArray(attributeLocation(name), tupleSize, type, normalize, stride, offset);
}

template<typename T>
inline void vc::wrapgl::Program::setAttributeArray(const char* name, bool normalize, uintptr_t offset, int stride)
{
    typedef typename vc::type_traits<T>::ChannelType cht;
    static constexpr int chcnt = vc::type_traits<T>::ChannelCount;
    static constexpr GLenum gltype = internal::GLTypeTraits<cht>::opengl_data_type;
    
    setAttributeArray(attributeLocation(name), chcnt, gltype, normalize, stride, offset);
}

inline void vc::wrapgl::Program::enableAttributeArray(int location)
{
    glEnableVertexAttribArray(location);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::enableAttributeArray(const char* name) 
{ 
    enableAttributeArray(attributeLocation(name)); 
}

inline void vc::wrapgl::Program::disableAttributeArray(int location)
{
    glDisableVertexAttribArray(location);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::disableAttributeArray(const char* name) 
{ 
    disableAttributeArray(attributeLocation(name)); 
}

inline GLint vc::wrapgl::Program::uniformLocation(const char* name) const 
{ 
    const GLint ret = glGetUniformLocation(progid, name); 
    WRAPGL_CHECK_ERROR();
    return ret;
}

inline void vc::wrapgl::Program::setUniformValue(int location, GLfloat value)
{
    glUniform1f(location, value);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValue(int location, GLint value)
{
    glUniform1i(location, value);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValue(int location, GLuint value)
{
    glUniform1ui(location, value);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValue(int location, GLfloat x, GLfloat y)
{
    glUniform2f(location, x, y);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValue(int location, GLfloat x, GLfloat y, GLfloat z)
{
    glUniform3f(location, x, y, z);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValue(int location, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
    glUniform4f(location, x, y, z, w);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValue(int location, const Eigen::Matrix<float,2,1>& value) 
{ 
    setUniformValue(location, value(0), value(1)); 
}

inline void vc::wrapgl::Program::setUniformValue(int location, const Eigen::Matrix<float,3,1>& value) 
{ 
    setUniformValue(location, value(0), value(1), value(2)); 
}

inline void vc::wrapgl::Program::setUniformValue(int location, const Eigen::Matrix<float,4,1>& value) 
{ 
    setUniformValue(location, value(0), value(1), value(2), value(3)); 
}

inline void vc::wrapgl::Program::setUniformValue(int location, const float2& value) 
{ 
    setUniformValue(location, value.x, value.y); 
}

inline void vc::wrapgl::Program::setUniformValue(int location, const float3& value) 
{ 
    setUniformValue(location, value.x, value.y, value.z); 
}

inline void vc::wrapgl::Program::setUniformValue(int location, const float4& value) 
{ 
    setUniformValue(location, value.x, value.y, value.z, value.w); 
}

inline void vc::wrapgl::Program::setUniformValue(int location, const Eigen::Matrix<float,2,2>& value) 
{ 
    glUniformMatrix2fv(location, 1, GL_FALSE, value.data()); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValue(int location, const Eigen::Matrix<float,2,3>& value) 
{ 
    glUniformMatrix2x3fv(location, 1, GL_FALSE, value.data()); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValue(int location, const Eigen::Matrix<float,2,4>& value) 
{ 
    glUniformMatrix2x4fv(location, 1, GL_FALSE, value.data()); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValue(int location, const Eigen::Matrix<float,3,2>& value) 
{ 
    glUniformMatrix3x2fv(location, 1, GL_FALSE, value.data()); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValue(int location, const Eigen::Matrix<float,3,3>& value) 
{ 
    glUniformMatrix3fv(location, 1, GL_FALSE, value.data());   
    WRAPGL_CHECK_ERROR();    
}

inline void vc::wrapgl::Program::setUniformValue(int location, const Eigen::Matrix<float,3,4>& value) 
{ 
    glUniformMatrix3x4fv(location, 1, GL_FALSE, value.data());
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValue(int location, const Eigen::Matrix<float,4,2>& value) 
{ 
    glUniformMatrix4x2fv(location, 1, GL_FALSE, value.data()); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValue(int location, const Eigen::Matrix<float,4,3>& value) 
{ 
    glUniformMatrix4x3fv(location, 1, GL_FALSE, value.data()); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValue(int location, const Eigen::Matrix<float,4,4>& value) 
{ 
    glUniformMatrix4fv(location, 1, GL_FALSE, value.data()); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValue(int location, const GLfloat value[2][2]) 
{ 
    glUniformMatrix2fv(location, 1, GL_FALSE, value[0]); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValue(int location, const GLfloat value[3][3]) 
{ 
    glUniformMatrix3fv(location, 1, GL_FALSE, value[0]); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValue(int location, const GLfloat value[4][4]) 
{ 
    glUniformMatrix4fv(location, 1, GL_FALSE, value[0]);       
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValue(int location, const Sophus::SE3f& value)
{
    const Sophus::SE3f::Transformation m = value.matrix();
    glUniformMatrix4fv(location, 1, GL_FALSE, m.data());
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, GLfloat value) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, GLint value) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, GLuint value) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, GLfloat x, GLfloat y) 
{ 
    setUniformValue(uniformLocation(name), x, y); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, GLfloat x, GLfloat y, GLfloat z) 
{ 
    setUniformValue(uniformLocation(name), x, y, z); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, GLfloat x, GLfloat y, GLfloat z, GLfloat w) 
{ 
    setUniformValue(uniformLocation(name), x, y, z, w); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, const Eigen::Matrix<float,2,1>& value) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, const Eigen::Matrix<float,3,1>& value) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, const Eigen::Matrix<float,4,1>& value) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, const float2& value) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, const float3& value) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, const float4& value) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, const Eigen::Matrix<float,2,2>& value) 
{
    setUniformValue(uniformLocation(name), value);  
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, const Eigen::Matrix<float,2,3>& value) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, const Eigen::Matrix<float,2,4>& value) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, const Eigen::Matrix<float,3,2>& value) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, const Eigen::Matrix<float,3,3>& value) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, const Eigen::Matrix<float,3,4>& value) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, const Eigen::Matrix<float,4,2>& value) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, const Eigen::Matrix<float,4,3>& value) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, const Eigen::Matrix<float,4,4>& value) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, const GLfloat value[2][2]) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, const GLfloat value[3][3]) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, const GLfloat value[4][4]) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValue(const char* name, const Sophus::SE3f& value) 
{ 
    setUniformValue(uniformLocation(name), value); 
}

inline void vc::wrapgl::Program::setUniformValueArray(int location, const GLfloat* values, int count, int tupleSize)
{
    if (tupleSize == 1)
    {
        glUniform1fv(location, count, values);
    }
    else if (tupleSize == 2)
    {
        glUniform2fv(location, count, values);
    }
    else if (tupleSize == 3)
    {
        glUniform3fv(location, count, values);
    }
    else if (tupleSize == 4)
    {
        glUniform4fv(location, count, values);
    }
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValueArray(int location, const GLint* values, int count) 
{ 
    glUniform1iv(location, count, values); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValueArray(int location, const GLuint* values, int count) 
{ 
    glUniform1uiv(location, count, values); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValueArray(int location, const Eigen::Matrix<float,2,1>* values, int count) 
{ 
    glUniform2fv(location, count, values[0].data()); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValueArray(int location, const Eigen::Matrix<float,3,1>* values, int count) 
{ 
    glUniform3fv(location, count, values[0].data()); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValueArray(int location, const Eigen::Matrix<float,4,1>* values, int count) 
{ 
    glUniform4fv(location, count, values[0].data()); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValueArray(int location, const Eigen::Matrix<float,2,2>* values, int count) 
{ 
    glUniformMatrix2fv(location, count, GL_FALSE, values[0].data()); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValueArray(int location, const Eigen::Matrix<float,2,3>* values, int count) 
{ 
    glUniformMatrix2x3fv(location, count, GL_FALSE, values[0].data()); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValueArray(int location, const Eigen::Matrix<float,2,4>* values, int count) 
{ 
    glUniformMatrix2x4fv(location, count, GL_FALSE, values[0].data());
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValueArray(int location, const Eigen::Matrix<float,3,2>* values, int count) 
{ 
    glUniformMatrix3x2fv(location, count, GL_FALSE, values[0].data()); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValueArray(int location, const Eigen::Matrix<float,3,3>* values, int count) 
{ 
    glUniformMatrix3fv(location, count, GL_FALSE, values[0].data()); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValueArray(int location, const Eigen::Matrix<float,3,4>* values, int count) 
{ 
    glUniformMatrix3x4fv(location, count, GL_FALSE, values[0].data()); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValueArray(int location, const Eigen::Matrix<float,4,2>* values, int count) 
{ 
    glUniformMatrix4x2fv(location, count, GL_FALSE, values[0].data()); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValueArray(int location, const Eigen::Matrix<float,4,3>* values, int count) 
{ 
    glUniformMatrix4x3fv(location, count, GL_FALSE, values[0].data()); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setUniformValueArray(int location, const Eigen::Matrix<float,4,4>* values, int count) 
{ 
    glUniformMatrix4fv(location, count, GL_FALSE, values[0].data()); 
    WRAPGL_CHECK_ERROR();
}

template<typename T, std::size_t N>
inline void vc::wrapgl::Program::setUniformValueArray(int location, const std::array<T,N>& values)
{
    setUniformValueArray(location, values.data(), N);
}

inline void vc::wrapgl::Program::setUniformValueArray(const char* name, const GLfloat* values, int count, int tupleSize) 
{ 
    setUniformValueArray(uniformLocation(name), values, count, tupleSize); 
}

inline void vc::wrapgl::Program::setUniformValueArray(const char* name, const GLint* values, int count) 
{ 
    setUniformValueArray(uniformLocation(name), values, count); 
}

inline void vc::wrapgl::Program::setUniformValueArray(const char* name, const GLuint* values, int count) 
{ 
    setUniformValueArray(uniformLocation(name), values, count); 
}

inline void vc::wrapgl::Program::setUniformValueArray(const char* name, const Eigen::Matrix<float,2,1>* values, int count) 
{ 
    setUniformValueArray(uniformLocation(name), values, count);   
}

inline void vc::wrapgl::Program::setUniformValueArray(const char* name, const Eigen::Matrix<float,3,1>* values, int count) 
{ 
    setUniformValueArray(uniformLocation(name), values, count); 
}

inline void vc::wrapgl::Program::setUniformValueArray(const char* name, const Eigen::Matrix<float,4,1>* values, int count) 
{ 
    setUniformValueArray(uniformLocation(name), values, count); 
}

inline void vc::wrapgl::Program::setUniformValueArray(const char* name, const Eigen::Matrix<float,2,2>* values, int count) 
{ 
    setUniformValueArray(uniformLocation(name), values, count); 
}

inline void vc::wrapgl::Program::setUniformValueArray(const char* name, const Eigen::Matrix<float,2,3>* values, int count) 
{ 
    setUniformValueArray(uniformLocation(name), values, count); 
}

inline void vc::wrapgl::Program::setUniformValueArray(const char* name, const Eigen::Matrix<float,2,4>* values, int count) 
{ 
    setUniformValueArray(uniformLocation(name), values, count); 
}

inline void vc::wrapgl::Program::setUniformValueArray(const char* name, const Eigen::Matrix<float,3,2>* values, int count) 
{ 
    setUniformValueArray(uniformLocation(name), values, count); 
}

inline void vc::wrapgl::Program::setUniformValueArray(const char* name, const Eigen::Matrix<float,3,3>* values, int count) 
{ 
    setUniformValueArray(uniformLocation(name), values, count); 
}

inline void vc::wrapgl::Program::setUniformValueArray(const char* name, const Eigen::Matrix<float,3,4>* values, int count) 
{ 
    setUniformValueArray(uniformLocation(name), values, count); 
}

inline void vc::wrapgl::Program::setUniformValueArray(const char* name, const Eigen::Matrix<float,4,2>* values, int count) 
{ 
    setUniformValueArray(uniformLocation(name), values, count); 
}

inline void vc::wrapgl::Program::setUniformValueArray(const char* name, const Eigen::Matrix<float,4,3>* values, int count) 
{ 
    setUniformValueArray(uniformLocation(name), values, count); 
}

inline void vc::wrapgl::Program::setUniformValueArray(const char* name, const Eigen::Matrix<float,4,4>* values, int count) 
{ 
    setUniformValueArray(uniformLocation(name), values, count); 
}

template<typename T, std::size_t N>
inline void vc::wrapgl::Program::setUniformValueArray(const char* name, const std::array<T,N>& values)
{
    setUniformValueArray(uniformLocation(name),values.data(), N);
}

inline void vc::wrapgl::Program::bindImage(const wrapgl::Texture2D& tex, GLuint unit, GLenum access, GLenum intfmt) const 
{
    glBindImageTexture(unit, /* unit, note that we're not offseting GL_TEXTURE0 */
                            tex.id(), /* a 2D texture for example */
                            0, /* miplevel */
                            GL_FALSE, /* we cannot use layered */
                            0, /* this is ignored */
                            access, /* we're only writing to it */
                            intfmt/* interpret format as 32-bit float */);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::unbindImage(GLuint unit)
{
    glBindImageTexture(unit, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);
    WRAPGL_CHECK_ERROR();
}

inline GLuint vc::wrapgl::Program::getMaxImageUnits() const 
{ 
    const GLuint ret = internal::getParameter<GLint>(GL_MAX_IMAGE_UNITS); 
    WRAPGL_CHECK_ERROR();
    return ret;
}

inline GLint vc::wrapgl::Program::getFragmentDataLocation(const char* name) const 
{ 
    const GLint ret = glGetFragDataLocation(progid, name); 
    WRAPGL_CHECK_ERROR();
    return ret;
}

inline void vc::wrapgl::Program::bindFragmentDataLocation(const char* name, GLuint color) 
{ 
    glBindFragDataLocation(progid, color, name); 
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setTransformFeedbackVaryings(GLsizei count, const char** varyings, GLenum bufmode)
{
    glTransformFeedbackVaryings(progid, count, varyings, bufmode);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::setTransformFeedbackVaryings(const std::vector<const char*>& varyings, GLenum bufmode)
{
    glTransformFeedbackVaryings(progid, varyings.size(), varyings.data(), bufmode);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::bindShaderStorageBlock(GLuint storageBlockIndex, GLuint storageBlockBinding)
{
    glShaderStorageBlockBinding(progid, storageBlockIndex, storageBlockBinding);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::bindBufferBase(GLuint location, const Buffer& buf, typename Buffer::Type bt)
{
    if(bt == Buffer::Type::Invalid) { bt = buf.type(); }
    glBindBufferBase(static_cast<GLenum>(bt), location, buf.id());
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::bindBufferBase(const char* name, const Buffer& buf, typename Buffer::Type bt)
{
    bindBufferBase(uniformBlockLocation(name),buf,bt);
}

inline void vc::wrapgl::Program::bindBufferRange(GLuint location, const Buffer& buf, GLintptr offset, GLsizeiptr size)
{
    glBindBufferRange(static_cast<GLenum>(buf.type()), location, buf.id(), offset, size);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::bindBufferRange(const char* name, const Buffer& buf, GLintptr offset, GLsizeiptr size)
{
    bindBufferRange(uniformBlockLocation(name), buf, offset, size);
}

inline GLuint vc::wrapgl::Program::uniformBlockLocation(const char* name) const 
{ 
    const GLuint ret = glGetUniformBlockIndex(progid, name); 
    WRAPGL_CHECK_ERROR();
    return ret;
}

inline void vc::wrapgl::Program::bindUniformBuffer(GLuint location, GLuint uniformBlockBinding)
{
    glUniformBlockBinding(progid, location, uniformBlockBinding);
    WRAPGL_CHECK_ERROR();
}

inline void vc::wrapgl::Program::bindUniformBuffer(const char* name, GLuint uniformBlockBinding)
{
    bindUniformBuffer(uniformBlockLocation(name),uniformBlockBinding);
}
 
#endif // VISIONCORE_WRAPGL_PROGRAM_IMPL_HPP
