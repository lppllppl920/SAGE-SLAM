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
 * Shaders.
 * ****************************************************************************
 */

#ifndef VISIONCORE_WRAPGL_PROGRAM_HPP
#define VISIONCORE_WRAPGL_PROGRAM_HPP

#include <string>
#include <sstream>
#include <fstream>
#include <array>

#include <VisionCore/WrapGL/WrapGLCommon.hpp>
#include <VisionCore/WrapGL/WrapGLTexture.hpp>
#include <VisionCore/WrapGL/WrapGLBuffer.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <sophus/so2.hpp>
#include <sophus/so3.hpp>
#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

/**
 * TODO:
 * 
 * - glGetActiveUniformName
 * - glGetUniformIndices
 * - glGetActiveUniformsiv
 * - glGetProgramInterfaceiv
 * - glGetProgramResourceIndex
 * - glGetProgramResourceiv
 * - glGetUniformSubroutine
 * - glGetTransformFeedbackVarying
 * - glShaderStorageBlockBinding
 * 
 * Add class: ProgramPipelines
 */

namespace vc
{

namespace wrapgl
{
    
class ShaderException : public std::runtime_error
{
public:
    ShaderException(const std::string& build_err) : std::runtime_error(build_err)
    {
    }
};
    
class Program
{
public:    
    typedef std::vector<std::string> IncPathVecT;
    typedef std::pair<bool,std::string> CompileRetT;
    typedef ScopeBinder<Program> Binder;
    
    enum class Type : GLuint
    {
        Vertex = (GLuint)GL_VERTEX_SHADER,
        Fragment = (GLuint)GL_FRAGMENT_SHADER,
        Geometry = (GLuint)GL_GEOMETRY_SHADER,
        TessellationControl = (GLuint)GL_TESS_CONTROL_SHADER,
        TessellationEvaluation = (GLuint)GL_TESS_EVALUATION_SHADER,
        Compute = (GLuint)GL_COMPUTE_SHADER,
    };
    
    inline Program();
    virtual ~Program();
    
    inline CompileRetT addShaderFromSourceCode(Type type, const std::string& source, const IncPathVecT& inc_path = IncPathVecT());
    inline CompileRetT addShaderFromSourceFile(Type type, const std::string& fn, const IncPathVecT& inc_path = IncPathVecT());
    inline void removeAllShaders();
    
    inline CompileRetT link();
    inline bool isLinked() const;
    inline bool isValid() const;
    
    inline CompileRetT validate();
        
    inline void bind() const;
    inline void unbind() const;
    
    inline void create();    
    inline void destroy();
    
    /// Program number
    inline GLuint id() const; 

    /// Dispatch a compute shader.
    inline void dispatchCompute(GLuint num_groups_x, GLuint num_groups_y = 0, GLuint num_groups_z = 0) const;
    /// Force a memory barrier
    inline void memoryBarrier(MemoryBarrierMask mbm =  GL_ALL_BARRIER_BITS);
    
    /// Assign location to in variable name
    inline void bindAttributeLocation(const char* name, int location);
    
    /// Get location for a in variable name
    inline GLint attributeLocation(const char* name) const;
    
    /// Rarely used set single value
    inline void setAttributeValue(int location, GLfloat value);
    inline void setAttributeValue(int location, GLfloat x, GLfloat y);
    inline void setAttributeValue(int location, GLfloat x, GLfloat y, GLfloat z);
    inline void setAttributeValue(int location, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
    inline void setAttributeValue(int location, const Eigen::Matrix<float,2,1>& value);
    inline void setAttributeValue(int location, const Eigen::Matrix<float,3,1>& value);
    inline void setAttributeValue(int location, const Eigen::Matrix<float,4,1>& value);
    inline void setAttributeValue(int location, const float2& value);
    inline void setAttributeValue(int location, const float3& value);
    inline void setAttributeValue(int location, const float4& value);
    inline void setAttributeValue(const char* name, GLfloat value);
    inline void setAttributeValue(const char* name, GLfloat x, GLfloat y);
    inline void setAttributeValue(const char* name, GLfloat x, GLfloat y, GLfloat z);
    inline void setAttributeValue(const char* name, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
    inline void setAttributeValue(const char* name, const Eigen::Matrix<float,2,1>& value);
    inline void setAttributeValue(const char* name, const Eigen::Matrix<float,3,1>& value);
    inline void setAttributeValue(const char* name, const Eigen::Matrix<float,4,1>& value);
    inline void setAttributeValue(const char* name, const float2& value);
    inline void setAttributeValue(const char* name, const float3& value);
    inline void setAttributeValue(const char* name, const float4& value);
    
    // Modern VAO based
    inline void setAttributeArray(int location, int tupleSize, GLenum type, 
                                  bool normalize = false, uintptr_t offset = 0, int stride = 0);
    template<typename T>
    inline void setAttributeArray(int location, bool normalize = false, uintptr_t offset = 0, int stride = 0);
    
    inline void setAttributeArray(const char* name, int tupleSize, GLenum type, 
                                  bool normalize = false, uintptr_t offset = 0, int stride = 0);
    template<typename T>
    inline void setAttributeArray(const char* name, bool normalize = false, uintptr_t offset = 0, int stride = 0);
    
    inline void enableAttributeArray(int location);
    inline void enableAttributeArray(const char* name);
    inline void disableAttributeArray(int location);
    inline void disableAttributeArray(const char* name);
    
    // Image Units
    inline void bindImage(const wrapgl::Texture2D& tex, GLuint unit, GLenum access = GL_READ_ONLY, GLenum intfmt = GL_R32F) const;
    inline void unbindImage(GLuint unit);
    inline GLuint getMaxImageUnits() const;
    
    // Fragment Shader Outputs (GL_COLOR_ATTACHMENT0 etc)
    inline GLint getFragmentDataLocation(const char* name) const;
    inline void bindFragmentDataLocation(const char* name, GLuint color);
    
    // Transform Feedback
    inline void setTransformFeedbackVaryings(GLsizei count, const char** varyings, GLenum bufmode = GL_INTERLEAVED_ATTRIBS);
    inline void setTransformFeedbackVaryings(const std::vector<const char*>& varyings, GLenum bufmode = GL_INTERLEAVED_ATTRIBS);
    
    // Shader Storage Blocks
    inline void bindShaderStorageBlock(GLuint storageBlockIndex, GLuint storageBlockBinding);
    
    // Binding Buffers (UBO,SSBO)
    inline void bindBufferBase(GLuint location, const Buffer& buf, typename Buffer::Type bt = Buffer::Type::Invalid);
    inline void bindBufferBase(const char* name, const Buffer& buf, typename Buffer::Type bt = Buffer::Type::Invalid);
    inline void bindBufferRange(GLuint location, const Buffer& buf, GLintptr offset, GLsizeiptr size);
    inline void bindBufferRange(const char* name, const Buffer& buf, GLintptr offset, GLsizeiptr size);
    
    /// Uniform variables
    inline GLint uniformLocation(const char* name) const;
    
    inline void setUniformValue(int location, GLfloat value);
    inline void setUniformValue(int location, GLint value);
    inline void setUniformValue(int location, GLuint value);
    inline void setUniformValue(int location, GLfloat x, GLfloat y);
    inline void setUniformValue(int location, GLfloat x, GLfloat y, GLfloat z);
    inline void setUniformValue(int location, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
    inline void setUniformValue(int location, const Eigen::Matrix<float,2,1>& value);
    inline void setUniformValue(int location, const Eigen::Matrix<float,3,1>& value);
    inline void setUniformValue(int location, const Eigen::Matrix<float,4,1>& value);
    inline void setUniformValue(int location, const float2& value);
    inline void setUniformValue(int location, const float3& value);
    inline void setUniformValue(int location, const float4& value);
    inline void setUniformValue(int location, const Eigen::Matrix<float,2,2>& value);
    inline void setUniformValue(int location, const Eigen::Matrix<float,2,3>& value);
    inline void setUniformValue(int location, const Eigen::Matrix<float,2,4>& value);
    inline void setUniformValue(int location, const Eigen::Matrix<float,3,2>& value);
    inline void setUniformValue(int location, const Eigen::Matrix<float,3,3>& value);
    inline void setUniformValue(int location, const Eigen::Matrix<float,3,4>& value);
    inline void setUniformValue(int location, const Eigen::Matrix<float,4,2>& value);
    inline void setUniformValue(int location, const Eigen::Matrix<float,4,3>& value);
    inline void setUniformValue(int location, const Eigen::Matrix<float,4,4>& value);
    inline void setUniformValue(int location, const GLfloat value[2][2]);
    inline void setUniformValue(int location, const GLfloat value[3][3]);
    inline void setUniformValue(int location, const GLfloat value[4][4]);
    inline void setUniformValue(int location, const Sophus::SE3f& value);
    
    inline void setUniformValue(const char* name, GLfloat value);
    inline void setUniformValue(const char* name, GLint value);
    inline void setUniformValue(const char* name, GLuint value);
    inline void setUniformValue(const char* name, GLfloat x, GLfloat y);
    inline void setUniformValue(const char* name, GLfloat x, GLfloat y, GLfloat z);
    inline void setUniformValue(const char* name, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
    inline void setUniformValue(const char* name, const Eigen::Matrix<float,2,1>& value);
    inline void setUniformValue(const char* name, const Eigen::Matrix<float,3,1>& value);
    inline void setUniformValue(const char* name, const Eigen::Matrix<float,4,1>& value);
    inline void setUniformValue(const char* name, const float2& value);
    inline void setUniformValue(const char* name, const float3& value);
    inline void setUniformValue(const char* name, const float4& value);
    inline void setUniformValue(const char* name, const Eigen::Matrix<float,2,2>& value);
    inline void setUniformValue(const char* name, const Eigen::Matrix<float,2,3>& value);
    inline void setUniformValue(const char* name, const Eigen::Matrix<float,2,4>& value);
    inline void setUniformValue(const char* name, const Eigen::Matrix<float,3,2>& value);
    inline void setUniformValue(const char* name, const Eigen::Matrix<float,3,3>& value);
    inline void setUniformValue(const char* name, const Eigen::Matrix<float,3,4>& value);
    inline void setUniformValue(const char* name, const Eigen::Matrix<float,4,2>& value);
    inline void setUniformValue(const char* name, const Eigen::Matrix<float,4,3>& value);
    inline void setUniformValue(const char* name, const Eigen::Matrix<float,4,4>& value);
    inline void setUniformValue(const char* name, const GLfloat value[2][2]);
    inline void setUniformValue(const char* name, const GLfloat value[3][3]);
    inline void setUniformValue(const char* name, const GLfloat value[4][4]);
    inline void setUniformValue(const char* name, const Sophus::SE3f& value);
    
    // Arrays of uniform
    inline void setUniformValueArray(int location, const GLfloat* values, int count, int tupleSize);
    inline void setUniformValueArray(int location, const GLint* values, int count);
    inline void setUniformValueArray(int location, const GLuint* values, int count);
    inline void setUniformValueArray(int location, const Eigen::Matrix<float,2,1>* values, int count);
    inline void setUniformValueArray(int location, const Eigen::Matrix<float,3,1>* values, int count);
    inline void setUniformValueArray(int location, const Eigen::Matrix<float,4,1>* values, int count);
    inline void setUniformValueArray(int location, const Eigen::Matrix<float,2,2>* values, int count);
    inline void setUniformValueArray(int location, const Eigen::Matrix<float,2,3>* values, int count);
    inline void setUniformValueArray(int location, const Eigen::Matrix<float,2,4>* values, int count);
    inline void setUniformValueArray(int location, const Eigen::Matrix<float,3,2>* values, int count);
    inline void setUniformValueArray(int location, const Eigen::Matrix<float,3,3>* values, int count);
    inline void setUniformValueArray(int location, const Eigen::Matrix<float,3,4>* values, int count);
    inline void setUniformValueArray(int location, const Eigen::Matrix<float,4,2>* values, int count);
    inline void setUniformValueArray(int location, const Eigen::Matrix<float,4,3>* values, int count);
    inline void setUniformValueArray(int location, const Eigen::Matrix<float,4,4>* values, int count);
    template<typename T, std::size_t N>
    inline void setUniformValueArray(int location, const std::array<T,N>& values);
    
    inline void setUniformValueArray(const char* name, const GLfloat* values, int count, int tupleSize);
    inline void setUniformValueArray(const char* name, const GLint* values, int count);
    inline void setUniformValueArray(const char* name, const GLuint* values, int count);
    inline void setUniformValueArray(const char* name, const Eigen::Matrix<float,2,1>* values, int count);
    inline void setUniformValueArray(const char* name, const Eigen::Matrix<float,3,1>* values, int count);
    inline void setUniformValueArray(const char* name, const Eigen::Matrix<float,4,1>* values, int count);
    inline void setUniformValueArray(const char* name, const Eigen::Matrix<float,2,2>* values, int count);
    inline void setUniformValueArray(const char* name, const Eigen::Matrix<float,2,3>* values, int count);
    inline void setUniformValueArray(const char* name, const Eigen::Matrix<float,2,4>* values, int count);
    inline void setUniformValueArray(const char* name, const Eigen::Matrix<float,3,2>* values, int count);
    inline void setUniformValueArray(const char* name, const Eigen::Matrix<float,3,3>* values, int count);
    inline void setUniformValueArray(const char* name, const Eigen::Matrix<float,3,4>* values, int count);
    inline void setUniformValueArray(const char* name, const Eigen::Matrix<float,4,2>* values, int count);
    inline void setUniformValueArray(const char* name, const Eigen::Matrix<float,4,3>* values, int count);
    inline void setUniformValueArray(const char* name, const Eigen::Matrix<float,4,4>* values, int count);
    template<typename T, std::size_t N>
    inline void setUniformValueArray(const char* name, const std::array<T,N>& values);
    
    // Uniform interface blocks
    inline GLuint uniformBlockLocation(const char* name) const;
    inline void bindUniformBuffer(GLuint location, GLuint uniformBlockBinding);
    inline void bindUniformBuffer(const char* name, GLuint uniformBlockBinding);
    
private:
    inline CompileRetT addPreprocessedShader(Type type, const std::string& source);
    inline bool parseShader(std::istream& buf_in, std::ostream& buf_out,
                            const IncPathVecT& inc_path, 
                            std::string& errout);
    
    GLuint progid;
    std::vector<GLhandleARB> shaders;
    bool linked;
};

}

}

#include <VisionCore/WrapGL/impl/WrapGLProgram_impl.hpp>

#endif // VISIONCORE_WRAPGL_PROGRAM_HPP
