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
 * Compile all of them as a sanity check.
 * ****************************************************************************
 */

#include <VisionCore/Platform.hpp>
#include <VisionCore/CUDAException.hpp>
#include <VisionCore/MemoryPolicy.hpp>
#include <VisionCore/LaunchUtils.hpp>

#include <VisionCore/Buffers/Buffer1D.hpp>
#include <VisionCore/Buffers/Buffer2D.hpp>
#include <VisionCore/Buffers/Buffer3D.hpp>
#include <VisionCore/Buffers/BufferPyramid.hpp>
#include <VisionCore/Buffers/CUDATexture.hpp>
#include <VisionCore/Buffers/GPUVariable.hpp>
#include <VisionCore/Buffers/Image2D.hpp>
#include <VisionCore/Buffers/ImagePyramid.hpp>
#include <VisionCore/Buffers/PyramidBase.hpp>
#include <VisionCore/Buffers/Volume.hpp>
#include <VisionCore/Control/PID.hpp>
#include <VisionCore/Control/VelocityProfile.hpp>
#include <VisionCore/Image/BufferOps.hpp>
#include <VisionCore/Image/ColorMap.hpp>
#include <VisionCore/Image/ConnectedComponents.hpp>
#include <VisionCore/Image/Filters.hpp>
#include <VisionCore/Image/ImagePatch.hpp>
#include <VisionCore/Image/PixelConvert.hpp>
#include <VisionCore/IO/File.hpp>
#include <VisionCore/IO/ImageIO.hpp>
#include <VisionCore/IO/PLYModel.hpp>
#include <VisionCore/Math/Angles.hpp>
#include <VisionCore/Math/Convolution.hpp>
#include <VisionCore/Math/DenavitHartenberg.hpp>
#include <VisionCore/Math/Divergence.hpp>
#include <VisionCore/Math/Fitting.hpp>
#include <VisionCore/Math/Fourier.hpp>
#include <VisionCore/Math/HammingDistance.hpp>
#include <VisionCore/Math/Kalman.hpp>
#include <VisionCore/Math/LeastSquares.hpp>
#include <VisionCore/Math/LiangBarsky.hpp>
//#include <VisionCore/Math/LocalParamSE3.hpp>
#include <VisionCore/Math/LossFunctions.hpp>
#include <VisionCore/Math/PolarSpherical.hpp>
#include <VisionCore/Math/Random.hpp>
#include <VisionCore/Math/RANSAC.hpp>
#include <VisionCore/Math/Statistics.hpp>
#include <VisionCore/Types/AxisAlignedBoundingBox.hpp>
#include <VisionCore/Types/CostVolumeElement.hpp>
#include <VisionCore/Types/Gaussian.hpp>
#include <VisionCore/Types/Hypersphere.hpp>
#include <VisionCore/Types/Polynomial.hpp>
#include <VisionCore/Types/Rectangle.hpp>
#include <VisionCore/Types/SDF.hpp>
#include <VisionCore/Types/SquareUpperTriangularMatrix.hpp>
#include <VisionCore/WrapGL/WrapGLBuffer.hpp>
#include <VisionCore/WrapGL/WrapGLCommon.hpp>
#include <VisionCore/WrapGL/WrapGLContext.hpp>
#include <VisionCore/WrapGL/WrapGLFramebuffer.hpp>
#include <VisionCore/WrapGL/WrapGL.hpp>
#include <VisionCore/WrapGL/WrapGLQuery.hpp>
#include <VisionCore/WrapGL/WrapGLSampler.hpp>
#include <VisionCore/WrapGL/WrapGLProgram.hpp>
#include <VisionCore/WrapGL/WrapGLTexture.hpp>
#include <VisionCore/WrapGL/WrapGLTransformFeedback.hpp>
#include <VisionCore/WrapGL/WrapGLVertexArrayObject.hpp>
#include <VisionCore/WrapGL/WrapGLVertexProcessor.hpp>
#include <VisionCore/WrapGL/WrapGLPixelProcessor.hpp>

