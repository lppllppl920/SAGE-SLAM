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
 * Denavit-Hartenberg robotic joint generator.
 * ****************************************************************************
 */

#ifndef VISIONCORE_MATH_DENAVIT_HARTENBERG_HPP
#define VISIONCORE_MATH_DENAVIT_HARTENBERG_HPP

#include <VisionCore/Platform.hpp>

namespace vc
{
    
namespace math
{

/**
 * Denavit-Hartenberg transformation generator.
 * 
 * n-1^T_n = Trans_z_n-1(dn) * Rot_z_n-1(theta_n) * Trans_x_n(rn) * Rot_x_n(alphan) 
 * 
 * @param dn offset along previous z to the common normal.
 * @param thetan angle about previous z, from old x to new x.
 * @param rn length of the common normal. Assuming a revolute joint, this is the radius about previous z.
 * @param alphan angle about common normal, from old z axis to new z axis.
 * @param output result.
 * 
 */
template<typename Scalar>
EIGEN_DEVICE_FUNC void generateDenavitHartenberg(Scalar dn, Scalar thetan, Scalar rn, Scalar alphan, Eigen::Matrix<Scalar,4,4>& output)
{
    output << 
    cos(thetan) , -sin(thetan) * cos(alphan) , sin(thetan) * sin(alphan), rn * cos(thetan),
    sin(thetan) , cos(thetan) * cos(alphan) , -cos(thetan) * sin(alphan) , rn * sin(thetan),
    Scalar(0.0) , sin(alphan) , cos(alphan) , dn,
    Scalar(0.0) , Scalar(0.0) , Scalar(0.0), Scalar(1.0);
}
    
}

}

#endif // VISIONCORE_MATH_DENAVIT_HARTENBERG_HPP
