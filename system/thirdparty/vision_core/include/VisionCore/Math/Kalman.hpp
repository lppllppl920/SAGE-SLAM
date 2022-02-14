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
 * Simple Kalman Filter.
 * ****************************************************************************
 */

#ifndef VISIONCORE_MATH_KALMAN_HPP
#define VISIONCORE_MATH_KALMAN_HPP

#include <VisionCore/Platform.hpp>

#include <Eigen/LU>

namespace vc
{
    
namespace math
{
    
#if 0 // TODO FIXME
template<int dimension, class StateType>
class ConstantProcess {
public:
    typedef Eigen::Matrix<double, StateType::DIM, 1> VecState;
    typedef Eigen::Matrix<double, StateType::DIM, StateType::DIM> MatStateState;
    
    
    /// Noise per second, per component
    VecState sigma;
    MatStateState noise;
    MatStateState jacobian;
    
    inline ConstantProcess() :
    sigma(VecState::Zero()),
    noise(MatStateState::Zero()),
    jacobian(MatStateState::Identity())
    {}
    
    MatStateState const& getJacobian(StateType const& /*state*/, const double /*dt*/) {
        /// No change over time due to process model
        return jacobian;
    }
    
    void updateState(StateType const& /*state*/, const double /*dt*/) {
        /// no-op - no change due to process model
    }
    
    MatStateState const& getNoiseCovariance(const double dt) {
        noise = (dt * sigma).asDiagonal();
        return noise;
    }
    
    void updateFromMeasurement(StateType & state, VecState const & innovation) {
        state.x += innovation;
    }
    
};

template<class StateType>
class AbsoluteMeasurement {
public:
    static const int DIM = StateType::DIM;
    typedef Eigen::Matrix<double, DIM, 1> VecMeas;
    typedef Eigen::Matrix<double, DIM, 1> VecState;
    typedef Eigen::Matrix<double, DIM, DIM> MatMeasMeas;
    typedef Eigen::Matrix<double, DIM, DIM> MatMeasState;
    
    VecMeas measurement;
    
    MatMeasState jacobian;
    MatMeasMeas covariance;
    
    AbsoluteMeasurement() :
    measurement(VecMeas::Zero()),
    jacobian(MatMeasState::Identity()),
    covariance(MatMeasMeas::Identity()) {}
    
    
    MatMeasState const& getJacobian(StateType const& /*state*/) {
        return jacobian;
    }
    
    /// Measurement noise covariance, aka uncertainty
    /// in measurement
    MatMeasMeas const& getCovariance(StateType const& /*state*/) {
        return covariance;
    }
    
    VecState const getInnovation(StateType const& state) {
        return measurement - state.x;
    }
    
};

template<typename T, typename TimeT = T>
class KalmanFilter 
{
public:
    typedef T Scalar;
    typedef TimeT TimeType;
    
    template<unsigned int dim>
    class StateT
    {
    public:
        static const int Dimension = dim;
        typedef Eigen::Matrix<Scalar, Dimension, 1> StateVectorT;
        typedef Eigen::Matrix<Scalar, Dimension, Dimension> CovarianceMatrixT;
        
        StateVectorT StateVector;
        CovarianceMatrixT Covariance;
        
        StateT() : StateVector(StateVectorT::Zero()), Covariance(CovarianceMatrixT::Identity()) 
        { 
            
        }  
    };
    
    template<unsigned int sdim, unsigned int mdim>
    class MeasurementT
    {
    public:
        typedef StateT<sdim> StateType;
        static const int Dimension = mdim;
        typedef Eigen::Matrix<Scalar, Dimension, 1> MeasurementVectorT;
        typedef Eigen::Matrix<Scalar, StateType::Dimension, 1> StateVectorT;
        typedef Eigen::Matrix<Scalar, Dimension, Dimension> JacobianMatrixT;
        typedef Eigen::Matrix<Scalar, Dimension, StateType::Dimension> HMatrixT;

        MeasurementT() 
        { 
            
        }  
    };
    
    template<unsigned int dim>
    void predict(StateT<dim>& state, TimeType dt) 
    {
        const StateT<dim>::CovarianceMatrixT A(processModel.getJacobian(state, dt));
        state.Covariance = A * state.Covariance * A.transpose() + processModel.getNoiseCovariance(dt);
        /// @todo symmetrize?
        processModel.updateState(state, dt);
    }
    
    template<class MeasurementType>
    void correct(MeasurementType & m) 
    {
        typedef Eigen::Matrix<Scalar, MeasurementType::DIM, StateType::DIM> MatMeasState;
        typedef Eigen::Matrix<Scalar, MeasurementType::DIM, MeasurementType::DIM> MatMeasMeas;
        typedef Eigen::Matrix<Scalar, MeasurementType::DIM, 1> VecMeas;
        /// @todo implement
        const MatMeasState & H = m.getJacobian(state);
        const MatMeasMeas & R = m.getCovariance(state);
        const VecState innovation = m.getInnovation(state);
        const MatMeasMeas S = H * state.covariance * H.transpose() + R;

        Eigen::eikfLUType<MatMeasMeas> luOfS = S.eikfLUFunc();
        MatStateState K = state.covariance * H.transpose() * luOfS.inverse();
        processModel.updateFromMeasurement(state, K * innovation);
        state.covariance = (MatStateState::Identity() - K * H) * state.covariance;
        
    }
    
    
};
#endif

}

}

#endif // VISIONCORE_MATH_KALMAN_HPP
