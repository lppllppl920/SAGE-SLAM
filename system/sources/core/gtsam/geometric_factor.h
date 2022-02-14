#ifndef DF_GEOMETRIC_FACTOR_H_
#define DF_GEOMETRIC_FACTOR_H_

#include <gtsam/linear/HessianFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <sophus/se3.hpp>
#include <torch/torch.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <Eigen/Dense>

#include "gtsam_traits.h"
#include "pinhole_camera.h"
#include "keyframe.h"
#include "mapping_utils.h"
#include "geometric_factor_kernels.h"

namespace df
{

  template <typename Scalar, int CS>
  class GeometricFactor : public gtsam::NonlinearFactor
  {
    typedef GeometricFactor<Scalar, CS> This;
    typedef gtsam::NonlinearFactor Base;
    typedef Sophus::SE3<Scalar> PoseT;
    typedef gtsam::Vector CodeT;
    typedef Keyframe<Scalar> KeyframeT;
    typedef typename KeyframeT::Ptr KeyframePtr;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /*!
   * \brief Constructor calculating the sparse sampling
   */
    GeometricFactor(const PinholeCamera<Scalar> &cam,
                    const KeyframePtr &kf0,
                    const KeyframePtr &kf1,
                    const gtsam::Key &pose0_key,
                    const gtsam::Key &pose1_key,
                    const gtsam::Key &code0_key,
                    const gtsam::Key &code1_key,
                    const gtsam::Key &scale0_key,
                    const gtsam::Key &scale1_key,
                    const Scalar &factor_weight,
                    const Scalar &loss_param,
                    const Scalar &dpt_eps);

    virtual ~GeometricFactor();

    /*!
   * \brief Calculate the error of the factor
   * \param c Values to evaluate the error at
   * \return The error
   */
    double error(const gtsam::Values &c) const override;

    /*!
   * \brief Linearizes this factor at a linearization point
   * \param c Linearization point
   * \return Linearized factor
   */
    boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values &c) const override;

    /*!
    * \brief Get the dimension of the factor (number of rows on linearization)
    * \return
    */
    size_t dim() const override { return 14 + 2 * CS; }

    /*!
   * \brief Return a string describing this factor
   * \return Factor description string
   */
    std::string Name() const;

    /*!
    * \brief Clone the factor
    * \return shared ptr to a cloned factor
    */
    virtual shared_ptr clone() const override
    {
      return shared_ptr(new This(cam_, kf0_, kf1_, pose0_key_, pose1_key_,
                                 code0_key_, code1_key_, scale0_key_, scale1_key_, factor_weight_, loss_param_, dpt_eps_));
    }

  private:
    Scalar ComputeError(const PoseT &pose0, const PoseT &pose1, const CodeT &code0,
                        const CodeT &code1, const Scalar &scale0, const Scalar &scale1) const;
    void ComputeJacobianAndError(const PoseT &pose0, const PoseT &pose1,
                                 const CodeT &code0, const CodeT &code1,
                                 const Scalar &scale0, const Scalar &scale1) const;

    /* variables we tie with this factor */
    gtsam::Key pose0_key_;
    gtsam::Key pose1_key_;
    gtsam::Key code0_key_;
    gtsam::Key code1_key_;
    gtsam::Key scale0_key_;
    gtsam::Key scale1_key_;

    PinholeCamera<Scalar> cam_;

    /* we do modify the keyframes in the factor */
    KeyframePtr kf0_;
    KeyframePtr kf1_;

    Scalar factor_weight_;
    Scalar loss_param_;
    
    Scalar dpt_eps_;

    mutable double error_;
    mutable Eigen::Matrix<double, 14 + 2 * CS, 14 + 2 * CS> AtA_;
    mutable Eigen::Matrix<double, 14 + 2 * CS, 1> Atb_;

  };

} // namespace df

#endif // DF_GEOMETRIC_FACTOR_H_
