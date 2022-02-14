#ifndef DF_REL_POSE_SCALE_FACTOR_H_
#define DF_REL_POSE_SCALE_FACTOR_H_

#include <gtsam/linear/HessianFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <sophus/se3.hpp>
#include <torch/torch.h>
#include <cmath>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <Eigen/Dense>

#include "gtsam_traits.h"
#include "pinhole_camera.h"
#include "keyframe.h"
#include "mapping_utils.h"

namespace df
{

  template <typename Scalar>
  class RelPoseScaleFactor : public gtsam::NonlinearFactor
  {
    typedef RelPoseScaleFactor<Scalar> This;
    typedef Sophus::SE3<Scalar> PoseT;
    typedef gtsam::NonlinearFactor Base;
    typedef Keyframe<Scalar> KeyframeT;
    typedef typename KeyframeT::Ptr KeyframePtr;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RelPoseScaleFactor(const KeyframePtr &kf0,
                       const KeyframePtr &kf1,
                       const gtsam::Key &pose0_key,
                       const gtsam::Key &pose1_key,
                       const gtsam::Key &scale0_key,
                       const gtsam::Key &scale1_key,
                       const PoseT &target_pose_10,
                       const Scalar target_scale0,
                       const Scalar target_scale1,
                       const Scalar factor_weight,
                       const Scalar rot_weight,
                       const Scalar scale_weight);

    virtual ~RelPoseScaleFactor();

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

    //   /*!
    //  * \brief Get the dimension of the factor (number of rows on linearization)
    //  * \return
    //  */
    size_t dim() const override { return 14; }

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
      return shared_ptr(new This(kf0_, kf1_, pose0_key_, pose1_key_, scale0_key_, scale1_key_,
                                 target_pose10_, target_scale0_, target_scale1_,
                                 factor_weight_, rot_weight_, scale_weight_));
    }

  private:
    Scalar ComputeError(const PoseT &pose0, const PoseT &pose1, const Scalar scale0, const Scalar scale1) const;
    void ComputeJacobianAndError(const PoseT &pose0, const PoseT &pose1, const Scalar scale0, const Scalar scale1) const;

    gtsam::Key pose0_key_;
    gtsam::Key pose1_key_;
    gtsam::Key scale0_key_;
    gtsam::Key scale1_key_;

    KeyframePtr kf0_;
    KeyframePtr kf1_;

    PoseT target_pose10_;
    Scalar target_scale0_;
    Scalar target_scale1_;

    Scalar factor_weight_;
    Scalar rot_weight_;
    Scalar scale_weight_;

    mutable double error_;
    mutable Eigen::Matrix<double, 14, 14> AtA_;
    mutable Eigen::Matrix<double, 14, 1> Atb_;

    Scalar log_target_scale_ratio10_;

    Eigen::Matrix<Scalar, 9, 9> transpose_perm_mat_;
  };

} // namespace df

#endif // DF_REL_POSE_SCALE_FACTOR_H_
