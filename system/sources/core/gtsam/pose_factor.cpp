#include "pose_factor.h"

namespace df
{

  template <typename Scalar>
  PoseFactor<Scalar>::PoseFactor(const KeyframePtr &kf,
                                 const gtsam::Key &pose_key,
                                 const PoseT &target_pose,
                                 const Scalar &factor_weight)
      : Base(gtsam::cref_list_of<1>(pose_key)),
        pose_key_(pose_key), kf_(kf), target_pose_(target_pose),
        factor_weight_(factor_weight), error_(0.0) 
        {
          AtA_.setZero();
          Atb_.setZero();
        }

  /* ************************************************************************* */
  template <typename Scalar>
  PoseFactor<Scalar>::~PoseFactor() {}

  /* ************************************************************************* */
  template <typename Scalar>
  double PoseFactor<Scalar>::error(const gtsam::Values &c) const
  {
    if (this->active(c))
    {
      // get values of the optimization variables
      PoseT p = c.at<PoseT>(pose_key_);
      Scalar error = ComputeError(p);
      return (double)error;
    }
    else
    {
      return 0.0;
    }
  }

  /* ************************************************************************* */
  template <typename Scalar>
  boost::shared_ptr<gtsam::GaussianFactor>
  PoseFactor<Scalar>::linearize(const gtsam::Values &c) const
  {
    // Only linearize if the factor is active
    if (!this->active(c))
    {
      return boost::shared_ptr<gtsam::HessianFactor>();
    }

    // recover our values
    PoseT p = c.at<PoseT>(pose_key_);
    ComputeJacobianAndError(p);

    const gtsam::FastVector<gtsam::Key> keys = {pose_key_};

    VLOG(3) << "-----------------------------------";
    VLOG(3) << "[PoseFactor<Scalar>::linearize] Asking to linearize " << Name() << " at values:";
    VLOG(3) << "[PoseFactor<Scalar>::linearize] curr pose: " << p.log().transpose();
    VLOG(3) << "[PoseFactor<Scalar>::linearize] tgt pose: " << target_pose_.log().transpose();
    VLOG(3) << "[PoseFactor<Scalar>::linearize] AtA: " << AtA_ << " Atb: " << Atb_;
    VLOG(3) << "[PoseFactor<Scalar>::linearize] error: " << error_;
    VLOG(3) << "-----------------------------------";

    std::vector<gtsam::Matrix> Gs;
    std::vector<gtsam::Vector> gs;

    Eigen::MatrixXd M = AtA_.template cast<double>();
    Eigen::MatrixXd corrected_AtA = NearestPsd(M);

    /*
     * Hessian composition
     *
     *      s0
     * s0 [ G11 ]
     */
    const Eigen::MatrixXd G11 = corrected_AtA.template block<6, 6>(0, 0);
    Gs.push_back(G11);

    /*
    * Jtr composition
    *
    * s0 [ g1 ]
    */
    const Eigen::MatrixXd g1 = Atb_.template block<6, 1>(0, 0);
    gs.push_back(g1);

    // create and return HessianFactor
    return boost::make_shared<gtsam::HessianFactor>(keys, Gs, gs, (double)error_);
  }

  /* ************************************************************************* */
  template <typename Scalar>
  std::string PoseFactor<Scalar>::Name() const
  {
    std::stringstream ss;
    auto fmt = gtsam::DefaultKeyFormatter;
    ss << "PoseFactor " << kf_->id << " keys = {" << fmt(pose_key_) << "}";
    return ss.str();
  }

  /* ************************************************************************* */
  template <typename Scalar>
  inline Scalar PoseFactor<Scalar>::ComputeError(const PoseT &pose) const
  {
    return factor_weight_ * (pose.log().array() - target_pose_.log().array()).square().sum();
  }

  /* ************************************************************************* */
  template <typename Scalar>
  inline void PoseFactor<Scalar>::ComputeJacobianAndError(const PoseT &pose) const
  {
    const auto pose_diff = (target_pose_.log().array() - pose.log().array());
    
    AtA_ = factor_weight_ * Eigen::Matrix<double, 6, 6>::Identity();
    Atb_ = factor_weight_ * pose_diff.template cast<double>();
    error_ = factor_weight_ * (pose.log().array() - target_pose_.log().array()).square().sum();
    
    return;
  }

  /* ************************************************************************* */
  // explicit instantiation
  template class PoseFactor<float>;

} // namespace df
