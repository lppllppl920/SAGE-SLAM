#include "scale_factor.h"

namespace df
{

  template <typename Scalar>
  ScaleFactor<Scalar>::ScaleFactor(const KeyframePtr &kf,
                                   const gtsam::Key &scale_key,
                                   const Scalar &init_scale,
                                   const Scalar &factor_weight)
      : Base(gtsam::cref_list_of<1>(scale_key)),
        scale_key_(scale_key), kf_(kf), init_scale_(init_scale),
        factor_weight_(factor_weight), error_(0.0)
        {
          AtA_.setZero();
          Atb_.setZero();
        }

  /* ************************************************************************* */
  template <typename Scalar>
  ScaleFactor<Scalar>::~ScaleFactor() {}

  /* ************************************************************************* */
  template <typename Scalar>
  double ScaleFactor<Scalar>::error(const gtsam::Values &c) const
  {
    if (this->active(c))
    {
      // get values of the optimization variables
      Scalar s = c.at<Scalar>(scale_key_);
      Scalar error = ComputeError(s);
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
  ScaleFactor<Scalar>::linearize(const gtsam::Values &c) const
  {
    // Only linearize if the factor is active
    if (!this->active(c))
    {
      return boost::shared_ptr<gtsam::HessianFactor>();
    }

    // recover our values
    Scalar scale = c.at<Scalar>(scale_key_);
    ComputeJacobianAndError(scale);

    const gtsam::FastVector<gtsam::Key> keys = {scale_key_};

    VLOG(3) << "-----------------------------------";
    VLOG(3) << "[ScaleFactor<Scalar>::linearize] Asking to linearize " << Name() << " at values:";
    VLOG(3) << "[ScaleFactor<Scalar>::linearize] scale: " << scale;
    VLOG(3) << "[ScaleFactor<Scalar>::linearize] AtA: " << AtA_ << " Atb: " << Atb_;
    VLOG(3) << "[ScaleFactor<Scalar>::linearize] error: " << error_ << " delta scale: " << scale - init_scale_;
    VLOG(3) << "-----------------------------------";

    std::vector<gtsam::Matrix> Gs;
    std::vector<gtsam::Vector> gs;

    Eigen::MatrixXd M = AtA_.template cast<double>();
    Eigen::MatrixXd corrected_AtA = NearestPsd(M);

    //   /*
    //  * Hessian composition
    //  *
    //  *      s0
    //  * s0 [ G11 ]
    const Eigen::MatrixXd G11 = corrected_AtA.template block<1, 1>(0, 0);
    Gs.push_back(G11);

    /*
    * Jtr composition
    *
    * s0 [ g1 ]
    */
    const Eigen::MatrixXd g1 = Atb_.template block<1, 1>(0, 0);
    gs.push_back(g1);

    // create and return HessianFactor
    return boost::make_shared<gtsam::HessianFactor>(keys, Gs, gs, (double)error_);
  }

  /* ************************************************************************* */
  template <typename Scalar>
  std::string ScaleFactor<Scalar>::Name() const
  {
    std::stringstream ss;
    auto fmt = gtsam::DefaultKeyFormatter;
    ss << "ScaleFactor " << kf_->id << " keys = {" << fmt(scale_key_) << "}";
    return ss.str();
  }

  /* ************************************************************************* */
  template <typename Scalar>
  inline Scalar ScaleFactor<Scalar>::ComputeError(const Scalar &scale) const
  {
    // safety check on scale being positive number
    if (scale <= 0)
    {
      LOG(FATAL) << "[ScaleFactor<Scalar>::ComputeError] scale in keyframe " << kf_->id << " is non-positive : " << scale;
    }

    return factor_weight_ * std::pow(std::log(scale) - std::log(init_scale_), 2);
  }

  /* ************************************************************************* */
  template <typename Scalar>
  inline void ScaleFactor<Scalar>::ComputeJacobianAndError(const Scalar &scale) const
  {

    // safety check on scale being positive number
    if (scale <= 0)
    {
      LOG(FATAL) << "[ScaleFactor<Scalar>::ComputeJacobianAndError] scale in keyframe " << kf_->id << " is non-positive : " << scale;
    }

    Scalar scale_log_diff = std::log(init_scale_) - std::log(scale);

    AtA_.coeffRef(0, 0) = factor_weight_ / (scale * scale);
    Atb_.coeffRef(0, 0) = factor_weight_ / scale * scale_log_diff;
    error_ = factor_weight_ * (scale_log_diff * scale_log_diff);
    return;
  }

  /* ************************************************************************* */
  // explicit instantiation
  template class ScaleFactor<float>;

} // namespace df
