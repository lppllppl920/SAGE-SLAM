#include "code_factor.h"

namespace df
{

  template <typename Scalar, int CS>
  CodeFactor<Scalar, CS>::CodeFactor(const KeyframePtr &kf,
                                     const gtsam::Key &code_key,
                                     const CodeT &init_code,
                                     const Scalar &factor_weight)
      : Base(gtsam::cref_list_of<1>(code_key)),
        code_key_(code_key), init_code_(init_code), kf_(kf),
        factor_weight_(factor_weight), error_(0.0)
  { 
    AtA_.setZero();
    Atb_.setZero();
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  CodeFactor<Scalar, CS>::~CodeFactor() {}

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  double CodeFactor<Scalar, CS>::error(const gtsam::Values &c) const
  {
    if (this->active(c))
    {
      // get values of the optimization variables
      CodeT code = c.at<CodeT>(code_key_);
      return ComputeError(code);
    }
    else
    {
      return 0.0;
    }
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  boost::shared_ptr<gtsam::GaussianFactor>
  CodeFactor<Scalar, CS>::linearize(const gtsam::Values &c) const
  {
    // Only linearize if the factor is active
    if (!this->active(c))
    {
      return boost::shared_ptr<gtsam::HessianFactor>();
    }
    // recover our values
    CodeT code = c.at<CodeT>(code_key_);
    error_ = ComputeError(code);


    AtA_ = factor_weight_ * Eigen::Matrix<double, CS, CS>::Identity();
    Atb_ = factor_weight_ * (init_code_ - code);

    const gtsam::FastVector<gtsam::Key> keys = {code_key_};

    std::vector<gtsam::Matrix> Gs;
    std::vector<gtsam::Vector> gs;

    VLOG(3) << "-----------------------------------";
    VLOG(3) << "[CodeFactor<Scalar, CS>::linearize] Asking to linearize " << Name() << " at values:";
    VLOG(3) << "[CodeFactor<Scalar, CS>::linearize] code: " << code.transpose();
    VLOG(3) << "[CodeFactor<Scalar, CS>::linearize] error: " << error_ << " delta code: " << (code - init_code_).transpose();
    VLOG(3) << "-----------------------------------";
    //   /*
    //  * Hessian composition
    //  *
    //  *      c0
    //  * c0 [ G11 ]
    const Eigen::MatrixXd G11 = AtA_.template block<CS, CS>(0, 0);
    Gs.push_back(G11);

    /*
    * Jtr composition
    *
    * c0 [ g1 ]
    */
    const Eigen::MatrixXd g1 = Atb_.template block<CS, 1>(0, 0);
    gs.push_back(g1);

    // create and return HessianFactor
    return boost::make_shared<gtsam::HessianFactor>(keys, Gs, gs, error_);
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  std::string CodeFactor<Scalar, CS>::Name() const
  {
    std::stringstream ss;
    auto fmt = gtsam::DefaultKeyFormatter;
    ss << "CodeFactor " << kf_->id << " keys = {" << fmt(code_key_) << "}";
    return ss.str();
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  inline double CodeFactor<Scalar, CS>::ComputeError(const CodeT &code) const
  {
    CodeT diff = init_code_ - code;
    
    return factor_weight_ * (diff.array().square().sum() / (double)code.size());
  }

  /* ************************************************************************* */
  // explicit instantiation
  template class CodeFactor<float, DF_CODE_SIZE>;

} // namespace df
