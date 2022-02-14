#include "loop_mg_factor.h"

namespace df
{

  template <typename Scalar>
  LoopMGFactor<Scalar>::LoopMGFactor(
      const KeyframePtr &kf0,
      const KeyframePtr &kf1,
      const gtsam::Key &pose0_key,
      const gtsam::Key &pose1_key,
      const gtsam::Key &scale0_key,
      const gtsam::Key &scale1_key,
      const at::Tensor matched_unscaled_dpts_0,
      const at::Tensor matched_unscaled_dpts_1,
      const at::Tensor matched_locations_homo_0,
      const at::Tensor matched_locations_homo_1,
      const Scalar &factor_weight,
      const Scalar &loss_param,
      const Scalar &dpt_eps)
      : Base(gtsam::cref_list_of<4>(pose0_key)(pose1_key)(scale0_key)(scale1_key)),
        pose0_key_(pose0_key),
        pose1_key_(pose1_key),
        scale0_key_(scale0_key),
        scale1_key_(scale1_key),
        kf0_(kf0), kf1_(kf1),
        factor_weight_(factor_weight), loss_param_(loss_param),
        dpt_eps_(dpt_eps), error_(0.0)
  {
    torch::NoGradGuard no_grad;
    matched_unscaled_dpts_0_ = matched_unscaled_dpts_0;
    matched_unscaled_dpts_1_ = matched_unscaled_dpts_1;
    matched_locations_homo_0_ = matched_locations_homo_0;
    matched_locations_homo_1_ = matched_locations_homo_1;
    AtA_.setZero();
    Atb_.setZero();
  }

  /* ************************************************************************* */
  template <typename Scalar>
  LoopMGFactor<Scalar>::~LoopMGFactor() {}

  /* ************************************************************************* */
  template <typename Scalar>
  double LoopMGFactor<Scalar>::error(const gtsam::Values &c) const
  {
    torch::NoGradGuard no_grad;
    if (this->active(c))
    {
      // get values of the optimization variables
      PoseT p0 = c.at<PoseT>(pose0_key_);
      PoseT p1 = c.at<PoseT>(pose1_key_);
      Scalar s0 = c.at<Scalar>(scale0_key_);
      Scalar s1 = c.at<Scalar>(scale1_key_);

      Scalar error = ComputeError(p0, p1, s0, s1);

      VLOG(3) << "[LoopMGFactor<Scalar>::error] error between " << kf0_->id << " " << kf1_->id << " : " << error;

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
  LoopMGFactor<Scalar>::linearize(const gtsam::Values &c) const
  {
    torch::NoGradGuard no_grad;
    // Only linearize if the factor is active
    if (!this->active(c))
    {
      return boost::shared_ptr<gtsam::HessianFactor>();
    }

    // recover our values
    PoseT p0 = c.at<PoseT>(pose0_key_);
    PoseT p1 = c.at<PoseT>(pose1_key_);
    Scalar s0 = c.at<Scalar>(scale0_key_);
    Scalar s1 = c.at<Scalar>(scale1_key_);

    ComputeJacobianAndError(p0, p1, s0, s1);

    // tic("[LoopMGFactor<Scalar>::linearize] psd " + std::to_string(kf0_->id) + " " + std::to_string(kf1_->id));
    Eigen::MatrixXd M = AtA_.template cast<double>();
    Eigen::MatrixXd corrected_AtA = NearestPsd(M);
    // toc("[LoopMGFactor<Scalar>::linearize] psd " + std::to_string(kf0_->id) + " " + std::to_string(kf1_->id));

    const gtsam::FastVector<gtsam::Key> keys = {pose0_key_, pose1_key_, scale0_key_, scale1_key_};

    std::vector<gtsam::Matrix> Gs;
    std::vector<gtsam::Vector> gs;

    //  * Hessian composition
    //  *
    //  *      p0   p1   s0   s1
    //  * p0 [ G11  G12  G13  G14 ]
    //  * p1 [      G22  G23  G24 ]
    //  * s0 [           G33  G34 ]
    //  * s1 [                G44 ]

    const Eigen::MatrixXd G11 = corrected_AtA.template block<6, 6>(0, 0);
    const Eigen::MatrixXd G12 = corrected_AtA.template block<6, 6>(0, 6);
    const Eigen::MatrixXd G13 = corrected_AtA.template block<6, 1>(0, 12);
    const Eigen::MatrixXd G14 = corrected_AtA.template block<6, 1>(0, 13);

    const Eigen::MatrixXd G22 = corrected_AtA.template block<6, 6>(6, 6);
    const Eigen::MatrixXd G23 = corrected_AtA.template block<6, 1>(6, 12);
    const Eigen::MatrixXd G24 = corrected_AtA.template block<6, 1>(6, 13);

    const Eigen::MatrixXd G33 = corrected_AtA.template block<1, 1>(12, 12);
    const Eigen::MatrixXd G34 = corrected_AtA.template block<1, 1>(12, 13);

    const Eigen::MatrixXd G44 = corrected_AtA.template block<1, 1>(13, 13);

    Gs.push_back(G11);
    Gs.push_back(G12);
    Gs.push_back(G13);
    Gs.push_back(G14);

    Gs.push_back(G22);
    Gs.push_back(G23);
    Gs.push_back(G24);

    Gs.push_back(G33);
    Gs.push_back(G34);

    Gs.push_back(G44);

    /*
    * Jtr composition
    *
    * p0 [ g1 ]
    * p1 [ g2 ]
    * s0 [ g3 ]
    * s1 [ g4 ]
    */
    const Eigen::MatrixXd g1 = Atb_.template block<6, 1>(0, 0);
    const Eigen::MatrixXd g2 = Atb_.template block<6, 1>(6, 0);
    const Eigen::MatrixXd g3 = Atb_.template block<1, 1>(12, 0);
    const Eigen::MatrixXd g4 = Atb_.template block<1, 1>(13, 0);

    gs.push_back(g1);
    gs.push_back(g2);
    gs.push_back(g3);
    gs.push_back(g4);

    VLOG(3) << "-----------------------------------";
    VLOG(3) << "[LoopMGFactor<Scalar>::linearize] Asking to linearize " << Name() << " at values:";
    VLOG(3) << "[LoopMGFactor<Scalar>::linearize] pose0: " << p0.log().transpose();
    VLOG(3) << "[LoopMGFactor<Scalar>::linearize] pose1: " << p1.log().transpose();
    VLOG(3) << "[LoopMGFactor<Scalar>::linearize] scale0: " << s0;
    VLOG(3) << "[LoopMGFactor<Scalar>::linearize] scale1: " << s1;
    VLOG(3) << "[LoopMGFactor<Scalar>::linearize] error between " << kf0_->id << " " << kf1_->id << " : " << error_;
    VLOG(3) << "-----------------------------------";

    // create and return HessianFactor
    return boost::make_shared<gtsam::HessianFactor>(keys, Gs, gs, (double)error_);
  }

  /* ************************************************************************* */
  template <typename Scalar>
  inline Scalar LoopMGFactor<Scalar>::ComputeError(const PoseT &pose0,
                                                   const PoseT &pose1,
                                                   const Scalar &scale0,
                                                   const Scalar &scale1) const
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;

    PoseT relpose10;
    RelativePose(pose1, pose0, relpose10);
    at::Tensor rotation, translation;
    SophusSE3ToTensor(relpose10, rotation, translation, kf0_->dpt_map_bias.options());

    Scalar cuerror = loop_mg_error_calculate(
        rotation, translation.reshape({-1}),
        matched_unscaled_dpts_0_,
        matched_unscaled_dpts_1_,
        matched_locations_homo_0_,
        matched_locations_homo_1_,
        scale0, scale1,
        loss_param_, factor_weight_);

    return cuerror;
  }

  template <typename Scalar>
  inline void LoopMGFactor<Scalar>::ComputeJacobianAndError(const PoseT &pose0,
                                                            const PoseT &pose1,
                                                            const Scalar &scale0,
                                                            const Scalar &scale1) const
  {
    at::Tensor rotation10, translation10;
    at::Tensor rotation0, translation0;
    at::Tensor rotation1, translation1;
    SophusSE3ToTensor(pose0, rotation0, translation0, kf0_->dpt_map_bias.options());
    SophusSE3ToTensor(pose1, rotation1, translation1, kf0_->dpt_map_bias.options());
    rotation10 = torch::matmul(rotation1.permute({1, 0}), rotation0);
    translation10 = torch::matmul(rotation1.permute({1, 0}), translation0 - translation1);

    at::Tensor cuAtA, cuAtb;
    float cuerror;

    tic("[LoopMGFactor<Scalar>::ComputeJacobianAndError] jac " + std::to_string(kf0_->id) + " " + std::to_string(kf1_->id));
    loop_mg_jac_error_calculate(cuAtA, cuAtb, cuerror,
                                rotation10, translation10.reshape({-1}),
                                rotation0, translation0.reshape({-1}),
                                rotation1, translation1.reshape({-1}),
                                matched_unscaled_dpts_0_,
                                matched_unscaled_dpts_1_,
                                matched_locations_homo_0_,
                                matched_locations_homo_1_,
                                scale0, scale1,
                                loss_param_, factor_weight_);

    // Pass the computed values to class variables
    error_ = cuerror;
    TensorToEigenMatrix(cuAtA.to(torch::kDouble), AtA_);
    TensorToEigenMatrix(cuAtb.to(torch::kDouble), Atb_);
    toc("[LoopMGFactor<Scalar>::ComputeJacobianAndError] jac " + std::to_string(kf0_->id) + " " + std::to_string(kf1_->id));

    return;
  }

  /* ************************************************************************* */
  template <typename Scalar>
  std::string LoopMGFactor<Scalar>::Name() const
  {
    std::stringstream ss;
    auto fmt = gtsam::DefaultKeyFormatter;
    ss << "LoopMGFactor " << fmt(pose0_key_) << " -> " << fmt(pose1_key_);
    return ss.str();
  }

  /* ************************************************************************* */
  // explicit instantiation
  template class LoopMGFactor<float>;

}