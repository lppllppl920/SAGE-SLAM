#include "geometric_factor.h"

namespace df
{

  template <typename Scalar, int CS>
  GeometricFactor<Scalar, CS>::GeometricFactor(const PinholeCamera<Scalar> &cam,
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
                                               const Scalar &dpt_eps)
      : Base(gtsam::cref_list_of<6>(pose0_key)(pose1_key)(code0_key)(code1_key)(scale0_key)(scale1_key)),
        pose0_key_(pose0_key),
        pose1_key_(pose1_key),
        code0_key_(code0_key),
        code1_key_(code1_key),
        scale0_key_(scale0_key),
        scale1_key_(scale1_key),
        cam_(cam),
        kf0_(kf0), kf1_(kf1),
        factor_weight_(factor_weight), loss_param_(loss_param),
        dpt_eps_(dpt_eps), error_(0.0)
  {
    AtA_.setZero();
    Atb_.setZero();
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  GeometricFactor<Scalar, CS>::~GeometricFactor() {}

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  double GeometricFactor<Scalar, CS>::error(const gtsam::Values &c) const
  {
    torch::NoGradGuard no_grad;
    if (this->active(c))
    {
      // get values of the optimization variables
      PoseT p0 = c.at<PoseT>(pose0_key_);
      PoseT p1 = c.at<PoseT>(pose1_key_);
      CodeT c0 = c.at<CodeT>(code0_key_);
      CodeT c1 = c.at<CodeT>(code1_key_);
      Scalar s0 = c.at<Scalar>(scale0_key_);
      Scalar s1 = c.at<Scalar>(scale1_key_);

      Scalar error = ComputeError(p0, p1, c0, c1, s0, s1);

      return error;
    }
    else
    {
      return 0.0;
    }
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  boost::shared_ptr<gtsam::GaussianFactor>
  GeometricFactor<Scalar, CS>::linearize(const gtsam::Values &c) const
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
    CodeT c0 = c.at<CodeT>(code0_key_);
    CodeT c1 = c.at<CodeT>(code1_key_);
    Scalar s0 = c.at<Scalar>(scale0_key_);
    Scalar s1 = c.at<Scalar>(scale1_key_);

    ComputeJacobianAndError(p0, p1, c0, c1, s0, s1);

    // Eigen::Matrix<double, 14 + 2 * CS, 14 + 2 * CS> oriAtA;
    // Eigen::Matrix<double, 14 + 2 * CS, 1> oriAtb;
    // double orierror;
    // oriAtA = AtA_;
    // oriAtb = Atb_;
    // orierror = error_;

    // Scalar eps = 1.0e-6;

    // PoseT modified_p0 = p0;
    // modified_p0.translation() = modified_p0.translation().array() + eps;
    // ComputeJacobianAndError(modified_p0, p1, c0, c1, s0, s1);
    // // (err(x+\deltax) - err(x)) = \deltax^T * cuAtA * \deltax + 2 * cuAtb^T * \deltax
    // double d_err_numeric = (error_ - orierror);
    // Eigen::Matrix<double, 14 + 2 * CS, 1> delta_x = Eigen::Matrix<double, 14 + 2 * CS, 1>::Zero();
    // delta_x(0, 0) = eps;
    // delta_x(1, 0) = eps;
    // delta_x(2, 0) = eps;
    // // delta_x(13 + 2 * CS, 0) = eps;
    // auto d_err_analytic = delta_x.transpose() * oriAtA * delta_x - 2.0 * oriAtb.transpose() * delta_x;
    // LOG(INFO) << "[GeometricFactor<Scalar, CS>::linearize] d_err numeric 1: " << d_err_numeric << " d_err analytic: " << d_err_analytic;

    // PoseT modified_p1 = p1;
    // modified_p0.translation() = modified_p0.translation().array() + eps;
    // ComputeJacobianAndError(p0, modified_p1, c0, c1, s0, s1);
    // // (err(x+\deltax) - err(x)) = \deltax^T * cuAtA * \deltax + 2 * cuAtb^T * \deltax
    // auto d_err_numeric2 = (error_ - orierror);
    // delta_x.setZero();
    // delta_x(6, 0) = eps;
    // delta_x(7, 0) = eps;
    // delta_x(8, 0) = eps;
    // // delta_x(12 + 2 * CS, 0) = eps;
    // auto d_err_analytic2 = delta_x.transpose() * oriAtA * delta_x - 2.0 * oriAtb.transpose() * delta_x;
    // LOG(INFO) << "[GeometricFactor<Scalar, CS>::linearize] d_err numeric 2: " << d_err_numeric2 << " d_err analytic: " << d_err_analytic2;

    // Only dynamic matrix can have thinV option in SVD so we first convert it to dynamic one
    Eigen::MatrixXd M = AtA_.template cast<double>();
    Eigen::MatrixXd corrected_AtA = NearestPsd(M);

    // WARNING: in poseT, trans is ahead of rot (3 + 3) !!
    // need to partition the mats here into separate ones
    // const gtsam::FastVector<gtsam::Key> keys = {pose0_key_, pose1_key_, code0_key_, scale0_key_};
    const gtsam::FastVector<gtsam::Key> keys = {pose0_key_, pose1_key_, code0_key_, code1_key_, scale0_key_, scale1_key_};

    std::vector<gtsam::Matrix> Gs;
    std::vector<gtsam::Vector> gs;

    //  * Hessian composition
    //  *
    //  *      p0   p1   c0   c1   s0   s1
    //  * p0 [ G11  G12  G13  G14  G15  G16 ]
    //  * p1 [      G22  G23  G24  G25  G26 ]
    //  * c0 [           G33  G34  G35  G36 ]
    //  * c1 [                G44  G45  G46 ]
    //  * s0 [                     G55  G56 ]
    //  * s1 [                          G66 ]

    const Eigen::MatrixXd G11 = corrected_AtA.template block<6, 6>(0, 0);
    const Eigen::MatrixXd G12 = corrected_AtA.template block<6, 6>(0, 6);
    const Eigen::MatrixXd G13 = corrected_AtA.template block<6, CS>(0, 12);
    const Eigen::MatrixXd G14 = corrected_AtA.template block<6, CS>(0, 12 + CS);
    const Eigen::MatrixXd G15 = corrected_AtA.template block<6, 1>(0, 12 + 2 * CS);
    const Eigen::MatrixXd G16 = corrected_AtA.template block<6, 1>(0, 13 + 2 * CS);

    const Eigen::MatrixXd G22 = corrected_AtA.template block<6, 6>(6, 6);
    const Eigen::MatrixXd G23 = corrected_AtA.template block<6, CS>(6, 12);
    const Eigen::MatrixXd G24 = corrected_AtA.template block<6, CS>(6, 12 + CS);
    const Eigen::MatrixXd G25 = corrected_AtA.template block<6, 1>(6, 12 + 2 * CS);
    const Eigen::MatrixXd G26 = corrected_AtA.template block<6, 1>(6, 13 + 2 * CS);

    const Eigen::MatrixXd G33 = corrected_AtA.template block<CS, CS>(12, 12);
    const Eigen::MatrixXd G34 = corrected_AtA.template block<CS, CS>(12, 12 + CS);
    const Eigen::MatrixXd G35 = corrected_AtA.template block<CS, 1>(12, 12 + 2 * CS);
    const Eigen::MatrixXd G36 = corrected_AtA.template block<CS, 1>(12, 13 + 2 * CS);

    const Eigen::MatrixXd G44 = corrected_AtA.template block<CS, CS>(12 + CS, 12 + CS);
    const Eigen::MatrixXd G45 = corrected_AtA.template block<CS, 1>(12 + CS, 12 + 2 * CS);
    const Eigen::MatrixXd G46 = corrected_AtA.template block<CS, 1>(12 + CS, 13 + 2 * CS);

    const Eigen::MatrixXd G55 = corrected_AtA.template block<1, 1>(12 + 2 * CS, 12 + 2 * CS);
    const Eigen::MatrixXd G56 = corrected_AtA.template block<1, 1>(12 + 2 * CS, 13 + 2 * CS);

    const Eigen::MatrixXd G66 = corrected_AtA.template block<1, 1>(13 + 2 * CS, 13 + 2 * CS);

    Gs.push_back(G11);
    Gs.push_back(G12);
    Gs.push_back(G13);
    Gs.push_back(G14);
    Gs.push_back(G15);
    Gs.push_back(G16);

    Gs.push_back(G22);
    Gs.push_back(G23);
    Gs.push_back(G24);
    Gs.push_back(G25);
    Gs.push_back(G26);

    Gs.push_back(G33);
    Gs.push_back(G34);
    Gs.push_back(G35);
    Gs.push_back(G36);

    Gs.push_back(G44);
    Gs.push_back(G45);
    Gs.push_back(G46);

    Gs.push_back(G55);
    Gs.push_back(G56);

    Gs.push_back(G66);
    /*
    * Jtr composition
    *
    * p0 [ g1 ]
    * p1 [ g2 ]
    * c0 [ g3 ]
    * c1 [ g4 ]
    * s0 [ g5 ]
    * s1 [ g6 ]
    */
    const Eigen::MatrixXd g1 = Atb_.template block<6, 1>(0, 0);
    const Eigen::MatrixXd g2 = Atb_.template block<6, 1>(6, 0);
    const Eigen::MatrixXd g3 = Atb_.template block<CS, 1>(12, 0);
    const Eigen::MatrixXd g4 = Atb_.template block<CS, 1>(12 + CS, 0);
    const Eigen::MatrixXd g5 = Atb_.template block<1, 1>(12 + 2 * CS, 0);
    const Eigen::MatrixXd g6 = Atb_.template block<1, 1>(13 + 2 * CS, 0);

    gs.push_back(g1);
    gs.push_back(g2);
    gs.push_back(g3);
    gs.push_back(g4);
    gs.push_back(g5);
    gs.push_back(g6);

    VLOG(3) << "-----------------------------------";
    VLOG(3) << "[GeometricFactor<Scalar, CS>::linearize] Asking to linearize " << Name() << " at values:";
    VLOG(3) << "[GeometricFactor<Scalar, CS>::linearize] pose0: " << p0.log().transpose();
    VLOG(3) << "[GeometricFactor<Scalar, CS>::linearize] pose1: " << p1.log().transpose();
    VLOG(3) << "[GeometricFactor<Scalar, CS>::linearize] code0: " << c0.transpose();
    VLOG(3) << "[GeometricFactor<Scalar, CS>::linearize] code1: " << c1.transpose();
    VLOG(3) << "[GeometricFactor<Scalar, CS>::linearize] scale0: " << s0;
    VLOG(3) << "[GeometricFactor<Scalar, CS>::linearize] scale1: " << s1;
    VLOG(3) << "[GeometricFactor<Scalar, CS>::linearize] error between " << kf0_->id << " " << kf1_->id << " : " << error_;
    VLOG(3) << "-----------------------------------";

    // create and return HessianFactor
    return boost::make_shared<gtsam::HessianFactor>(keys, Gs, gs, (double)error_);
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  std::string GeometricFactor<Scalar, CS>::Name() const
  {
    std::stringstream ss;
    auto fmt = gtsam::DefaultKeyFormatter;
    ss << "GeometricFactor " << kf0_->id << " -> " << kf1_->id << " keys = {"
       << fmt(pose0_key_) << ", " << fmt(pose1_key_) << ", " << fmt(code0_key_) << ", " << fmt(code1_key_) << ", "
       << fmt(scale0_key_) << ", " << fmt(scale1_key_)
       << "}";
    return ss.str();
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  inline Scalar GeometricFactor<Scalar, CS>::ComputeError(const PoseT &pose0,
                                                          const PoseT &pose1,
                                                          const CodeT &code0,
                                                          const CodeT &code1,
                                                          const Scalar &scale0,
                                                          const Scalar &scale1) const
  {
    torch::NoGradGuard no_grad;
    PoseT relpose;
    at::Tensor rotation, translation;

    RelativePose(pose1, pose0, relpose);
    SophusSE3ToTensor(relpose, rotation, translation, kf0_->dpt_map_bias.options());
    translation = translation.reshape({-1});

    const long height = cam_.height();
    const long width = cam_.width();

    // C_code
    const at::Tensor code_tensor0 = EigenVectorToTensor(code0, kf0_->dpt_map_bias.options());
    const at::Tensor code_tensor1 = EigenVectorToTensor(code1, kf1_->dpt_map_bias.options());
    // H x W
    const at::Tensor unscaled_dpt_map_1 = kf1_->dpt_map_bias.reshape({height, width}) +
                                          torch::matmul(kf1_->dpt_jac_code, code_tensor1).reshape({height, width});

    float error =
        geometric_error_calculate<CS>(rotation, translation,
                                      kf0_->dpt_map_bias.reshape({-1}), kf0_->dpt_jac_code,
                                      code_tensor0, unscaled_dpt_map_1 * scale1,
                                      *(kf1_->video_mask_ptr),
                                      kf0_->sampled_locations_1d.to(torch::kInt32),
                                      kf0_->sampled_locations_homo,
                                      scale0, cam_, dpt_eps_, loss_param_,
                                      factor_weight_);

    // float error =
    //     geometric_error_calculate_unbiased<CS>(rotation, translation,
    //                                            kf0_->dpt_map_bias.reshape({-1}), kf0_->dpt_jac_code,
    //                                            code_tensor0, unscaled_dpt_map_1,
    //                                            *(kf1_->video_mask_ptr),
    //                                            kf0_->sampled_locations_1d.to(torch::kInt32),
    //                                            kf0_->sampled_locations_homo,
    //                                            scale0, scale1, cam_, dpt_eps_, loss_param_,
    //                                            factor_weight_);

    return error;
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  inline void GeometricFactor<Scalar, CS>::ComputeJacobianAndError(const PoseT &pose0,
                                                                   const PoseT &pose1,
                                                                   const CodeT &code0,
                                                                   const CodeT &code1,
                                                                   const Scalar &scale0,
                                                                   const Scalar &scale1) const
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    namespace F = torch::nn::functional;

    const long height = cam_.height();
    const long width = cam_.width();

    const at::Tensor code_tensor0 = EigenVectorToTensor(code0, kf0_->dpt_map_bias.options());
    const at::Tensor code_tensor1 = EigenVectorToTensor(code1, kf1_->dpt_map_bias.options());

    const at::Tensor unscaled_dpt_map_1 = kf1_->dpt_map_bias.reshape({height, width}) +
                                          torch::matmul(kf1_->dpt_jac_code, code_tensor1).reshape({height, width});
    at::Tensor unscaled_dpt_map_grad_1;
    ComputeSpatialGrad(unscaled_dpt_map_1.reshape({1, 1, height, width}), unscaled_dpt_map_grad_1);

    at::Tensor rotation10, translation10;
    at::Tensor rotation0, translation0;
    at::Tensor rotation1, translation1;
    SophusSE3ToTensor(pose0, rotation0, translation0, kf0_->dpt_map_bias.options());
    SophusSE3ToTensor(pose1, rotation1, translation1, kf0_->dpt_map_bias.options());
    rotation10 = torch::matmul(rotation1.permute({1, 0}), rotation0);
    translation10 = torch::matmul(rotation1.permute({1, 0}), translation0 - translation1);

    at::Tensor cuAtA, cuAtb;
    float cuerror;

    tic("[GeometricFactor<Scalar, CS>::ComputeJacobianAndError] jac " + std::to_string(kf0_->id) + " " + std::to_string(kf1_->id));

    geometric_jac_error_calculate<CS>(cuAtA, cuAtb, cuerror,
                                      rotation10, translation10.reshape({-1}),
                                      rotation0, translation0.reshape({-1}),
                                      rotation1, translation1.reshape({-1}),
                                      kf0_->dpt_map_bias.reshape({-1}), kf0_->dpt_jac_code,
                                      code_tensor0, scale1 * unscaled_dpt_map_1.reshape({height, width}),
                                      scale1 * unscaled_dpt_map_grad_1.reshape({2, height, width}),
                                      kf1_->dpt_jac_code.reshape({height, width, CS}),
                                      *(kf1_->video_mask_ptr),
                                      kf0_->sampled_locations_1d.to(torch::kInt32),
                                      kf0_->sampled_locations_homo,
                                      scale0, scale1, cam_, dpt_eps_,
                                      loss_param_, factor_weight_);

    toc("[GeometricFactor<Scalar, CS>::ComputeJacobianAndError] jac " + std::to_string(kf0_->id) + " " + std::to_string(kf1_->id));

    // Pass the computed values to class variables
    error_ = cuerror;
    TensorToEigenMatrix(cuAtA.to(torch::kDouble), AtA_);
    TensorToEigenMatrix(cuAtb.to(torch::kDouble), Atb_);
    return;
  }

  /* ***********************************************DF_CODE_SIZE************************** */
  // explicit instantiation
  template class GeometricFactor<float, DF_CODE_SIZE>;

} // namespace df
