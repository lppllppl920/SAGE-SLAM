#include <cuda.h>

#include "photometric_factor.h"

namespace df
{

  template <typename Scalar, int CS>
  PhotometricFactor<Scalar, CS>::PhotometricFactor(
      const CameraPyramid<Scalar> &camera_pyramid,
      const KeyframePtr &kf,
      const FramePtr &fr,
      const gtsam::Key &pose0_key,
      const gtsam::Key &pose1_key,
      const gtsam::Key &code0_key,
      const gtsam::Key &scale0_key,
      const std::vector<Scalar> &factor_weights,
      const Scalar &dpt_eps,
      const bool display_stats)
      : Base(gtsam::cref_list_of<4>(pose0_key)(pose1_key)(code0_key)(scale0_key)),
        pose0_key_(pose0_key),
        pose1_key_(pose1_key),
        code0_key_(code0_key),
        scale0_key_(scale0_key),
        camera_pyramid_(camera_pyramid),
        kf_(kf), fr_(fr),
        dpt_eps_(dpt_eps),
        display_stats_(display_stats),
        error_(0.0)
  {
    factor_weights_ = torch::from_blob((void *)factor_weights.data(), {static_cast<long>(factor_weights.size())},
                                       torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    AtA_.setZero();
    Atb_.setZero();
  }

  template <typename Scalar, int CS>
  PhotometricFactor<Scalar, CS>::PhotometricFactor(
      const CameraPyramid<Scalar> &camera_pyramid,
      const KeyframePtr &kf,
      const FramePtr &fr,
      const gtsam::Key &pose0_key,
      const gtsam::Key &pose1_key,
      const gtsam::Key &code0_key,
      const gtsam::Key &scale0_key,
      const at::Tensor factor_weights,
      const Scalar &dpt_eps,
      const bool display_stats)
      : Base(gtsam::cref_list_of<4>(pose0_key)(pose1_key)(code0_key)(scale0_key)),
        pose0_key_(pose0_key),
        pose1_key_(pose1_key),
        code0_key_(code0_key),
        scale0_key_(scale0_key),
        camera_pyramid_(camera_pyramid),
        kf_(kf), fr_(fr),
        dpt_eps_(dpt_eps),
        display_stats_(display_stats),
        error_(0.0)
  {
    AtA_.setZero();
    Atb_.setZero();
    factor_weights_ = factor_weights;
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  PhotometricFactor<Scalar, CS>::~PhotometricFactor() {}

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  double PhotometricFactor<Scalar, CS>::error(const gtsam::Values &c) const
  {
    torch::NoGradGuard no_grad;
    if (this->active(c))
    {
      // get values of the optimization variables
      PoseT p0 = c.at<PoseT>(pose0_key_);
      PoseT p1 = c.at<PoseT>(pose1_key_);
      CodeT c0 = c.at<CodeT>(code0_key_);
      Scalar s0 = c.at<Scalar>(scale0_key_);

      // compute photometric error with updated data
      Scalar error = ComputeError(p0, p1, c0, s0);

      VLOG(3) << "-----------------------------------";
      VLOG(3) << "[PhotometricFactor<Scalar, CS>::error] Asking for error " << Name() << " at values:";
      VLOG(3) << "[PhotometricFactor<Scalar, CS>::error] pose0: " << p0.log().transpose();
      VLOG(3) << "[PhotometricFactor<Scalar, CS>::error] pose1: " << p1.log().transpose();
      VLOG(3) << "[PhotometricFactor<Scalar, CS>::error] code0: " << c0.transpose();
      VLOG(3) << "[PhotometricFactor<Scalar, CS>::error] scale0: " << s0;
      VLOG(3) << "[PhotometricFactor<Scalar, CS>::error] error between " << kf_->id << " " << fr_->id << " : " << error_;
      VLOG(3) << "-----------------------------------";

      return (double)error;
    }
    else
    {
      return 0.0;
    }
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  boost::shared_ptr<gtsam::GaussianFactor>
  PhotometricFactor<Scalar, CS>::linearize(const gtsam::Values &c) const
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
    Scalar s0 = c.at<Scalar>(scale0_key_);

    ComputeJacobianAndError(p0, p1, c0, s0);


    // Eigen::Matrix<double, 13 + CS, 13 + CS> oriAtA;
    // Eigen::Matrix<double, 13 + CS, 1> oriAtb;
    // double orierror;
    // oriAtA = AtA_;
    // oriAtb = Atb_;
    // orierror = error_;

    // Scalar eps = 1.0e-5;
    // PoseT modified_p0 = p0;
    // modified_p0.translation() = modified_p0.translation().array() + eps;
    // ComputeJacobianAndError(modified_p0, p1, c0, s0);
    // // ComputeJacobianAndError(p0, p1, c0, c1, s0 + eps, s1);
    // // (err(x+\deltax) - err(x)) = \deltax^T * cuAtA * \deltax + 2 * cuAtb^T * \deltax
    // double d_err_numeric = (error_ - orierror);
    // Eigen::Matrix<double, 13 + CS, 1> delta_x = Eigen::Matrix<double, 13 + CS, 1>::Zero();
    // delta_x(0, 0) = eps;
    // delta_x(1, 0) = eps;
    // delta_x(2, 0) = eps;
    // auto d_err_analytic = delta_x.transpose() * oriAtA * delta_x - 2.0 * oriAtb.transpose() * delta_x;
    // LOG(INFO) << "[PhotometricFactor<Scalar, CS>::linearize] d_err numeric: " << d_err_numeric << " d_err analytic: " << d_err_analytic;


    Eigen::MatrixXd M = AtA_.template cast<double>();
    Eigen::MatrixXd corrected_AtA = NearestPsd(M);

    // WARNING: in poseT, trans is ahead of rot (3 + 3) !! (XT)
    // need to partition the mats here into separate ones
    const gtsam::FastVector<gtsam::Key> keys = {pose0_key_, pose1_key_, code0_key_, scale0_key_};

    std::vector<gtsam::Matrix> Gs;
    std::vector<gtsam::Vector> gs;

    //
    //  * Hessian composition
    //  *
    //  *      p0   p1   c0   s0
    //  * p0 [ G11  G12  G13  G14 ]
    //  * p1 [      G22  G23  G24 ]
    //  * c0 [           G33  G34 ]
    //  * s0 [                G44 ]
    //  */
    const Eigen::MatrixXd G11 = corrected_AtA.template block<6, 6>(0, 0);
    const Eigen::MatrixXd G12 = corrected_AtA.template block<6, 6>(0, 6);
    const Eigen::MatrixXd G13 = corrected_AtA.template block<6, CS>(0, 12);
    const Eigen::MatrixXd G14 = corrected_AtA.template block<6, 1>(0, 12 + CS);
    const Eigen::MatrixXd G22 = corrected_AtA.template block<6, 6>(6, 6);
    const Eigen::MatrixXd G23 = corrected_AtA.template block<6, CS>(6, 12);
    const Eigen::MatrixXd G24 = corrected_AtA.template block<6, 1>(6, 12 + CS);
    const Eigen::MatrixXd G33 = corrected_AtA.template block<CS, CS>(12, 12);
    const Eigen::MatrixXd G34 = corrected_AtA.template block<CS, 1>(12, 12 + CS);
    const Eigen::MatrixXd G44 = corrected_AtA.template block<1, 1>(12 + CS, 12 + CS);

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
    * c0 [ g3 ]
    * s0 [ g4 ]
    */
    const Eigen::MatrixXd g1 = Atb_.template block<6, 1>(0, 0);
    const Eigen::MatrixXd g2 = Atb_.template block<6, 1>(6, 0);
    const Eigen::MatrixXd g3 = Atb_.template block<CS, 1>(12, 0);
    const Eigen::MatrixXd g4 = Atb_.template block<1, 1>(12 + CS, 0);

    gs.push_back(g1);
    gs.push_back(g2);
    gs.push_back(g3);
    gs.push_back(g4);

    VLOG(3) << "-----------------------------------";
    VLOG(3) << "[PhotometricFactor<Scalar, CS>::linearize] Asking to linearize " << Name() << " at values:";
    VLOG(3) << "[PhotometricFactor<Scalar, CS>::linearize] pose0: " << p0.log().transpose();
    VLOG(3) << "[PhotometricFactor<Scalar, CS>::linearize] pose1: " << p1.log().transpose();
    VLOG(3) << "[PhotometricFactor<Scalar, CS>::linearize] code0: " << c0.transpose();
    VLOG(3) << "[PhotometricFactor<Scalar, CS>::linearize] scale0: " << s0;
    VLOG(3) << "[PhotometricFactor<Scalar, CS>::linearize] error between " << kf_->id << " " << fr_->id << " : " << error_;
    VLOG(3) << "-----------------------------------";

    // create and return HessianFactor
    return boost::make_shared<gtsam::HessianFactor>(keys, Gs, gs, (double)error_);
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  std::string PhotometricFactor<Scalar, CS>::Name() const
  {
    std::stringstream ss;
    auto fmt = gtsam::DefaultKeyFormatter;
    ss << "PhotometricFactor " << fmt(pose0_key_) << " -> " << fmt(pose1_key_);
    return ss.str();
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  inline Scalar PhotometricFactor<Scalar, CS>::ComputeError(const PoseT &pose0,
                                                            const PoseT &pose1,
                                                            const CodeT &code0,
                                                            const Scalar &scale0) const
  {
    torch::NoGradGuard no_grad;
    namespace F = torch::nn::functional;
    using namespace torch::indexing;

    const at::Tensor code_tensor0 = EigenVectorToTensor(code0, kf_->dpt_map_bias.options());

    PoseT relpose;
    RelativePose(pose1, pose0, relpose);
    at::Tensor rotation, translation;
    SophusSE3ToTensor(relpose, rotation, translation, kf_->dpt_map_bias.options());
    rotation = rotation.to(kf_->dpt_map.device());
    translation = translation.to(kf_->dpt_map.device());

    Scalar cuerror = photometric_error_calculate<DF_FEAT_SIZE>(
        rotation, translation.reshape({-1}),
        kf_->dpt_map_bias.reshape({-1}), kf_->dpt_jac_code,
        code_tensor0, *(fr_->video_mask_ptr),
        kf_->sampled_locations_1d, kf_->sampled_locations_homo,
        kf_->feat_map_pyramid, fr_->feat_map_pyramid,
        *(fr_->level_offsets_ptr),
        scale0, *(fr_->camera_pyramid_ptr), dpt_eps_, factor_weights_);

    return cuerror;
  }

  template <typename Scalar, int CS>
  inline void PhotometricFactor<Scalar, CS>::ComputeJacobianAndError(const PoseT &pose0,
                                                                     const PoseT &pose1,
                                                                     const CodeT &code0,
                                                                     const Scalar &scale0) const
  {
    torch::NoGradGuard no_grad;
    namespace F = torch::nn::functional;
    using namespace torch::indexing;
    // C_code
    const at::Tensor code_tensor0 = EigenVectorToTensor(code0, kf_->dpt_map_bias.options());

    at::Tensor rotation10, translation10;
    at::Tensor rotation0, translation0;
    at::Tensor rotation1, translation1;
    SophusSE3ToTensor(pose0, rotation0, translation0, kf_->dpt_map_bias.options());
    SophusSE3ToTensor(pose1, rotation1, translation1, kf_->dpt_map_bias.options());
    rotation10 = torch::matmul(rotation1.permute({1, 0}), rotation0);
    translation10 = torch::matmul(rotation1.permute({1, 0}), translation0 - translation1);

    at::Tensor cuAtA, cuAtb;
    float cuerror;

    tic("[PhotometricFactor<Scalar, CS>::ComputeJacobianAndError] jac " + std::to_string(kf_->id) + " " + std::to_string(fr_->id));

    photometric_jac_error_calculate<CS, DF_FEAT_SIZE>(
        cuAtA, cuAtb, cuerror,
        rotation10, translation10.reshape({-1}),
        rotation0, translation0.reshape({-1}),
        rotation1, translation1.reshape({-1}),
        kf_->dpt_map_bias.reshape({-1}), kf_->dpt_jac_code,
        code_tensor0,
        *(fr_->video_mask_ptr),
        kf_->sampled_locations_1d,
        kf_->sampled_locations_homo,
        kf_->feat_map_pyramid, fr_->feat_map_pyramid,
        fr_->feat_map_grad_pyramid,
        *(fr_->level_offsets_ptr),
        scale0, *(fr_->camera_pyramid_ptr),
        dpt_eps_, factor_weights_);

    error_ = cuerror;
    TensorToEigenMatrix(cuAtA.to(torch::kDouble), AtA_);
    TensorToEigenMatrix(cuAtb.to(torch::kDouble), Atb_);

    toc("[PhotometricFactor<Scalar, CS>::ComputeJacobianAndError] jac " + std::to_string(kf_->id) + " " + std::to_string(fr_->id));

    return;
  }

  /* ************************************************************************* */
  // explicit instantiation
  template class PhotometricFactor<float, DF_CODE_SIZE>;

} // namespace df
