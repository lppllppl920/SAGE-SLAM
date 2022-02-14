#include "match_geometry_factor.h"

namespace df
{

  template <typename Scalar, int CS>
  MatchGeometryFactor<Scalar, CS>::MatchGeometryFactor(
      const PinholeCamera<Scalar> &cam,
      const KeyframePtr &kf0,
      const KeyframePtr &kf1,
      const gtsam::Key &pose0_key,
      const gtsam::Key &pose1_key,
      const gtsam::Key &code0_key,
      const gtsam::Key &code1_key,
      const gtsam::Key &scale0_key,
      const gtsam::Key &scale1_key,
      const long &num_keypoints,
      const Scalar &cyc_consis_thresh,
      const Scalar &factor_weight,
      const Scalar &loss_param,
      const Scalar &dpt_eps, // for the purpose of displaying 2d matches
      const std::string robust_loss_type,
      const teaser::RobustRegistrationSolver::Params &teaser_params,
      const bool display_stats)
      : Base(gtsam::cref_list_of<6>(pose0_key)(pose1_key)(code0_key)(code1_key)(scale0_key)(scale1_key)),
        pose0_key_(pose0_key),
        pose1_key_(pose1_key),
        code0_key_(code0_key),
        code1_key_(code1_key),
        scale0_key_(scale0_key),
        scale1_key_(scale1_key),
        cam_(cam), kf0_(kf0), kf1_(kf1),
        cyc_consis_thresh_(cyc_consis_thresh), factor_weight_(factor_weight), loss_param_(loss_param),
        dpt_eps_(dpt_eps), display_stats_(display_stats),
        robust_loss_type_(robust_loss_type), teaser_params_(teaser_params), error_(0.0)
  {
    torch::NoGradGuard no_grad;
    // Generate a fixed set of random selected within-mask locations as the keypoint location for the keyframe 0, keep it within this reprojection factor
    // this may allow us to use a different set of locations for different pairs that involve the same keyframe (robustness?).
    using namespace torch::indexing;
    const at::Tensor valid_locations_1d = kf0_->valid_locations_1d;
    long num_points = valid_locations_1d.size(0);
    long width = cam.width();
    long height = cam.height();

    num_keypoints_ = (num_points >= num_keypoints) ? num_keypoints : num_points;

    std::vector<long> indices(num_points);
    std::iota(indices.begin(), indices.end(), 0);
    // shuffle and take the first set of indices as the randomly selected ones
    // std::random_device rd;
    const long seed = kf0_->id * kf1_->id;
    std::mt19937 g;
    g.seed(seed);
    std::shuffle(indices.begin(), indices.end(), g);

    keypoint_indexes_ = torch::from_blob(static_cast<long *>(indices.data()),
                                         {num_keypoints_}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
                            .to(valid_locations_1d.device())
                            .clone();

    // K
    const at::Tensor keypoint_locations_1d = valid_locations_1d.index({keypoint_indexes_});
    const at::Tensor keypoint_locations_2d_x = torch::fmod(keypoint_locations_1d, (float)width);
    const at::Tensor keypoint_locations_2d_y = torch::floor(keypoint_locations_1d / (float)width);

    const at::Tensor feature_desc_0 = kf0_->feat_desc;
    const at::Tensor feature_desc_1 = kf1_->feat_desc;
    long channel = feature_desc_0.size(1);

    // C_feat x K
    const at::Tensor keypoint_features_0 = feature_desc_0.reshape({channel, height * width}).index({Slice(), keypoint_locations_1d});
    // K x H*W
    const at::Tensor feature_response_1 = -torch::sum(torch::square(keypoint_features_0.reshape({channel, num_keypoints_, 1}) -
                                                                    feature_desc_1.reshape({channel, 1, height * width})),
                                                      0, false);
    // K
    const at::Tensor raw_matched_locations_1d_1 = std::get<1>(torch::max(feature_response_1, 1, false));
    // C_feat x K
    const at::Tensor raw_matched_features_1 = feature_desc_1.reshape({channel, height * width}).index({Slice(), raw_matched_locations_1d_1});

    // K x H*W
    const at::Tensor feature_response_0 = -torch::sum(torch::square(raw_matched_features_1.reshape({channel, num_keypoints_, 1}) -
                                                                    feature_desc_0.reshape({channel, 1, height * width})),
                                                      0, false);

    // K
    const at::Tensor cyc_matched_locations_1d_0 = std::get<1>(torch::max(feature_response_0, 1, false));
    const at::Tensor cyc_matched_locations_2d_x = torch::fmod(cyc_matched_locations_1d_0, (float)width);
    const at::Tensor cyc_matched_locations_2d_y = torch::floor(cyc_matched_locations_1d_0 / (float)width);

    const at::Tensor cyc_distances_sq = torch::square(keypoint_locations_2d_x - cyc_matched_locations_2d_x) +
                                        torch::square(keypoint_locations_2d_y - cyc_matched_locations_2d_y);

    const at::Tensor inlier_keypoint_indexes = torch::nonzero(cyc_distances_sq <= (cyc_consis_thresh * cyc_consis_thresh)).reshape({-1});

    matched_keypoint_indexes_ = keypoint_indexes_.index({inlier_keypoint_indexes});

    // N x 3
    const at::Tensor valid_locations_homo_0 = kf0_->valid_locations_homo;
    // N
    const at::Tensor valid_locations_1d_0 = kf0_->valid_locations_1d;

    // M x 3
    matched_locations_homo_0_ = valid_locations_homo_0.index({matched_keypoint_indexes_, Slice()});
    // M
    matched_locations_1d_0_ = valid_locations_1d_0.index({matched_keypoint_indexes_}).to(torch::kInt32);
    // M
    matched_locations_1d_1_ = raw_matched_locations_1d_1.index({inlier_keypoint_indexes}).to(torch::kInt32);

    // M x 2
    const at::Tensor matched_locations_2d_1 =
        torch::stack({torch::fmod(matched_locations_1d_1_, (float)width), torch::floor(matched_locations_1d_1_ / (float)width)}, 1);

    // M
    const at::Tensor matched_locations_homo_1_x = (matched_locations_2d_1.index({Slice(), 0}) - cam_.u0()) / cam_.fx();
    const at::Tensor matched_locations_homo_1_y = (matched_locations_2d_1.index({Slice(), 1}) - cam_.v0()) / cam_.fy();
    // M x 3
    matched_locations_homo_1_ = torch::stack({matched_locations_homo_1_x, matched_locations_homo_1_y,
                                              torch::ones_like(matched_locations_homo_1_x)},
                                             1);
    // M x 1
    const at::Tensor matched_dpt_bias_0 = kf0_->dpt_map_bias.reshape({-1, 1}).index({matched_locations_1d_0_.to(torch::kLong), Slice()});
    const at::Tensor matched_dpt_bias_1 = kf1_->dpt_map_bias.reshape({-1, 1}).index({matched_locations_1d_1_.to(torch::kLong), Slice()});

    // M x 3
    const at::Tensor matched_locations_3d_0 = matched_dpt_bias_0 * matched_locations_homo_0_;
    const at::Tensor matched_locations_3d_1 = matched_dpt_bias_1 * matched_locations_homo_1_;

    // 3 x M
    Eigen::Matrix<double, 3, Eigen::Dynamic> eigen_matched_locations_3d_0;
    Eigen::Matrix<double, 3, Eigen::Dynamic> eigen_matched_locations_3d_1;

    eigen_matched_locations_3d_0.resize(Eigen::NoChange, matched_dpt_bias_0.size(0));
    eigen_matched_locations_3d_1.resize(Eigen::NoChange, matched_dpt_bias_1.size(0));

    // 3 x M col major in eigen matrix is the same as M x 3 row major in torch tensor in terms of memory arrangement
    std::memcpy(static_cast<double *>(eigen_matched_locations_3d_0.data()),
                matched_locations_3d_0.to(torch::kCPU).to(torch::kDouble).contiguous().data_ptr(), sizeof(double) * matched_locations_3d_0.numel());
    std::memcpy(static_cast<double *>(eigen_matched_locations_3d_1.data()),
                matched_locations_3d_1.to(torch::kCPU).to(torch::kDouble).contiguous().data_ptr(), sizeof(double) * matched_locations_3d_1.numel());

    // TEASER
    const at::Tensor avg_dpt_bias_1 = torch::mean(matched_dpt_bias_1);
    float avg_dpt = avg_dpt_bias_1.item<float>();
    float focal_length = (cam_.fx() + cam_.fy()) / 2.0;

    // Use dst_point_cloud for noise bound(which is 1 in our case)
    teaser_params_.noise_bound = teaser_params_.noise_bound_multiplier * avg_dpt / focal_length;
    teaser::RobustRegistrationSolver solver(teaser_params_);

    at::Tensor noise_bounds_tensor = teaser_params_.noise_bound_multiplier * matched_dpt_bias_1 / focal_length;
    // TODO: hard-coded minimum noise bound here
    noise_bounds_tensor = torch::clamp_min(noise_bounds_tensor, 5.0e-4);
    Eigen::Matrix<double, 1, Eigen::Dynamic> noise_bounds;
    noise_bounds.resize(Eigen::NoChange, matched_dpt_bias_1.size(0));
    std::memcpy(static_cast<double *>(noise_bounds.data()),
                noise_bounds_tensor.to(torch::kCPU).to(torch::kDouble).contiguous().data_ptr(), sizeof(double) * noise_bounds_tensor.numel());

    // src and dst are 3-by-N Eigen matrices
    // solver.solve(eigen_matched_locations_3d_0, eigen_matched_locations_3d_1);
    solver.solve(eigen_matched_locations_3d_0, eigen_matched_locations_3d_1, noise_bounds);

    std::vector<int> inlier_indexes_in_clique_vec = solver.getTranslationInliers();
    Eigen::Matrix<int, 1, Eigen::Dynamic> clique_indexes_in_ori_vec = solver.getTranslationInliersMap();
    auto translation_inliers_mask = solver.getTranslationInliersMask();

    const at::Tensor inlier_indexes_in_clique = torch::from_blob(static_cast<int *>(inlier_indexes_in_clique_vec.data()),
                                                                 {static_cast<long>(inlier_indexes_in_clique_vec.size())},
                                                                 torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU))
                                                    .to(matched_locations_3d_0.device())
                                                    .to(torch::kLong)
                                                    .clone();

    const at::Tensor clique_indexes_in_ori = torch::from_blob(static_cast<int *>(clique_indexes_in_ori_vec.data()),
                                                              {static_cast<long>(clique_indexes_in_ori_vec.size())},
                                                              torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU))
                                                 .to(matched_locations_3d_0.device())
                                                 .to(torch::kLong)
                                                 .clone();

    const at::Tensor inlier_indexes = clique_indexes_in_ori.index({inlier_indexes_in_clique});

    // RAW_LOG_INFO("[MatchGeometryFactor<Scalar, CS>::MatchGeometryFactor] Teaser inliers between %d and %d : %d / %d",
    //              kf0_->id, kf1_->id, inlier_indexes.size(0), matched_locations_homo_0_.size(0));
    // " << kf0_->id << " and " << kf1_->id
    // inlier_indexes.size(0)
    // matched_locations_homo_0_.size(0)

    matched_locations_homo_0_ = matched_locations_homo_0_.index({inlier_indexes, Slice()});
    matched_locations_homo_1_ = matched_locations_homo_1_.index({inlier_indexes, Slice()});
    matched_locations_1d_0_ = matched_locations_1d_0_.index({inlier_indexes});
    matched_locations_1d_1_ = matched_locations_1d_1_.index({inlier_indexes});

    num_inlier_matches_ = inlier_indexes.size(0);

    desc_inlier_ratio_ = (Scalar)inlier_indexes.size(0) / (Scalar)num_keypoints_;
    inlier_multiplier_ = desc_inlier_ratio_; // 1.0;

    AtA_.setZero();
    Atb_.setZero();
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  MatchGeometryFactor<Scalar, CS>::~MatchGeometryFactor() {}

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  double MatchGeometryFactor<Scalar, CS>::error(const gtsam::Values &c) const
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

      VLOG(3) << "[MatchGeometryFactor<Scalar, CS>::error] error between " << kf0_->id << " " << kf1_->id << " : " << error;

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
  MatchGeometryFactor<Scalar, CS>::linearize(const gtsam::Values &c) const
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

    // Scalar eps = 1.0e-4;

    // PoseT modified_p0 = p0;
    // modified_p0.translation() = modified_p0.translation().array() + eps;
    // ComputeJacobianAndError(modified_p0, p1, c0, c1, s0, s1);
    // // ComputeJacobianAndError(p0, p1, c0, c1, s0 + eps, s1);
    // // (err(x+\deltax) - err(x)) = \deltax^T * cuAtA * \deltax + 2 * cuAtb^T * \deltax
    // double d_err_numeric = (error_ - orierror);
    // Eigen::Matrix<double, 14 + 2 * CS, 1> delta_x = Eigen::Matrix<double, 14 + 2 * CS, 1>::Zero();
    // delta_x(0, 0) = eps;
    // delta_x(1, 0) = eps;
    // delta_x(2, 0) = eps;
    // // delta_x(12 + 2 * CS, 0) = eps;
    // auto d_err_analytic = delta_x.transpose() * oriAtA * delta_x - 2.0 * oriAtb.transpose() * delta_x;
    // LOG(INFO) << "[MatchGeometryFactor<Scalar, CS>::linearize] d_err numeric 1: " << d_err_numeric << " d_err analytic: " << d_err_analytic;

    // PoseT modified_p1 = p1;
    // modified_p0.translation() = modified_p0.translation().array() + eps;
    // ComputeJacobianAndError(p0, modified_p1, c0, c1, s0, s1);
    // // ComputeJacobianAndError(p0, p1, c0, c1, s0, s1 + eps);
    // // (err(x+\deltax) - err(x)) = \deltax^T * cuAtA * \deltax + 2 * cuAtb^T * \deltax
    // auto d_err_numeric2 = (error_ - orierror);
    // delta_x.setZero();
    // delta_x(6, 0) = eps;
    // delta_x(7, 0) = eps;
    // delta_x(8, 0) = eps;
    // // delta_x(13 + 2 * CS, 0) = eps;
    // auto d_err_analytic2 = delta_x.transpose() * oriAtA * delta_x - 2.0 * oriAtb.transpose() * delta_x;
    // LOG(INFO) << "[MatchGeometryFactor<Scalar, CS>::linearize] d_err numeric 2: " << d_err_numeric2 << " d_err analytic: " << d_err_analytic2;

    Eigen::MatrixXd M = AtA_.template cast<double>();
    Eigen::MatrixXd corrected_AtA = NearestPsd(M);

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
    VLOG(3) << "[MatchGeometryFactor<Scalar, CS>::linearize] Asking to linearize " << Name() << " at values:";
    VLOG(3) << "[MatchGeometryFactor<Scalar, CS>::linearize] pose0: " << p0.log().transpose();
    VLOG(3) << "[MatchGeometryFactor<Scalar, CS>::linearize] pose1: " << p1.log().transpose();
    VLOG(3) << "[MatchGeometryFactor<Scalar, CS>::linearize] code0: " << c0.transpose();
    VLOG(3) << "[MatchGeometryFactor<Scalar, CS>::linearize] code1: " << c1.transpose();
    VLOG(3) << "[MatchGeometryFactor<Scalar, CS>::linearize] scale0: " << s0;
    VLOG(3) << "[MatchGeometryFactor<Scalar, CS>::linearize] scale1: " << s1;
    VLOG(3) << "[MatchGeometryFactor<Scalar, CS>::linearize] error between " << kf0_->id << " " << kf1_->id << " : " << error_;
    VLOG(3) << "-----------------------------------";

    // create and return HessianFactor
    return boost::make_shared<gtsam::HessianFactor>(keys, Gs, gs, (double)error_);
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  inline Scalar MatchGeometryFactor<Scalar, CS>::ComputeError(const PoseT &pose0,
                                                              const PoseT &pose1,
                                                              const CodeT &code0,
                                                              const CodeT &code1,
                                                              const Scalar &scale0,
                                                              const Scalar &scale1) const
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;

    PoseT relpose10;
    RelativePose(pose1, pose0, relpose10);
    at::Tensor rotation, translation;
    SophusSE3ToTensor(relpose10, rotation, translation, kf0_->dpt_map_bias.options());

    const at::Tensor code_tensor0 = EigenVectorToTensor(code0, kf0_->dpt_map_bias.options());
    const at::Tensor code_tensor1 = EigenVectorToTensor(code1, kf0_->dpt_map_bias.options());

    Scalar cuerror = match_geometry_error_calculate<CS>(
        rotation, translation.reshape({-1}),
        kf0_->dpt_map_bias.reshape({-1}),
        kf1_->dpt_map_bias.reshape({-1}),
        kf0_->dpt_jac_code,
        kf1_->dpt_jac_code,
        code_tensor0, code_tensor1,
        matched_locations_homo_0_,
        matched_locations_homo_1_,
        matched_locations_1d_0_,
        matched_locations_1d_1_,
        scale0, scale1,
        loss_param_, inlier_multiplier_ * factor_weight_,
        robust_loss_type_);

    return cuerror;
  }

  template <typename Scalar, int CS>
  inline void MatchGeometryFactor<Scalar, CS>::ComputeJacobianAndError(const PoseT &pose0,
                                                                       const PoseT &pose1,
                                                                       const CodeT &code0,
                                                                       const CodeT &code1,
                                                                       const Scalar &scale0,
                                                                       const Scalar &scale1) const
  {
    const at::Tensor code_tensor0 = EigenVectorToTensor(code0, kf0_->dpt_map_bias.options());
    const at::Tensor code_tensor1 = EigenVectorToTensor(code1, kf0_->dpt_map_bias.options());

    at::Tensor rotation10, translation10;
    at::Tensor rotation0, translation0;
    at::Tensor rotation1, translation1;
    SophusSE3ToTensor(pose0, rotation0, translation0, kf0_->dpt_map_bias.options());
    SophusSE3ToTensor(pose1, rotation1, translation1, kf0_->dpt_map_bias.options());
    rotation10 = torch::matmul(rotation1.permute({1, 0}), rotation0);
    translation10 = torch::matmul(rotation1.permute({1, 0}), translation0 - translation1);

    at::Tensor cuAtA, cuAtb;
    float cuerror;

    tic("[MatchGeometryFactor<Scalar, CS>::ComputeJacobianAndError] jac " + std::to_string(kf0_->id) + " " + std::to_string(kf1_->id));

    match_geometry_jac_error_calculate<CS>(cuAtA, cuAtb, cuerror,
                                           rotation10, translation10.reshape({-1}),
                                           rotation0, translation0.reshape({-1}),
                                           rotation1, translation1.reshape({-1}),
                                           kf0_->dpt_map_bias.reshape({-1}),
                                           kf1_->dpt_map_bias.reshape({-1}),
                                           kf0_->dpt_jac_code,
                                           kf1_->dpt_jac_code,
                                           code_tensor0, code_tensor1,
                                           matched_locations_homo_0_,
                                           matched_locations_homo_1_,
                                           matched_locations_1d_0_,
                                           matched_locations_1d_1_,
                                           scale0, scale1,
                                           loss_param_, inlier_multiplier_ * factor_weight_,
                                           robust_loss_type_);

    // Pass the computed values to class variables
    error_ = cuerror;
    TensorToEigenMatrix(cuAtA.to(torch::kDouble), AtA_);
    TensorToEigenMatrix(cuAtb.to(torch::kDouble), Atb_);

    toc("[MatchGeometryFactor<Scalar, CS>::ComputeJacobianAndError] jac " + std::to_string(kf0_->id) + " " + std::to_string(kf1_->id));

    return;
  }

  // /* ************************************************************************* */
  template <typename Scalar, int CS>
  std::tuple<cv::Mat, cv::Mat> MatchGeometryFactor<Scalar, CS>::DrawMatches()
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    // TODO: hard coded display_size here
    const long display_size = std::min(100L, matched_locations_1d_0_.size(0));
    const long height = cam_.height();
    const long width = cam_.width();

    const double ratio = (double)kf0_->color_img.cols / (double)width;

    at::Tensor rotation, translation;
    PoseT relpose10;
    RelativePose(kf1_->pose_wk, kf0_->pose_wk, relpose10);
    SophusSE3ToTensor(relpose10, rotation, translation, kf0_->dpt_map_bias.options());

    if (keypoints_0_.empty())
    {
      // M x 2
      const at::Tensor display_matched_locations_2d_0 =
          torch::stack({torch::fmod(matched_locations_1d_0_.index({Slice(None, display_size)}), (float)width),
                        torch::floor(matched_locations_1d_0_.index({Slice(None, display_size)}) / (float)width)},
                       1);

      // M x 2
      const at::Tensor display_matched_locations_2d_1 =
          torch::stack({torch::fmod(matched_locations_1d_1_.index({Slice(None, display_size)}), (float)width),
                        torch::floor(matched_locations_1d_1_.index({Slice(None, display_size)}) / (float)width)},
                       1);

      for (int i = 0; i < display_size; ++i)
      {
        keypoints_0_.emplace_back(display_matched_locations_2d_0.index({i, 0}).item<float>() * ratio,
                                  display_matched_locations_2d_0.index({i, 1}).item<float>() * ratio, 0.1f);
        keypoints_1_.emplace_back(display_matched_locations_2d_1.index({i, 0}).item<float>() * ratio,
                                  display_matched_locations_2d_1.index({i, 1}).item<float>() * ratio, 0.1f);
        desc_matches_.emplace_back(i, i, 0.0f);
      }
    }

    // 1 x 1 x H x W
    at::Tensor dpt_map_0;
    {
      std::shared_lock<std::shared_mutex> lock(kf0_->mutex);
      dpt_map_0 = kf0_->dpt_map;
    }

    // M
    const at::Tensor sampled_dpts_0 =
        dpt_map_0.reshape({height * width}).index({matched_locations_1d_0_.index({Slice(None, display_size)}).to(torch::kLong)});

    // M x 2, M x 1
    at::Tensor display_reproj_locations_2d_in_1, sampled_valid_mask;
    GenerateReproj2DLocationsNoClamp(sampled_dpts_0, matched_locations_homo_0_.index({Slice(None, display_size), Slice()}),
                                     rotation, translation, dpt_eps_,
                                     display_reproj_locations_2d_in_1, sampled_valid_mask, cam_, false);

    display_corrs_1_.clear();
    for (int i = 0; i < display_size; ++i)
    {
      display_corrs_1_.emplace_back(display_reproj_locations_2d_in_1.index({i, 0}).item<float>() * ratio,
                                    display_reproj_locations_2d_in_1.index({i, 1}).item<float>() * ratio, 0.1f);
    }

    // draw matches for debug
    cv::Mat img_matches, img_corrs;
    cv::drawMatches(kf0_->color_img, keypoints_0_, kf1_->color_img,
                    keypoints_1_, desc_matches_, img_matches);

    cv::drawMatches(kf0_->color_img, keypoints_0_, kf1_->color_img,
                    display_corrs_1_, desc_matches_, img_corrs);

    return std::make_tuple(img_matches, img_corrs);
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  cv::Mat MatchGeometryFactor<Scalar, CS>::ErrorImage() const
  {
    torch::NoGradGuard no_grad;

    cv::Mat kf1_color = kf1_->color_img.clone();

    for (long i = 0; i < static_cast<long>(keypoints_1_.size()); ++i)
    {
      const cv::KeyPoint &corr = display_corrs_1_[i];
      const cv::KeyPoint &kp = keypoints_1_[i];

      cv::Point p1(corr.pt.x, corr.pt.y);
      cv::Point p2(kp.pt.x, kp.pt.y);

      // draw lines
      cv::arrowedLine(kf1_color, p1, p2, cv::Scalar(0, 255, 0), 1, cv::LineTypes::LINE_AA, 0, 0.2);
      cv::circle(kf1_color, p2, 1, cv::Scalar(0, 255, 255), 1, cv::LineTypes::LINE_AA);
    }

    return kf1_color;
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  std::string MatchGeometryFactor<Scalar, CS>::Name() const
  {
    std::stringstream ss;
    auto fmt = gtsam::DefaultKeyFormatter;
    ss << "MatchGeometryFactor " << fmt(pose0_key_) << " -> " << fmt(pose1_key_);
    return ss.str();
  }

  /* ************************************************************************* */
  // explicit instantiation
  template class MatchGeometryFactor<float, DF_CODE_SIZE>;

}