#include "camera_tracker.h"

#include <torch/script.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

BOOST_GEOMETRY_REGISTER_POINT_2D(df::Point, float, cs::cartesian, x, y)

namespace df
{

  CameraTracker::CameraTracker(const TrackerConfig &config, const bool display)
      : config_(config), display_(display)
  {
    GenerateCheckerboard(checkerboard_, config_.net_output_size);

    pose_ck_ = Sophus::SE3f();
    photo_weights_tensor_ = torch::from_blob((void *)config_.photo_factor_weights.data(), {static_cast<long>(config_.photo_factor_weights.size())},
                                             torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                                .to(torch::kCUDA, config_.cuda_id);

    // TEASER
    teaser_params_.max_clique_time_limit = config_.teaser_max_clique_time_limit;
    teaser_params_.kcore_heuristic_threshold = config_.teaser_kcore_heuristic_threshold;
    teaser_params_.rotation_max_iterations = config_.teaser_rotation_max_iterations;
    teaser_params_.rotation_cost_threshold = config_.teaser_rotation_cost_threshold;
    teaser_params_.rotation_gnc_factor = config_.teaser_rotation_gnc_factor;

    std::string rotation_estimation_algorithm = boost::algorithm::to_lower_copy(config_.teaser_rotation_estimation_algorithm);
    if (rotation_estimation_algorithm == std::string("gnc_tls"))
    {
      teaser_params_.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    }
    else if (rotation_estimation_algorithm == std::string("fgr"))
    {
      teaser_params_.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::FGR;
    }
    else
    {
      LOG(FATAL) << "[CameraTracker::CameraTracker] rotation_estimation_algorithm type not supported: " << rotation_estimation_algorithm;
    }

    std::string rotation_tim_graph = boost::algorithm::to_lower_copy(config_.teaser_rotation_tim_graph);
    if (rotation_tim_graph == std::string("chain"))
    {
      teaser_params_.rotation_tim_graph = teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION::CHAIN;
    }
    else if (rotation_tim_graph == std::string("complete"))
    {
      teaser_params_.rotation_tim_graph = teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION::COMPLETE;
    }
    else
    {
      LOG(FATAL) << "[CameraTracker::CameraTracker] rotation_tim_graph type not supported: " << rotation_tim_graph;
    }

    std::string inlier_selection_mode = boost::algorithm::to_lower_copy(config_.teaser_inlier_selection_mode);
    if (inlier_selection_mode == std::string("pmc_exact"))
    {
      teaser_params_.inlier_selection_mode = teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_EXACT;
    }
    else if (inlier_selection_mode == std::string("pmc_heu"))
    {
      teaser_params_.inlier_selection_mode = teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_HEU;
    }
    else if (inlier_selection_mode == std::string("kcore_heu"))
    {
      teaser_params_.inlier_selection_mode = teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::KCORE_HEU;
    }
    else if (inlier_selection_mode == std::string("none"))
    {
      teaser_params_.inlier_selection_mode = teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::NONE;
    }
    else
    {
      LOG(FATAL) << "[CameraTracker::CameraTracker] inlier_selection_mode type not supported: " << inlier_selection_mode;
    }

    tracker_name_ = "";
    error_ = 0.0;
    warp_area_ratio_ = 0.0;
    inlier_ratio_ = 0.0;
    average_motion_ = 0.0;
    desc_match_inlier_ratio_ = 0.0;
    relative_desc_match_inlier_ratio_ = 0.0;
    inlier_multiplier_ = 0.0;
  }

  CameraTracker::~CameraTracker()
  {
    VLOG(2) << "[CameraTracker::~CameraTracker] deconstructor called -- " << tracker_name_;
  }

  void CameraTracker::ComputeAreaInlierRatio(const at::Tensor valid_dpts_0, const at::Tensor valid_locations_homo_0,
                                             const at::Tensor guess_rotation, const at::Tensor guess_translation,
                                             const df::PinholeCamera<float> &camera,
                                             const at::Tensor video_mask, float &area_ratio, float &inlier_ratio,
                                             float &average_motion)
  {
    torch::NoGradGuard no_grad;
    namespace F = torch::nn::functional;
    using namespace torch::indexing;
    using Polygon = boost::geometry::model::polygon<Point>;

    at::Tensor valid_locations_2d_in_1, pos_depth_mask_1;
    GenerateReproj2DLocationsNoClamp(valid_dpts_0, valid_locations_homo_0, guess_rotation,
                                     guess_translation, config_.dpt_eps,
                                     valid_locations_2d_in_1,
                                     pos_depth_mask_1, camera, false);

    const at::Tensor normalized_valid_locations_2d_in_1 = torch::stack({2.0 * valid_locations_2d_in_1.index({Slice(), 0}) / camera.width() - 1.0,
                                                                        2.0 * valid_locations_2d_in_1.index({Slice(), 1}) / camera.height() - 1.0},
                                                                       1);

    // 1 x N
    const at::Tensor valid_mask_1 = F::grid_sample(video_mask, normalized_valid_locations_2d_in_1.reshape({1, 1, -1, 2}),
                                                   F::GridSampleFuncOptions().mode(torch::kNearest).padding_mode(torch::kZeros).align_corners(true))
                                        .reshape({-1}) *
                                    pos_depth_mask_1.reshape({-1});

    at::Tensor valid_locations_homo_2d_0 =
        valid_locations_homo_0.to(torch::kFloat32).index({Slice(), Slice(None, 2)}).to(torch::kCPU).contiguous();

    at::Tensor valid_locations_2d_0 = torch::stack({valid_locations_homo_2d_0.index({Slice(), 0}) * camera.fx() + camera.u0(),
                                                    valid_locations_homo_2d_0.index({Slice(), 1}) * camera.fy() + camera.v0()},
                                                   1)
                                          .to(torch::kFloat32);

    std::vector<Point> points(valid_locations_homo_0.size(0));
    std::memcpy(points.data(), valid_locations_2d_0.contiguous().data_ptr(), sizeof(float) * valid_locations_2d_0.numel());

    Polygon poly, hull;
    poly.outer().assign(points.begin(), points.end());
    boost::geometry::convex_hull(poly, hull);

    input_area_ = boost::geometry::area(hull);

    const at::Tensor nonzero_indexes = torch::nonzero(valid_mask_1 > 0.5).reshape({-1});
    const at::Tensor pos_depth_indexes = torch::nonzero(pos_depth_mask_1.reshape({-1}) > 0.5).reshape({-1});
    // M x 2
    const at::Tensor valid_locations_2d_in_1_within_mask =
        valid_locations_2d_in_1.index({nonzero_indexes, Slice()}).to(torch::kCPU).contiguous();

    std::vector<Point> warp_points(valid_locations_2d_in_1_within_mask.size(0));
    std::memcpy(warp_points.data(), valid_locations_2d_in_1_within_mask.contiguous().data_ptr(),
                sizeof(float) * valid_locations_2d_in_1_within_mask.numel());

    Polygon warp_poly, warp_hull;
    warp_poly.outer().assign(warp_points.begin(), warp_points.end());
    boost::geometry::convex_hull(warp_poly, warp_hull);
    const float warp_area_within_mask = boost::geometry::area(warp_hull);

    // This two metrics cannot account for in-plane rotation, we use the average magnitude of rigid flow to describe this
    area_ratio = warp_area_within_mask / input_area_;
    inlier_ratio = (float)nonzero_indexes.size(0) / (float)valid_locations_homo_0.size(0);
    valid_locations_2d_0 = valid_locations_2d_0.to(valid_locations_2d_in_1.device());
    average_motion = torch::mean(torch::sqrt(torch::sum(torch::square(valid_locations_2d_in_1.index({pos_depth_indexes, Slice()}) -
                                                                      valid_locations_2d_0.index({pos_depth_indexes, Slice()})),
                                                        1, false)))
                         .item<float>() /
                     sqrt(camera.width() * camera.width() + camera.height() * camera.height());

    VLOG(3) << "[CameraTracker::TrackFrame] warp area ratio and inlier ratio: " << area_ratio << " " << inlier_ratio;
    VLOG(3) << "[CameraTracker::TrackFrame] translation and rotation: " << guess_translation << " " << guess_rotation;
    VLOG(3) << "[CameraTracker::TrackFrame] average 2d motion magnitude: " << average_motion;

    return;
  }

  void CameraTracker::ComputePhotoError(const CameraTracker::FrameT &frame_1, const at::Tensor cat_photo_features_0,
                                        const at::Tensor photo_dpts_0, const at::Tensor photo_locations_homo_0,
                                        const at::Tensor guess_rotation, const at::Tensor guess_translation,
                                        float &error)
  {
    torch::NoGradGuard no_grad;
    error = tracker_photo_error_calculate<DF_FEAT_SIZE>(
        guess_rotation, guess_translation.reshape({-1}),
        *(frame_1.video_mask_ptr),
        photo_dpts_0, photo_locations_homo_0,
        cat_photo_features_0,
        frame_1.feat_map_pyramid,
        *(frame_1.level_offsets_ptr),
        *(frame_1.camera_pyramid_ptr),
        config_.dpt_eps,
        photo_weights_tensor_);
  }

  void CameraTracker::ComputeMatchGeomError(const FrameT &reference_frame,
                                            const at::Tensor keypoint_dpts_0, const at::Tensor keypoint_locations_homo_0,
                                            const at::Tensor matched_dpts_1, const at::Tensor matched_locations_homo_1,
                                            const at::Tensor guess_rotation, const at::Tensor guess_translation,
                                            float &error)
  {
    torch::NoGradGuard no_grad;
    float match_geom_loss_param = config_.match_geom_loss_param_factor * reference_frame.avg_squared_dpt_bias;
    error = tracker_match_geom_error_calculate(guess_rotation, guess_translation.reshape({-1}),
                                               keypoint_dpts_0, matched_dpts_1,
                                               keypoint_locations_homo_0, matched_locations_homo_1,
                                               match_geom_loss_param, inlier_multiplier_ * config_.match_geom_factor_weight);

    return;
  }

  void CameraTracker::ComputeReprojError(const CameraTracker::FrameT &frame_to_track, const at::Tensor keypoint_dpts_0,
                                         const at::Tensor keypoint_locations_homo_0, const at::Tensor matched_locations_2d_1,
                                         const at::Tensor guess_rotation, const at::Tensor guess_translation,
                                         float &error)
  {
    torch::NoGradGuard no_grad;

    error = tracker_reproj_error_calculate(guess_rotation, guess_translation.reshape({-1}), keypoint_dpts_0,
                                           keypoint_locations_homo_0,
                                           matched_locations_2d_1, (*(frame_to_track.camera_pyramid_ptr))[0],
                                           config_.dpt_eps, reproj_loss_param_, inlier_multiplier_ * config_.reproj_factor_weight);

    return;
  }

  void CameraTracker::ComputeError(const CameraTracker::FrameT &frame_to_track, const at::Tensor cat_photo_features_0,
                                   const at::Tensor photo_dpts_0, const at::Tensor photo_locations_homo_0,
                                   const at::Tensor keypoint_dpts_0, const at::Tensor keypoint_locations_homo_0,
                                   const at::Tensor matched_locations_2d_1,
                                   const at::Tensor guess_rotation, const at::Tensor guess_translation,
                                   const bool use_photo, const bool use_reproj, float &error)
  {
    torch::NoGradGuard no_grad;
    float photo_error = 0, reproj_error = 0;

    if (use_photo)
    {
      ComputePhotoError(frame_to_track, cat_photo_features_0,
                        photo_dpts_0, photo_locations_homo_0,
                        guess_rotation, guess_translation,
                        photo_error);
    }

    if (use_reproj)
    {
      ComputeReprojError(frame_to_track, keypoint_dpts_0,
                         keypoint_locations_homo_0, matched_locations_2d_1,
                         guess_rotation, guess_translation,
                         reproj_error);
    }

    error = photo_error + reproj_error;

    return;
  }

  void CameraTracker::ComputeError(const CameraTracker::FrameT &reference_frame, const at::Tensor cat_photo_features_0,
                                   const at::Tensor unscaled_photo_dpts_0, const at::Tensor photo_locations_homo_0,
                                   const at::Tensor unscaled_keypoint_dpts_0, const at::Tensor keypoint_locations_homo_0,
                                   const at::Tensor matched_dpts_1, const at::Tensor matched_locations_homo_1,
                                   const at::Tensor guess_rotation, const at::Tensor guess_translation, const float guess_scale_0,
                                   const bool use_photo, const bool use_match_geom, float &error)
  {
    torch::NoGradGuard no_grad;
    float match_geom_error = 0, photo_error = 0;

    if (use_photo)
    {
      ComputePhotoError(reference_frame, cat_photo_features_0,
                        guess_scale_0 * unscaled_photo_dpts_0, photo_locations_homo_0,
                        guess_rotation, guess_translation,
                        photo_error);
    }

    if (use_match_geom)
    {
      ComputeMatchGeomError(reference_frame,
                            guess_scale_0 * unscaled_keypoint_dpts_0, keypoint_locations_homo_0,
                            matched_dpts_1, matched_locations_homo_1,
                            guess_rotation, guess_translation,
                            match_geom_error);
    }
    error = photo_error + match_geom_error;

    return;
  }

  void CameraTracker::ComputeJacobianAndError(const CameraTracker::FrameT &frame_1, const at::Tensor cat_photo_features_0,
                                              const at::Tensor photo_dpts_0, const at::Tensor photo_locations_homo_0,
                                              const at::Tensor keypoint_dpts_0, const at::Tensor keypoint_locations_homo_0,
                                              const at::Tensor matched_locations_2d_1,
                                              const at::Tensor guess_rotation, const at::Tensor guess_translation,
                                              const bool use_photo, const bool use_reproj,
                                              const bool update_error, at::Tensor &AtA, at::Tensor &Atb, float &error)
  {
    torch::NoGradGuard no_grad;

    // relative pose
    AtA = torch::zeros({6, 6}, photo_dpts_0.options());
    Atb = torch::zeros({6, 1}, photo_dpts_0.options());

    at::Tensor reproj_AtA, reproj_Atb;
    at::Tensor photo_AtA, photo_Atb;

    float reproj_error = 0;
    float photo_error = 0;

    if (use_photo)
    {
      ComputePhotoJacobianAndError(frame_1, cat_photo_features_0,
                                   photo_dpts_0, photo_locations_homo_0,
                                   guess_rotation, guess_translation,
                                   photo_AtA, photo_Atb, photo_error);
      AtA += photo_AtA;
      Atb += photo_Atb;
    }

    if (use_reproj)
    {
      ComputeReprojJacobianAndError(frame_1,
                                    keypoint_dpts_0, keypoint_locations_homo_0,
                                    matched_locations_2d_1,
                                    guess_rotation, guess_translation,
                                    reproj_AtA, reproj_Atb, reproj_error);
      AtA += reproj_AtA;
      Atb += reproj_Atb;
    }

    VLOG(3) << "[CameraTracker::ComputeJacobianAndError] photo and reproj error: " << photo_error << " " << reproj_error;

    if (update_error)
    {
      error = photo_error + reproj_error;
    }

    return;
  }

  void CameraTracker::ComputeJacobianAndError(const CameraTracker::FrameT &reference_frame, const at::Tensor cat_photo_features_0,
                                              const at::Tensor unscaled_photo_dpts_0, const at::Tensor photo_locations_homo_0,
                                              const at::Tensor unscaled_keypoint_dpts_0, const at::Tensor keypoint_locations_homo_0,
                                              const at::Tensor matched_dpts_1, const at::Tensor matched_locations_homo_1,
                                              const at::Tensor guess_rotation, const at::Tensor guess_translation,
                                              const float guess_scale_0, const bool use_photo, const bool use_match_geom, const bool update_error,
                                              at::Tensor &AtA, at::Tensor &Atb, float &error)
  {
    torch::NoGradGuard no_grad;

    // relative pose + scale_0
    AtA = torch::zeros({7, 7}, unscaled_photo_dpts_0.options());
    Atb = torch::zeros({7, 1}, unscaled_photo_dpts_0.options());

    at::Tensor match_geom_AtA, match_geom_Atb;
    at::Tensor photo_AtA, photo_Atb;

    float match_geom_error = 0;
    float photo_error = 0;

    if (use_photo)
    {
      ComputePhotoJacobianAndErrorWithScale(reference_frame, cat_photo_features_0,
                                            unscaled_photo_dpts_0, photo_locations_homo_0,
                                            guess_rotation, guess_translation, guess_scale_0,
                                            photo_AtA, photo_Atb, photo_error);
      AtA += photo_AtA;
      Atb += photo_Atb;
    }

    if (use_match_geom)
    {
      ComputeMatchGeomJacobianAndErrorWithScale(reference_frame,
                                                unscaled_keypoint_dpts_0, keypoint_locations_homo_0,
                                                matched_dpts_1, matched_locations_homo_1,
                                                guess_rotation, guess_translation, guess_scale_0,
                                                match_geom_AtA, match_geom_Atb, match_geom_error);
      AtA += match_geom_AtA;
      Atb += match_geom_Atb;
    }

    VLOG(3) << "[CameraTracker::ComputeJacobianAndError] photo and match geom error: " << photo_error << " " << match_geom_error;

    if (update_error)
    {
      error = photo_error + match_geom_error;
    }

    return;
  }

  void CameraTracker::ComputeReprojJacobianAndError(const FrameT &frame_1, const at::Tensor sampled_dpts_0,
                                                    const at::Tensor sampled_locations_homo_0,
                                                    const at::Tensor matched_locations_2d_1,
                                                    const at::Tensor guess_rotation,
                                                    const at::Tensor guess_translation, at::Tensor &AtA, at::Tensor &Atb,
                                                    float &error)
  {
    torch::NoGradGuard no_grad;

    tracker_reproj_jac_error_calculate(AtA, Atb, error,
                                       guess_rotation, guess_translation.reshape({-1}),
                                       sampled_dpts_0, sampled_locations_homo_0, matched_locations_2d_1,
                                       (*(frame_1.camera_pyramid_ptr))[0],
                                       config_.dpt_eps, reproj_loss_param_, inlier_multiplier_ * config_.reproj_factor_weight);

    return;
  }

  void CameraTracker::ComputeMatchGeomJacobianAndErrorWithScale(const FrameT &reference_frame,
                                                                const at::Tensor unscaled_keypoint_dpts_0, const at::Tensor keypoint_locations_homo_0,
                                                                const at::Tensor matched_dpts_1, const at::Tensor matched_locations_homo_1,
                                                                const at::Tensor guess_rotation, const at::Tensor guess_translation,
                                                                const float scale_0, at::Tensor &AtA, at::Tensor &Atb, float &error)
  {
    torch::NoGradGuard no_grad;
    float match_geom_loss_param = config_.match_geom_loss_param_factor * reference_frame.avg_squared_dpt_bias;
    tracker_match_geom_jac_error_calculate_with_scale(AtA, Atb, error,
                                                      guess_rotation, guess_translation.reshape({-1}),
                                                      scale_0 * unscaled_keypoint_dpts_0, matched_dpts_1,
                                                      keypoint_locations_homo_0, matched_locations_homo_1,
                                                      scale_0, match_geom_loss_param, inlier_multiplier_ * config_.match_geom_factor_weight);

    return;
  }

  void CameraTracker::ComputePhotoJacobianAndErrorWithScale(const FrameT &reference_frame, const at::Tensor cat_photo_features_0,
                                                            const at::Tensor unscaled_photo_dpts_0, const at::Tensor photo_locations_homo_0,
                                                            const at::Tensor guess_rotation, const at::Tensor guess_translation,
                                                            const float scale_0,
                                                            at::Tensor &AtA, at::Tensor &Atb, float &error)
  {
    torch::NoGradGuard no_grad;

    tracker_photo_jac_error_calculate_with_scale<DF_FEAT_SIZE>(
        AtA, Atb, error,
        guess_rotation, guess_translation.reshape({-1}),
        *(reference_frame.video_mask_ptr),
        scale_0 * unscaled_photo_dpts_0, // N
        photo_locations_homo_0,          // N x 2
        cat_photo_features_0,            // L x N x C_feat
        reference_frame.feat_map_pyramid,
        reference_frame.feat_map_grad_pyramid,
        *(reference_frame.level_offsets_ptr),
        *(reference_frame.camera_pyramid_ptr),
        scale_0,
        config_.dpt_eps,
        photo_weights_tensor_);

    return;
  }

  void CameraTracker::ComputePhotoJacobianAndError(const FrameT &frame_1, const at::Tensor cat_photo_features_0,
                                                   const at::Tensor photo_dpts_0, const at::Tensor photo_locations_homo_0,
                                                   const at::Tensor guess_rotation, const at::Tensor guess_translation,
                                                   at::Tensor &AtA, at::Tensor &Atb, float &error)
  {
    torch::NoGradGuard no_grad;
    tracker_photo_jac_error_calculate<DF_FEAT_SIZE>(
        AtA, Atb, error,
        guess_rotation, guess_translation.reshape({-1}),
        *(frame_1.video_mask_ptr),
        photo_dpts_0,           // N
        photo_locations_homo_0, // N x 2
        cat_photo_features_0,   // L x N x C_feat
        frame_1.feat_map_pyramid,
        frame_1.feat_map_grad_pyramid,
        *(frame_1.level_offsets_ptr),
        *(frame_1.camera_pyramid_ptr),
        config_.dpt_eps,
        photo_weights_tensor_);
    return;
  }

  void CameraTracker::UpdateVariables(const Eigen::Matrix<float, 7, 1> &solution, const at::Tensor curr_rot_mat, const at::Tensor curr_trans_vec,
                                      const float curr_scale, at::Tensor &updated_rot_mat, at::Tensor &updated_trans_vec, float &updated_scale)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    at::Tensor delta_rot, delta_trans;

    auto eigen_omega = solution.block(3, 0, 3, 1);
    auto eigen_v = solution.block(0, 0, 3, 1);
    float delta_scale = solution(6, 0);
    Eigen::Matrix<float, 3, 3> eigen_delta_rot_mat;
    Eigen::Matrix<float, 3, 1> eigen_delta_trans_vec;

    se3_exp<float>(eigen_omega, eigen_v, eigen_delta_rot_mat, eigen_delta_trans_vec);

    const at::Tensor delta_rot_mat = EigenMatToTensor(eigen_delta_rot_mat, curr_rot_mat.options());
    const at::Tensor delta_trans_vec = EigenVectorToTensor(eigen_delta_trans_vec, curr_rot_mat.options());

    updated_rot_mat = torch::matmul(delta_rot_mat, curr_rot_mat);
    updated_trans_vec = torch::matmul(delta_rot_mat, curr_trans_vec.reshape({3, 1})) + delta_trans_vec.reshape({3, 1});
    updated_scale = curr_scale + delta_scale;
    return;
  }

  void CameraTracker::UpdateVariables(const Eigen::Matrix<float, 6, 1> &solution, const at::Tensor curr_rot_mat, const at::Tensor curr_trans_vec,
                                      at::Tensor &updated_rot_mat, at::Tensor &updated_trans_vec)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    at::Tensor delta_rot, delta_trans;

    auto eigen_omega = solution.block(3, 0, 3, 1);
    auto eigen_v = solution.block(0, 0, 3, 1);

    Eigen::Matrix<float, 3, 3> eigen_delta_rot_mat;
    Eigen::Matrix<float, 3, 1> eigen_delta_trans_vec;

    se3_exp<float>(eigen_omega, eigen_v, eigen_delta_rot_mat, eigen_delta_trans_vec);

    const at::Tensor delta_rot_mat = EigenMatToTensor(eigen_delta_rot_mat, curr_rot_mat.options());
    const at::Tensor delta_trans_vec = EigenVectorToTensor(eigen_delta_trans_vec, curr_rot_mat.options());

    updated_rot_mat = torch::matmul(delta_rot_mat, curr_rot_mat);
    updated_trans_vec = torch::matmul(delta_rot_mat, curr_trans_vec.reshape({3, 1})) + delta_trans_vec.reshape({3, 1});
    return;
  }

  // void CameraTracker::UpdateVariables(const at::Tensor solution, const at::Tensor curr_rot_mat, const at::Tensor curr_trans_vec,
  //                                     const float curr_scale, at::Tensor &updated_rot_mat, at::Tensor &updated_trans_vec, float &updated_scale)
  // {
  //   torch::NoGradGuard no_grad;
  //   using namespace torch::indexing;
  //   at::Tensor delta_rot, delta_trans;
  //   SE3Exp(solution.index({Slice(3, 6)}), solution.index({Slice(None, 3)}), delta_rot, delta_trans);
  //   updated_rot_mat = torch::matmul(delta_rot, curr_rot_mat);
  //   updated_trans_vec = torch::matmul(delta_rot, curr_trans_vec.reshape({3, 1})) + delta_trans.reshape({3, 1});
  //   updated_scale = curr_scale + solution.index({6}).item<float>();
  //   return;
  // }

  bool CameraTracker::LMConvergence(const at::Tensor guess_rotation, const at::Tensor guess_translation, const float guess_scale,
                                    const at::Tensor Atb, const at::Tensor solution)
  {
    torch::NoGradGuard no_grad;
    const at::Tensor guess_rotvec = RotationToAngleAxis(guess_rotation, 1.0e-6);
    const float max_grad = torch::max(torch::abs(Atb)).item<float>();
    const float max_param_inc = torch::max(solution /
                                           (torch::abs(torch::cat({guess_translation.reshape({-1}), guess_rotvec,
                                                                   guess_scale * torch::ones({1}, guess_rotvec.options())},
                                                                  0)) +
                                            1.0e-8))
                                    .item<float>();

    // The optimization has converged
    if (max_grad < config_.min_grad_thresh || max_param_inc < config_.min_param_inc_thresh)
    {
      VLOG(2) << "[CameraTracker::LMConvergence] convergence max grad: " << max_grad << " max param relative increment: " << max_param_inc << " " << tracker_name_;
      return true;
    }
    else
    {
      return false;
    }
  }

  bool CameraTracker::LMConvergence(const at::Tensor guess_rotation, const at::Tensor guess_translation,
                                    const at::Tensor Atb, const at::Tensor solution)
  {
    torch::NoGradGuard no_grad;
    const at::Tensor guess_rotvec = RotationToAngleAxis(guess_rotation, 1.0e-6);
    const float max_grad = torch::max(torch::abs(Atb)).item<float>();
    const float max_param_inc = torch::max(solution /
                                           (torch::abs(torch::cat({guess_translation.reshape({-1}), guess_rotvec}, 0)) +
                                            1.0e-8))
                                    .item<float>();

    // The optimization has converged
    if (max_grad < config_.min_grad_thresh || max_param_inc < config_.min_param_inc_thresh)
    {
      VLOG(2) << "[CameraTracker::LMConvergence] convergence max grad: " << max_grad << " max param relative increment: " << max_param_inc << " " << tracker_name_;
      return true;
    }
    else
    {
      return false;
    }
  }

  void CameraTracker::FeatureMatchingGeo(const FrameT &frame_to_track, const at::Tensor valid_locations_1d_0,
                                         const at::Tensor valid_locations_homo_0, const at::Tensor valid_dpts_0,
                                         const at::Tensor dpt_map_1, const PinholeCamera<float> camera,
                                         at::Tensor &inlier_keypoint_locations_homo_0, at::Tensor &inlier_keypoint_dpts_0,
                                         at::Tensor &matched_locations_homo_1, at::Tensor &matched_dpts_1,
                                         at::Tensor &guess_rotation, at::Tensor &guess_translation, float &guess_scale,
                                         float &relative_desc_match_inlier_ratio, float &desc_match_inlier_ratio,
                                         float &inlier_multiplier)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;

    tic("[CameraTracker::FeatureMatchingGeo] cycle feature matching " + tracker_name_);

    const long seed = kf_->id;
    std::mt19937 g;
    g.seed(seed);
    std::vector<long> indices(valid_locations_1d_0.size(0));
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), g);

    const at::Tensor feature_desc_0 = frame_to_track.feat_desc;
    const at::Tensor feature_desc_1 = kf_->feat_desc;

    const long height = feature_desc_0.size(2);
    const long width = feature_desc_0.size(3);

    // K
    const at::Tensor keypoint_indexes = torch::from_blob(static_cast<long *>(indices.data()),
                                                         {std::min(config_.desc_num_samples, valid_locations_1d_0.size(0))},
                                                         torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
                                            .to(valid_locations_1d_0.device())
                                            .clone();
    const at::Tensor keypoint_locations_1d_0 = valid_locations_1d_0.index({keypoint_indexes});
    const at::Tensor keypoint_locations_2d_0_x = torch::fmod(keypoint_locations_1d_0, (float)width);
    const at::Tensor keypoint_locations_2d_0_y = torch::floor(keypoint_locations_1d_0 / (float)width);

    long channel = feature_desc_0.size(1);

    // C_feat x K
    const at::Tensor keypoint_features_0 = feature_desc_0.reshape({channel, height * width}).index({Slice(), keypoint_locations_1d_0});
    // K x H*W
    const at::Tensor feature_response_1 = -torch::sum(torch::square(keypoint_features_0.reshape({channel, config_.desc_num_samples, 1}) -
                                                                    feature_desc_1.reshape({channel, 1, height * width})),
                                                      0, false);
    // K
    const at::Tensor raw_matched_locations_1d_1 = std::get<1>(torch::max(feature_response_1, 1, false));
    // C_feat x K
    const at::Tensor raw_matched_features_1 = feature_desc_1.reshape({channel, height * width}).index({Slice(), raw_matched_locations_1d_1});

    // K x H*W
    const at::Tensor feature_response_0 = -torch::sum(torch::square(raw_matched_features_1.reshape({channel, config_.desc_num_samples, 1}) -
                                                                    feature_desc_0.reshape({channel, 1, height * width})),
                                                      0, false);

    // K
    const at::Tensor cyc_matched_locations_1d_0 = std::get<1>(torch::max(feature_response_0, 1, false));
    const at::Tensor cyc_matched_locations_2d_0_x = torch::fmod(cyc_matched_locations_1d_0, (float)width);
    const at::Tensor cyc_matched_locations_2d_0_y = torch::floor(cyc_matched_locations_1d_0 / (float)width);

    const at::Tensor cyc_distances_sq = torch::square(keypoint_locations_2d_0_x - cyc_matched_locations_2d_0_x) +
                                        torch::square(keypoint_locations_2d_0_y - cyc_matched_locations_2d_0_y);

    const at::Tensor inlier_within_keypoint_indexes =
        torch::nonzero(cyc_distances_sq <= (config_.desc_cyc_consis_thresh * config_.desc_cyc_consis_thresh)).reshape({-1});

    // Check if there are any matches that are within the distance threshold
    if (inlier_within_keypoint_indexes.size(0) <= 0)
    {
      relative_desc_match_inlier_ratio = 0;
      desc_match_inlier_ratio = 0;
      inlier_multiplier = 0;
      inlier_keypoint_locations_homo_0 = torch::zeros_like(inlier_within_keypoint_indexes).to(torch::kFloat);
      inlier_keypoint_dpts_0 = torch::zeros_like(inlier_within_keypoint_indexes).to(torch::kFloat);
      matched_locations_homo_1 = torch::zeros_like(inlier_within_keypoint_indexes).to(torch::kFloat);
      matched_dpts_1 = torch::zeros_like(inlier_within_keypoint_indexes).to(torch::kFloat);
      return;
    } 

    const at::Tensor inlier_keypoint_indexes = keypoint_indexes.index({inlier_within_keypoint_indexes});

    // M x 3
    inlier_keypoint_locations_homo_0 = valid_locations_homo_0.index({inlier_keypoint_indexes, Slice()});
    // M
    inlier_keypoint_dpts_0 = valid_dpts_0.index({inlier_keypoint_indexes});

    // Feature descriptor matching
    // M
    const at::Tensor matched_locations_1d_1 = raw_matched_locations_1d_1.index({inlier_within_keypoint_indexes});
    // M x 2
    const at::Tensor matched_locations_2d_1 =
        torch::stack({torch::fmod(matched_locations_1d_1, (float)width), torch::floor(matched_locations_1d_1 / (float)width)}, 1);
    // M
    const at::Tensor matched_locations_2d_1_x = (matched_locations_2d_1.index({Slice(), 0}) - camera.u0()) / camera.fx();
    const at::Tensor matched_locations_2d_1_y = (matched_locations_2d_1.index({Slice(), 1}) - camera.v0()) / camera.fy();
    // M x 3
    matched_locations_homo_1 = torch::stack({matched_locations_2d_1_x, matched_locations_2d_1_y,
                                             torch::ones_like(matched_locations_2d_1_x)},
                                            1);
    matched_dpts_1 = dpt_map_1.reshape({-1}).index({matched_locations_1d_1});

    // M x 3
    const at::Tensor inlier_keypoint_locations_3d_0 = inlier_keypoint_dpts_0.reshape({-1, 1}) / frame_to_track.dpt_scale * inlier_keypoint_locations_homo_0;
    const at::Tensor matched_locations_3d_1 = matched_dpts_1.reshape({-1, 1}) * matched_locations_homo_1;

    // 3 x M
    Eigen::Matrix<double, 3, Eigen::Dynamic> eigen_inlier_keypoint_locations_3d_0;
    Eigen::Matrix<double, 3, Eigen::Dynamic> eigen_matched_locations_3d_1;

    eigen_inlier_keypoint_locations_3d_0.resize(Eigen::NoChange, inlier_keypoint_locations_3d_0.size(0));
    eigen_matched_locations_3d_1.resize(Eigen::NoChange, matched_locations_3d_1.size(0));

    // 3 x M col major in eigen matrix is the same as M x 3 row major in torch tensor in terms of memory arrangement
    std::memcpy(static_cast<double *>(eigen_inlier_keypoint_locations_3d_0.data()),
                inlier_keypoint_locations_3d_0.to(torch::kCPU).to(torch::kDouble).contiguous().data_ptr(), sizeof(double) * inlier_keypoint_locations_3d_0.numel());
    std::memcpy(static_cast<double *>(eigen_matched_locations_3d_1.data()),
                matched_locations_3d_1.to(torch::kCPU).to(torch::kDouble).contiguous().data_ptr(), sizeof(double) * matched_locations_3d_1.numel());

    // at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    toc("[CameraTracker::FeatureMatchingGeo] cycle feature matching " + tracker_name_);

    tic("[CameraTracker::FeatureMatchingGeo] teaser filtering " + tracker_name_);

    const at::Tensor avg_dpt_1 = torch::mean(matched_dpts_1);
    float avg_dpt = avg_dpt_1.item<float>();
    float focal_length = (camera.fx() + camera.fy()) / 2.0;

    // What noise bound to use?? should use dst_point_cloud (which is 1 in our case)
    // We modify the source code of TEASER to include different noise per point
    teaser_params_.noise_bound = config_.teaser_noise_bound_multiplier * avg_dpt / focal_length;
    teaser::RobustRegistrationSolver solver(teaser_params_);

    at::Tensor noise_bounds_tensor = config_.teaser_noise_bound_multiplier * matched_dpts_1 / focal_length;
    // hard-coded minimum noise bound here
    noise_bounds_tensor = torch::clamp_min(noise_bounds_tensor, 5.0e-4);
    Eigen::Matrix<double, 1, Eigen::Dynamic> noise_bounds;
    noise_bounds.resize(Eigen::NoChange, matched_dpts_1.size(0));
    std::memcpy(static_cast<double *>(noise_bounds.data()),
                noise_bounds_tensor.to(torch::kCPU).to(torch::kDouble).contiguous().data_ptr(), sizeof(double) * noise_bounds_tensor.numel());

    // src and dst are 3-by-N Eigen matrices
    solver.solve(eigen_inlier_keypoint_locations_3d_0, eigen_matched_locations_3d_1, noise_bounds);

    auto solution = solver.getSolution();
    std::vector<int> inlier_indexes_in_clique_vec = solver.getTranslationInliers();
    Eigen::Matrix<int, 1, Eigen::Dynamic> clique_indexes_in_ori_vec = solver.getTranslationInliersMap();
    auto translation_inliers_mask = solver.getTranslationInliersMask();

    const at::Tensor inlier_indexes_in_clique = torch::from_blob(static_cast<int *>(inlier_indexes_in_clique_vec.data()),
                                                                 {static_cast<long>(inlier_indexes_in_clique_vec.size())},
                                                                 torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU))
                                                    .to(inlier_keypoint_locations_3d_0.device())
                                                    .to(torch::kLong)
                                                    .clone();

    const at::Tensor clique_indexes_in_ori = torch::from_blob(static_cast<int *>(clique_indexes_in_ori_vec.data()), {clique_indexes_in_ori_vec.size()},
                                                              torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU))
                                                 .to(inlier_keypoint_locations_3d_0.device())
                                                 .to(torch::kLong)
                                                 .clone();

    const at::Tensor inlier_indexes = clique_indexes_in_ori.index({inlier_indexes_in_clique});

    // This inlier ratio should give us information on how dissimilar two images are
    relative_desc_match_inlier_ratio = static_cast<float>(inlier_indexes.size(0)) / static_cast<float>(inlier_keypoint_locations_3d_0.size(0));
    desc_match_inlier_ratio = static_cast<float>(inlier_indexes.size(0)) / static_cast<float>(config_.desc_num_samples);
    inlier_multiplier = desc_match_inlier_ratio; // 1.0; //

    matched_locations_homo_1 = matched_locations_homo_1.index({inlier_indexes, Slice()});
    matched_dpts_1 = matched_dpts_1.index({inlier_indexes});
    inlier_keypoint_locations_homo_0 = inlier_keypoint_locations_homo_0.index({inlier_indexes, Slice()});
    inlier_keypoint_dpts_0 = inlier_keypoint_dpts_0.index({inlier_indexes});

    // T^1_0
    guess_rotation = torch::from_blob(static_cast<double *>(solution.rotation.data()), {3, 3},
                                      torch::TensorOptions().device(torch::kCPU).dtype(torch::kDouble))
                         .to(matched_locations_homo_1.device())
                         .to(matched_locations_homo_1.dtype())
                         .permute({1, 0})
                         .clone();
    guess_translation = torch::from_blob(static_cast<double *>(solution.translation.data()), {3, 1},
                                         torch::TensorOptions().device(torch::kCPU).dtype(torch::kDouble))
                            .to(matched_locations_homo_1.device())
                            .to(matched_locations_homo_1.dtype())
                            .clone();
    guess_scale = solution.scale;

    // stream = at::cuda::getCurrentCUDAStream();
    // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    toc("[CameraTracker::FeatureMatchingGeo] teaser filtering " + tracker_name_);

    return;
  }

  void CameraTracker::FeatureMatchingGeo(const at::Tensor feature_desc_0, const at::Tensor feature_desc_1,
                                         const at::Tensor valid_locations_1d_0,
                                         const at::Tensor valid_locations_homo_0, const at::Tensor valid_dpts_0,
                                         const at::Tensor dpt_map_1, const PinholeCamera<float> camera,
                                         at::Tensor &inlier_keypoint_locations_homo_0, at::Tensor &inlier_keypoint_dpts_0,
                                         at::Tensor &matched_locations_2d_1,
                                         float &relative_desc_match_inlier_ratio, float &desc_match_inlier_ratio,
                                         float &inlier_multiplier)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;

    tic("[CameraTracker::FeatureMatchingGeo] cycle feature matching " + tracker_name_);

    const long seed = kf_->id;
    std::mt19937 g;
    g.seed(seed);
    std::vector<long> indices(valid_locations_1d_0.size(0));
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), g);

    const long height = feature_desc_0.size(2);
    const long width = feature_desc_0.size(3);

    // K
    const at::Tensor keypoint_indexes = torch::from_blob(static_cast<long *>(indices.data()),
                                                         {std::min(config_.desc_num_samples, valid_locations_1d_0.size(0))},
                                                         torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
                                            .to(valid_locations_1d_0.device())
                                            .clone();
    const at::Tensor keypoint_locations_1d_0 = valid_locations_1d_0.index({keypoint_indexes});
    const at::Tensor keypoint_locations_2d_0_x = torch::fmod(keypoint_locations_1d_0, (float)width);
    const at::Tensor keypoint_locations_2d_0_y = torch::floor(keypoint_locations_1d_0 / (float)width);

    long channel = feature_desc_0.size(1);

    // C_feat x K
    const at::Tensor keypoint_features_0 = feature_desc_0.reshape({channel, height * width}).index({Slice(), keypoint_locations_1d_0});
    // C_feat x K x 1 - C_feat x 1 x H*W -> K x H*W
    const at::Tensor feature_response_1 = -torch::sum(torch::square(keypoint_features_0.reshape({channel, config_.desc_num_samples, 1}) -
                                                                    feature_desc_1.reshape({channel, 1, height * width})),
                                                      0, false);

    // K
    const at::Tensor raw_matched_locations_1d_1 = std::get<1>(torch::max(feature_response_1, 1, false));

    // C_feat x K
    const at::Tensor raw_matched_features_1 = feature_desc_1.reshape({channel, height * width}).index({Slice(), raw_matched_locations_1d_1});

    // K x H*W
    const at::Tensor feature_response_0 = -torch::sum(torch::square(raw_matched_features_1.reshape({channel, config_.desc_num_samples, 1}) -
                                                                    feature_desc_0.reshape({channel, 1, height * width})),
                                                      0, false);

    // K
    const at::Tensor cyc_matched_locations_1d_0 = std::get<1>(torch::max(feature_response_0, 1, false));

    const at::Tensor cyc_matched_locations_2d_0_x = torch::fmod(cyc_matched_locations_1d_0, (float)width);
    const at::Tensor cyc_matched_locations_2d_0_y = torch::floor(cyc_matched_locations_1d_0 / (float)width);

    const at::Tensor cyc_distances_sq = torch::square(keypoint_locations_2d_0_x - cyc_matched_locations_2d_0_x) +
                                        torch::square(keypoint_locations_2d_0_y - cyc_matched_locations_2d_0_y);

    const at::Tensor inlier_within_keypoint_indexes =
        torch::nonzero(cyc_distances_sq <= (config_.desc_cyc_consis_thresh * config_.desc_cyc_consis_thresh)).reshape({-1});

    // Check if there are any matches that are within the distance threshold
    if (inlier_within_keypoint_indexes.size(0) <= 0)
    {
      LOG(WARNING) << "[CameraTracker::FeatureMatchingGeo] zero inlier keypoint matches have been found";
      relative_desc_match_inlier_ratio = 0;
      desc_match_inlier_ratio = 0;
      inlier_multiplier = 0;
      inlier_keypoint_locations_homo_0 = torch::zeros_like(inlier_within_keypoint_indexes).to(torch::kFloat);
      inlier_keypoint_dpts_0 = torch::zeros_like(inlier_within_keypoint_indexes).to(torch::kFloat);
      matched_locations_2d_1 = torch::zeros_like(inlier_within_keypoint_indexes).to(torch::kFloat);
      return;
    }

    const at::Tensor inlier_keypoint_indexes = keypoint_indexes.index({inlier_within_keypoint_indexes});

    // M x 3
    inlier_keypoint_locations_homo_0 = valid_locations_homo_0.index({inlier_keypoint_indexes, Slice()});
    // M
    inlier_keypoint_dpts_0 = valid_dpts_0.index({inlier_keypoint_indexes});
    // Feature descriptor matching
    // M
    const at::Tensor matched_locations_1d_1 = raw_matched_locations_1d_1.index({inlier_within_keypoint_indexes});
    // M x 2
    matched_locations_2d_1 =
        torch::stack({torch::fmod(matched_locations_1d_1, (float)width), torch::floor(matched_locations_1d_1 / (float)width)}, 1);

    // M
    const at::Tensor matched_locations_2d_1_x = (matched_locations_2d_1.index({Slice(), 0}) - camera.u0()) / camera.fx();
    const at::Tensor matched_locations_2d_1_y = (matched_locations_2d_1.index({Slice(), 1}) - camera.v0()) / camera.fy();
    // M x 3
    const at::Tensor matched_locations_homo_1 = torch::stack({matched_locations_2d_1_x, matched_locations_2d_1_y,
                                                              torch::ones_like(matched_locations_2d_1_x)},
                                                             1);
    const at::Tensor matched_dpts_1 = dpt_map_1.reshape({-1}).index({matched_locations_1d_1});

    // M x 3
    const at::Tensor inlier_keypoint_locations_3d_0 = inlier_keypoint_dpts_0.reshape({-1, 1}) * inlier_keypoint_locations_homo_0;
    const at::Tensor matched_locations_3d_1 = matched_dpts_1.reshape({-1, 1}) * matched_locations_homo_1;

    // 3 x M
    Eigen::Matrix<double, 3, Eigen::Dynamic> eigen_inlier_keypoint_locations_3d_0;
    Eigen::Matrix<double, 3, Eigen::Dynamic> eigen_matched_locations_3d_1;

    eigen_inlier_keypoint_locations_3d_0.resize(Eigen::NoChange, inlier_keypoint_locations_3d_0.size(0));
    eigen_matched_locations_3d_1.resize(Eigen::NoChange, matched_locations_3d_1.size(0));

    // 3 x M col major in eigen matrix is the same as M x 3 row major in torch tensor in terms of memory arrangement
    std::memcpy(static_cast<double *>(eigen_inlier_keypoint_locations_3d_0.data()),
                inlier_keypoint_locations_3d_0.to(torch::kCPU).to(torch::kDouble).contiguous().data_ptr(), sizeof(double) * inlier_keypoint_locations_3d_0.numel());
    std::memcpy(static_cast<double *>(eigen_matched_locations_3d_1.data()),
                matched_locations_3d_1.to(torch::kCPU).to(torch::kDouble).contiguous().data_ptr(), sizeof(double) * matched_locations_3d_1.numel());

    // at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    toc("[CameraTracker::FeatureMatchingGeo] cycle feature matching " + tracker_name_);

    tic("[CameraTracker::FeatureMatchingGeo] teaser filtering " + tracker_name_);

    const at::Tensor avg_dpt_0 = torch::mean(inlier_keypoint_dpts_0);
    float avg_dpt = avg_dpt_0.item<float>();
    float focal_length = (camera.fx() + camera.fy()) / 2.0;

    // What noise bound to use?? should use dst_point_cloud (which is 1 in our case)
    // We modify the source code of TEASER to include different noise per point
    teaser_params_.noise_bound = config_.teaser_noise_bound_multiplier * avg_dpt / focal_length;
    teaser::RobustRegistrationSolver solver(teaser_params_);

    at::Tensor noise_bounds_tensor = config_.teaser_noise_bound_multiplier * inlier_keypoint_dpts_0 / focal_length;
    // hard-coded minimum noise bound here
    noise_bounds_tensor = torch::clamp_min(noise_bounds_tensor, 5.0e-4);
    Eigen::Matrix<double, 1, Eigen::Dynamic> noise_bounds;
    noise_bounds.resize(Eigen::NoChange, inlier_keypoint_dpts_0.size(0));
    std::memcpy(static_cast<double *>(noise_bounds.data()),
                noise_bounds_tensor.to(torch::kCPU).to(torch::kDouble).contiguous().data_ptr(), sizeof(double) * noise_bounds_tensor.numel());

    // src and dst are 3-by-N Eigen matrices
    solver.solve(eigen_matched_locations_3d_1, eigen_inlier_keypoint_locations_3d_0, noise_bounds);

    std::vector<int> inlier_indexes_in_clique_vec = solver.getTranslationInliers();
    Eigen::Matrix<int, 1, Eigen::Dynamic> clique_indexes_in_ori_vec = solver.getTranslationInliersMap();
    auto translation_inliers_mask = solver.getTranslationInliersMask();

    const at::Tensor inlier_indexes_in_clique = torch::from_blob(static_cast<int *>(inlier_indexes_in_clique_vec.data()),
                                                                 {static_cast<long>(inlier_indexes_in_clique_vec.size())},
                                                                 torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU))
                                                    .to(inlier_keypoint_locations_3d_0.device())
                                                    .to(torch::kLong)
                                                    .clone();

    const at::Tensor clique_indexes_in_ori = torch::from_blob(static_cast<int *>(clique_indexes_in_ori_vec.data()), {clique_indexes_in_ori_vec.size()},
                                                              torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU))
                                                 .to(inlier_keypoint_locations_3d_0.device())
                                                 .to(torch::kLong)
                                                 .clone();

    const at::Tensor inlier_indexes = clique_indexes_in_ori.index({inlier_indexes_in_clique});

    // This inlier ratio should give us information on how dissimilar two images are
    relative_desc_match_inlier_ratio = static_cast<float>(inlier_indexes.size(0)) / static_cast<float>(inlier_keypoint_locations_3d_0.size(0));
    desc_match_inlier_ratio = static_cast<float>(inlier_indexes.size(0)) / static_cast<float>(config_.desc_num_samples);
    inlier_multiplier = desc_match_inlier_ratio; //

    matched_locations_2d_1 = matched_locations_2d_1.index({inlier_indexes, Slice()});
    inlier_keypoint_locations_homo_0 = inlier_keypoint_locations_homo_0.index({inlier_indexes, Slice()});
    inlier_keypoint_dpts_0 = inlier_keypoint_dpts_0.index({inlier_indexes});

    // stream = at::cuda::getCurrentCUDAStream();
    // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    toc("[CameraTracker::FeatureMatchingGeo] teaser filtering " + tracker_name_);

    return;
  }

  bool CameraTracker::TrackMatchGeoCheck(FrameT &frame_to_track)
  {
    // frame_to_track should be frame 1 and reference keyframe should be frame 0
    // so that we could get the initialization and refinement of frame_to_track scale and camera pose from the TrackFrame method here
    torch::NoGradGuard no_grad;
    namespace F = torch::nn::functional;
    using namespace torch::indexing;

    if (!kf_)
    {
      throw std::runtime_error("[CameraTracker::MatchGeoCheck] call before a reference keyframe was set " + tracker_name_);
    }

    // N x 3
    const at::Tensor valid_locations_homo_0 = frame_to_track.valid_locations_homo;
    // N
    const at::Tensor valid_locations_1d_0 = frame_to_track.valid_locations_1d;
    // 1 x 1 x H x W
    const at::Tensor dpt_map_0 = frame_to_track.dpt_map;
    const at::Tensor valid_dpts_0 = dpt_map_0.reshape({-1}).index({valid_locations_1d_0});

    tic("[CameraTracker::MatchGeoCheck] match geometric preparation " + tracker_name_);
    // kf is 0 and fram_to_track is 1
    FeatureMatchingGeo(kf_->feat_desc, frame_to_track.feat_desc,
                       valid_locations_1d_0, valid_locations_homo_0, valid_dpts_0,
                       frame_to_track.dpt_map, (*(frame_to_track.camera_pyramid_ptr))[0],
                       inlier_keypoint_locations_homo_0_, inlier_keypoint_dpts_0_,
                       matched_locations_2d_1_, relative_desc_match_inlier_ratio_,
                       desc_match_inlier_ratio_, inlier_multiplier_);

    if (matched_locations_2d_1_.size(0) <= 3)
    {
      VLOG(2) << "[CameraTracker::MatchGeoCheck] Not enough feature matches detected for frame to track " << frame_to_track.id << " and keyframe " << kf_->id << " " << tracker_name_;
      // This means the two keyframes probably are not spatially close and we should abandon this pair.
      return false;
    }

    return true;
  }

  bool CameraTracker::MatchGeoCheck(FrameT &frame_to_track)
  {
    // frame_to_track should be frame 1 and reference keyframe should be frame 0
    // so that we could get the initialization and refinement of frame_to_track scale and camera pose from the TrackFrame method here
    torch::NoGradGuard no_grad;
    namespace F = torch::nn::functional;
    using namespace torch::indexing;

    if (!kf_)
    {
      throw std::runtime_error("[CameraTracker::MatchGeoCheck] call before a reference keyframe was set " + tracker_name_);
    }

    // N x 3
    const at::Tensor valid_locations_homo_0 = frame_to_track.valid_locations_homo;
    // N
    const at::Tensor valid_locations_1d_0 = frame_to_track.valid_locations_1d;
    // 1 x 1 x H x W
    const at::Tensor dpt_map_0 = frame_to_track.dpt_map;
    const at::Tensor valid_dpts_0 = dpt_map_0.reshape({-1}).index({valid_locations_1d_0});

    tic("[CameraTracker::MatchGeoCheck] match geometric preparation " + tracker_name_);

    // N
    FeatureMatchingGeo(frame_to_track, valid_locations_1d_0,
                       valid_locations_homo_0, valid_dpts_0,
                       kf_->dpt_map, (*(kf_->camera_pyramid_ptr))[0],
                       inlier_keypoint_locations_homo_0_, inlier_keypoint_dpts_0_,
                       matched_locations_homo_1_, matched_dpts_1_,
                       guess_rotation_, guess_translation_, guess_scale_,
                       relative_desc_match_inlier_ratio_, desc_match_inlier_ratio_,
                       inlier_multiplier_);

    toc("[CameraTracker::MatchGeoCheck] match geometric preparation " + tracker_name_);

    if (matched_locations_homo_1_.size(0) <= 3)
    {
      VLOG(2) << "[CameraTracker::MatchGeoCheck] Not enough feature matches detected for frame to track " << frame_to_track.id << " and keyframe " << kf_->id << " " << tracker_name_;
      // This means the two keyframes probably are not spatially close and we should abandon this pair.
      return false;
    }

    return true;
  }

  bool CameraTracker::TrackNewFrame(FrameT &frame_to_track, bool use_photo, bool use_reproj, bool match_geom_pre_checked, bool update_frame, bool display_image)
  {
    torch::NoGradGuard no_grad;
    namespace F = torch::nn::functional;
    using namespace torch::indexing;

    if (!kf_)
    {
      throw std::runtime_error("[CameraTracker::TrackFrame] call before a reference keyframe was set " + tracker_name_);
    }

    Eigen::Matrix<float, 6, 6> eigen_AtA;
    Eigen::Matrix<float, 6, 1> eigen_Atb;
    Eigen::Matrix<float, 6, 1> eigen_solution;
    Eigen::Matrix<float, 6, 6> eigen_damped_AtA, eigen_AtA_diag;

    at::Tensor AtA, Atb, AtA_diag, solution;
    at::Tensor candidate_rotation_ck, candidate_translation_ck;
    cv::Mat warp_display;
    cv::Mat init_warp_display;

    bool update_jac = true;
    float prev_error = 0;
    float curr_error = 1;
    float candidate_error;
    long curr_iter = 0;
    float curr_damp = config_.init_damp;
    int num_levels = kf_->camera_pyramid_ptr->Levels();

    // Here the rotation and translation should be T^c_k
    // 3 x 3
    at::Tensor guess_rotation_10 = SophusMatrixToTensor(Sophus::Matrix<float, 3, 3>(pose_ck_.so3().matrix()), kf_->dpt_map_bias.options());
    // 3 x 1
    at::Tensor guess_translation_10 = SophusVectorToTensor(Sophus::Vector<float, 3>(pose_ck_.translation()), kf_->dpt_map_bias.options());

    // In this method, 0 is keyframe, while 1 is the frame to track
    // 1 x 1 x H x W
    const at::Tensor dpt_map_0 = kf_->dpt_map;

    long height = dpt_map_0.size(2);
    long width = dpt_map_0.size(3);

    reproj_loss_param_ = config_.reproj_loss_param_factor * static_cast<float>(width * width);

    // N x 3
    const at::Tensor valid_locations_homo_0 = kf_->valid_locations_homo;
    const at::Tensor photo_locations_homo_0 = kf_->sampled_locations_homo;
    // N
    const at::Tensor valid_locations_1d_0 = kf_->valid_locations_1d;
    const at::Tensor photo_locations_1d_0 = kf_->sampled_locations_1d;

    const at::Tensor valid_dpts_0 = dpt_map_0.reshape({-1}).index({valid_locations_1d_0});

    const long num_photo_points = photo_locations_1d_0.size(0);

    // Photometric Factor preparation
    at::Tensor photo_dpts_0 = dpt_map_0.reshape({-1}).index({photo_locations_1d_0});
    // N x 2
    const at::Tensor photo_locations_2d_0 =
        torch::stack({torch::fmod(photo_locations_1d_0.to(torch::kFloat32), width),
                      torch::floor(photo_locations_1d_0.to(torch::kFloat32) / width)},
                     1);

    // 1 x 1 x N x 2
    const at::Tensor photo_locations_normalized_2d_0 =
        torch::stack({(photo_locations_2d_0.index({Slice(), 0}) + 0.5f) * (2.0f / width) - 1.0,
                      (photo_locations_2d_0.index({Slice(), 1}) + 0.5f) * (2.0f / height) - 1.0},
                     1)
            .reshape({1, 1, -1, 2});

    std::vector<at::Tensor> stack_photo_features_0(num_levels);
    long feat_channel = kf_->feat_map_pyramid.size(0);
    long offset = 0;
    for (int i = 0; i < num_levels; ++i)
    {
      long cur_width = (*(kf_->camera_pyramid_ptr))[i].width();
      long cur_height = (*(kf_->camera_pyramid_ptr))[i].height();
      // N x C_feat
      stack_photo_features_0[i] = F::grid_sample(kf_->feat_map_pyramid.index({Slice(),
                                                                              Slice(offset, offset + cur_height * cur_width)})
                                                     .reshape({1, feat_channel, cur_height, cur_width}),
                                                 photo_locations_normalized_2d_0,
                                                 F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(false))
                                      .reshape({-1, num_photo_points})
                                      .permute({1, 0});

      offset += cur_width * cur_height;
    }
    // L x N x C_feat
    at::Tensor cat_photo_features_0 = torch::stack(stack_photo_features_0, 0);

    at::Tensor inlier_keypoint_locations_homo_0, inlier_keypoint_dpts_0, matched_locations_2d_1;

    if (use_reproj && !match_geom_pre_checked)
    {
      FeatureMatchingGeo(kf_->feat_desc, frame_to_track.feat_desc,
                         valid_locations_1d_0, valid_locations_homo_0, valid_dpts_0,
                         frame_to_track.dpt_map, (*(frame_to_track.camera_pyramid_ptr))[0],
                         inlier_keypoint_locations_homo_0_, inlier_keypoint_dpts_0_,
                         matched_locations_2d_1_, relative_desc_match_inlier_ratio_, desc_match_inlier_ratio_,
                         inlier_multiplier_);
    }

    if (use_reproj)
    {
      matched_locations_2d_1 = matched_locations_2d_1_;
      inlier_keypoint_locations_homo_0 = inlier_keypoint_locations_homo_0_;
      inlier_keypoint_dpts_0 = inlier_keypoint_dpts_0_;

      if (inlier_keypoint_locations_homo_0.size(0) <= 3)
      {
        VLOG(2) << "[CameraTracker::TrackFrame] Not enough feature matches detected for frame to track " << frame_to_track.id << " and keyframe " << kf_->id << " " << tracker_name_;
        // This means the two keyframes probably are not spatially close and we should abandon this pair.
        return false;
      }
    }

    if (display_image)
    {
      init_warp_display = DisplaySE3Warp(*kf_, frame_to_track, guess_rotation_10, guess_translation_10, kf_->dpt_scale, config_.dpt_eps, checkerboard_);
    }

    while (true)
    {
      // Skip jacobian computation if the error decrease is too small in the last step
      if (fabs(curr_error - prev_error) / prev_error > config_.jac_update_err_inc_threshold)
      {
        ComputeJacobianAndError(frame_to_track, cat_photo_features_0,
                                photo_dpts_0, photo_locations_homo_0,
                                inlier_keypoint_dpts_0, inlier_keypoint_locations_homo_0,
                                matched_locations_2d_1,
                                guess_rotation_10, guess_translation_10,
                                use_photo, use_reproj,
                                (curr_iter == 0) ? true : false,
                                AtA, Atb, curr_error);

        TensorToEigenMatrix(AtA, eigen_AtA);
        TensorToEigenMatrix(torch::diag(torch::diag(AtA)), eigen_AtA_diag);
        TensorToEigenMatrix(Atb, eigen_Atb);
        update_jac = true;
      }
      else
      {
        update_jac = false;
      }

      curr_iter += 1;

      eigen_damped_AtA = eigen_AtA + curr_damp * eigen_AtA_diag;
      eigen_solution = eigen_damped_AtA.colPivHouseholderQr().solve(eigen_Atb);
      solution = EigenVectorToTensor(eigen_solution, AtA.options());

      if (LMConvergence(guess_rotation_10, guess_translation_10, Atb, solution))
      {
        VLOG(2) << "[CameraTracker::TrackFrame] step too small, optimization converged.";
        break;
      }

      while (true)
      {
        // stream = at::cuda::getCurrentCUDAStream();
        // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        // tic("[CameraTracker::TrackFrame] UpdateVariables " + tracker_name_);

        UpdateVariables(eigen_solution, guess_rotation_10, guess_translation_10, candidate_rotation_ck, candidate_translation_ck);

        // stream = at::cuda::getCurrentCUDAStream();
        // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        // toc("[CameraTracker::TrackFrame] UpdateVariables " + tracker_name_);

        // stream = at::cuda::getCurrentCUDAStream();
        // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        // tic("[CameraTracker::TrackFrame] ComputeError " + tracker_name_);

        ComputeError(frame_to_track, cat_photo_features_0,
                     photo_dpts_0, photo_locations_homo_0,
                     inlier_keypoint_dpts_0, inlier_keypoint_locations_homo_0,
                     matched_locations_2d_1,
                     candidate_rotation_ck, candidate_translation_ck,
                     use_photo, use_reproj, candidate_error);
        // stream = at::cuda::getCurrentCUDAStream();
        // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        // toc("[CameraTracker::TrackFrame] ComputeError " + tracker_name_);

        if (candidate_error < curr_error)
        {
          VLOG(3) << "[CameraTracker::TrackKeyframe] update step accepted " + tracker_name_;
          break;
        }
        else if (curr_damp < config_.max_damp)
        {
          // stream = at::cuda::getCurrentCUDAStream();
          // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
          // tic("[CameraTracker::TrackFrame] err solution compute " + tracker_name_);
          // Increase the damp value to try to find a better optim location
          curr_damp = std::min(std::max(config_.min_damp, curr_damp * config_.damp_inc_factor), config_.max_damp);

          eigen_damped_AtA = eigen_AtA + curr_damp * eigen_AtA_diag;
          eigen_solution = eigen_damped_AtA.colPivHouseholderQr().solve(eigen_Atb);
          solution = EigenVectorToTensor(eigen_solution, AtA.options());

          VLOG(3) << "[CameraTracker::TrackKeyframe] update step rejected, increasing damp to: " << curr_damp << " " << tracker_name_;

          // stream = at::cuda::getCurrentCUDAStream();
          // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
          // toc("[CameraTracker::TrackFrame] err solution compute " + tracker_name_);
        }
        else
        {
          // If damp value reaches maximum and still no better solution found, then determin the optimization as converged
          break;
        }
      }

      // stream = at::cuda::getCurrentCUDAStream();
      // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
      // toc("[CameraTracker::TrackFrame] several err computation " + tracker_name_);

      // stream = at::cuda::getCurrentCUDAStream();
      // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
      // toc("[CameraTracker::TrackFrame] one iteration of LM optimization " + tracker_name_);

      if (candidate_error >= curr_error && curr_damp >= config_.max_damp)
      {
        VLOG(2) << "[CameraTracker::TrackKeyframe] damp reaches maximum, optimization converged " + tracker_name_;
        break;
      }

      // The solution is accepted and the estimates are updated. Damp value is decreased
      guess_rotation_10 = candidate_rotation_ck;
      guess_translation_10 = candidate_translation_ck;
      if (update_jac)
      {
        prev_error = curr_error;
      }
      curr_error = candidate_error;
      curr_damp = std::min(std::max(config_.min_damp, curr_damp / config_.damp_dec_factor), config_.max_damp);

      VLOG(3) << "[CameraTracker::TrackKeyframe] updated error: " << candidate_error << " previous error: " << curr_error << " damp: " << curr_damp << " " << tracker_name_;

      if (curr_iter >= config_.max_num_iters)
      {
        VLOG(2) << "[CameraTracker::TrackKeyframe] reaching maximum LM steps, finishing optimization " << tracker_name_;
        break;
      }
    }

    if (display_image)
    {
      // Draw the frame alignment
      warp_display = DisplaySE3Warp(*kf_, frame_to_track, guess_rotation_10, guess_translation_10, kf_->dpt_scale, config_.dpt_eps, checkerboard_);
      std::unique_lock<std::shared_mutex> lock(warp_image_mutex_);
      cv::vconcat(init_warp_display, warp_display, warp_image_);
    }

    ComputeAreaInlierRatio(valid_dpts_0, valid_locations_homo_0,
                           guess_rotation_10, guess_translation_10,
                           (*(kf_->camera_pyramid_ptr))[0],
                           kf_->feat_video_mask.reshape({1, 1, height, width}),
                           warp_area_ratio_, inlier_ratio_, average_motion_);

    // update the estimate of relative pose between the current frame and the reference keyframe
    Eigen::Matrix<float, 3, 3> rotation;
    TensorToEigenMatrix(guess_rotation_10, rotation);
    pose_ck_.so3() = Sophus::SO3f(rotation);
    TensorToEigenMatrix(guess_translation_10, pose_ck_.translation());
    error_ = curr_error;

    if (update_frame)
    {
      std::unique_lock<std::shared_mutex> lock(frame_to_track.mutex);
      // Update the variables within frame_to_track
      frame_to_track.pose_wk = kf_->pose_wk * pose_ck_.inverse();
    }

    return true;
  }

  bool CameraTracker::TrackFrame(FrameT &frame_to_track, bool use_photo, bool use_match_geom, bool display_image, bool update_frame, bool match_geom_pre_checked)
  {
    // frame_to_track should be frame 1 and reference keyframe should be frame 0
    // so that we could get the initialization and refinement of frame_to_track scale and camera pose from the TrackFrame method here
    torch::NoGradGuard no_grad;
    namespace F = torch::nn::functional;
    using namespace torch::indexing;

    frame_to_track_id_ = frame_to_track.id;

    if (!kf_)
    {
      throw std::runtime_error("[CameraTracker::TrackFrame] call before a reference keyframe was set " + tracker_name_);
    }

    if (!use_photo && !use_match_geom)
    {
      throw std::runtime_error("[CameraTracker::TrackFrame] at least one factor should be enabled " + tracker_name_);
    }

    Eigen::Matrix<float, 7, 7> eigen_AtA;
    Eigen::Matrix<float, 7, 1> eigen_Atb;

    at::Tensor AtA, Atb, AtA_diag, solution;
    at::Tensor candidate_rotation, candidate_translation;
    float candidate_scale;
    cv::Mat warp_display;
    cv::Mat init_warp_display;

    bool update_jac = true;
    float prev_error = 0;
    float curr_error = 1;
    float candidate_error;
    long curr_iter = 0;
    float curr_damp = config_.init_damp;
    int num_levels = kf_->camera_pyramid_ptr->Levels();

    // Here the rotation, translation, and scale should be T^k_c
    // 3 x 3
    at::Tensor guess_rotation = SophusMatrixToTensor(Sophus::Matrix<float, 3, 3>(pose_ck_.so3().matrix()), kf_->dpt_map_bias.options());
    guess_rotation = guess_rotation.permute({1, 0});
    // 3 x 1
    at::Tensor guess_translation = SophusVectorToTensor(Sophus::Vector<float, 3>(pose_ck_.translation()), kf_->dpt_map_bias.options());
    guess_translation = torch::matmul(-guess_rotation, guess_translation);
    //
    float guess_scale = frame_to_track.dpt_scale;

    at::Tensor dpt_map_0;
    at::Tensor dpt_map_1;
    float dpt_scale_0;
    // 1 x 1 x H x W
    {
      std::shared_lock<std::shared_mutex> lock(frame_to_track.mutex);
      dpt_map_0 = frame_to_track.dpt_map.clone();
      dpt_scale_0 = frame_to_track.dpt_scale;
    }

    {
      std::shared_lock<std::shared_mutex> lock(kf_->mutex);
      dpt_map_1 = kf_->dpt_map.clone();
      ref_kf_scale_ = kf_->dpt_scale;
    }

    long height = dpt_map_0.size(2);
    long width = dpt_map_0.size(3);

    // N x 3
    const at::Tensor valid_locations_homo_0 = frame_to_track.valid_locations_homo;
    const at::Tensor photo_locations_homo_0 = frame_to_track.sampled_locations_homo;
    // N
    const at::Tensor valid_locations_1d_0 = frame_to_track.valid_locations_1d;
    const at::Tensor photo_locations_1d_0 = frame_to_track.sampled_locations_1d;

    const at::Tensor valid_dpts_0 = dpt_map_0.reshape({-1}).index({valid_locations_1d_0});

    const long num_photo_points = photo_locations_1d_0.size(0);

    at::Tensor cat_photo_features_0, unscaled_photo_dpts_0;
    at::Tensor inlier_keypoint_locations_homo_0, inlier_keypoint_dpts_0,
        matched_locations_homo_1, matched_dpts_1, unscaled_inlier_keypoint_dpts_0;
    Eigen::Matrix<float, 7, 1> eigen_solution;
    Eigen::Matrix<float, 7, 7> eigen_damped_AtA, eigen_AtA_diag;

    tic("[CameraTracker::TrackFrame] match geometric preparation " + tracker_name_);
    if (use_match_geom && !match_geom_pre_checked)
    {
      FeatureMatchingGeo(frame_to_track, valid_locations_1d_0,
                         valid_locations_homo_0, valid_dpts_0,
                         dpt_map_1, (*(kf_->camera_pyramid_ptr))[0],
                         inlier_keypoint_locations_homo_0_, inlier_keypoint_dpts_0_,
                         matched_locations_homo_1_, matched_dpts_1_,
                         guess_rotation_, guess_translation_, guess_scale_,
                         relative_desc_match_inlier_ratio_, desc_match_inlier_ratio_,
                         inlier_multiplier_);

      if (matched_locations_homo_1_.size(0) <= 3)
      {
        VLOG(2) << "[CameraTracker::TrackFrame] Not enough feature matches detected for frame to track " << frame_to_track.id << " and keyframe " << kf_->id << " " << tracker_name_;
        // This means the two keyframes probably are not spatially close and we should abandon this pair.
        return false;
      }
    }

    if (use_match_geom)
    {
      guess_rotation = guess_rotation_;
      guess_translation = guess_translation_;
      guess_scale = guess_scale_;
      inlier_keypoint_locations_homo_0 = inlier_keypoint_locations_homo_0_;
      inlier_keypoint_dpts_0 = inlier_keypoint_dpts_0_;
      unscaled_inlier_keypoint_dpts_0 = inlier_keypoint_dpts_0_ / dpt_scale_0;
      matched_locations_homo_1 = matched_locations_homo_1_;
      matched_dpts_1 = matched_dpts_1_;
    }

    if (use_photo)
    {
      unscaled_photo_dpts_0 = dpt_map_0.reshape({-1}).index({photo_locations_1d_0}) / dpt_scale_0;
      // N x 2
      const at::Tensor photo_locations_2d_0 =
          torch::stack({torch::fmod(photo_locations_1d_0.to(torch::kFloat32), width),
                        torch::floor(photo_locations_1d_0.to(torch::kFloat32) / width)},
                       1);

      // 1 x 1 x N x 2
      const at::Tensor photo_locations_normalized_2d_0 =
          torch::stack({(photo_locations_2d_0.index({Slice(), 0}) + 0.5f) * (2.0f / width) - 1.0,
                        (photo_locations_2d_0.index({Slice(), 1}) + 0.5f) * (2.0f / height) - 1.0},
                       1)
              .reshape({1, 1, -1, 2});

      std::vector<at::Tensor> stack_photo_features_0(num_levels);
      long feat_channel = frame_to_track.feat_map_pyramid.size(0);
      long offset = 0;
      for (int i = 0; i < num_levels; ++i)
      {
        long cur_width = (*(frame_to_track.camera_pyramid_ptr))[i].width();
        long cur_height = (*(frame_to_track.camera_pyramid_ptr))[i].height();

        // N x C_feat
        stack_photo_features_0[i] = F::grid_sample(frame_to_track.feat_map_pyramid.index({Slice(),
                                                                                          Slice(offset, offset + cur_height * cur_width)})
                                                       .reshape({1, feat_channel, cur_height, cur_width}),
                                                   photo_locations_normalized_2d_0,
                                                   F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(false))
                                        .reshape({-1, num_photo_points})
                                        .permute({1, 0});

        offset += cur_width * cur_height;
      }

      // L x N x C_feat
      cat_photo_features_0 = torch::stack(stack_photo_features_0, 0);
    }

    toc("[CameraTracker::TrackFrame] match geometric preparation " + tracker_name_);

    if (display_image)
    {
      init_warp_display = DisplaySE3Warp(frame_to_track, *kf_,
                                         guess_rotation, guess_translation,
                                         guess_scale, config_.dpt_eps, checkerboard_);
    }

    // at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    // tic("[CameraTracker::TrackFrame] LM optimization " + tracker_name_);
    while (true)
    {
      // stream = at::cuda::getCurrentCUDAStream();
      // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
      // tic("[CameraTracker::TrackFrame] one iteration of LM optimization " + tracker_name_);
      // frame_to_track is 0, reference kf is 1

      // Skip jacobian computation if the error decrease is too small in the last step
      if (fabs(curr_error - prev_error) / prev_error > config_.jac_update_err_inc_threshold)
      {
        // stream = at::cuda::getCurrentCUDAStream();
        // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        // tic("[CameraTracker::TrackFrame] jac computation " + tracker_name_);
        ComputeJacobianAndError(*kf_, cat_photo_features_0,
                                unscaled_photo_dpts_0, photo_locations_homo_0,
                                unscaled_inlier_keypoint_dpts_0, inlier_keypoint_locations_homo_0,
                                matched_dpts_1, matched_locations_homo_1,
                                guess_rotation, guess_translation, guess_scale,
                                use_photo, use_match_geom, (curr_iter == 0) ? true : false,
                                AtA, Atb, curr_error);

        TensorToEigenMatrix(AtA, eigen_AtA);
        TensorToEigenMatrix(torch::diag(torch::diag(AtA)), eigen_AtA_diag);
        TensorToEigenMatrix(Atb, eigen_Atb);

        // stream = at::cuda::getCurrentCUDAStream();
        // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        // toc("[CameraTracker::TrackFrame] jac computation " + tracker_name_);
        update_jac = true;
      }
      else
      {
        update_jac = false;
      }

      // Only if no overlap and match geometry factor is not enabled, we quit the camera tracking with failure indication
      if (curr_error >= torch::sum(photo_weights_tensor_).item<float>() * 9.9 && !use_match_geom)
      {
        VLOG(2) << "[CameraTracker::TrackKeyframe] no overlap between frame to track: " << frame_to_track.id << " and keyframe: " << kf_->id << " " << tracker_name_;
        return false;
      }

      curr_iter += 1;

      eigen_damped_AtA = eigen_AtA + curr_damp * eigen_AtA_diag;
      eigen_solution = eigen_damped_AtA.colPivHouseholderQr().solve(eigen_Atb);
      solution = EigenVectorToTensor(eigen_solution, AtA.options());
      // solution = std::get<0>(torch::solve(Atb, AtA + curr_damp * AtA_diag)).reshape({-1});

      if (LMConvergence(guess_rotation, guess_translation, guess_scale, Atb, solution))
      {
        VLOG(3) << "[CameraTracker::TrackKeyframe] step too small, optimization converged " + tracker_name_;
        break;
      }

      // stream = at::cuda::getCurrentCUDAStream();
      // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
      // tic("[CameraTracker::TrackFrame] several err computation " + tracker_name_);
      while (true)
      {
        // stream = at::cuda::getCurrentCUDAStream();
        // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        // tic("[CameraTracker::TrackFrame] UpdateVariables " + tracker_name_);

        UpdateVariables(eigen_solution, guess_rotation, guess_translation, guess_scale, candidate_rotation, candidate_translation, candidate_scale);

        // SE3Exp(solution.index({Slice(3, 6)}), solution.index({Slice(None, 3)}), delta_rot, delta_trans);
        // update the relative pose based on update step estimated above
        // UpdateVariables(solution, guess_rotation, guess_translation, guess_scale, candidate_rotation, candidate_translation, candidate_scale);

        // stream = at::cuda::getCurrentCUDAStream();
        // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        // toc("[CameraTracker::TrackFrame] UpdateVariables " + tracker_name_);

        // stream = at::cuda::getCurrentCUDAStream();
        // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        // tic("[CameraTracker::TrackFrame] ComputeError " + tracker_name_);
        ComputeError(*kf_, cat_photo_features_0,
                     unscaled_photo_dpts_0, photo_locations_homo_0,
                     unscaled_inlier_keypoint_dpts_0, inlier_keypoint_locations_homo_0,
                     matched_dpts_1, matched_locations_homo_1,
                     candidate_rotation, candidate_translation, candidate_scale,
                     use_photo, use_match_geom, candidate_error);
        // stream = at::cuda::getCurrentCUDAStream();
        // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        // toc("[CameraTracker::TrackFrame] ComputeError " + tracker_name_);

        if (candidate_error < curr_error)
        {
          VLOG(3) << "[CameraTracker::TrackKeyframe] update step accepted " + tracker_name_;
          break;
        }
        else if (curr_damp < config_.max_damp)
        {
          // stream = at::cuda::getCurrentCUDAStream();
          // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
          // tic("[CameraTracker::TrackFrame] err solution compute " + tracker_name_);
          // Increase the damp value to try to find a better optim location
          curr_damp = std::min(std::max(config_.min_damp, curr_damp * config_.damp_inc_factor), config_.max_damp);

          eigen_damped_AtA = eigen_AtA + curr_damp * eigen_AtA_diag;
          eigen_solution = eigen_damped_AtA.colPivHouseholderQr().solve(eigen_Atb);
          solution = EigenVectorToTensor(eigen_solution, AtA.options());

          // // Re-compute the solution based on updated damp value
          // solution = std::get<0>(torch::solve(Atb, AtA + curr_damp * AtA_diag)).reshape({-1});
          VLOG(3) << "[CameraTracker::TrackKeyframe] update step rejected, increasing damp to: " << curr_damp << " " << tracker_name_;

          // stream = at::cuda::getCurrentCUDAStream();
          // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
          // toc("[CameraTracker::TrackFrame] err solution compute " + tracker_name_);
        }
        else
        {
          // If damp value reaches maximum and still no better solution found, then determin the optimization as converged
          break;
        }
      }

      // stream = at::cuda::getCurrentCUDAStream();
      // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
      // toc("[CameraTracker::TrackFrame] several err computation " + tracker_name_);

      // stream = at::cuda::getCurrentCUDAStream();
      // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
      // toc("[CameraTracker::TrackFrame] one iteration of LM optimization " + tracker_name_);

      if (candidate_error >= curr_error && curr_damp >= config_.max_damp)
      {
        VLOG(3) << "[CameraTracker::TrackKeyframe] damp reaches maximum, optimization converged " + tracker_name_;
        break;
      }

      // The solution is accepted and the estimates are updated. Damp value is decreased
      guess_rotation = candidate_rotation;
      guess_translation = candidate_translation;
      guess_scale = candidate_scale;
      if (update_jac)
      {
        prev_error = curr_error;
      }
      curr_error = candidate_error;
      curr_damp = std::min(std::max(config_.min_damp, curr_damp / config_.damp_dec_factor), config_.max_damp);

      VLOG(3) << "[CameraTracker::TrackKeyframe] updated error: " << candidate_error << " previous error: " << curr_error << " damp: " << curr_damp << " " << tracker_name_;

      if (curr_iter >= config_.max_num_iters)
      {
        VLOG(3) << "[CameraTracker::TrackKeyframe] reaching maximum LM steps, finishing optimization " << tracker_name_;
        break;
      }
    }
    // stream = at::cuda::getCurrentCUDAStream();
    // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    // toc("[CameraTracker::TrackFrame] LM optimization " + tracker_name_);

    if (display_image)
    {
      // Draw the frame alignment
      warp_display = DisplaySE3Warp(frame_to_track, *kf_, guess_rotation, guess_translation, guess_scale, config_.dpt_eps, checkerboard_);
      std::unique_lock<std::shared_mutex> lock(warp_image_mutex_);
      cv::vconcat(init_warp_display, warp_display, warp_image_);
    }

    // Compute area ratio within mask and point number ratio within mask as a way of measuring the distances between two frames
    ComputeAreaInlierRatio(valid_dpts_0 * (guess_scale / dpt_scale_0), valid_locations_homo_0,
                           guess_rotation, guess_translation,
                           (*(kf_->camera_pyramid_ptr))[0],
                           kf_->feat_video_mask.reshape({1, 1, height, width}),
                           warp_area_ratio_, inlier_ratio_, average_motion_);

    // update the estimate of relative pose between the current frame and the reference keyframe
    Eigen::Matrix<float, 3, 3> rotation;
    guess_rotation = guess_rotation.permute({1, 0});
    TensorToEigenMatrix(guess_rotation, rotation);
    pose_ck_.so3() = Sophus::SO3f(rotation); // pose_ck_ -> T_^0_1
    guess_translation = torch::matmul(-guess_rotation, guess_translation);
    TensorToEigenMatrix(guess_translation, pose_ck_.translation());
    error_ = curr_error;

    if (update_frame)
    {
      std::unique_lock<std::shared_mutex> lock(frame_to_track.mutex);
      // Update the variables within frame_to_track
      frame_to_track.dpt_map = dpt_map_0 * (guess_scale / dpt_scale_0);
      frame_to_track.dpt_scale = guess_scale;
      frame_to_track.pose_wk = kf_->pose_wk * pose_ck_.inverse();
      // frame_to_track.scale_ratio_cur_ref = guess_scale / ref_kf_scale_;
    }

    guess_scale_ = guess_scale;

    return true;
  }

  void CameraTracker::Reset()
  {
    pose_ck_ = Sophus::SE3f();
  }

  Sophus::SE3f CameraTracker::GetRelativePoseEstimate()
  {
    return pose_ck_;
  }

  Sophus::SE3f CameraTracker::GetWorldPoseEstimate()
  {
    // convert the between-frame pose to frame-to-world pose
    return kf_->pose_wk * pose_ck_.inverse();
  }

  void CameraTracker::SetRefKeyframe(std::shared_ptr<KeyframeT> new_kf)
  {
    // change the current pose estimate to proper camera frame
    if (kf_)
    {
      auto pose_wc = kf_->pose_wk * pose_ck_.inverse();
      pose_ck_ = pose_wc.inverse() * new_kf->pose_wk;
    }

    // set the pointer
    kf_ = new_kf;
  }

  void CameraTracker::SetWorldPose(const Sophus::SE3f &pose_wc)
  {
    // calculate pose w.r.t new keyframe
    pose_ck_ = pose_wc.inverse() * kf_->pose_wk;
  }

  void CameraTracker::SetConfig(const CameraTracker::TrackerConfig &new_cfg)
  {
    config_ = new_cfg;
  }

} // namespace df