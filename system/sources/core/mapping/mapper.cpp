#include "mapper.h"

#include <sstream>
#include <opencv2/imgproc/types_c.h>

namespace df
{

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  Mapper<Scalar, CS>::Mapper(const MapperOptions &opts,
                             const cv::Mat &video_mask,
                             const df::CameraPyramid<Scalar> &out_cam_pyr,
                             const df::CodeDepthNetwork::Ptr &depth_network,
                             const df::FeatureNetwork::Ptr &feature_network)
      : depth_network_(depth_network), feature_network_(feature_network),
        opts_(opts)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    // Set cuda device id
    cuda_id_ = opts.cuda_id;
    work_manager_ = new work::WorkManager();

    namespace F = torch::nn::functional;
    // create a shared ptr of camera pyramid
    output_camera_pyramid_ptr_ = std::make_shared<df::CameraPyramid<Scalar>>(out_cam_pyr);

    // kernel building for gauss smmothing
    std::vector<float> kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    gauss_kernel_ = torch::from_blob(static_cast<float *>(kernel.data()), {1, 1, 3, 3},
                                     torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32))
                        .to(torch::kCUDA, cuda_id_)
                        .clone() /
                    16.0;
    // options used by pytorch
    gauss_conv_options_ = F::Conv2dFuncOptions().stride(2).padding(1);
    mask_interp_options_ = F::InterpolateFuncOptions().mode(torch::kNearest);

    at::Tensor valid_normalized_locations_2d, valid_locations_1d, valid_locations_homo;

    // create keyframe map
    map_ = std::make_shared<MapT>();

    isam_graph_ = std::make_shared<gtsam::ISAM2>(opts_.isam_params);

    // create video mask
    cv::Mat input_mask, output_mask, feat_mask;

    cv::resize(video_mask, input_mask, cv::Size2l(opts_.net_input_size[1], opts_.net_input_size[0]), 0, 0, CV_INTER_NN);
    cv::resize(video_mask, output_mask, cv::Size2l(opts_.net_output_size[1], opts_.net_output_size[0]), 0, 0, CV_INTER_NN);

    input_video_mask_ = torch::from_blob(static_cast<unsigned char *>(input_mask.data),
                                         {1, 1, opts_.net_input_size[0], opts_.net_input_size[1]},
                                         torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8))
                            .to(torch::kFloat32)
                            .to(torch::kCUDA, cuda_id_)
                            .clone();
    output_video_mask_ = torch::from_blob(static_cast<unsigned char *>(output_mask.data), {1, 1, opts_.net_output_size[0], opts_.net_output_size[1]},
                                          torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8))
                             .to(torch::kFloat32)
                             .to(torch::kCUDA, cuda_id_)
                             .clone();

    // generate video mask pyramid
    output_mask_ptr_ = std::make_shared<at::Tensor>(output_video_mask_.reshape({opts_.net_output_size[0], opts_.net_output_size[1]}));
    output_mask_pyramid_ptr_ = std::make_shared<std::vector<at::Tensor>>();
    df::GenerateMaskPyramid(output_video_mask_, out_cam_pyr.Levels(), output_mask_pyramid_ptr_);
    // TODO: hard-coded erode size
    cv::Mat erode_element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::erode(output_mask, feat_mask, erode_element, cv::Point(-1, -1), 6);

    feat_video_mask_ = torch::from_blob(static_cast<unsigned char *>(feat_mask.data), {1, 1, opts_.net_output_size[0], opts_.net_output_size[1]},
                                        torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8))
                           .to(torch::kFloat32)
                           .to(torch::kCUDA, cuda_id_)
                           .clone();

    // generate valid 1d locations and homogeneous locations
    // When sampling the locations, we should probably first shrink the output mask to remove the border pixels
    GenerateValidLocations(feat_video_mask_, (*output_camera_pyramid_ptr_)[0], valid_normalized_locations_2d, valid_locations_1d, valid_locations_homo);

    // GenerateValidLocations(output_video_mask_, (*output_camera_pyramid_ptr_)[0], valid_normalized_locations_2d, valid_locations_1d, valid_locations_homo);

    valid_locations_1d_ = valid_locations_1d.to(torch::kLong);
    valid_locations_homo_ = valid_locations_homo;

    at::Tensor level_offsets = torch::ones({static_cast<long>(output_camera_pyramid_ptr_->Levels())},
                                           torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, cuda_id_));
    int offset = 0;
    for (int i = 0; i < static_cast<int>(output_mask_pyramid_ptr_->size()); ++i)
    {
      level_offsets.index_put_({i}, offset);
      offset += (*output_mask_pyramid_ptr_)[i].size(2) * (*output_mask_pyramid_ptr_)[i].size(3);
    }

    level_offsets_ptr_ = std::make_shared<at::Tensor>(level_offsets);

    GenerateCheckerboard(checkerboard_, opts_.net_output_size);

    // TEASER++ related
    teaser_params_.noise_bound_multiplier = opts_.teaser_noise_bound_multiplier;
    teaser_params_.max_clique_time_limit = opts_.teaser_max_clique_time_limit;
    teaser_params_.kcore_heuristic_threshold = opts_.teaser_kcore_heuristic_threshold;
    teaser_params_.rotation_max_iterations = opts_.teaser_rotation_max_iterations;
    teaser_params_.rotation_cost_threshold = opts_.teaser_rotation_cost_threshold;
    teaser_params_.rotation_gnc_factor = opts_.teaser_rotation_gnc_factor;

    std::string rotation_estimation_algorithm = boost::algorithm::to_lower_copy(opts_.teaser_rotation_estimation_algorithm);
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
      LOG(FATAL) << "[Mapper<Scalar, CS>::Mapper] rotation_estimation_algorithm type not supported: " << rotation_estimation_algorithm;
    }

    std::string rotation_tim_graph = boost::algorithm::to_lower_copy(opts_.teaser_rotation_tim_graph);
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
      LOG(FATAL) << "[Mapper<Scalar, CS>::Mapper] rotation_tim_graph type not supported: " << rotation_tim_graph;
    }

    std::string inlier_selection_mode = boost::algorithm::to_lower_copy(opts_.teaser_inlier_selection_mode);
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
      LOG(FATAL) << "[Mapper<Scalar, CS>::Mapper] inlier_selection_mode type not supported: " << inlier_selection_mode;
    }

    new_match_imgs_ = false;

    global_loop_count_ = 0;
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void Mapper<Scalar, CS>::InitOneFrame(double timestamp, const cv::Mat &color)
  {
    torch::NoGradGuard no_grad;
    Reset();
    auto kf = BuildKeyframe(timestamp, color, SE3T{});

    {
      std::unique_lock<std::shared_mutex> lock(new_keyframe_mutex_);
      map_->AddKeyframe(kf);
    }

    {
      std::unique_lock<std::shared_mutex> lock(new_factor_mutex_);

      // To make the magnitude of trajectory more consistent throughout different videos and depth networks, re-scale the depth map of the first keyframe.
      const at::Tensor valid_locations_1d = kf->valid_locations_1d;
      const at::Tensor median_dpt = torch::median(kf->dpt_map.reshape({-1}).index({valid_locations_1d}));
      VLOG(2) << "[Mapper<Scalar, CS>::InitOneFrame] median depth of the initial keyframe will be normalized - " << median_dpt.item<Scalar>();
      kf->dpt_map = kf->dpt_map / median_dpt;
      kf->dpt_scale = kf->dpt_scale / median_dpt.item<Scalar>();
      // This line will initialize all optimization variables of a keyframe.
      // Later on it will be passed to ISAM2 graph for optimization with optional prior factor included in this work

      // Fix the pose and scale of the first keyframe as the reference
      work_manager_->AddWork<work::InitVariables<Scalar, CS>>(kf, 1.0 / opts_.init_pose_prior_weight);
      work_manager_->AddWork<work::OptimizeScale<Scalar>>(kf, opts_.factor_iters, kf->dpt_scale,
                                                          opts_.init_scale_prior_weight, false);

      const gtsam::Vector zero_code = gtsam::Vector::Zero(CS);
      work_manager_->AddWork<work::OptimizeCode<Scalar, CS>>(kf, opts_.factor_iters, zero_code,
                                                             opts_.code_factor_weight, false);
    }
  }

  // /* ************************************************************************* */
  // template <typename Scalar, int CS>
  // std::vector<typename Mapper<Scalar, CS>::FrameId> Mapper<Scalar, CS>::NonmarginalizedFrames()
  // {
  //   torch::NoGradGuard no_grad;
  //   std::vector<FrameId> nonmrg_frames;
  //   for (auto &id : map_->frames.Ids())
  //     if (!map_->frames.Get(id)->marginalized)
  //     {
  //       nonmrg_frames.push_back(id);
  //     }
  //   return nonmrg_frames;
  // }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  std::unordered_map<typename Mapper<Scalar, CS>::KeyframeId, bool> Mapper<Scalar, CS>::KeyframeRelinearization()
  {
    torch::NoGradGuard no_grad;
    std::unordered_map<KeyframeId, bool> info;
    auto &var_status = (*isam_res_.detail).variableStatus;

    std::vector<KeyframeId> kf_ids;
    {
      kf_ids = map_->keyframes.Ids();
    }

    for (auto &id : kf_ids)
    {
      auto pkey = PoseKey(id);
      auto ckey = CodeKey(id);
      auto skey = ScaleKey(id);
      info[id] = var_status[pkey].isRelinearized || var_status[ckey].isRelinearized || var_status[skey].isRelinearized;
    }
    return info;
  }
  template <typename Scalar, int CS>
  void Mapper<Scalar, CS>::CorrectDepthScale(const KeyframePtr &kf_to_scale, const KeyframePtr &reference_kf)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    namespace F = torch::nn::functional;

    PoseT relpose10;
    PoseT pose0 = reference_kf->pose_wk;
    PoseT pose1 = kf_to_scale->pose_wk;
    RelativePose(pose1, pose0, relpose10);

    at::Tensor rotation, translation;
    SophusSE3ToTensor(relpose10, rotation, translation, reference_kf->dpt_map_bias.options());

    auto camera = (*(reference_kf->camera_pyramid_ptr))[0];

    // 1 x 1 x H x W
    const at::Tensor valid_mask_1 = (*(kf_to_scale->video_mask_ptr)).reshape({1, 1, static_cast<long>(camera.height()), static_cast<long>(camera.width())});
    // N x 3
    const at::Tensor valid_locations_homo_0 = reference_kf->valid_locations_homo;
    // N
    const at::Tensor valid_locations_1d_0 = reference_kf->valid_locations_1d;

    at::Tensor valid_dpts_0;
    {
      std::shared_lock<std::shared_mutex> lock(reference_kf->mutex);
      // N
      valid_dpts_0 = reference_kf->dpt_map.reshape({-1}).index({valid_locations_1d_0});
    }

    // N x 3
    const at::Tensor valid_rotated_locations_homo = torch::matmul(rotation, valid_locations_homo_0.permute({1, 0})).permute({1, 0});
    // N x 3
    const at::Tensor valid_locations_3d_in_1 = valid_dpts_0.reshape({-1, 1}) * valid_rotated_locations_homo + translation.reshape({1, 3});
    at::Tensor pos_depth_mask_1;
    pos_depth_mask_1 = valid_locations_3d_in_1.index({Slice(), Slice(2, 3)}) > opts_.dpt_eps;
    // Locations3DNegDepthClamp(valid_locations_3d_in_1, opts_.dpt_eps, pos_depth_mask_1, clamped_valid_locations_3d_in_1); clamped_valid_locations_3d_in_1,

    // N x 3
    const at::Tensor valid_locations_homo_in_1 = valid_locations_3d_in_1 / valid_locations_3d_in_1.index({Slice(), Slice(2, 3)});
    // N x 1
    const at::Tensor valid_locations_2d_x = valid_locations_homo_in_1.index({Slice(), Slice(0, 1)}) * camera.fx() + camera.u0();
    const at::Tensor valid_locations_2d_y = valid_locations_homo_in_1.index({Slice(), Slice(1, 2)}) * camera.fy() + camera.v0();
    // 1 x 1 x N x 2
    const at::Tensor valid_locations_2d_in_1 = torch::cat({(valid_locations_2d_x + 0.5f) * (2.0f / camera.width()) - 1.0,
                                                           (valid_locations_2d_y + 0.5f) * (2.0f / camera.height()) - 1.0},
                                                          1)
                                                   .reshape({1, 1, -1, 2});
    // N
    const at::Tensor valid_dpts_1 = F::grid_sample(kf_to_scale->dpt_map_bias.reshape({1, 1,
                                                                                      static_cast<long>(camera.height()), static_cast<long>(camera.width())}),
                                                   valid_locations_2d_in_1,
                                                   F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(false))
                                        .reshape({-1});
    const at::Tensor valid_valid_mask_1 = F::grid_sample(valid_mask_1, valid_locations_2d_in_1,
                                                         F::GridSampleFuncOptions().mode(torch::kNearest).padding_mode(torch::kZeros).align_corners(false))
                                              .reshape({-1}) *
                                          pos_depth_mask_1;

    at::Tensor ratios = ((valid_locations_3d_in_1.index({Slice(), 2}) * valid_valid_mask_1) / valid_dpts_1).reshape({-1});
    at::Tensor ratio = torch::median(ratios.index({torch::nonzero(valid_valid_mask_1 > 0.5).reshape({-1})}));
    // at::Tensor ratio = torch::mean(ratios.index({torch::nonzero(valid_valid_mask_1 > 0.5).reshape({-1})}));

    {
      std::unique_lock<std::shared_mutex> lock(kf_to_scale->mutex);
      kf_to_scale->dpt_scale = ratio.item<Scalar>();
      UpdateDepth(kf_to_scale->dpt_map_bias, kf_to_scale->dpt_jac_code, kf_to_scale->code, kf_to_scale->dpt_scale, kf_to_scale->dpt_map);
    }

    VLOG(2) << "[Mapper<Scalar, CS>::CorrectDepthScale] depth ratio for keyframe " << kf_to_scale->id << " against reference keyframe " << reference_kf->id << " : " << kf_to_scale->dpt_scale;

    return;
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  typename Mapper<Scalar, CS>::KeyframePtr
  Mapper<Scalar, CS>::EnqueueKeyframe(const FramePtr &frame_ptr,
                                      const std::vector<FrameId> &conns)
  {
    torch::NoGradGuard no_grad;
    // conns store ids of all previous keyframes that will be connected to the new keyframe in the factor graph
    // add to map
    const std::shared_ptr<df::Keyframe<Scalar>> kf = BuildKeyframe(frame_ptr);
    kf->temporal_connections = conns;

    {
      std::unique_lock<std::shared_mutex> lock(new_keyframe_mutex_);
      map_->AddKeyframe(kf);
      // What if we dont scale the depth at all
      CorrectDepthScale(kf, map_->keyframes.Get(kf->temporal_connections.front()));
    }

    {
      std::unique_lock<std::shared_mutex> lock(new_factor_mutex_);

      work_manager_->AddWork<work::InitVariables<Scalar, CS>>(kf);

      const gtsam::Vector zero_code = gtsam::Vector::Zero(CS);
      work_manager_->AddWork<work::OptimizeCode<Scalar, CS>>(kf, opts_.factor_iters, zero_code,
                                                             opts_.code_factor_weight, false);

      for (auto id : conns)
      {
        auto back_kf = map_->keyframes.Get(id);

        work::WorkManager::WorkPtr ptr;

        // add photometric error both ways
        if (opts_.use_photometric)
        {
          work_manager_->AddWork<work::OptimizePhoto<Scalar, CS>>(kf, back_kf, opts_.factor_iters, *output_camera_pyramid_ptr_,
                                                                  opts_.photo_factor_weights, opts_.dpt_eps, false);
          work_manager_->AddWork<work::OptimizePhoto<Scalar, CS>>(back_kf, kf, opts_.factor_iters, *output_camera_pyramid_ptr_,
                                                                  opts_.photo_factor_weights, opts_.dpt_eps, false);
        }

        // add reprojection error both ways
        if (opts_.use_reprojection)
        {
          Scalar rep_loss_param = opts_.reproj_loss_param_factor * std::pow((*output_camera_pyramid_ptr_)[0].width(), 2);
          work_manager_->AddWork<work::OptimizeRep<Scalar, CS>>(kf, back_kf, opts_.factor_iters, (*output_camera_pyramid_ptr_)[0],
                                                                opts_.desc_num_keypoints, opts_.desc_cyc_consis_thresh, rep_loss_param,
                                                                opts_.reproj_factor_weight, opts_.dpt_eps, false, teaser_params_);
          work_manager_->AddWork<work::OptimizeRep<Scalar, CS>>(back_kf, kf, opts_.factor_iters, (*output_camera_pyramid_ptr_)[0],
                                                                opts_.desc_num_keypoints, opts_.desc_cyc_consis_thresh, rep_loss_param,
                                                                opts_.reproj_factor_weight, opts_.dpt_eps, false, teaser_params_);
        }

        // add geometric error to be optimized after one of the photometric errors
        if (opts_.use_geometric)
        {
          const Scalar geo_loss_param = opts_.geo_loss_param_factor * kf->avg_squared_dpt_bias;
          work_manager_->AddWork<work::OptimizeGeo<Scalar, CS>>(kf, back_kf, opts_.factor_iters, (*output_camera_pyramid_ptr_)[0],
                                                                opts_.geo_factor_weight, geo_loss_param, opts_.dpt_eps, false);
          work_manager_->AddWork<work::OptimizeGeo<Scalar, CS>>(back_kf, kf, opts_.factor_iters, (*output_camera_pyramid_ptr_)[0],
                                                                opts_.geo_factor_weight, geo_loss_param, opts_.dpt_eps, false);
        }
      }
    }

    // Directional links
    {
      std::unique_lock<std::shared_mutex> lock(new_links_mutex_);
      for (auto id : conns)
      {
        map_->keyframes.AddLink(kf->id, id);
        map_->keyframes.AddLink(id, kf->id);
      }
    }

    return kf;
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void Mapper<Scalar, CS>::EnqueueLink(FrameId id0, FrameId id1, bool photo, bool match_geom, bool geo, bool global_loop)
  {
    torch::NoGradGuard no_grad;
    auto kf0 = map_->keyframes.Get(id0);
    auto kf1 = map_->keyframes.Get(id1);

    work::WorkManager::WorkPtr ptr;

    {
      std::unique_lock<std::shared_mutex> lock(new_factor_mutex_);
      // add photometric error both ways
      if (photo)
      {
        const double photo_multiplier = 1.0;

        std::vector<float> photo_factor_weights = opts_.photo_factor_weights;
        for (auto &weight : photo_factor_weights)
        {
          weight = weight * photo_multiplier;
        }

        work_manager_->AddWork<work::OptimizePhoto<Scalar, CS>>(kf0, kf1, opts_.factor_iters, *output_camera_pyramid_ptr_,
                                                                photo_factor_weights, opts_.dpt_eps, false, global_loop);
        work_manager_->AddWork<work::OptimizePhoto<Scalar, CS>>(kf1, kf0, opts_.factor_iters, *output_camera_pyramid_ptr_,
                                                                photo_factor_weights, opts_.dpt_eps, false, global_loop);
      }

      // add reprojection error both ways
      if (match_geom)
      {
        const double match_geo_multiplier = 1.0;

        Scalar rep_loss_param = opts_.reproj_loss_param_factor * std::pow((*output_camera_pyramid_ptr_)[0].width(), 2);

        work_manager_->AddWork<work::OptimizeRep<Scalar, CS>>(kf0, kf1, opts_.factor_iters, (*output_camera_pyramid_ptr_)[0],
                                                              opts_.desc_num_keypoints, opts_.desc_cyc_consis_thresh, rep_loss_param,
                                                              match_geo_multiplier * opts_.reproj_factor_weight, opts_.dpt_eps, false,
                                                              teaser_params_, global_loop);
        work_manager_->AddWork<work::OptimizeRep<Scalar, CS>>(kf1, kf0, opts_.factor_iters, (*output_camera_pyramid_ptr_)[0],
                                                              opts_.desc_num_keypoints, opts_.desc_cyc_consis_thresh, rep_loss_param,
                                                              match_geo_multiplier * opts_.reproj_factor_weight, opts_.dpt_eps, false,
                                                              teaser_params_, global_loop);
      }

      if (geo)
      {
        const Scalar geo_loss_param = opts_.geo_loss_param_factor * kf0->avg_squared_dpt_bias;
        const double geo_multiplier = 1.0;
        work_manager_->AddWork<work::OptimizeGeo<Scalar, CS>>(kf0, kf1, opts_.factor_iters, (*output_camera_pyramid_ptr_)[0],
                                                              geo_multiplier * opts_.geo_factor_weight, geo_loss_param, opts_.dpt_eps, false);
        work_manager_->AddWork<work::OptimizeGeo<Scalar, CS>>(kf1, kf0, opts_.factor_iters, (*output_camera_pyramid_ptr_)[0],
                                                              geo_multiplier * opts_.geo_factor_weight, geo_loss_param, opts_.dpt_eps, false);
      }
    }

    {
      std::unique_lock<std::shared_mutex> lock(new_links_mutex_);
      map_->keyframes.AddLink(id0, id1);
      map_->keyframes.AddLink(id1, id0);
    }
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void Mapper<Scalar, CS>::Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                                       gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                                       gtsam::Values &var_init,
                                       gtsam::Values &var_update)
  {
    torch::NoGradGuard no_grad;
    work_manager_->Bookkeeping(new_factors, remove_indices, var_init, var_update);
    work_manager_->Update();
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void Mapper<Scalar, CS>::MappingStep(const std::vector<KeyframeId> &visited_kf_ids, const bool force_relinearize)
  {
    torch::NoGradGuard no_grad;

    if (VLOG_IS_ON(3))
    {
      // display keyframes before mapping step
      keyframes_display_ = DisplayKeyframes<Scalar>(map_, 5);
    }

    // determine new factors to add and initialization of new variables
    gtsam::Values var_init;
    gtsam::Values var_update;
    gtsam::NonlinearFactorGraph new_factors;
    gtsam::FastVector<gtsam::FactorIndex> remove_indices;
    gtsam::FastList<gtsam::Key> extra_eliminate_keys;
    gtsam::FastMap<gtsam::Key, int> constrained_keys;

    int global_loop_count = 0;
    // here only newly added factors will be added to new_factors.
    // remove_indices stand for the factor indices in the ISAM2 graph that will be removed
    {
      std::shared_lock<std::shared_mutex> lock(new_factor_mutex_);
      global_loop_count = global_loop_count_;
      Bookkeeping(new_factors, remove_indices, var_init, var_update);
    }

    if (!var_update.empty())
    {
      auto keys = var_update.keys();
      extra_eliminate_keys = gtsam::FastList<gtsam::Key>(keys.begin(), keys.end());
    }

    if (VLOG_IS_ON(3))
    {
      PrintFactors();
      new_factors.print("[Mapper<Scalar, CS>::MappingStep] new_factors: \n");
      var_init.print("[Mapper<Scalar, CS>::MappingStep] new_vars: \n");

      if (!remove_indices.empty())
      {
        VLOG(3) << "[Mapper<Scalar, CS>::MappingStep] remove indices:";
        std::stringstream remove_indices_str;
        for (auto id : remove_indices)
        {
          remove_indices_str << " " << id;
        }
        VLOG(3) << remove_indices_str.str();
      }

      if (!var_update.empty())
      {
        var_update.print("[Mapper<Scalar, CS>::MappingStep] updated variables: \n");
      }
    }

    // for (long i = 0; i < static_cast<long>(visited_kf_ids.size()); ++i)
    // {
    //   const KeyframeId kf_id = visited_kf_ids[i];
    //   constrained_keys[PoseKey(kf_id)] = i + 1;
    //   constrained_keys[CodeKey(kf_id)] = i + 1;
    //   constrained_keys[ScaleKey(kf_id)] = i + 1;
    // }

    /*
    cmember is an optional vector of length n. It defines the constraints on the column ordering. 
    If cmember(j) = c, then column j is in constraint set c (c must be in the range 1 to n). 
    In the output permutation p, all columns in set 1 appear first, followed by all columns in set 2, 
    and so on. cmember = ones (1,n) if not present or empty. ccolamd (S, [], 1 : n) returns 1 : n
    */

    tic("[Mapper<Scalar, CS>::MappingStep] isam graph update");
    // WARNING: XT if a factor in new_factors is an empty one, segmentation fault will be called here!
    if (var_update.empty())
    {
      isam_res_ = isam_graph_->update(new_factors, var_init, remove_indices,
                                      boost::none, boost::none, boost::none, force_relinearize);
    }
    else
    {
      // If the var_update is not empty, we shoud call this version of update to manually reinitialize some variables (for now only keyframe poses)
      isam_res_ = isam_graph_->update(new_factors, var_init, var_update, remove_indices,
                                      boost::none, boost::none, extra_eliminate_keys, true); // extra_eliminate_keys constrained_keys

      // Reset reinitialized after isam_graph optimization so that UpdateMap method can continue change their values
      // if equals, that means all global loops have been taken care of
      {
        std::shared_lock<std::shared_mutex> lock(global_loop_mutex_);
        if (global_loop_count == global_loop_count_)
        {
          VLOG(2) << "[Mapper<Scalar, CS>::MappingStep] no more global loops left to be integrated to ISAM2 graph";
          for (const auto &var : var_update)
          {
            if (gtsam::Symbol(var.key).chr() == 'p')
            {
              auto kf = map_->keyframes.Get(gtsam::Symbol(var.key).index());
              kf->reinitialize_count.store(0, std::memory_order_relaxed);
            }
          }
        }
        else
        {
          VLOG(2) << "[Mapper<Scalar, CS>::MappingStep] there are global loops left to be integrated to ISAM2 graph";
        }
      }
    }

    estimate_ = isam_graph_->calculateEstimate();
    auto delta = isam_graph_->getDelta();
    toc("[Mapper<Scalar, CS>::MappingStep] isam graph update");

    VLOG(2) << "[Mapper<Scalar, CS>::MappingStep] factorsRecalculated: " << isam_res_.factorsRecalculated;
    VLOG(2) << "[Mapper<Scalar, CS>::MappingStep] variablesReeliminated: " << isam_res_.variablesReeliminated;
    VLOG(1) << "[Mapper<Scalar, CS>::MappingStep] variablesRelinearized: " << isam_res_.variablesRelinearized;

    {
      if (!visited_kf_ids.empty())
      {
        UpdateMap(estimate_, delta, visited_kf_ids.back());
      }
      else
      {
        UpdateMap(estimate_, delta);
      }
    }

    // distribute indices of new factor to work items that they originated from
    work_manager_->DistributeIndices(isam_res_.newFactorsIndices);

    // display all factors if we've added any this step
    if (VLOG_IS_ON(4) && !new_factors.empty())
    {
      std::string out_dir = (!opts_.log_dir.empty() ? opts_.log_dir : "~") + "/factor_graph/";
      VLOG(4) << "[Mapper<Scalar, CS>::MappingStep] graph saved to " << out_dir;
      df::CreateDirIfNotExists(out_dir);
      SaveGraphs(out_dir, "isam_graph");
      PrintFactors();
    }

    if (VLOG_IS_ON(3))
    {
      work_manager_->PrintWork();
    }
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void Mapper<Scalar, CS>::Reset()
  {
    torch::NoGradGuard no_grad;
    map_->Clear();
    estimate_.clear();
    isam_graph_ = std::make_unique<gtsam::ISAM2>(opts_.isam_params);
    work_manager_->Clear();

    new_match_imgs_ = false;
    match_imgs_.clear();
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void Mapper<Scalar, CS>::SaveGraphs(const std::string &save_dir, const std::string &prefix)
  {
    torch::NoGradGuard no_grad;
    // Prune the unsafe graph of null pointers to factors
    // If we dont do this, functions like NonlinearFactorGraph::SaveGraph crash
    auto graph = isam_graph_->getFactorsUnsafe();
    auto it = std::remove_if(graph.begin(), graph.end(), [](auto &f) -> bool { return f == nullptr; });
    graph.erase(it, graph.end());

    // save graph
    std::string filename = save_dir + prefix + "_factors.dot";
    std::ofstream file(filename);
    if (!file.is_open())
    {
      LOG(FATAL) << "Failed to open file for writing: " << filename;
    }
    graph.saveGraph(file);
    file.close();

    // try to save ISAM2 bayes tree
    isam_graph_->saveGraph(save_dir + prefix + "_tree.dot");
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void Mapper<Scalar, CS>::PrintFactors()
  {
    torch::NoGradGuard no_grad;
    auto graph = isam_graph_->getFactorsUnsafe();
    for (auto factor : graph)
    {
      if (factor)
      {
        auto ptr = factor.get();
        // print all types of factors that we use
        auto photo = dynamic_cast<PhotometricFactor<Scalar, CS> *>(ptr);
        if (photo)
        {
          VLOG(3) << "[Mapper<Scalar, CS>::PrintFactors] " << photo->Name();
        }
        auto geo = dynamic_cast<GeometricFactor<Scalar, CS> *>(ptr);
        if (geo)
        {
          VLOG(3) << "[Mapper<Scalar, CS>::PrintFactors] " << geo->Name();
        }
        auto scale = dynamic_cast<ScaleFactor<Scalar> *>(ptr);
        if (scale)
        {
          VLOG(3) << "[Mapper<Scalar, CS>::PrintFactors] " << scale->Name();
        }
        auto code = dynamic_cast<CodeFactor<Scalar, CS> *>(ptr);
        if (code)
        {
          VLOG(3) << "[Mapper<Scalar, CS>::PrintFactors] " << code->Name();
        }
        auto priorpose = dynamic_cast<gtsam::PriorFactor<Sophus::SE3f> *>(ptr);
        if (priorpose)
        {
          VLOG(3) << "[Mapper<Scalar, CS>::PrintFactors] PosePrior on " << gtsam::DefaultKeyFormatter(priorpose->key());
        }
        auto lcf = dynamic_cast<gtsam::LinearContainerFactor *>(ptr);
        if (lcf)
        {
          VLOG(3) << "[Mapper<Scalar, CS>::PrintFactors] LinearContainerFactor on ";
          gtsam::PrintKeyVector(lcf->keys());
        }
      }
    }
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void Mapper<Scalar, CS>::PrintDebugInfo()
  {
    torch::NoGradGuard no_grad;
    PrintFactors();
    work_manager_->PrintWork();
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void Mapper<Scalar, CS>::DisplayMatches()
  {
    torch::NoGradGuard no_grad;
    if (!match_imgs_.empty() || new_match_imgs_)
    {
      cv::imshow("matches", CreateMosaic(match_imgs_));
      new_match_imgs_ = false;
    }
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  cv::Mat Mapper<Scalar, CS>::DisplayReprojectionErrors(int N)
  {
    torch::NoGradGuard no_grad;
    auto graph = isam_graph_->getFactorsUnsafe();
    auto ids = map_->keyframes.Ids();
    std::vector<cv::Mat> array;

    for (int i = ids.size() - 1; i >= std::max((int)ids.size() - N, 0); --i)
    {
      // for each connection find the reprojection error
      auto kf = map_->keyframes.Get(ids[i]);
      std::vector<FrameId> conns = map_->keyframes.GetConnections(kf->id, true);
      for (auto c : conns)
      {
        if (c > kf->id) // only do back connections
        {
          continue;
        }

        for (auto it = std::make_reverse_iterator(graph.end());
             it != std::make_reverse_iterator(graph.begin());
             ++it)
        {
          auto factor = *it;
          if (!factor)
          {
            continue;
          }
          auto rep = dynamic_cast<ReprojectionFactor<Scalar, CS> *>(factor.get());
          if (!rep)
          {
            continue;
          }

          bool front_conn = rep->GetKeyframe()->id == ids[i] && rep->GetFrame()->id == c;
          bool back_conn = rep->GetKeyframe()->id == c && rep->GetFrame()->id == ids[i];
          if (front_conn || back_conn)
          {
            auto matches_corrs = rep->DrawMatches();
            cv::Mat error = rep->ErrorImage();
            cv::Mat matches = std::get<0>(matches_corrs);
            cv::Mat corrs = std::get<1>(matches_corrs);

            // here it is splitting the match image to two in order to display using the mosaic function
            cv::Mat m1(matches, cv::Range::all(), cv::Range(0, matches.cols / 2));
            cv::Mat m2(matches, cv::Range::all(), cv::Range(matches.cols / 2, matches.cols));
            cv::Mat c1(corrs, cv::Range::all(), cv::Range(0, corrs.cols / 2));
            cv::Mat c2(corrs, cv::Range::all(), cv::Range(corrs.cols / 2, corrs.cols));

            at::Tensor dpt_map;
            {
              std::shared_lock<std::shared_mutex> lock(kf->mutex);
              dpt_map = kf->dpt_map * output_video_mask_;
            }

            const at::Tensor max = torch::max(dpt_map);
            float max_dpt = max.item<float>();
            cv::Mat dpt_display = Tensor2Mat(torch::clamp_min(dpt_map / max_dpt, 0.0));

            array.push_back(m1);
            array.push_back(m2);
            array.push_back(c1);
            array.push_back(c2);
            array.push_back(error);
            array.push_back(apply_colormap(dpt_display));
          }
        }
      }
    }

    if (array.size())
    {
      cv::Mat mosaic;
      const double ratio = 128.0 / array[0].rows;
      cv::resize(CreateMosaic(array, array.size() / 6, 6), mosaic, cv::Size(0, 0), ratio, ratio);
      cv::imshow("reprojection errors", mosaic);
      return mosaic;
    }

    return cv::Mat{};
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void Mapper<Scalar, CS>::SaveReprojectionDebug(std::string rep_dir)
  {
    torch::NoGradGuard no_grad;
    auto graph = isam_graph_->getFactorsUnsafe();
    auto ids = map_->keyframes.Ids();

    for (int i = ids.size() - 1; i >= 0; --i)
    {
      // for each connection find the reprojection error
      auto kf = map_->keyframes.Get(ids[i]);
      std::vector<FrameId> conns = map_->keyframes.GetConnections(kf->id, true);
      for (auto c : conns)
      {
        if (c > kf->id) // only do back connections
        {
          continue;
        }

        for (auto it = std::make_reverse_iterator(graph.end());
             it != std::make_reverse_iterator(graph.begin());
             ++it)
        {
          auto factor = *it;
          if (!factor)
          {
            continue;
          }
          auto rep = dynamic_cast<ReprojectionFactor<Scalar, CS> *>(factor.get());
          if (!rep)
          {
            continue;
          }

          bool front_conn = rep->GetKeyframe()->id == ids[i] && rep->GetFrame()->id == c;
          bool back_conn = rep->GetKeyframe()->id == c && rep->GetFrame()->id == ids[i];
          if (front_conn || back_conn)
          {
            auto matches_corrs = rep->DrawMatches();
            cv::Mat matches = std::get<0>(matches_corrs);
            cv::Mat corrs = std::get<1>(matches_corrs);
            cv::Mat m1(matches, cv::Range::all(), cv::Range(0, matches.cols / 2));
            cv::Mat m2(matches, cv::Range::all(), cv::Range(matches.cols / 2, matches.cols));
            cv::Mat c1(corrs, cv::Range::all(), cv::Range(0, corrs.cols / 2));
            cv::Mat c2(corrs, cv::Range::all(), cv::Range(corrs.cols / 2, corrs.cols));

            std::vector<cv::Mat> array;
            array.push_back(m1);
            array.push_back(m2);
            array.push_back(c1);
            array.push_back(c2);

            std::string ids_str = std::to_string(ids[i]);
            std::string c_str = std::to_string(c);
            std::string name = front_conn ? ids_str + "-" + c_str : c_str + "-" + ids_str;
            cv::Mat mosaic;
            const double ratio = 128.0 / array[0].rows;
            cv::resize(CreateMosaic(array, 1, 4), mosaic, cv::Size(0, 0), ratio, ratio);
            cv::imwrite(rep_dir + "/" + name + ".png", mosaic);
          }
        }
      }
    }
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  cv::Mat Mapper<Scalar, CS>::DisplayMatchGeometryErrors(int N)
  {
    torch::NoGradGuard no_grad;
    auto graph = isam_graph_->getFactorsUnsafe();
    auto ids = map_->keyframes.Ids();
    std::vector<cv::Mat> array;

    for (int i = ids.size() - 1; i >= std::max((int)ids.size() - N, 0); --i)
    {
      // for each connection find the reprojection error
      auto kf = map_->keyframes.Get(ids[i]);
      std::vector<FrameId> conns = map_->keyframes.GetConnections(kf->id, true);
      for (auto c : conns)
      {
        if (c > kf->id) // only do back connections
        {
          continue;
        }

        for (auto it = std::make_reverse_iterator(graph.end());
             it != std::make_reverse_iterator(graph.begin());
             ++it)
        {
          auto factor = *it;
          if (!factor)
          {
            continue;
          }
          auto rep = dynamic_cast<MatchGeometryFactor<Scalar, CS> *>(factor.get());
          if (!rep)
          {
            continue;
          }

          bool front_conn = rep->GetKeyframe(0)->id == ids[i] && rep->GetKeyframe(1)->id == c;
          bool back_conn = rep->GetKeyframe(0)->id == c && rep->GetKeyframe(1)->id == ids[i];
          if (front_conn || back_conn)
          {
            auto matches_corrs = rep->DrawMatches();
            cv::Mat error = rep->ErrorImage();
            cv::Mat matches = std::get<0>(matches_corrs);
            cv::Mat corrs = std::get<1>(matches_corrs);

            // here it is splitting the match image to two in order to display using the mosaic function
            cv::Mat m1(matches, cv::Range::all(), cv::Range(0, matches.cols / 2));
            cv::Mat m2(matches, cv::Range::all(), cv::Range(matches.cols / 2, matches.cols));
            cv::Mat c1(corrs, cv::Range::all(), cv::Range(0, corrs.cols / 2));
            cv::Mat c2(corrs, cv::Range::all(), cv::Range(corrs.cols / 2, corrs.cols));

            at::Tensor dpt_map;
            {
              std::shared_lock<std::shared_mutex> lock(kf->mutex);
              dpt_map = kf->dpt_map * output_video_mask_;
            }

            const at::Tensor max = torch::max(dpt_map);
            float max_dpt = max.item<float>();
            cv::Mat dpt_display = Tensor2Mat(torch::clamp_min(dpt_map / max_dpt, 0.0));

            array.push_back(m1);
            array.push_back(m2);
            array.push_back(c1);
            array.push_back(c2);
            array.push_back(error);
            array.push_back(apply_colormap(dpt_display));
          }
        }
      }
    }

    if (array.size())
    {
      cv::Mat mosaic;
      const double ratio = 128.0 / array[0].rows;
      cv::resize(CreateMosaic(array, array.size() / 6, 6), mosaic, cv::Size(0, 0), ratio, ratio);
      cv::imshow("reprojection errors", mosaic);
      return mosaic;
    }

    return cv::Mat{};
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  cv::Mat Mapper<Scalar, CS>::DisplayPhotometricErrors(int N)
  {
    torch::NoGradGuard no_grad;
    typedef typename MapT::FrameGraphT::LinkT LinkT;

    std::vector<LinkT> links;
    auto ids = map_->keyframes.Ids();
    for (int i = ids.size() - 1; i >= std::max((int)ids.size() - N, 0); --i)
    {
      std::vector<FrameId> conns = map_->keyframes.GetConnections(ids[i], true);
      for (auto c : conns)
      {
        if (c < ids[i]) // show only connections to the older frames
        {
          links.push_back({ids[i], c});
        }
      }
    }
    return DisplayPairs<Scalar>(map_, links, *output_mask_ptr_, (*output_camera_pyramid_ptr_)[0], opts_.dpt_eps, 10, checkerboard_);
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void Mapper<Scalar, CS>::DisplayLoopClosures(std::vector<std::pair<long, long>> &links, int N)
  {
    torch::NoGradGuard no_grad;
    std::vector<cv::Mat> array;
    auto graph = isam_graph_->getFactorsUnsafe();
    for (auto it = std::make_reverse_iterator(graph.end());
         it != std::make_reverse_iterator(graph.begin());
         ++it)
    {
      auto factor = *it;
      if (!factor)
      {
        continue;
      }

      auto ptr = factor.get();
      auto match_geom = dynamic_cast<MatchGeometryFactor<Scalar, CS> *>(ptr);
      if (match_geom)
      {
        bool is_our = false;
        // src and tgt frame involved in this reprojection factor
        int kf0_id = (int)match_geom->GetKeyframe(0)->id;
        int kf1_id = (int)match_geom->GetKeyframe(1)->id;
        for (auto &link : links)
        {
          if ((kf0_id == link.first && kf1_id == link.second) ||
              (kf0_id == link.second && kf1_id == link.first))
          {
            is_our = true;
            break;
          }
        }

        if (is_our)
        {
          auto matches_corrs = match_geom->DrawMatches();
          cv::Mat matches = std::get<0>(matches_corrs);
          cv::Mat corrs = std::get<1>(matches_corrs);
          cv::Mat m1(matches, cv::Range::all(), cv::Range(0, matches.cols / 2));
          cv::Mat m2(matches, cv::Range::all(), cv::Range(matches.cols / 2, matches.cols));
          cv::Mat c1(corrs, cv::Range::all(), cv::Range(0, corrs.cols / 2));
          cv::Mat c2(corrs, cv::Range::all(), cv::Range(corrs.cols / 2, corrs.cols));
          array.push_back(m1);
          array.push_back(m2);
          array.push_back(c1);
          array.push_back(c2);
          N--;
        }

        if (N <= 0)
        {
          break;
        }
      }
    }

    if (array.size())
    {
      cv::Mat mosaic;
      const double ratio = 128.0 / array[0].rows;
      cv::resize(CreateMosaic(array, array.size() / 4, 4), mosaic, cv::Size(0, 0), ratio, ratio);
      cv::imshow("loop closures", mosaic);
    }
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void Mapper<Scalar, CS>::SavePhotometricDebug(std::string save_dir)
  {
    torch::NoGradGuard no_grad;
    typedef typename MapT::FrameGraphT::LinkT LinkT;
    auto ids = map_->keyframes.Ids();
    for (int i = ids.size() - 1; i >= 0; --i)
    {
      std::vector<FrameId> conns = map_->keyframes.GetConnections(ids[i], true);
      for (auto c : conns)
      {
        if (c < ids[i]) // show only back connections
        {
          std::vector<LinkT> links{{ids[i], c}};
          cv::Mat img = DisplayPairs(map_, links, *output_mask_ptr_, (*output_camera_pyramid_ptr_)[0], opts_.dpt_eps, 1, checkerboard_);
          std::string outname = save_dir + "/photo" + std::to_string(ids[i]) + "-" + std::to_string(c) + ".png";
          cv::imwrite(outname, img);
        }
      }
    }
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void Mapper<Scalar, CS>::SaveMatchGeometryDebug(std::string rep_dir)
  {
    torch::NoGradGuard no_grad;
    auto graph = isam_graph_->getFactorsUnsafe();
    auto ids = map_->keyframes.Ids();

    for (int i = ids.size() - 1; i >= 0; --i)
    {
      // for each connection find the reprojection error
      auto kf = map_->keyframes.Get(ids[i]);
      std::vector<FrameId> conns = map_->keyframes.GetConnections(kf->id, true);
      for (auto c : conns)
      {
        if (c > kf->id) // only do back connections
        {
          continue;
        }

        for (auto it = std::make_reverse_iterator(graph.end());
             it != std::make_reverse_iterator(graph.begin());
             ++it)
        {
          auto factor = *it;
          if (!factor)
          {
            continue;
          }
          auto rep = dynamic_cast<MatchGeometryFactor<Scalar, CS> *>(factor.get());
          if (!rep)
          {
            continue;
          }

          bool front_conn = rep->GetKeyframe(0)->id == ids[i] && rep->GetKeyframe(1)->id == c;
          bool back_conn = rep->GetKeyframe(0)->id == c && rep->GetKeyframe(1)->id == ids[i];
          if (front_conn || back_conn)
          {
            auto matches_corrs = rep->DrawMatches();
            cv::Mat matches = std::get<0>(matches_corrs);
            cv::Mat corrs = std::get<1>(matches_corrs);
            cv::Mat m1(matches, cv::Range::all(), cv::Range(0, matches.cols / 2));
            cv::Mat m2(matches, cv::Range::all(), cv::Range(matches.cols / 2, matches.cols));
            cv::Mat c1(corrs, cv::Range::all(), cv::Range(0, corrs.cols / 2));
            cv::Mat c2(corrs, cv::Range::all(), cv::Range(corrs.cols / 2, corrs.cols));

            std::vector<cv::Mat> array;
            array.push_back(m1);
            array.push_back(m2);
            array.push_back(c1);
            array.push_back(c2);

            std::string ids_str = std::to_string(ids[i]);
            std::string c_str = std::to_string(c);
            std::string name = front_conn ? ids_str + "-" + c_str : c_str + "-" + ids_str;
            cv::Mat mosaic;
            const double ratio = 128.0 / array[0].rows;
            cv::resize(CreateMosaic(array, 1, 4), mosaic, cv::Size(0, 0), ratio, ratio);
            cv::imwrite(rep_dir + "/" + name + ".png", mosaic);
          }
        }
      }
    }
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void Mapper<Scalar, CS>::SaveGeometricDebug(std::string save_dir)
  {
  }
  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void Mapper<Scalar, CS>::UpdateMap(const gtsam::Values &vals, const gtsam::VectorValues &delta, boost::optional<const FrameId> curr_kf_id)
  {
    torch::NoGradGuard no_grad;
    // vals here should be the latest values estimated by ISAM2 graph
    // and delta is the update step that reaches the current estimated values
    std::vector<FrameId> changed_kfs;

    std::vector<KeyframeId> kf_ids;
    {
      kf_ids = map_->keyframes.Ids();
    }

    {
      std::unique_lock<std::shared_mutex> lock(new_factor_mutex_);
      // modify each changed keyframe
      // for (const auto &id : changed_kfs)
      for (const auto &id : kf_ids)
      {
        if (delta.find(PoseKey(id)) != delta.end())
        {
          auto kf = map_->keyframes.Get(id);
          {
            std::unique_lock<std::shared_mutex> lock(kf->mutex);
            if (kf->reinitialize_count.load(std::memory_order_relaxed) <= 0) // Only update the pose_wk based on the optimization variables if it will not be reinitialized
            {
              Eigen::Matrix<Scalar, CS, 1> updated_code = vals.at(CodeKey(id)).template cast<gtsam::Vector>().template cast<Scalar>();
              kf->code = df::EigenVectorToTensor(updated_code, kf->dpt_map_bias.options());
              kf->pose_wk = vals.at(PoseKey(id)).template cast<SE3T>();
              VLOG(3) << "[Mapper<Scalar, CS>::UpdateMap] keyframe " << kf->id << " scale changed from " << kf->dpt_scale << " to " << vals.at(ScaleKey(id)).template cast<Scalar>();
              kf->dpt_scale = vals.at(ScaleKey(id)).template cast<Scalar>();
              UpdateDepth(kf->dpt_map_bias, kf->dpt_jac_code, kf->code, kf->dpt_scale, kf->dpt_map);
            }
          }
        }
      }
    }

    // tell the one who has registered the callback function that a newly updated map is available now
    NotifyMapObservers();
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  typename Mapper<Scalar, CS>::FramePtr
  Mapper<Scalar, CS>::BuildFrame(double timestamp, const cv::Mat &color, const SE3T &pose_init)
  {
    using namespace torch::indexing;
    torch::NoGradGuard no_grad;
    cv::Mat resized_input_color, resized_output_color;
    // create a new empty frame
    auto fr = std::make_shared<FrameT>();
    fr->pose_wk = pose_init;
    cv::resize(color, resized_input_color, cv::Size2l(input_video_mask_.size(3), input_video_mask_.size(2)));
    cv::resize(color, resized_output_color, cv::Size2l(output_video_mask_.size(3), output_video_mask_.size(2)));

    fr->dpt_scale = (Scalar)1.0;
    fr->color_img = resized_input_color.clone();
    fr->timestamp = timestamp;
    fr->video_mask_ptr = output_mask_ptr_;
    fr->camera_pyramid_ptr = output_camera_pyramid_ptr_;
    fr->level_offsets_ptr = level_offsets_ptr_;
    fr->feat_video_mask = feat_video_mask_;

    // build color tensor from cv Mat
    const at::Tensor fine_color_tensor = torch::from_blob(static_cast<unsigned char *>(resized_input_color.data),
                                                          {1, input_video_mask_.size(2), input_video_mask_.size(3), 3},
                                                          torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8))
                                             .to(torch::kCUDA, cuda_id_)
                                             .to(torch::kFloat32)
                                             .permute({0, 3, 1, 2})
                                             .clone() /
                                         255.0;

    at::Tensor feat_map;

    feature_network_->GenerateFeatureMaps(fine_color_tensor, input_video_mask_, feat_map, fr->feat_desc);

    GenerateGaussianPyramidWithGrad(feat_map, output_camera_pyramid_ptr_->Levels(),
                                    fr->feat_map_pyramid, fr->feat_map_grad_pyramid);

    // Randomly subsample valid locations for photo and geom factors
    std::vector<long> indices(valid_locations_1d_.size(0));
    std::iota(indices.begin(), indices.end(), 0);
    // shuffle and take the first set of indices as the randomly selected ones
    const long seed = timestamp;
    std::mt19937 g;
    g.seed(seed);
    std::shuffle(indices.begin(), indices.end(), g);
    // Take top several for indexes
    const at::Tensor indexes = torch::from_blob(static_cast<long *>(indices.data()),
                                                {std::min(static_cast<long>(opts_.pho_num_samples), valid_locations_1d_.size(0))},
                                                torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
                                   .to(valid_locations_1d_.device())
                                   .clone();

    fr->sampled_locations_homo = valid_locations_homo_.index({indexes, Slice()});
    fr->sampled_locations_1d = valid_locations_1d_.index({indexes});
    fr->valid_locations_homo = valid_locations_homo_;
    fr->valid_locations_1d = valid_locations_1d_;

    // Initial code will be a zero dpt code
    fr->code = torch::zeros({CS, 1}, torch::TensorOptions().device(torch::kCUDA, cuda_id_).dtype(torch::kFloat32));
    // Generate depth map bias and depth jacobian wrt code
    tic("[Mapper<Scalar, CS>::BuildKeyframe] Generate depth bias and jacobian");
    depth_network_->GenerateDepthBiasAndJacobian(fine_color_tensor, input_video_mask_, fr->dpt_map_bias, fr->dpt_jac_code);
    toc("[Mapper<Scalar, CS>::BuildKeyframe] Generate depth bias and jacobian");

    const at::Tensor sum = (torch::sum(torch::square(fr->dpt_map_bias * (*output_mask_ptr_))) /
                            torch::sum(*output_mask_ptr_));
    fr->avg_squared_dpt_bias = sum.item<Scalar>();
    // fr->scale_ratio_cur_ref = 1.0;
    UpdateDepth(fr->dpt_map_bias, fr->dpt_jac_code, fr->code, fr->dpt_scale, fr->dpt_map);

    return fr;
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  typename Mapper<Scalar, CS>::KeyframePtr
  Mapper<Scalar, CS>::BuildKeyframe(const FramePtr &frame_ptr)
  {
    using namespace torch::indexing;
    namespace F = torch::nn::functional;
    torch::NoGradGuard no_grad;
    // create a new empty kf
    auto kf = std::make_shared<df::Keyframe<float>>();
    kf->pose_wk = frame_ptr->pose_wk;
    kf->timestamp = frame_ptr->timestamp;
    kf->color_img = frame_ptr->color_img;
    kf->video_mask_ptr = frame_ptr->video_mask_ptr;
    kf->feat_video_mask = frame_ptr->feat_video_mask;
    kf->camera_pyramid_ptr = frame_ptr->camera_pyramid_ptr;
    kf->feat_desc = frame_ptr->feat_desc;
    kf->feat_map_pyramid = frame_ptr->feat_map_pyramid;
    kf->feat_map_grad_pyramid = frame_ptr->feat_map_grad_pyramid;
    kf->level_offsets_ptr = frame_ptr->level_offsets_ptr;
    kf->dpt_map_bias = frame_ptr->dpt_map_bias;
    kf->dpt_jac_code = frame_ptr->dpt_jac_code;
    kf->code = frame_ptr->code;
    kf->dpt_scale = frame_ptr->dpt_scale;
    kf->dpt_map = frame_ptr->dpt_map;
    kf->avg_squared_dpt_bias = frame_ptr->avg_squared_dpt_bias;

    // kf->scale_ratio_cur_ref = frame_ptr->scale_ratio_cur_ref;

    kf->sampled_locations_homo = frame_ptr->sampled_locations_homo;
    kf->sampled_locations_1d = frame_ptr->sampled_locations_1d;
    kf->valid_locations_1d = frame_ptr->valid_locations_1d;
    kf->valid_locations_homo = frame_ptr->valid_locations_homo;

    kf->reinitialize_count = frame_ptr->reinitialize_count.load(std::memory_order_relaxed);

    return kf;
  }

  template <typename Scalar, int CS>
  bool Mapper<Scalar, CS>::HasWork()
  {
    return !work_manager_->Empty();
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  typename Mapper<Scalar, CS>::KeyframePtr
  Mapper<Scalar, CS>::BuildKeyframe(double timestamp, const cv::Mat &color, const SE3T &pose_init)
  {
    using namespace torch::indexing;
    namespace F = torch::nn::functional;
    torch::NoGradGuard no_grad;
    // create a new empty kf
    auto kf = std::make_shared<df::Keyframe<float>>();

    // color_img in kf and fr has the fine-resolution input size
    kf->pose_wk = pose_init;
    // resize color image to the required one
    cv::Mat resized_input_color, resized_output_color;
    cv::resize(color, resized_input_color, cv::Size2l(input_video_mask_.size(3), input_video_mask_.size(2)));
    cv::resize(color, resized_output_color, cv::Size2l(output_video_mask_.size(3), output_video_mask_.size(2)));

    kf->dpt_scale = (Scalar)1.0;
    kf->color_img = resized_input_color.clone();
    kf->timestamp = timestamp;
    // assign the pointer of sampled locations to keyframes
    // because all frames should have the same set of valid locations within video mask (video mask remains the same for all frames)

    // Randomly subsample valid locations for photo and geom factors
    std::vector<long> indices(valid_locations_1d_.size(0));
    std::iota(indices.begin(), indices.end(), 0);
    // shuffle and take the first set of indices as the randomly selected ones
    const long seed = timestamp;
    std::mt19937 g;
    g.seed(seed);
    std::shuffle(indices.begin(), indices.end(), g);
    // Take top several for indexes
    const at::Tensor indexes = torch::from_blob(static_cast<long *>(indices.data()),
                                                {std::min(static_cast<long>(opts_.pho_num_samples), valid_locations_1d_.size(0))},
                                                torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
                                   .to(valid_locations_1d_.device())
                                   .clone();

    kf->sampled_locations_homo = valid_locations_homo_.index({indexes, Slice()});
    kf->sampled_locations_1d = valid_locations_1d_.index({indexes});
    kf->valid_locations_homo = valid_locations_homo_;
    kf->valid_locations_1d = valid_locations_1d_;
    kf->video_mask_ptr = output_mask_ptr_;
    kf->camera_pyramid_ptr = output_camera_pyramid_ptr_;
    kf->level_offsets_ptr = level_offsets_ptr_;
    kf->feat_video_mask = feat_video_mask_;

    // build torch tensor from opencv Mat
    const at::Tensor fine_color_tensor = torch::from_blob(static_cast<unsigned char *>(resized_input_color.data),
                                                          {1, input_video_mask_.size(2), input_video_mask_.size(3), 3},
                                                          torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8))
                                             .to(torch::kCUDA, cuda_id_)
                                             .to(torch::kFloat32)
                                             .permute({0, 3, 1, 2})
                                             .clone() /
                                         255.0;


    // Generate feature map and feature descriptor
    at::Tensor feat_map;

    feature_network_->GenerateFeatureMaps(fine_color_tensor, input_video_mask_, feat_map, kf->feat_desc);

    GenerateGaussianPyramidWithGrad(feat_map, output_camera_pyramid_ptr_->Levels(),
                                    kf->feat_map_pyramid, kf->feat_map_grad_pyramid);

    // Initial code will be a zero dpt code
    kf->code = torch::zeros({CS, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, cuda_id_));
    // Generate depth map bias and depth jacobian wrt code
    depth_network_->GenerateDepthBiasAndJacobian(fine_color_tensor, input_video_mask_, kf->dpt_map_bias, kf->dpt_jac_code);

    UpdateDepth(kf->dpt_map_bias, kf->dpt_jac_code, kf->code, kf->dpt_scale, kf->dpt_map);

    const at::Tensor sum = (torch::sum(torch::square(kf->dpt_map_bias * (*output_mask_ptr_))) /
                            torch::sum(*output_mask_ptr_));
    kf->avg_squared_dpt_bias = sum.item<Scalar>();
    // kf->scale_ratio_cur_ref = 1.0;

    return kf;
  }

  template <typename Scalar, int CS>
  void Mapper<Scalar, CS>::GenerateGaussianPyramidWithGrad(const at::Tensor feat_map,
                                                           const int &num_levels,
                                                           at::Tensor &feat_map_pyramid,
                                                           at::Tensor &feat_map_grad_pyramid)
  {
    torch::NoGradGuard no_grad;
    namespace F = torch::nn::functional;

    const long feat_channel = feat_map.size(1);
    long height = feat_map.size(2);
    long width = feat_map.size(3);

    at::Tensor curr_feat_map = feat_map.reshape({feat_channel, 1, height, width});
    at::Tensor raw_feat_map, raw_mask, curr_feat_map_grad;

    ComputeSpatialGrad(curr_feat_map.reshape({1, feat_channel, height, width}), curr_feat_map_grad);

    std::vector<at::Tensor> feats_vec, feat_grads_vec;
    feats_vec.push_back(curr_feat_map.reshape({feat_channel, height * width}));
    feat_grads_vec.push_back(curr_feat_map_grad.reshape({2, feat_channel, height * width}));

    at::Tensor curr_valid_mask;
    for (int i = 0; i < num_levels - 1; ++i)
    {
      curr_valid_mask = (*output_mask_pyramid_ptr_)[i];
      raw_feat_map = F::conv2d(curr_feat_map * curr_valid_mask, gauss_kernel_, gauss_conv_options_);
      raw_mask = F::conv2d(curr_valid_mask, gauss_kernel_, gauss_conv_options_);
      height = raw_feat_map.size(2);
      width = raw_feat_map.size(3);
      curr_feat_map = raw_feat_map / (raw_mask + 1.0e-8);

      ComputeSpatialGrad(curr_feat_map.reshape({1, feat_channel, height, width}), curr_feat_map_grad);
      feats_vec.push_back(curr_feat_map.reshape({feat_channel, height * width}));
      feat_grads_vec.push_back(curr_feat_map_grad.reshape({2, feat_channel, height * width}));
    }
    // C_feat x (N0 + N1 + ...)
    feat_map_pyramid = torch::cat(feats_vec, 1).clone();
    // 2 x C_feat x (N0 + N1 + ...)
    feat_map_grad_pyramid = torch::cat(feat_grads_vec, 2).clone();

    return;
  }


  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void Mapper<Scalar, CS>::NotifyMapObservers()
  {
    if (map_callback_)
    {
      map_callback_(map_);
    }
  }

  /* ************************************************************************* */
  // explicit instantiation
  template class Mapper<float, DF_CODE_SIZE>;

} // namespace df
