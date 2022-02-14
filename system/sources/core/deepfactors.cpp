
#include "deepfactors.h"
#include "rel_pose_factor.h"
#include "rel_pose_scale_factor.h"

#include <gtsam/slam/PriorFactor.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

namespace df
{
  /* ************************************************************************* */
  template <typename Scalar, int CS>
  DeepFactors<Scalar, CS>::DeepFactors()
      : force_keyframe_(false), bootstrapped_(false),
        curr_kf_(0), enable_loop_(false), quit_loop_(false),
        enable_mapping_(false), quit_mapping_(false)
  {
    pose_kc_ = SE3T();
    depth_network_ptr_ = nullptr;
    feature_network_ptr_ = nullptr;
    tracker_ = nullptr;
    mapper_ = nullptr;
    loop_detector_ = nullptr;
    live_frame_ptr_ = nullptr;
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  DeepFactors<Scalar, CS>::~DeepFactors()
  {
    VLOG(2) << "[DeepFactors<Scalar, CS>::~DeepFactors] deconstructor called";
    enable_mapping_ = false;
    quit_mapping_ = true;
    enable_loop_ = false;
    quit_loop_ = true;

    if (mapping_thread_ != 0)
    {
      pthread_join(mapping_thread_, NULL);
      mapping_thread_ = 0;
    }

    if (global_loop_detect_thread_ != 0)
    {
      pthread_join(global_loop_detect_thread_, NULL);
      global_loop_detect_thread_ = 0;
    }

    if (local_loop_detect_thread_ != 0)
    {
      pthread_join(local_loop_detect_thread_, NULL);
      local_loop_detect_thread_ = 0;
    }
  }

  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::JoinMappingThreads()
  {
    VLOG(2) << "[DeepFactors<Scalar, CS>::JoinMappingThreads] join backend mapping thread";
    enable_mapping_ = false;
    quit_mapping_ = true;
    pthread_join(mapping_thread_, NULL);
    mapping_thread_ = 0;
  }

  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::JoinLoopThreads()
  {
    VLOG(2) << "[DeepFactors<Scalar, CS>::JoinLoopThreads] join backend loop closure thread";
    enable_loop_ = false;
    quit_loop_ = true;
    pthread_join(global_loop_detect_thread_, NULL);
    pthread_join(local_loop_detect_thread_, NULL);
    global_loop_detect_thread_ = 0;
    local_loop_detect_thread_ = 0;
  }

  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::LoopClosurePoseScaleEstimate(
      const std::map<std::pair<KeyframeId, KeyframeId>, std::pair<Scalar, Scalar>> &pre_global_loops,
      const std::vector<std::tuple<KeyframeId, KeyframeId, SE3T, Scalar, Scalar>> &new_global_loop_vec)
  {
    // The global loop contains keyframe Id pair and the corresponding matching 2D locations.
    // Graph optimization with LoopMG on depth scale and pose
    gtsam::ISAM2Params isam_params;
    isam_params.optimizationParams =
        gtsam::ISAM2DoglegParams(1.0, opts_.isam_wildfire_threshold, gtsam::DoglegOptimizerImpl::ONE_STEP_PER_ITERATION, false);
    isam_params.enableDetailedResults = false;
    isam_params.enablePartialRelinearizationCheck = false;
    isam_params.factorization = gtsam::ISAM2Params::CHOLESKY;
    isam_params.cacheLinearizedFactors = true;
    isam_params.findUnusedFactorSlots = false;
    isam_params.relinearizeSkip = 1;
    gtsam::FastMap<char, gtsam::Vector> thresholds;
    thresholds['p'] = gtsam::Vector::Ones(6) * opts_.loop_pose_linearize_threshold;
    thresholds['s'] = gtsam::Vector::Ones(1) * opts_.loop_scale_linearize_threshold;
    isam_params.relinearizeThreshold = thresholds;

    gtsam::ISAM2 pose_scale_graph(isam_params);
    gtsam::Values var_init;
    gtsam::NonlinearFactorGraph new_factors;
    gtsam::FastVector<gtsam::FactorIndex> remove_indices;

    // Initialize all poses as optimization variables
    std::set<KeyframeId> keyframe_id_set;
    std::vector<std::pair<KeyframeId, KeyframeId>> links;
    {
      std::shared_lock<std::shared_mutex> lock(mapper_->new_links_mutex_);
      links = mapper_->GetMap()->keyframes.GetLinks();
    }

    // Add the zero prior for the first pose to anchor all poses
    gtsam::Vector prior_sigmas = gtsam::Vector::Constant(6, 1, 1.0e-4);
    const auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas);
    const auto init_kf = mapper_->GetMap()->keyframes.Get(1);
    new_factors.emplace_shared<gtsam::PriorFactor<SE3T>>(PoseKey(init_kf->id), init_kf->pose_wk, prior_noise);
    new_factors.emplace_shared<ScaleFactor<Scalar>>(init_kf, ScaleKey(init_kf->id), init_kf->dpt_scale, 100.0);

    SE3T tgt_relpose10;

    std::unordered_map<KeyframeId, SE3T> pose_map;
    std::unordered_map<KeyframeId, Scalar> scale_map;
    // Build pose factors for local links first
    for (auto const &link : links)
    {
      if (link.first < link.second)
      {
        auto id_pair = std::make_pair(link.first, link.second);
        if (pre_global_loops.find(id_pair) == pre_global_loops.end()) // If the link is not a global loop one
        {
          keyframe_id_set.insert(link.first);
          keyframe_id_set.insert(link.second);

          const auto kf0 = mapper_->GetMap()->keyframes.Get(link.first);
          const auto kf1 = mapper_->GetMap()->keyframes.Get(link.second);
          SE3T kf0_pose, kf1_pose;
          Scalar kf0_scale, kf1_scale;

          if (pose_map.find(link.first) == pose_map.end())
          {
            std::shared_lock<std::shared_mutex> lock(kf0->mutex);
            kf0_pose = kf0->pose_wk;
            pose_map[link.first] = kf0_pose;
            scale_map[link.first] = kf0->dpt_scale;
          }

          kf0_pose = pose_map[link.first];
          kf0_scale = scale_map[link.first];

          if (pose_map.find(link.second) == pose_map.end())
          {
            std::shared_lock<std::shared_mutex> lock(kf1->mutex);
            kf1_pose = kf1->pose_wk;
            pose_map[link.second] = kf1_pose;
            scale_map[link.second] = kf1->dpt_scale;
          }

          kf1_pose = pose_map[link.second];
          kf1_scale = scale_map[link.second];

          tgt_relpose10 = kf1_pose.inverse() * kf0_pose;

          new_factors.emplace_shared<RelPoseScaleFactor<Scalar>>(
              kf0, kf1,
              PoseKey(kf0->id), PoseKey(kf1->id),
              ScaleKey(kf0->id), ScaleKey(kf1->id),
              tgt_relpose10, kf0_scale, kf1_scale,
              opts_.pose_graph_local_link_weight,
              opts_.pose_graph_rot_weight,
              opts_.pose_graph_scale_weight);

          new_factors.emplace_shared<RelPoseScaleFactor<Scalar>>(
              kf1, kf0,
              PoseKey(kf1->id), PoseKey(kf0->id),
              ScaleKey(kf1->id), ScaleKey(kf0->id),
              tgt_relpose10.inverse(), kf1_scale, kf0_scale,
              opts_.pose_graph_local_link_weight,
              opts_.pose_graph_rot_weight,
              opts_.pose_graph_scale_weight);
        }
      }
    }

    KeyframeId kf0_id, kf1_id;
    for (auto const &loop_pair : pre_global_loops)
    {
      kf0_id = std::get<0>(loop_pair.first);
      kf1_id = std::get<1>(loop_pair.first);

      keyframe_id_set.insert(kf0_id);
      keyframe_id_set.insert(kf1_id);

      const auto kf0 = mapper_->GetMap()->keyframes.Get(kf0_id);
      const auto kf1 = mapper_->GetMap()->keyframes.Get(kf1_id);
      SE3T kf0_pose, kf1_pose;
      Scalar kf0_scale, kf1_scale;
      {
        std::shared_lock<std::shared_mutex> lock(kf0->mutex);
        kf0_pose = kf0->pose_wk;
        pose_map[kf0_id] = kf0_pose;
        kf0_scale = std::get<0>(loop_pair.second);
        scale_map[kf0_id] = std::get<0>(loop_pair.second);
      }

      {
        std::shared_lock<std::shared_mutex> lock(kf1->mutex);
        kf1_pose = kf1->pose_wk;
        pose_map[kf1_id] = kf1_pose;
        kf1_scale = std::get<1>(loop_pair.second);
        scale_map[kf1_id] = kf1_scale;
      }

      tgt_relpose10 = kf1_pose.inverse() * kf0_pose;

      new_factors.emplace_shared<RelPoseScaleFactor<Scalar>>(
          kf0, kf1,
          PoseKey(kf0_id), PoseKey(kf1_id),
          ScaleKey(kf0_id), ScaleKey(kf1_id),
          tgt_relpose10, kf0_scale, kf1_scale,
          opts_.pose_graph_global_link_weight,
          opts_.pose_graph_rot_weight,
          opts_.pose_graph_scale_weight);

      new_factors.emplace_shared<RelPoseScaleFactor<Scalar>>(
          kf1, kf0,
          PoseKey(kf1_id), PoseKey(kf0_id),
          ScaleKey(kf1_id), ScaleKey(kf0_id),
          tgt_relpose10.inverse(), kf1_scale, kf0_scale,
          opts_.pose_graph_global_link_weight,
          opts_.pose_graph_rot_weight,
          opts_.pose_graph_scale_weight);
    }

    const Scalar ref_scale0 = std::get<3>(new_global_loop_vec[0]);

    for (int i = 0; i < new_global_loop_vec.size(); i++)
    {
      const auto new_global_loop = new_global_loop_vec[i];

      // Latest global loop pair
      kf0_id = std::get<0>(new_global_loop);
      kf1_id = std::get<1>(new_global_loop);

      keyframe_id_set.insert(kf0_id);
      keyframe_id_set.insert(kf1_id);

      tgt_relpose10 = std::get<2>(new_global_loop);
      const Scalar target_scale0 = ref_scale0;
      const Scalar target_scale1 = target_scale0 * std::get<4>(new_global_loop) / std::get<3>(new_global_loop);

      auto kf0 = mapper_->GetMap()->keyframes.Get(kf0_id);
      auto kf1 = mapper_->GetMap()->keyframes.Get(kf1_id);

      new_factors.emplace_shared<RelPoseScaleFactor<Scalar>>(
          kf0, kf1,
          PoseKey(kf0_id), PoseKey(kf1_id),
          ScaleKey(kf0_id), ScaleKey(kf1_id),
          tgt_relpose10, target_scale0, target_scale1,
          opts_.pose_graph_global_link_weight,
          opts_.pose_graph_rot_weight,
          opts_.pose_graph_scale_weight);

      new_factors.emplace_shared<RelPoseScaleFactor<Scalar>>(
          kf1, kf0,
          PoseKey(kf1_id), PoseKey(kf0_id),
          ScaleKey(kf1_id), ScaleKey(kf0_id),
          tgt_relpose10.inverse(), target_scale1, target_scale0,
          opts_.pose_graph_global_link_weight,
          opts_.pose_graph_rot_weight,
          opts_.pose_graph_scale_weight);

      VLOG(2) << "[DeepFactors<Scalar, CS>::LoopClosurePoseScaleEstimate] current global loop info keyframe " << std::get<0>(new_global_loop) << " and "
              << std::get<1>(new_global_loop) << " target scale: " << target_scale0 << " and " << target_scale1;

      if (i == 0)
      {
        // Only the last new global loop get the scale prior factor
        new_factors.emplace_shared<ScaleFactor<Scalar>>(kf0, ScaleKey(kf0_id), target_scale0, opts_.pose_graph_scale_prior_weight);
        new_factors.emplace_shared<ScaleFactor<Scalar>>(kf1, ScaleKey(kf1_id), target_scale1, opts_.pose_graph_scale_prior_weight);
      }
    }

    // Initialize all keyframes involved in the match geometry factors above
    for (auto id : keyframe_id_set)
    {
      auto kf = mapper_->GetMap()->keyframes.Get(id);
      std::shared_lock<std::shared_mutex> lock(kf->mutex);
      var_init.insert(PoseKey(id), kf->pose_wk);
      var_init.insert(ScaleKey(id), kf->dpt_scale);
    }

    // Pose graph optimization
    VLOG(2) << "[DeepFactors<Scalar, CS>::LoopClosurePoseScaleEstimate] starting graph optimization";
    auto isam_res = pose_scale_graph.update(new_factors, var_init, remove_indices,
                                            boost::none, boost::none, boost::none, false);
    int num_update_steps = opts_.pose_scale_graph_max_iters;
    int num_no_relinearize_steps = 0;
    while (--num_update_steps >= 0 && num_no_relinearize_steps <= opts_.pose_scale_graph_no_relin_max_iters)
    {
      isam_res = pose_scale_graph.update();
      if (isam_res.variablesRelinearized == 0)
      {
        num_no_relinearize_steps += 1;
      }
      else
      {
        num_no_relinearize_steps = 0;
      }
      VLOG(3) << "[DeepFactors<Scalar, CS>::LoopClosurePoseScaleEstimate] graph optimization step: " << num_update_steps << " num relinearized: " << isam_res.variablesRelinearized;
    }
    VLOG(2) << "[DeepFactors<Scalar, CS>::LoopClosurePoseScaleEstimate] finished graph optimization";

    // Update all pose_wk and dpt_scale from the result of pose_graph here. dpt_map and dpt_map_grad also needs to be rescaled based on the dpt_scale changes.
    auto estimate = pose_scale_graph.calculateEstimate();

    {
      std::scoped_lock lock(mapper_->new_factor_mutex_, mapper_->new_keyframe_mutex_, mapper_->global_loop_mutex_);
      auto it = std::max_element(keyframe_id_set.begin(), keyframe_id_set.end());
      const KeyframeId max_kf_id = *it;
      const std::vector<KeyframeId> updated_keyframe_ids = mapper_->GetMap()->keyframes.Ids();
      const std::set<KeyframeId> updated_keyframe_id_set(updated_keyframe_ids.begin(), updated_keyframe_ids.end());
      std::vector<KeyframeId> increment_keyframe_ids;

      if (updated_keyframe_id_set.size() > keyframe_id_set.size())
      {
        increment_keyframe_ids = std::vector<KeyframeId>(updated_keyframe_id_set.size() - keyframe_id_set.size());
        std::set_difference(updated_keyframe_id_set.begin(), updated_keyframe_id_set.end(),
                            keyframe_id_set.begin(), keyframe_id_set.end(),
                            increment_keyframe_ids.begin());
        VLOG(2) << "[DeepFactors<Scalar, CS>::LoopClosurePoseScaleEstimate] incremental keyframe ids since loop closure optimzation: " << increment_keyframe_ids;
      }

      // We should modify both the pose and the dpt scale based on the update of the last keyframe in the graph optimization here
      SE3T prev_last_pose_wk;
      SE3T updated_last_pose_wk;
      Scalar prev_last_dpt_scale;
      Scalar updated_last_dpt_scale;

      auto kf = mapper_->GetMap()->keyframes.Get(max_kf_id);
      {
        std::shared_lock<std::shared_mutex> lock(kf->mutex);
        prev_last_pose_wk = kf->pose_wk;
        prev_last_dpt_scale = kf->dpt_scale;
      }
      updated_last_pose_wk = estimate.at(PoseKey(max_kf_id)).template cast<SE3T>();
      updated_last_dpt_scale = estimate.at(ScaleKey(max_kf_id)).template cast<Scalar>();
      const Scalar last_dpt_scale_ratio = updated_last_dpt_scale / prev_last_dpt_scale;

      for (auto &id : keyframe_id_set)
      {
        auto kf = mapper_->GetMap()->keyframes.Get(id);
        {
          std::unique_lock<std::shared_mutex> lock(kf->mutex);
          kf->reinitialize_count.fetch_add(1, std::memory_order_relaxed);
          kf->pose_wk = estimate.at(PoseKey(id)).template cast<SE3T>();
          Scalar updated_dpt_scale = estimate.at(ScaleKey(id)).template cast<Scalar>();
          kf->dpt_map = kf->dpt_map * (updated_dpt_scale / kf->dpt_scale);
          VLOG(2) << "[DeepFactors<Scalar, CS>::LoopClosurePoseScaleEstimate] keyframe " << kf->id << " dpt scale from " << kf->dpt_scale << " to " << updated_dpt_scale;
          kf->dpt_scale = updated_dpt_scale;
        }
      }

      // Use previous relative pose between the pre-updated keyframe and the later ones
      // and the updated keyframe pose to calculate the latest poses for all the keyframes that were not in the graph optimization.
      // dpt_scale is also changed based on the previous ratio
      // Scale should also affect the previous pose increments. Scale translation also?
      for (auto &id : increment_keyframe_ids)
      {
        auto kf = mapper_->GetMap()->keyframes.Get(id);
        {
          std::unique_lock<std::shared_mutex> lock(kf->mutex);
          kf->reinitialize_count.fetch_add(1, std::memory_order_relaxed);
          SE3T scaled_rel_pose = prev_last_pose_wk.inverse() * kf->pose_wk;
          scaled_rel_pose.translation() = scaled_rel_pose.translation() * last_dpt_scale_ratio;
          kf->pose_wk = updated_last_pose_wk * scaled_rel_pose;
          kf->dpt_map = kf->dpt_map * last_dpt_scale_ratio;
          VLOG(2) << "[DeepFactors<Scalar, CS>::LoopClosurePoseScaleEstimate] rough keyframe " << kf->id << " dpt scale from " << kf->dpt_scale << " to " << kf->dpt_scale * last_dpt_scale_ratio;
          kf->dpt_scale = kf->dpt_scale * last_dpt_scale_ratio;
        }
      }

      mapper_->AddGlobalLoopCount();
    }
  }

  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::LoopClosurePoseScaleMGEstimate(
      const std::set<std::pair<KeyframeId, KeyframeId>> &pre_global_loops,
      const std::tuple<KeyframeId, KeyframeId, at::Tensor, at::Tensor, at::Tensor, at::Tensor, Scalar> &cur_global_loop)
  {
    // The global loop contains keyframe Id pair and the corresponding matching 2D locations.
    // Graph optimization with LoopMG on depth scale and pose
    gtsam::ISAM2Params isam_params;
    isam_params.optimizationParams =
        gtsam::ISAM2DoglegParams(1.0, opts_.isam_wildfire_threshold, gtsam::DoglegOptimizerImpl::ONE_STEP_PER_ITERATION, false);
    isam_params.enableDetailedResults = false;
    isam_params.enablePartialRelinearizationCheck = false;
    isam_params.factorization = gtsam::ISAM2Params::CHOLESKY;
    isam_params.cacheLinearizedFactors = true;
    isam_params.findUnusedFactorSlots = false;
    isam_params.relinearizeSkip = 1;
    gtsam::FastMap<char, gtsam::Vector> thresholds;
    thresholds['p'] = gtsam::Vector::Ones(6) * opts_.loop_pose_linearize_threshold;
    thresholds['s'] = gtsam::Vector::Ones(1) * opts_.loop_scale_linearize_threshold;
    isam_params.relinearizeThreshold = thresholds;

    gtsam::ISAM2 loop_mg_graph(isam_params);
    gtsam::Values var_init;
    gtsam::NonlinearFactorGraph new_factors;
    gtsam::FastVector<gtsam::FactorIndex> remove_indices;

    // Initialize all poses as optimization variables
    std::set<KeyframeId> keyframe_id_set;
    std::vector<std::pair<KeyframeId, KeyframeId>> links;
    {
      std::shared_lock<std::shared_mutex> lock(mapper_->new_links_mutex_);
      links = mapper_->GetMap()->keyframes.GetLinks();
    }
    std::set<std::pair<KeyframeId, KeyframeId>> link_set(links.begin(), links.end());
    // Add the zero prior for the first pose to anchor all poses
    gtsam::Vector prior_sigmas = gtsam::Vector::Constant(6, 1, 1.0e-4);
    const auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas);
    new_factors.emplace_shared<gtsam::PriorFactor<SE3T>>(PoseKey(1), SE3T{}, prior_noise);
    new_factors.emplace_shared<ScaleFactor<Scalar>>(mapper_->GetMap()->keyframes.Get(1), ScaleKey(1), 1.0, 100.0);

    std::set<std::pair<KeyframeId, KeyframeId>> mg_link_set;
    const auto graph = mapper_->GetISAMGraph()->getFactorsUnsafe();

    // mapper_->new_factor_mutex_.lock_shared();
    auto rend = std::make_reverse_iterator(graph.end());
    auto rbegin = std::make_reverse_iterator(graph.begin());
    // mapper_->new_factor_mutex_.unlock_shared();

    at::Tensor matched_unscaled_dpts_0, matched_unscaled_dpts_1;
    for (auto it = rend; it != rbegin; ++it)
    {
      auto factor = *it;
      if (!factor)
      {
        continue;
      }
      auto mg = dynamic_cast<MatchGeometryFactor<Scalar, CS> *>(factor.get());
      if (!mg)
      {
        continue;
      }

      const auto kf0 = mapper_->GetMap()->keyframes.Get(mg->GetKeyframe(0)->id);
      const auto kf1 = mapper_->GetMap()->keyframes.Get(mg->GetKeyframe(1)->id);

      const at::Tensor matched_locations_1d_0 = mg->GetMatchLocations1D(0);
      const at::Tensor matched_locations_1d_1 = mg->GetMatchLocations1D(1);
      const at::Tensor matched_locations_homo_0 = mg->GetMatchLocationsHomo(0);
      const at::Tensor matched_locations_homo_1 = mg->GetMatchLocationsHomo(1);

      Scalar factor_weight;

      std::pair<KeyframeId, KeyframeId> pair = std::make_pair(std::min(kf0->id, kf1->id), std::max(kf0->id, kf1->id));

      if (pre_global_loops.find(pair) == pre_global_loops.end())
      {
        factor_weight = opts_.pose_graph_local_link_weight * std::pow(mg->DescInlierRatio(), 0.5);
      }
      else
      {
        factor_weight = opts_.pose_graph_global_link_weight * std::pow(mg->DescInlierRatio(), 0.5);
      }

      const Scalar loss_param = mg->GetLossParam();

      {
        std::scoped_lock lock(kf0->mutex, kf1->mutex);
        matched_unscaled_dpts_0 = kf0->dpt_map.reshape({-1}).index({matched_locations_1d_0}) / kf0->dpt_scale;
        matched_unscaled_dpts_1 = kf1->dpt_map.reshape({-1}).index({matched_locations_1d_1}) / kf1->dpt_scale;
      }

      new_factors.emplace_shared<LoopMGFactor<Scalar>>(
          kf0, kf1, PoseKey(kf0->id), PoseKey(kf1->id),
          ScaleKey(kf0->id), ScaleKey(kf1->id),
          matched_unscaled_dpts_0, matched_unscaled_dpts_1,
          matched_locations_homo_0, matched_locations_homo_1,
          factor_weight, loss_param, opts_.warp_dpt_eps);

      mg_link_set.insert(std::make_pair(kf0->id, kf1->id));

      keyframe_id_set.insert(kf0->id);
      keyframe_id_set.insert(kf1->id);
    }

    // The current global loop matching information come from the camera tracker of loop detector
    const KeyframeId kf0_id = std::get<0>(cur_global_loop);
    const KeyframeId kf1_id = std::get<1>(cur_global_loop);

    const auto kf0 = mapper_->GetMap()->keyframes.Get(kf0_id);
    const auto kf1 = mapper_->GetMap()->keyframes.Get(kf1_id);

    const Scalar loss_param = opts_.match_geom_loss_param_factor * kf0->avg_squared_dpt_bias;
    const Scalar desc_inlier_ratio = std::get<6>(cur_global_loop);
    const Scalar factor_weight = opts_.pose_graph_global_link_weight * std::pow(desc_inlier_ratio, 0.5);

    const at::Tensor matched_locations_1d_0 = std::get<2>(cur_global_loop);
    const at::Tensor matched_locations_1d_1 = std::get<3>(cur_global_loop);
    const at::Tensor matched_locations_homo_0 = std::get<4>(cur_global_loop);
    const at::Tensor matched_locations_homo_1 = std::get<5>(cur_global_loop);

    VLOG(2) << "[DeepFactors<Scalar, CS>::LoopClosurePoseScaleMGEstimate] global loop " << kf0_id << " " << kf1_id << " match size: " << matched_locations_1d_0.size(0) << " desc inlier ratio: " << desc_inlier_ratio;

    {
      std::scoped_lock lock(kf0->mutex, kf1->mutex);
      matched_unscaled_dpts_0 = kf0->dpt_map.reshape({-1}).index({matched_locations_1d_0}) / kf0->dpt_scale;
      matched_unscaled_dpts_1 = kf1->dpt_map.reshape({-1}).index({matched_locations_1d_1}) / kf1->dpt_scale;
    }

    new_factors.emplace_shared<LoopMGFactor<Scalar>>(
        kf0, kf1, PoseKey(kf0->id), PoseKey(kf1->id),
        ScaleKey(kf0->id), ScaleKey(kf1->id),
        matched_unscaled_dpts_0, matched_unscaled_dpts_1,
        matched_locations_homo_0, matched_locations_homo_1,
        factor_weight, loss_param, opts_.warp_dpt_eps);

    new_factors.emplace_shared<LoopMGFactor<Scalar>>(
        kf1, kf0, PoseKey(kf1->id), PoseKey(kf0->id),
        ScaleKey(kf1->id), ScaleKey(kf0->id),
        matched_unscaled_dpts_1, matched_unscaled_dpts_0,
        matched_locations_homo_1, matched_locations_homo_0,
        factor_weight, loss_param, opts_.warp_dpt_eps);

    keyframe_id_set.insert(kf0->id);
    keyframe_id_set.insert(kf1->id);

    if (link_set.size() > mg_link_set.size())
    {
      std::vector<std::pair<KeyframeId, KeyframeId>> no_mg_links(link_set.size() - mg_link_set.size());
      std::set_difference(link_set.begin(), link_set.end(),
                          mg_link_set.begin(), mg_link_set.end(),
                          no_mg_links.begin());
      VLOG(2) << "[DeepFactors<Scalar, CS>::LoopClosurePoseScaleMGEstimate] no match geometry links: " << no_mg_links;
    }

    // Initialize all keyframes involved in the match geometry factors above
    for (auto id : keyframe_id_set)
    {
      auto kf = mapper_->GetMap()->keyframes.Get(id);
      std::shared_lock<std::shared_mutex> lock(kf->mutex);
      var_init.insert(PoseKey(id), kf->pose_wk);
      var_init.insert(ScaleKey(id), kf->dpt_scale);
    }

    // Pose graph optimization
    VLOG(2) << "[DeepFactors<Scalar, CS>::LoopClosurePoseScaleMGEstimate] starting graph optimization";
    auto isam_res = loop_mg_graph.update(new_factors, var_init, remove_indices,
                                         boost::none, boost::none, boost::none, false);
    int num_update_steps = opts_.pose_scale_graph_max_iters;
    int num_no_relinearize_steps = 0;
    while (--num_update_steps >= 0 && num_no_relinearize_steps <= opts_.pose_scale_graph_no_relin_max_iters)
    {
      isam_res = loop_mg_graph.update();
      if (isam_res.variablesRelinearized == 0)
      {
        num_no_relinearize_steps += 1;
      }
      else
      {
        num_no_relinearize_steps = 0;
      }
      VLOG(3) << "[DeepFactors<Scalar, CS>::LoopClosurePoseScaleMGEstimate] graph optimization step: " << num_update_steps << " num relinearized: " << isam_res.variablesRelinearized;
    }
    VLOG(2) << "[DeepFactors<Scalar, CS>::LoopClosurePoseScaleMGEstimate] finished graph optimization";

    // Update all pose_wk and dpt_scale from the result of pose_graph here. dpt_map and dpt_map_grad also needs to be rescaled based on the dpt_scale changes.
    auto estimate = loop_mg_graph.calculateEstimate();

    {
      std::scoped_lock lock(mapper_->new_factor_mutex_, mapper_->new_keyframe_mutex_);

      auto it = std::max_element(keyframe_id_set.begin(), keyframe_id_set.end());
      const KeyframeId max_kf_id = *it;
      const std::vector<KeyframeId> updated_keyframe_ids = mapper_->GetMap()->keyframes.Ids();
      const std::set<KeyframeId> updated_keyframe_id_set(updated_keyframe_ids.begin(), updated_keyframe_ids.end());
      std::vector<KeyframeId> increment_keyframe_ids;

      if (updated_keyframe_id_set.size() > keyframe_id_set.size())
      {
        increment_keyframe_ids = std::vector<KeyframeId>(updated_keyframe_id_set.size() - keyframe_id_set.size());
        std::set_difference(updated_keyframe_id_set.begin(), updated_keyframe_id_set.end(),
                            keyframe_id_set.begin(), keyframe_id_set.end(),
                            increment_keyframe_ids.begin());
        VLOG(2) << "[DeepFactors<Scalar, CS>::LoopClosurePoseScaleMGEstimate] incremental keyframe ids since loop closure optimzation: " << increment_keyframe_ids;
      }

      // We should modify both the pose and the dpt scale based on the update of the last keyframe in the graph optimization here
      SE3T prev_last_pose_wk;
      SE3T updated_last_pose_wk;
      Scalar prev_last_dpt_scale;
      Scalar updated_last_dpt_scale;

      auto kf = mapper_->GetMap()->keyframes.Get(max_kf_id);
      {
        std::shared_lock<std::shared_mutex> lock(kf->mutex);
        prev_last_pose_wk = kf->pose_wk;
        prev_last_dpt_scale = kf->dpt_scale;
      }
      updated_last_pose_wk = estimate.at(PoseKey(max_kf_id)).template cast<SE3T>();
      updated_last_dpt_scale = estimate.at(ScaleKey(max_kf_id)).template cast<Scalar>();

      for (auto &id : keyframe_id_set)
      {
        auto kf = mapper_->GetMap()->keyframes.Get(id);
        {

          std::unique_lock<std::shared_mutex> lock(kf->mutex);
          kf->pose_wk = estimate.at(PoseKey(id)).template cast<SE3T>();
          Scalar updated_dpt_scale = estimate.at(ScaleKey(id)).template cast<Scalar>();
          VLOG(2) << "[DeepFactors<Scalar, CS>::LoopClosurePoseScaleEstimate] keyframe " << kf->id << " dpt scale from " << kf->dpt_scale << " to " << updated_dpt_scale;
          kf->dpt_map = kf->dpt_map * (updated_dpt_scale / kf->dpt_scale);
          kf->dpt_scale = updated_dpt_scale;
          kf->reinitialize_count.fetch_add(1, std::memory_order_relaxed);
        }
      }

      // Use previous relative pose between the pre-updated keyframe and the later ones
      // and the updated keyframe pose to calculate the latest poses for all the keyframes that were not in the graph optimization.
      // dpt_scale is also changed based on the previous ratio
      // Scale should also affect the previous pose increments. Scale translation also?
      for (auto &id : increment_keyframe_ids)
      {
        auto kf = mapper_->GetMap()->keyframes.Get(id);
        {
          std::unique_lock<std::shared_mutex> lock(kf->mutex);
          SE3T scaled_rel_pose = prev_last_pose_wk.inverse() * kf->pose_wk;
          scaled_rel_pose.translation() = scaled_rel_pose.translation() * (updated_last_dpt_scale / prev_last_dpt_scale);
          kf->pose_wk = updated_last_pose_wk * scaled_rel_pose;
          kf->dpt_map = kf->dpt_map * (updated_last_dpt_scale / prev_last_dpt_scale);
          VLOG(2) << "[DeepFactors<Scalar, CS>::LoopClosurePoseScaleEstimate] rough keyframe " << kf->id << " dpt scale from " << kf->dpt_scale << " to " << kf->dpt_scale * (updated_last_dpt_scale / prev_last_dpt_scale);
          kf->dpt_scale = kf->dpt_scale * (updated_last_dpt_scale / prev_last_dpt_scale);
          kf->reinitialize_count.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
  }

  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::LoopClosurePoseEstimate(const std::set<std::pair<KeyframeId, KeyframeId>> &pre_global_loops,
                                                        const std::tuple<KeyframeId, KeyframeId, SE3T, Scalar, Scalar> &cur_global_loop)
  {
    gtsam::ISAM2Params isam_params;
    isam_params.optimizationParams =
        gtsam::ISAM2DoglegParams(1.0, opts_.isam_wildfire_threshold, gtsam::DoglegOptimizerImpl::ONE_STEP_PER_ITERATION, false);
    isam_params.enableDetailedResults = false;
    isam_params.enablePartialRelinearizationCheck = false;
    isam_params.factorization = gtsam::ISAM2Params::CHOLESKY;
    isam_params.cacheLinearizedFactors = true;
    isam_params.findUnusedFactorSlots = false;
    isam_params.relinearizeSkip = 1;
    gtsam::FastMap<char, gtsam::Vector> thresholds;
    thresholds['p'] = gtsam::Vector::Ones(6) * opts_.loop_pose_linearize_threshold;
    isam_params.relinearizeThreshold = thresholds;

    gtsam::ISAM2 pose_graph(isam_params);
    gtsam::Values var_init;
    gtsam::NonlinearFactorGraph new_factors;
    gtsam::FastVector<gtsam::FactorIndex> remove_indices;

    // Initialize all poses as optimization variables
    std::set<KeyframeId> keyframe_id_set;
    std::vector<std::pair<KeyframeId, KeyframeId>> links;
    {
      std::shared_lock<std::shared_mutex> lock(mapper_->new_links_mutex_);
      links = mapper_->GetMap()->keyframes.GetLinks();
    }

    // Add the zero prior for the first pose to anchor all poses
    gtsam::Vector prior_sigmas = gtsam::Vector::Constant(6, 1, 1.0e-4);
    auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas);
    new_factors.emplace_shared<gtsam::PriorFactor<SE3T>>(PoseKey(1), SE3T{}, prior_noise);

    SE3T tgt_relpose10;

    std::unordered_map<KeyframeId, SE3T> pose_map;
    // Build pose factors for local links first
    for (auto const &link : links)
    {
      if (link.first < link.second)
      {
        auto id_pair = std::make_pair(link.first, link.second);
        if (pre_global_loops.find(id_pair) == pre_global_loops.end()) // If the link is not a global loop one
        {
          keyframe_id_set.insert(link.first);
          keyframe_id_set.insert(link.second);

          const auto kf0 = mapper_->GetMap()->keyframes.Get(link.first);
          const auto kf1 = mapper_->GetMap()->keyframes.Get(link.second);
          SE3T kf0_pose, kf1_pose;

          if (pose_map.find(link.first) == pose_map.end())
          {
            std::shared_lock<std::shared_mutex> lock(kf0->mutex);
            kf0_pose = kf0->pose_wk;
            pose_map[link.first] = kf0_pose;
          }
          else
          {
            kf0_pose = pose_map[link.first];
          }

          if (pose_map.find(link.second) == pose_map.end())
          {
            std::shared_lock<std::shared_mutex> lock(kf1->mutex);
            kf1_pose = kf1->pose_wk;
            pose_map[link.second] = kf1_pose;
          }
          else
          {
            kf1_pose = pose_map[link.second];
          }

          tgt_relpose10 = kf1_pose.inverse() * kf0_pose;

          new_factors.emplace_shared<RelPoseFactor<Scalar>>(
              kf0, kf1,
              PoseKey(link.first), PoseKey(link.second),
              tgt_relpose10,
              opts_.pose_graph_local_link_weight,
              opts_.pose_graph_rot_weight);

          new_factors.emplace_shared<RelPoseFactor<Scalar>>(
              kf1, kf0,
              PoseKey(link.second), PoseKey(link.first),
              tgt_relpose10.inverse(),
              opts_.pose_graph_local_link_weight,
              opts_.pose_graph_rot_weight);
        }
      }
    }

    KeyframeId kf0_id, kf1_id;
    for (auto const &loop_pair : pre_global_loops)
    {
      kf0_id = std::get<0>(loop_pair);
      kf1_id = std::get<1>(loop_pair);

      keyframe_id_set.insert(kf0_id);
      keyframe_id_set.insert(kf1_id);

      const auto kf0 = mapper_->GetMap()->keyframes.Get(kf0_id);
      const auto kf1 = mapper_->GetMap()->keyframes.Get(kf1_id);
      SE3T kf0_pose, kf1_pose;

      if (pose_map.find(kf0_id) == pose_map.end())
      {
        std::shared_lock<std::shared_mutex> lock(kf0->mutex);
        kf0_pose = kf0->pose_wk;
        pose_map[kf0_id] = kf0_pose;
      }
      else
      {
        kf0_pose = pose_map[kf0_id];
      }

      if (pose_map.find(kf1_id) == pose_map.end())
      {
        std::shared_lock<std::shared_mutex> lock(kf1->mutex);
        kf1_pose = kf1->pose_wk;
        pose_map[kf1_id] = kf1_pose;
      }
      else
      {
        kf1_pose = pose_map[kf1_id];
      }

      tgt_relpose10 = kf1_pose.inverse() * kf0_pose;

      new_factors.emplace_shared<RelPoseFactor<Scalar>>(
          kf0, kf1,
          PoseKey(kf0_id), PoseKey(kf1_id),
          tgt_relpose10, //relpose10 1 is cur with larger id and 0 is ref with smaller id
          opts_.pose_graph_global_link_weight,
          opts_.pose_graph_rot_weight);

      new_factors.emplace_shared<RelPoseFactor<Scalar>>(
          kf1, kf0,
          PoseKey(kf1_id), PoseKey(kf0_id),
          tgt_relpose10.inverse(), //relpose10 1 is cur with larger id and 0 is ref with smaller id
          opts_.pose_graph_global_link_weight,
          opts_.pose_graph_rot_weight);
    }

    // Latest global loop pair
    kf0_id = std::get<0>(cur_global_loop);
    kf1_id = std::get<1>(cur_global_loop);

    keyframe_id_set.insert(kf0_id);
    keyframe_id_set.insert(kf1_id);

    tgt_relpose10 = std::get<2>(cur_global_loop);
    auto kf0 = mapper_->GetMap()->keyframes.Get(kf0_id);
    auto kf1 = mapper_->GetMap()->keyframes.Get(kf1_id);
    new_factors.emplace_shared<RelPoseFactor<Scalar>>(
        kf0, kf1,
        PoseKey(kf0_id), PoseKey(kf1_id),
        tgt_relpose10, //relpose10 1 is cur with larger id and 0 is ref with smaller id
        opts_.pose_graph_global_link_weight,
        opts_.pose_graph_rot_weight);

    new_factors.emplace_shared<RelPoseFactor<Scalar>>(
        kf1, kf0,
        PoseKey(kf1_id), PoseKey(kf0_id),
        tgt_relpose10.inverse(), //relpose10 1 is cur with larger id and 0 is ref with smaller id
        opts_.pose_graph_global_link_weight,
        opts_.pose_graph_rot_weight);

    for (auto id : keyframe_id_set)
    {
      var_init.insert(PoseKey(id), mapper_->GetMap()->keyframes.Get(id)->pose_wk);
    }

    // Pose graph optimization
    auto isam_res = pose_graph.update(new_factors, var_init, remove_indices,
                                      boost::none, boost::none, boost::none, true);
    int num_update_steps = opts_.pose_graph_max_iters;
    int num_no_relinearize_steps = 0;
    while (--num_update_steps >= 0 && num_no_relinearize_steps <= opts_.pose_graph_no_relin_max_iters)
    {
      isam_res = pose_graph.update();
      if (isam_res.variablesRelinearized == 0)
      {
        num_no_relinearize_steps += 1;
      }
      else
      {
        num_no_relinearize_steps = 0;
      }
    }
    // Update all pose_wk from the result of pose_graph here
    auto estimate = pose_graph.calculateEstimate();
    // We don't want the main thread to do bookkeeping while we are updating the pose_wk here.
    // Because this may cause bookkeeping to only have subset of updated poses

    {
      std::scoped_lock lock(mapper_->new_factor_mutex_, mapper_->new_keyframe_mutex_);
      // std::set<KeyframeId> updated_keyframe_id_set;
      // {
      //   updated_keyframe_id_set = mapper_->GetBookKeepedIds();
      // }
      std::vector<KeyframeId> updated_keyframe_ids;
      updated_keyframe_ids = mapper_->GetMap()->keyframes.Ids();
      SE3T prev_last_pose_wk;
      SE3T updated_last_pose_wk;

      auto it = std::max_element(keyframe_id_set.begin(), keyframe_id_set.end());
      const KeyframeId max_kf_id = *it;
      std::set<KeyframeId> updated_keyframe_id_set(updated_keyframe_ids.begin(), updated_keyframe_ids.end());
      std::vector<KeyframeId> increment_keyframe_ids;

      if (updated_keyframe_id_set.size() > keyframe_id_set.size())
      {
        increment_keyframe_ids = std::vector<KeyframeId>(updated_keyframe_id_set.size() - keyframe_id_set.size());
        std::set_difference(updated_keyframe_id_set.begin(), updated_keyframe_id_set.end(),
                            keyframe_id_set.begin(), keyframe_id_set.end(),
                            increment_keyframe_ids.begin());
      }

      auto kf = mapper_->GetMap()->keyframes.Get(max_kf_id);
      {
        std::shared_lock<std::shared_mutex> lock(kf->mutex);
        prev_last_pose_wk = kf->pose_wk;
      }
      updated_last_pose_wk = estimate.at(PoseKey(max_kf_id)).template cast<SE3T>();

      for (auto &id : keyframe_id_set)
      {
        auto kf = mapper_->GetMap()->keyframes.Get(id);
        {
          std::unique_lock<std::shared_mutex> lock(kf->mutex);
          kf->pose_wk = estimate.at(PoseKey(id)).template cast<SE3T>();
          kf->reinitialize_count.fetch_add(1, std::memory_order_relaxed);
        }
      }

      // Use previous relative pose between the pre-updated keyframe and the later ones
      // and the updated keyframe pose to calculate the latest poses for all the keyframes that were not in the pose graph optimization.
      for (auto &id : increment_keyframe_ids)
      {
        auto kf = mapper_->GetMap()->keyframes.Get(id);
        kf->pose_wk = updated_last_pose_wk * (prev_last_pose_wk.inverse() * kf->pose_wk);
        kf->reinitialize_count.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  /*

  */

  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::LocalLoopDetectBackend()
  {

    while (!enable_loop_ && !quit_loop_)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    if (quit_loop_)
    {
      return;
    }

    bool new_keyframe = false;
    const auto opts = opts_;
    const double loop_time = 1000.0f / opts.loop_detection_frequency;
    long remaining_ms;

    KeyframeId query_kf_id;

    while (true)
    {
      auto loop_start = std::chrono::steady_clock::now();
      double elapsed_ms = 0;

      {
        std::shared_lock<std::shared_mutex> lock(visited_keyframes_mutex_);
        if (visited_keyframe_ids_.size() > local_loop_curr_keyframe_ids_.size())
        {
          // Here both vector should already be sorted
          local_loop_prev_keyframe_ids_ = local_loop_curr_keyframe_ids_;
          local_loop_curr_keyframe_ids_ = visited_keyframe_ids_;

          new_keyframe = true;
        }
        else
        {
          new_keyframe = false;
        }
      }

      if (new_keyframe)
      {
        std::set<KeyframeId> local_loop_curr_keyframe_id_set(local_loop_curr_keyframe_ids_.begin(), local_loop_curr_keyframe_ids_.end());
        std::set<KeyframeId> local_loop_prev_keyframe_id_set(local_loop_prev_keyframe_ids_.begin(), local_loop_prev_keyframe_ids_.end());
        std::vector<KeyframeId> increment_keyframe_ids(local_loop_curr_keyframe_id_set.size() - local_loop_prev_keyframe_id_set.size());
        // The difference of two sets is formed by the elements that are present in the first set, but not in the second one
        std::set_difference(local_loop_curr_keyframe_id_set.begin(), local_loop_curr_keyframe_id_set.end(),
                            local_loop_prev_keyframe_id_set.begin(), local_loop_prev_keyframe_id_set.end(),
                            increment_keyframe_ids.begin());

        std::set<KeyframeId> increment_keyframe_id_set(increment_keyframe_ids.begin(), increment_keyframe_ids.end());
        for (auto &id : increment_keyframe_id_set)
        {
          VLOG(2) << "[DeepFactors<Scalar, CS>::LocalLoopDetectBackend] adding new keyframe " << id << " to loop detector database";
          loop_detector_->AddKeyframe(mapper_->GetMap()->keyframes.Get(id));
        }

        {
          // Here, after adding all incoming keyframes to the loop detector database,
          // we can then tell the global loop detector to try to detect global loops for these keyframes
          std::unique_lock<std::shared_mutex> lock(global_loop_ids_mutex_);
          global_loop_curr_keyframe_ids_ = local_loop_curr_keyframe_ids_;
        }
      }

      if ((!new_keyframe && !enable_loop_) || quit_loop_)
      {
        break;
      }

      // Use the last item in the current keyframe id list as the query item to search for a local loop.
      if (new_keyframe)
      {
        query_kf_id = local_loop_curr_keyframe_ids_.back();
        auto query_kf = mapper_->GetMap()->keyframes.Get(query_kf_id);
        // detect local loops
        if (!query_kf->local_loop_searched)
        {
          VLOG(2) << "[DeepFactors<Scalar, CS>::LocalLoopDetectBackend] start local loop detection for keyframe " << query_kf_id;
          query_kf->local_loop_searched = true;
          auto it = local_loop_curr_keyframe_ids_.rbegin();
          auto local_loop_info = loop_detector_->DetectLocalLoop(query_kf, local_loop_curr_keyframe_ids_, it);

          if (local_loop_info.detected)
          {
            query_kf->local_loop_connections.push_back(local_loop_info.id_ref);

            mapper_->GetMap()->keyframes.Get(local_loop_info.id_ref)->local_loop_connections.push_back(query_kf_id);

            VLOG(1) << "[DeepFactors<Scalar, CS>::LocalLoopDetectBackend] Local loop detected -- keyframe " << local_loop_info.id_ref << " for keyframe " << query_kf->id;
            // include the local loop frame link to the factor graph
            {
              // std::unique_lock<std::shared_mutex> lock(mapper_->new_factor_mutex_);
              mapper_->EnqueueLink(query_kf_id, local_loop_info.id_ref, opts.use_photometric, opts.use_reprojection, opts.use_geometric, false);
            }
            loop_links.push_back({query_kf_id, local_loop_info.id_ref});
          }
        }
        new_keyframe = false;
      }
      else
      {
        // We should select one previous frame that still has not been searched for
        query_kf_id = -1;
        auto it = local_loop_curr_keyframe_ids_.rbegin();
        while (true)
        {
          if (it == local_loop_curr_keyframe_ids_.rend())
          {
            break;
          }

          auto kf = mapper_->GetMap()->keyframes.Get(*it);
          if (!kf->local_loop_searched)
          {
            query_kf_id = *it;
            break;
          }
          else
          {
            it++;
          }
        }

        // If there is a keyframe that has not been searched for local loop
        if (query_kf_id != -1)
        {
          auto query_kf = mapper_->GetMap()->keyframes.Get(query_kf_id);
          query_kf->local_loop_searched = true;
          VLOG(2) << "[DeepFactors<Scalar, CS>::LocalLoopDetectBackend] start local loop detection for previous keyframe " << query_kf_id;
          auto local_loop_info = loop_detector_->DetectLocalLoop(query_kf, local_loop_curr_keyframe_ids_, it);

          if (local_loop_info.detected)
          {
            query_kf->local_loop_connections.push_back(local_loop_info.id_ref);
            mapper_->GetMap()->keyframes.Get(local_loop_info.id_ref)->local_loop_connections.push_back(query_kf_id);
            VLOG(1) << "[DeepFactors<Scalar, CS>::LocalLoopDetectBackend] Local loop detected -- keyframe " << local_loop_info.id_ref << " for keyframe " << query_kf->id;
            // include the local loop frame link to the factor graph
            {
              // std::unique_lock<std::shared_mutex> lock(mapper_->new_factor_mutex_);
              mapper_->EnqueueLink(query_kf_id, local_loop_info.id_ref, opts.use_photometric, opts.use_reprojection, opts.use_geometric, false);
            }
            loop_links.push_back({query_kf_id, local_loop_info.id_ref});
          }
        }
      }

      auto loop_end = std::chrono::steady_clock::now();
      elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();

      remaining_ms = loop_time - elapsed_ms;
      if (remaining_ms > 0)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(remaining_ms));
      }
    }

    VLOG(2) << "[DeepFactors<Scalar, CS>::LocalLoopDetectBackend] quit local loop backend";
    return;
  }

  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::GlobalLoopDetectBackend()
  {

    while (!enable_loop_ && !quit_loop_)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    if (quit_loop_)
    {
      VLOG(2) << "[DeepFactors<Scalar, CS>::GlobalLoopDetectBackend] quit global loop backend";
      return;
    }

    bool new_keyframe = false;
    const auto opts = opts_;
    const double loop_time = 1000.0f / opts.loop_detection_frequency;
    long remaining_ms;

    KeyframeId query_kf_id;
    std::vector<KeyframeId> curr_keyframe_ids;

    while (true)
    {
      auto loop_start = std::chrono::steady_clock::now();
      double elapsed_ms = 0;

      {
        std::shared_lock<std::shared_mutex> lock(global_loop_ids_mutex_);
        if (global_loop_curr_keyframe_ids_.size() > curr_keyframe_ids.size())
        {
          curr_keyframe_ids = global_loop_curr_keyframe_ids_;
          new_keyframe = true;
        }
        else
        {
          new_keyframe = false;
        }
      }

      // Use the last id of global_loop_curr_keyframe_ids_ as the query_kf_id and do searching.
      // (all the previous keyframes should already be added to the loop detector database)
      // do other frame searching when no new keyframe is available.

      if ((!new_keyframe && !enable_loop_) || quit_loop_)
      {
        break;
      }

      std::vector<std::tuple<KeyframeId, KeyframeId, SE3T, Scalar, Scalar>> new_global_loop_vec;

      if (new_keyframe)
      {
        query_kf_id = curr_keyframe_ids.back();
        auto query_kf = mapper_->GetMap()->keyframes.Get(query_kf_id);
        // detect global loop
        if (!query_kf->global_loop_searched)
        {
          VLOG(2) << "[DeepFactors<Scalar, CS>::GlobalLoopDetectBackend] start global loop detection for keyframe " << query_kf_id;
          tic("[DeepFactors<Scalar, CS>::GlobalLoopDetectBackend] Global loop closure");
          const std::vector<typename LoopDetector<Scalar>::LoopInfo> global_loop_info_vec = loop_detector_->DetectLoop(query_kf);
          query_kf->global_loop_searched = true;

          if (!global_loop_info_vec.empty())
          {
            for (const auto &loop_info : global_loop_info_vec)
            {
              VLOG(2) << "[DeepFactors<Scalar, CS>::GlobalLoopDetectBackend] Real-time connecting keyframe " << query_kf_id << " to " << loop_info.id_ref << "with scale " << loop_info.query_scale << " " << loop_info.ref_scale;
              new_global_loop_vec.push_back(std::make_tuple(query_kf->id, loop_info.id_ref,
                                                            loop_info.pose_cur_ref.inverse(),
                                                            loop_info.query_scale, loop_info.ref_scale));
            }

            LoopClosurePoseScaleEstimate(global_loops_info_, new_global_loop_vec);

            for (int i = 0; i < new_global_loop_vec.size(); i++)
            {
              global_loops_info_[std::make_pair(std::get<0>(new_global_loop_vec[i]), std::get<1>(new_global_loop_vec[i]))] =
                  std::make_pair(std::get<3>(new_global_loop_vec[i]), std::get<4>(new_global_loop_vec[i]));

              query_kf->global_loop_connections.push_back(global_loop_info_vec[i].id_ref);
              mapper_->GetMap()->keyframes.Get(global_loop_info_vec[i].id_ref)->global_loop_connections.push_back(query_kf_id);

              VLOG(1) << "[DeepFactors<Scalar, CS>::GlobalLoopDetectBackend] Global loop detected -- keyframe "
                      << query_kf->id << " against keyframe " << global_loop_info_vec[i].id_ref;

              {
                mapper_->EnqueueLink(query_kf_id, global_loop_info_vec[i].id_ref, true,
                                     true, true, true);
              }

              loop_links.push_back({query_kf_id, global_loop_info_vec[i].id_ref});
            }
          }

          toc("[DeepFactors<Scalar, CS>::GlobalLoopDetectBackend] Global loop closure");
        }
        new_keyframe = false;
      }
      else
      {
        // We should select one previous frame that still has not been searched for
        query_kf_id = -1;
        auto it = curr_keyframe_ids.rbegin();

        std::vector<KeyframeId> searched_kf_ids;
        while (true)
        {
          if (it == curr_keyframe_ids.rend())
          {
            break;
          }

          auto kf = mapper_->GetMap()->keyframes.Get(*it);
          searched_kf_ids.push_back(*it);

          if (!(kf->global_loop_searched))
          {
            query_kf_id = *it;
            break;
          }
          else
          {
            it++;
          }
        }

        // If there is a keyframe that has not been searched for global loop
        if (query_kf_id != -1)
        {
          tic("[DeepFactors<Scalar, CS>::GlobalLoopDetectBackend] Global loop closure");
          VLOG(2) << "[DeepFactors<Scalar, CS>::GlobalLoopDetectBackend] start global loop detection for previous keyframe " << query_kf_id;
          auto query_kf = mapper_->GetMap()->keyframes.Get(query_kf_id);
          const std::vector<typename LoopDetector<Scalar>::LoopInfo> global_loop_info_vec = loop_detector_->DetectLoop(query_kf);
          std::tuple<KeyframeId, KeyframeId, SE3T, Scalar, Scalar> cur_global_loop;
          query_kf->global_loop_searched = true;

          if (!global_loop_info_vec.empty())
          {
            for (const auto &loop_info : global_loop_info_vec)
            {
              VLOG(2) << "[DeepFactors<Scalar, CS>::GlobalLoopDetectBackend] Retrospective connecting keyframe " << query_kf_id << " to " << loop_info.id_ref << "with scale " << loop_info.query_scale << " " << loop_info.ref_scale;
              new_global_loop_vec.push_back(std::make_tuple(query_kf->id, loop_info.id_ref,
                                                            loop_info.pose_cur_ref.inverse(),
                                                            loop_info.query_scale, loop_info.ref_scale));
            }

            LoopClosurePoseScaleEstimate(global_loops_info_, new_global_loop_vec);

            for (int i = 0; i < new_global_loop_vec.size(); i++)
            {
              global_loops_info_[std::make_pair(std::get<0>(new_global_loop_vec[i]), std::get<1>(new_global_loop_vec[i]))] =
                  std::make_pair(std::get<3>(new_global_loop_vec[i]), std::get<4>(new_global_loop_vec[i]));

              query_kf->global_loop_connections.push_back(global_loop_info_vec[i].id_ref);
              mapper_->GetMap()->keyframes.Get(global_loop_info_vec[i].id_ref)->global_loop_connections.push_back(query_kf_id);

              VLOG(1) << "[DeepFactors<Scalar, CS>::GlobalLoopDetectBackend] Global loop detected -- keyframe "
                      << query_kf->id << " against keyframe " << global_loop_info_vec[i].id_ref;

              {
                mapper_->EnqueueLink(query_kf_id, global_loop_info_vec[i].id_ref, true,
                                     true, true, true);
              }

              loop_links.push_back({query_kf_id, global_loop_info_vec[i].id_ref});
            }
          }

          toc("[DeepFactors<Scalar, CS>::GlobalLoopDetectBackend] Global loop closure");
        }
      }

      auto loop_end = std::chrono::steady_clock::now();
      elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();

      remaining_ms = loop_time - elapsed_ms;
      if (remaining_ms > 0)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(remaining_ms));
      }
    }

    VLOG(2) << "[DeepFactors<Scalar, CS>::GlobalLoopDetectBackend] quit global loop backend";
    return;
  }

  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::MappingBackend()
  {
    LOG(INFO) << "[DeepFactors<Scalar, CS>::MappingBackend] Mapping update thread started";

    const auto opts = opts_;
    const double loop_time = 1000.0f / opts.mapping_update_frequency;
    double elapsed_ms = 0;
    long remaining_ms;

    std::vector<KeyframeId> keyframe_ids;
    while (!enable_mapping_ && !quit_mapping_)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    if (quit_mapping_)
    {
      VLOG(2) << "[DeepFactors<Scalar, CS>::MappingBackend] quit mapping backend";
      return;
    }

    while (enable_mapping_ && !quit_mapping_)
    {
      VLOG(3) << "[DeepFactors<Scalar, CS>::MappingBackend] one iteration of mapping step";
      auto loop_start = std::chrono::steady_clock::now();

      {
        std::shared_lock<std::shared_mutex> lock(visited_keyframes_mutex_);
        if (visited_keyframe_ids_.size() > keyframe_ids.size())
        {
          keyframe_ids = visited_keyframe_ids_;
        }
      }

      mapper_->MappingStep(keyframe_ids, false);

      VLOG(3) << "[DeepFactors<Scalar, CS>::MappingBackend] mapping step finished";

      // We only have one KeyframeRelinearization call at the same time so no need for mutex lock
      if (opts_.isam_enable_detailed_results)
      {
        stats_.relin_info = mapper_->KeyframeRelinearization();
        NotifyStatsObservers();
      }

      auto loop_end = std::chrono::steady_clock::now();
      elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();

      remaining_ms = loop_time - elapsed_ms;
      if (remaining_ms > 0)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(remaining_ms));
      }
    }

    VLOG(2) << "[DeepFactors<Scalar, CS>::MappingBackend] quit mapping backend";
    return;
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::Init(const df::PinholeCamera<Scalar> &cam, const cv::Mat &video_mask, const DeepFactorsOptions &opts)
  {
    opts_ = opts;

    // Resize the camera parameter to the output spatial size
    output_cam_ = cam;
    output_cam_.ResizeViewport(opts.net_output_size[1], opts.net_output_size[0]);
    // create a pyramid from the network camera
    output_cam_pyr_ = df::CameraPyramid<float>(output_cam_, opts_.num_pyramid_levels);

    // Load depth network and feature network
    depth_network_ptr_ = std::make_shared<df::CodeDepthNetwork>(opts_.depth_network_path, opts_.cuda_id);
    feature_network_ptr_ = std::make_shared<df::FeatureNetwork>(opts_.feature_network_path);

    // Create tracker
    df::CameraTracker::TrackerConfig tracker_cfg;
    tracker_cfg.cuda_id = opts.cuda_id;
    tracker_cfg.max_num_iters = opts.tracking_max_num_iters;
    tracker_cfg.min_grad_thresh = opts.tracking_min_grad_thresh;
    tracker_cfg.min_param_inc_thresh = opts.tracking_min_param_inc_thresh;
    tracker_cfg.min_damp = opts.tracking_min_max_damp[0];
    tracker_cfg.max_damp = opts.tracking_min_max_damp[1];
    tracker_cfg.init_damp = opts.tracking_init_damp;
    tracker_cfg.damp_dec_factor = opts.tracking_damp_dec_inc_factor[0];
    tracker_cfg.damp_inc_factor = opts.tracking_damp_dec_inc_factor[1];
    tracker_cfg.dpt_eps = opts.warp_dpt_eps;
    tracker_cfg.jac_update_err_inc_threshold = opts.tracking_jac_update_err_inc_threshold;
    tracker_cfg.photo_factor_weights = opts.photo_factor_weights;
    tracker_cfg.desc_num_samples = opts.tracking_desc_num_keypoints;
    tracker_cfg.desc_cyc_consis_thresh = opts.desc_cyc_consis_thresh;
    tracker_cfg.match_geom_factor_weight = opts.tracker_match_geom_factor_weight;
    tracker_cfg.match_geom_loss_param_factor = opts.match_geom_loss_param_factor;
    tracker_cfg.reproj_factor_weight = opts.tracker_reproj_factor_weight;
    tracker_cfg.reproj_loss_param_factor = opts.reproj_loss_param_factor;
    tracker_cfg.net_output_size = opts.net_output_size;
    // TEASER++ related
    tracker_cfg.teaser_max_clique_time_limit = opts.teaser_max_clique_time_limit;
    tracker_cfg.teaser_kcore_heuristic_threshold = opts.teaser_kcore_heuristic_threshold;
    tracker_cfg.teaser_rotation_max_iterations = opts.teaser_rotation_max_iterations;
    tracker_cfg.teaser_rotation_cost_threshold = opts.teaser_rotation_cost_threshold;
    tracker_cfg.teaser_rotation_gnc_factor = opts.teaser_rotation_gnc_factor;
    tracker_cfg.teaser_rotation_estimation_algorithm = opts.teaser_rotation_estimation_algorithm;
    tracker_cfg.teaser_rotation_tim_graph = opts.teaser_rotation_tim_graph;
    tracker_cfg.teaser_inlier_selection_mode = opts.teaser_tracker_inlier_selection_mode;
    tracker_cfg.teaser_noise_bound_multiplier = opts.teaser_noise_bound_multiplier;

    tracker_ = std::make_shared<df::CameraTracker>(tracker_cfg);
    tracker_->SetName("camera tracking");

    // Create mapper
    df::MapperOptions mapper_opts;

    mapper_opts.log_dir = opts.log_dir;
    // device id
    mapper_opts.cuda_id = opts.cuda_id;
    // network input output size and video mask path
    mapper_opts.net_input_size = opts.net_input_size;
    mapper_opts.net_output_size = opts.net_output_size;
    // prior
    mapper_opts.init_pose_prior_weight = opts.init_pose_prior_weight;
    mapper_opts.init_scale_prior_weight = opts.init_scale_prior_weight;
    // epsilon
    mapper_opts.dpt_eps = opts.warp_dpt_eps;

    // factors
    mapper_opts.factor_iters = opts.factor_iters;

    /* photo */
    mapper_opts.use_photometric = opts.use_photometric;
    mapper_opts.pho_num_samples = opts.pho_num_samples;
    mapper_opts.photo_factor_weights = opts.photo_factor_weights;
    /* reprojection */
    mapper_opts.use_reprojection = opts.use_reprojection;
    mapper_opts.factor_iters = opts.factor_iters;
    mapper_opts.desc_num_keypoints = opts.desc_num_keypoints;
    mapper_opts.desc_cyc_consis_thresh = opts.desc_cyc_consis_thresh;
    /* geometric */
    mapper_opts.use_geometric = opts.use_geometric;
    mapper_opts.factor_iters = opts.factor_iters;
    mapper_opts.geo_factor_weight = opts.geo_factor_weight;
    mapper_opts.geo_loss_param_factor = opts.geo_loss_param_factor;

    // match geometry factor
    mapper_opts.match_geom_factor_weight = opts.match_geom_factor_weight;
    mapper_opts.match_geom_loss_param_factor = opts.match_geom_loss_param_factor;

    // reprojection factor
    mapper_opts.reproj_factor_weight = opts.reproj_factor_weight;
    mapper_opts.reproj_loss_param_factor = opts.reproj_loss_param_factor;

    // scale factor
    mapper_opts.factor_iters = opts.factor_iters;

    // code factor
    mapper_opts.code_factor_weight = opts.code_factor_weight;

    /* keyframe connections */
    mapper_opts.temporal_max_back_connections = opts.temporal_max_back_connections;
    /* ISAM2 */
    /*
    * If this is SEARCH_EACH_ITERATION, then the trust region radius will be increased potentially 
    * multiple times during one iteration until increasing it further no longer decreases the error. 
    * If this is ONE_STEP_PER_ITERATION, then the step in one iteration will not exceed the current trust region radius, 
    * but the radius will be increased for the next iteration if the error decrease is good. 
    * The former will generally result in slower iterations, but sometimes larger steps in early iterations. 
    * The latter generally results in faster iterations but it may take several iterations 
    * before the trust region radius is increased to the optimal value. Generally ONE_STEP_PER_ITERATION should be used, 
    * corresponding to most published descriptions of the algorithm.
    */
    //   double initialDelta; ///< The initial trust region radius for Dogleg
    //   double wildfireThreshold; ///< Continue updating the linear delta only when changes are above this threshold (default: 0.00001)
    mapper_opts.isam_params.optimizationParams =
        gtsam::ISAM2DoglegParams(1.0, opts.isam_wildfire_threshold, gtsam::DoglegOptimizerImpl::ONE_STEP_PER_ITERATION, false);
    mapper_opts.isam_params.enableDetailedResults = opts.isam_enable_detailed_results;
    // partial relin check will speed up the optimization but reduce correctness of the results
    mapper_opts.isam_params.enablePartialRelinearizationCheck = opts.isam_partial_relin_check;
    mapper_opts.isam_params.factorization = gtsam::ISAM2Params::CHOLESKY;
    mapper_opts.isam_params.cacheLinearizedFactors = true;
    mapper_opts.isam_params.findUnusedFactorSlots = false;
    // How many updates to skip before relinearize happens
    mapper_opts.isam_params.relinearizeSkip = opts.isam_relinearize_skip;
    // threshold for relinearization
    gtsam::FastMap<char, gtsam::Vector> thresholds;
    thresholds['p'] = gtsam::Vector::Ones(6) * opts.isam_pose_lin_eps;
    thresholds['a'] = gtsam::Vector::Ones(6) * opts.isam_pose_lin_eps;
    thresholds['c'] = gtsam::Vector::Ones(CS) * opts.isam_code_lin_eps;
    thresholds['s'] = gtsam::Vector::Ones(1) * opts.isam_scale_lin_eps;
    mapper_opts.isam_params.relinearizeThreshold = thresholds;

    // TEASER++ related
    mapper_opts.teaser_max_clique_time_limit = opts.teaser_max_clique_time_limit;
    mapper_opts.teaser_kcore_heuristic_threshold = opts.teaser_kcore_heuristic_threshold;
    mapper_opts.teaser_rotation_max_iterations = opts.teaser_rotation_max_iterations;
    mapper_opts.teaser_rotation_cost_threshold = opts.teaser_rotation_cost_threshold;
    mapper_opts.teaser_rotation_gnc_factor = opts.teaser_rotation_gnc_factor;
    mapper_opts.teaser_rotation_estimation_algorithm = opts.teaser_rotation_estimation_algorithm;
    mapper_opts.teaser_rotation_tim_graph = opts.teaser_rotation_tim_graph;
    mapper_opts.teaser_inlier_selection_mode = opts.teaser_inlier_selection_mode;
    mapper_opts.teaser_noise_bound_multiplier = opts.teaser_noise_bound_multiplier;

    mapper_ = std::make_unique<MapperT>(mapper_opts, video_mask, output_cam_pyr_, depth_network_ptr_, feature_network_ptr_);

    // loop detector
    LoopDetectorConfig detector_cfg;
    detector_cfg.dpt_eps = opts.warp_dpt_eps;
    detector_cfg.temporal_max_back_connections = opts.temporal_max_back_connections;
    detector_cfg.max_candidates = opts.loop_max_candidates;
    detector_cfg.min_area_ratio = opts.loop_min_area_ratio;
    detector_cfg.min_inlier_ratio = opts.loop_min_inlier_ratio;
    detector_cfg.min_desc_inlier_ratio = opts.loop_min_desc_inlier_ratio;
    detector_cfg.global_redundant_range = opts.global_redundant_range;
    detector_cfg.use_match_geom = opts.loop_use_match_geom;
    detector_cfg.local_active_window = opts.loop_local_active_window;
    detector_cfg.local_dist_ratio = opts.loop_local_dist_ratio;
    detector_cfg.global_active_window = opts.loop_global_active_window;
    detector_cfg.trans_weight = opts.pose_dist_trans_weight;
    detector_cfg.rot_weight = opts.pose_dist_rot_weight;
    detector_cfg.global_sim_ratio = opts.loop_global_sim_ratio;
    detector_cfg.global_metric_ratio = opts.loop_global_metric_ratio;
    detector_cfg.local_metric_ratio = opts.loop_local_metric_ratio;
    
    // Here the device is used to store descriptors
    detector_cfg.device = c10::Device(torch::kCUDA, opts.cuda_id);
    detector_cfg.tracker_cfg = tracker_cfg;
    detector_cfg.tracker_cfg.max_num_iters = opts.loop_tracking_max_num_iters;
    detector_cfg.tracker_cfg.min_grad_thresh = opts.loop_tracking_min_grad_thresh;
    detector_cfg.tracker_cfg.min_param_inc_thresh = opts.loop_tracking_min_param_inc_thresh;
    detector_cfg.tracker_cfg.match_geom_factor_weight = opts.match_geom_factor_weight;
    detector_cfg.tracker_cfg.match_geom_loss_param_factor = opts.match_geom_loss_param_factor;
    detector_cfg.tracker_cfg.reproj_factor_weight = opts.reproj_factor_weight;
    detector_cfg.tracker_cfg.reproj_loss_param_factor = opts.reproj_loss_param_factor;
    detector_cfg.tracker_cfg.teaser_inlier_selection_mode = opts.teaser_inlier_selection_mode;
    detector_cfg.tracker_cfg.desc_num_samples = opts.desc_num_keypoints;
    detector_cfg.tracker_cfg.damp_dec_factor = opts.loop_tracking_damp_dec_inc_factor[0];
    detector_cfg.tracker_cfg.damp_inc_factor = opts.loop_tracking_damp_dec_inc_factor[1];

    try
    {
      loop_detector_ = std::make_unique<LoopDetectorT>(opts.vocabulary_path, mapper_->GetMap(), detector_cfg);
    }
    catch (std::exception &e)
    {
      LOG(FATAL) << "[DeepFactors<Scalar, CS>::Init] Loop detector initialization failed";
    }
  
    if (opts.use_global_loop)
    {
      pthread_create(&global_loop_detect_thread_, NULL, &DeepFactors<Scalar, CS>::GlobalLoopDetectBackendWrapper, this);
    }
    
    if (opts.use_local_loop)
    {
      pthread_create(&local_loop_detect_thread_, NULL, &DeepFactors<Scalar, CS>::LocalLoopDetectBackendWrapper, this);
    }
    
    pthread_create(&mapping_thread_, NULL, &DeepFactors<Scalar, CS>::MappingBackendWrapper, this);
    
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::Reset()
  {
    // clear all data
    loop_detector_->Reset();
    tracker_->Reset();
    mapper_->Reset();
    NotifyMapObservers();
    bootstrapped_ = false;
    tracking_lost_ = false;
    force_keyframe_ = false;
  }

  template <typename Scalar, int CS>
  bool DeepFactors<Scalar, CS>::RefineMapping()
  {
    std::vector<KeyframeId> empty;
    static int num_no_linearize = opts_.refine_mapping_num_no_linearize;
    static int num_refinement = opts_.refine_mapping_iters;

    if (opts_.enable_gui && VLOG_IS_ON(1))
    {
      cv::Mat warp_image = loop_detector_->GetWarpImage();
      if (!warp_image.empty())
      {
        cv::imshow("loop detection", warp_image);
        cv::waitKey(1);
      }
    }

    if (--num_refinement >= 0)
    {
      tic("[DeepFactors<Scalar, CS>::RefineMapping] mapping step");
      mapper_->MappingStep(empty, false);
      toc("[DeepFactors<Scalar, CS>::RefineMapping] mapping step");

      NotifyMapObservers();

      if (opts_.isam_enable_detailed_results)
      {
        stats_.relin_info = mapper_->KeyframeRelinearization();
        NotifyStatsObservers();
      }

      if (mapper_->GetResults().variablesRelinearized == 0)
      {
        --num_no_linearize;
        if (num_no_linearize <= 0)
        {
          VLOG(1) << "[DeepFactors<Scalar, CS>::RefineMapping] Refine mapping converged";
          return true;
        }
      }

      return false;
    }
    else
    {
      return true;
    }
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::ProcessFrame(double timestamp, const cv::Mat &frame)
  {

    if (!bootstrapped_)
    {
      throw std::runtime_error("[DeepFactors<Scalar, CS>::ProcessFrame] Calling ProcessFrame before system is bootstrapped!");
    }

    tic("[DeepFactors<Scalar, CS>::ProcessFrame] Build frame");
    // build a frame pointer to be used in trackframe or relocalize
    live_frame_ptr_ = mapper_->BuildFrame(timestamp, frame, mapper_->GetMap()->keyframes.Get(curr_kf_)->pose_wk * pose_kc_);
    toc("[DeepFactors<Scalar, CS>::ProcessFrame] Build frame");

    // pose_kc_ and curr_kf_ has been modified inside TrackFrame()
    TrackFrame();
    auto curr_kf = mapper_->GetMap()->keyframes.Get(curr_kf_);
    // We accept the new tracking as good
    NotifyPoseObservers(curr_kf->pose_wk * pose_kc_);

    {
      std::unique_lock<std::shared_mutex> lock(visited_keyframes_mutex_);
      if (visited_keyframe_ids_.back() != curr_kf_)
      {
        visited_keyframe_ids_.push_back(curr_kf_);
      }
    }

    if (opts_.enable_gui && VLOG_IS_ON(3))
    {
      cv::Mat warp_image = tracker_->GetWarpImage();
      if (!warp_image.empty())
      {
        cv::imshow("camera tracking", warp_image);
        cv::waitKey(1);
      }
    }

    if (opts_.enable_gui && VLOG_IS_ON(1))
    {
      cv::Mat warp_image = loop_detector_->GetWarpImage();
      if (!warp_image.empty())
      {
        cv::imshow("loop detection", warp_image);
        cv::waitKey(1);
      }
    }

    // Check whether we should add a new keyframe
    if (NewKeyframeRequired())
    {
      // If a new keyframe is required, we should update the live_frame_ptr
      // and calculate other required variables that are not in the frame class

      std::unique_lock<std::shared_mutex> lock(global_loop_mutex_);
      std::vector<KeyframeId> back_conns;
      auto reverse_iter = visited_keyframe_ids_.rbegin();

      // Find the back connections that are within the local active window to prevent some erroneous connections
      int trial_count = 0;
      while (true)
      {
        if (reverse_iter == visited_keyframe_ids_.rend())
        {
          break;
        }

        if (std::find(back_conns.begin(), back_conns.end(), *reverse_iter) == back_conns.end())
        {
          if (back_conns.empty())
          {
            back_conns.push_back(*reverse_iter);
            trial_count += 1;
          }
          else
          {
            // Check inlier ratio etc before connecting
            tracker_->SetRefKeyframe(mapper_->GetMap()->keyframes.Get(*reverse_iter));
            tracker_->SetWorldPose(curr_kf->pose_wk);

            if (tracker_->MatchGeoCheck(*curr_kf))
            {
              // Add a match geometry matching pre-check to save computation when inlier ratio is small
              if (tracker_->GetDescInlierRatio() >= opts_.temporal_min_desc_inlier_ratio)
              {
                back_conns.push_back(*reverse_iter);
              }
              else
              {
                VLOG(2) << "[DeepFactors<Scalar, CS>::ProcessFrame] small desc inlier ratio -- Temporal connection candidate keyframe " << *reverse_iter << " not accepted " << tracker_->GetDescInlierRatio();
              }
            }
            else
            {
              VLOG(2) << "[DeepFactors<Scalar, CS>::ProcessFrame] desc matching failed -- Temporal connection candidate keyframe " << *reverse_iter << " not accepted";
            }
            trial_count += 1;
          }

          if (trial_count >= static_cast<int>(opts_.temporal_max_back_connections))
          {
            break;
          }
        }

        ++reverse_iter;
      }

      // back_conns
      typename KeyframeT::Ptr kf;
      {
        // std::unique_lock<std::shared_mutex> lock(mapper_->new_factor_mutex_);
        kf = mapper_->EnqueueKeyframe(live_frame_ptr_, back_conns);
        // kf->pose_wk = curr_kf->pose_wk * pose_kc_;
        // Scalar ori_dpt_scale = kf->dpt_scale;
        // kf->dpt_scale = curr_kf->dpt_scale * kf->scale_ratio_cur_ref;
        // if (kf->dpt_scale / ori_dpt_scale < 0.999 || kf->dpt_scale / ori_dpt_scale > 1.001)
        // {
        //   VLOG(2) << "[DeepFactors<Scalar, CS>::ProcessFrame] reference keyframe dpt scale changed during new keyframe creation: ori scale, new scale" << ori_dpt_scale << " " << kf->dpt_scale;
        //   kf->dpt_map = kf->dpt_map * (kf->dpt_scale / ori_dpt_scale);
        // }
      }

      VLOG(1) << "[DeepFactors<Scalar, CS>::ProcessFrame] Tracker switching from keyframe " << curr_kf_ << " to new keyframe " << kf->id << " to track";

      SE3T pose_wc = curr_kf->pose_wk * pose_kc_;
      pose_kc_ = mapper_->GetMap()->keyframes.Get(kf->id)->pose_wk.inverse() * pose_wc;
      {
        // std::unique_lock<std::shared_mutex> lock(kf_id_mutex_);
        curr_kf_ = kf->id;
      }

      VLOG(2) << "[DeepFactors<Scalar, CS>::ProcessFrame] Back connections for new keyframe " << kf->id << " : " << back_conns;

      enable_loop_ = true;

      NotifyMapObservers();

      {
        std::unique_lock<std::shared_mutex> lock(visited_keyframes_mutex_);
        if (visited_keyframe_ids_.back() != curr_kf_)
        {
          visited_keyframe_ids_.push_back(curr_kf_);
        }
      }

      // WARNING: this return statement must be kept to maintain some following logics!
      return;
    }

    enable_mapping_ = true;

    if (opts_.enable_gui && VLOG_IS_ON(3))
    {
      const cv::Mat &keyframes_display = mapper_->GetDisplayKeyframes();
      if (!keyframes_display.empty())
      {
        cv::imshow("keyframes", keyframes_display);
      }

      tic("[DeepFactors<Scalar, CS>::ProcessFrame] Draw debug images");
      DebugImages dbg_imgs;

      if (loop_links.size() > 0)
      {
        mapper_->DisplayLoopClosures(loop_links, 10);
        // display loop closure links
        cv::Mat result =
            DisplayPairs(mapper_->GetMap(), loop_links, *(mapper_->output_mask_ptr_), output_cam_pyr_[0], opts_.warp_dpt_eps, 10, mapper_->GetCheckerboard());
        cv::imshow("se3 warping", result);
      }

      if (opts_.use_reprojection)
      {
        // dbg_imgs.reprojection_errors = mapper_->DisplayMatchGeometryErrors();
        dbg_imgs.reprojection_errors = mapper_->DisplayReprojectionErrors();
      }

      if (opts_.use_photometric)
      {
        dbg_imgs.photometric_errors = mapper_->DisplayPhotometricErrors();
      }

      debug_buffer_.push_back(dbg_imgs);

      if (debug_buffer_.size() > 20)
      {
        debug_buffer_.pop_front();
      }

      toc("[DeepFactors<Scalar, CS>::ProcessFrame] Draw debug images");
    }

    if (opts_.enable_gui && VLOG_IS_ON(3))
    {
      cv::waitKey(1);
    }
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::ForceKeyframe()
  {
    force_keyframe_ = true;
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::BootstrapOneFrame(double timestamp, const cv::Mat &img)
  {
    CHECK_EQ(img.type(), CV_8UC3) << "[DeepFactors<Scalar, CS>::BootstrapOneFrame] Invalid input image to DeepFactors";

    Reset();

    mapper_->InitOneFrame(timestamp, img);
    bootstrapped_ = true;

    curr_kf_ = mapper_->GetMap()->keyframes.LastId();
    tracker_->SetRefKeyframe(mapper_->GetMap()->keyframes.Get(curr_kf_));
    tracker_->Reset();
    pose_kc_ = tracker_->GetRelativePoseEstimate();

    {
      std::unique_lock<std::shared_mutex> lock(visited_keyframes_mutex_);
      visited_keyframe_ids_.push_back(curr_kf_);
    }
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::SetOptions(DeepFactorsOptions new_opts)
  {
    if (new_opts.cuda_id != opts_.cuda_id ||
        new_opts.depth_network_path != opts_.depth_network_path ||
        new_opts.feature_network_path != opts_.feature_network_path)
    {
      LOG(FATAL) << "[DeepFactors<Scalar, CS>::SetOptions] Online changes to GPU or network path are not allowed";
    }

    // save the options
    opts_ = new_opts;

    // set new tracking options
    df::CameraTracker::TrackerConfig tracker_cfg;
    tracker_cfg.cuda_id = new_opts.cuda_id;
    tracker_cfg.max_num_iters = new_opts.tracking_max_num_iters;
    tracker_cfg.min_grad_thresh = new_opts.tracking_min_grad_thresh;
    tracker_cfg.min_param_inc_thresh = new_opts.tracking_min_param_inc_thresh;
    tracker_cfg.min_damp = new_opts.tracking_min_max_damp[0];
    tracker_cfg.max_damp = new_opts.tracking_min_max_damp[1];
    tracker_cfg.init_damp = new_opts.tracking_init_damp;
    tracker_cfg.damp_dec_factor = new_opts.tracking_damp_dec_inc_factor[0];
    tracker_cfg.damp_inc_factor = new_opts.tracking_damp_dec_inc_factor[1];
    tracker_cfg.jac_update_err_inc_threshold = new_opts.tracking_jac_update_err_inc_threshold;
    tracker_cfg.dpt_eps = new_opts.warp_dpt_eps;
    tracker_cfg.photo_factor_weights = new_opts.photo_factor_weights;
    tracker_cfg.match_geom_factor_weight = new_opts.tracker_match_geom_factor_weight;
    tracker_cfg.match_geom_loss_param_factor = new_opts.match_geom_loss_param_factor;
    tracker_cfg.reproj_factor_weight = new_opts.reproj_factor_weight;
    tracker_cfg.reproj_loss_param_factor = new_opts.reproj_loss_param_factor;
    tracker_cfg.desc_num_samples = new_opts.tracking_desc_num_keypoints;
    tracker_cfg.desc_cyc_consis_thresh = new_opts.desc_cyc_consis_thresh;
    tracker_cfg.net_output_size = new_opts.net_output_size;
    // TEASER++ related
    tracker_cfg.teaser_max_clique_time_limit = new_opts.teaser_max_clique_time_limit;
    tracker_cfg.teaser_kcore_heuristic_threshold = new_opts.teaser_kcore_heuristic_threshold;
    tracker_cfg.teaser_rotation_max_iterations = new_opts.teaser_rotation_max_iterations;
    tracker_cfg.teaser_rotation_cost_threshold = new_opts.teaser_rotation_cost_threshold;
    tracker_cfg.teaser_rotation_gnc_factor = new_opts.teaser_rotation_gnc_factor;
    tracker_cfg.teaser_rotation_estimation_algorithm = new_opts.teaser_rotation_estimation_algorithm;
    tracker_cfg.teaser_rotation_tim_graph = new_opts.teaser_rotation_tim_graph;
    tracker_cfg.teaser_inlier_selection_mode = new_opts.teaser_tracker_inlier_selection_mode;

    tracker_->SetConfig(tracker_cfg);
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::NotifyPoseObservers(const SE3T &pose_wc)
  {
    if (pose_callback_)
    {
      pose_callback_(pose_wc);
    }
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::NotifyMapObservers()
  {
    if (map_callback_)
    {
      map_callback_(mapper_->GetMap());
    }
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::NotifyStatsObservers()
  {
    if (stats_callback_)
    {
      stats_callback_(stats_);
    }
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::SaveInfo(std::string dir)
  {
    // display photometric factor errors
    if (opts_.use_photometric)
    {
      std::string photo_dir = dir + "/photo";
      df::CreateDirIfNotExists(photo_dir);
      mapper_->SavePhotometricDebug(photo_dir);
    }

    // display reprojection factor errors and matches
    if (opts_.use_reprojection)
    {
      std::string rep_dir = dir + "/repr";
      df::CreateDirIfNotExists(rep_dir);
      // mapper_->SaveMatchGeometryDebug(rep_dir);
      mapper_->SaveReprojectionDebug(rep_dir);
    }

    // save all keyframes
    SaveKeyframes(dir);

    // save all debug images
    int num = 0;
    for (auto it = debug_buffer_.rbegin(); it != debug_buffer_.rend(); ++it)
    {
      auto &buf = *it;
      std::string rep_name = dir + "/reprojection_errors_" + std::to_string(num) + ".png";
      std::string pho_name = dir + "/photometric_errors_" + std::to_string(num) + ".png";
      num += 1;

      if (opts_.use_reprojection)
      {
        cv::imwrite(rep_name, buf.reprojection_errors);
      }

      if (opts_.use_photometric)
      {
        cv::imwrite(pho_name, buf.photometric_errors);
      }
    }
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::SaveKeyframes(std::string dir)
  {
    // create directory
    std::string keyframe_dir = dir + "/keyframes";
    df::CreateDirIfNotExists(keyframe_dir);

    // save keyframes
    for (auto &kf : GetMap()->keyframes)
    {
      const at::Tensor video_mask = *(kf.second->video_mask_ptr);
      std::string timestamp_str = std::to_string(kf.second->id) + "_" + std::to_string(static_cast<long>(kf.second->timestamp));
      std::string rgb_filename = "rgb_" + timestamp_str + ".png";
      std::string display_dpt_filename = "display_dpt_" + timestamp_str + ".png";
      std::string dpt_filename = "dpt_" + timestamp_str + ".pt";

      cv::imwrite(keyframe_dir + "/" + rgb_filename, kf.second->color_img);

      const at::Tensor masked_dpt_map = kf.second->dpt_map * video_mask;
      torch::save(masked_dpt_map, keyframe_dir + "/" + dpt_filename);
      const at::Tensor max = torch::max(masked_dpt_map);
      float max_dpt = max.item<float>();
      cv::Mat dpt_display = Tensor2Mat(torch::clamp_min(masked_dpt_map / max_dpt, 0.0));
      cv::imwrite(keyframe_dir + "/" + display_dpt_filename, apply_colormap(dpt_display));
    }

    {
      std::ofstream intr_file(keyframe_dir + "/intrinsics.txt");
      intr_file << output_cam_.fx() << " " << output_cam_.fy() << " " << output_cam_.u0() << " "
                << output_cam_.v0() << " " << output_cam_.width() << " " << output_cam_.height();
    }
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::SaveResults(std::string dir)
  {
    // save trajectory file
    {
      std::ofstream f(dir + "/stamped_traj_estimate.txt");
      std::set<KeyframeId> kf_set(visited_keyframe_ids_.begin(), visited_keyframe_ids_.end());
      std::vector<KeyframeId> kf_vec(kf_set.begin(), kf_set.end());
      std::sort(kf_vec.begin(), kf_vec.end());

      for (auto &kf_id : kf_vec)
      {
        auto kf = GetMap()->keyframes.Get(kf_id);
        // save keyframe pose to trajectory file
        auto pose_wk = kf->pose_wk;
        Eigen::Quaternionf q = pose_wk.so3().unit_quaternion();
        Eigen::Vector3f t = pose_wk.translation();
        TumPose pose{kf->timestamp, q, t};
        f << pose << std::endl;
      }
    }

    // save keyframes
    SaveKeyframes(dir);

    LOG(INFO) << "[DeepFactors<Scalar, CS>::SaveResults] Saved results to " << dir;
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  void DeepFactors<Scalar, CS>::TrackFrame()
  {
    SE3T new_pose_kc;
    bool geo_checked = false;
    // find the active keyframe to track against
    tic("[DeepFactors<Scalar, CS>::TrackFrame] Select keyframe");
    auto newkfid = SelectKeyframe(geo_checked);
    toc("[DeepFactors<Scalar, CS>::TrackFrame] Select keyframe");
    if (VLOG_IS_ON(1) && newkfid != curr_kf_)
    {
      VLOG(1) << "[DeepFactors<Scalar, CS>::TrackFrame] Tracker switching from keyframe " << curr_kf_ << " to previous keyframe " << newkfid << " to track";
    }

    auto prev_kf = mapper_->GetMap()->keyframes.Get(curr_kf_);
    auto new_kf = mapper_->GetMap()->keyframes.Get(newkfid);

    const SE3T pose_wc = prev_kf->pose_wk * pose_kc_;
    curr_kf_ = newkfid;

    // We do not reset the pose_ck_ here because we want to keep pose_wc.
    tracker_->SetRefKeyframe(new_kf);
    tracker_->SetWorldPose(pose_wc);
    // track live frame against kf
    tic("[DeepFactors<Scalar, CS>::TrackFrame] Tracking current frame");
    tracker_->TrackNewFrame(*live_frame_ptr_, opts_.use_tracker_photometric, opts_.use_tracker_reprojection, geo_checked, true, VLOG_IS_ON(3));
    toc("[DeepFactors<Scalar, CS>::TrackFrame] Tracking current frame");
    pose_kc_ = tracker_->GetRelativePoseEstimate().inverse();
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  bool DeepFactors<Scalar, CS>::NewKeyframeRequired()
  {
    if (force_keyframe_)
    {
      force_keyframe_ = false;
      return true;
    }

    float warp_area_ratio = tracker_->GetAreaRatio();
    float inlier_ratio = tracker_->GetInlierRatio();
    float desc_inlier_ratio = tracker_->GetRelDescInlierRatio();
    float average_motion = tracker_->GetAverageMotion();

    switch (opts_.keyframe_mode)
    {
    case DeepFactorsOptions::AUTO:
    {
      const bool frame_too_far = (warp_area_ratio < opts_.new_kf_max_area_ratio) ||
                                 (inlier_ratio < opts_.new_kf_max_inlier_ratio) ||
                                 (average_motion > opts_.new_kf_min_average_motion);
      const bool frame_too_unlike = desc_inlier_ratio < opts_.new_kf_max_desc_inlier_ratio;
      if (!frame_too_far && !frame_too_unlike)
      {
        VLOG(2) << "[DeepFactors<Scalar, CS>::NewKeyframeRequired] current frame is too close and too similar to the current keyframe, new keyframe not required. Warp ratio, inlier ratio, motion, rel desc ratio: " << warp_area_ratio << " " << inlier_ratio << " " << average_motion << " " << desc_inlier_ratio;
        return false;
      }

      VLOG(2) << "[DeepFactors<Scalar, CS>::NewKeyframeRequired] NEW keyframe required. Warp ratio, inlier ratio, motion, rel desc ratio: " << warp_area_ratio << " " << inlier_ratio << " " << average_motion << " " << desc_inlier_ratio;
      return true;
    }
    case DeepFactorsOptions::NEVER:
      return false;
    }

    return false;
  }

  /* ************************************************************************* */
  template <typename Scalar, int CS>
  typename DeepFactors<Scalar, CS>::KeyframeId DeepFactors<Scalar, CS>::SelectKeyframe(bool &geo_checked)
  {
    geo_checked = false;
    KeyframeId kfid = curr_kf_;
    if (opts_.tracking_mode == DeepFactorsOptions::LAST)
    {
      kfid = mapper_->GetMap()->keyframes.LastId();
    }
    else if (opts_.tracking_mode == DeepFactorsOptions::CLOSEST)
    {
      // find closest keyframe
      const auto kfmap = mapper_->GetMap();
      Scalar closest_dist = std::numeric_limits<Scalar>::infinity();

      for (auto id : kfmap->keyframes.Ids())
      {
        Scalar dist = df::PoseDistance(kfmap->keyframes.Get(id)->pose_wk, kfmap->keyframes.Get(curr_kf_)->pose_wk * pose_kc_,
                                       opts_.pose_dist_trans_weight, opts_.pose_dist_rot_weight);
        if (dist < closest_dist)
        {
          closest_dist = dist;
          kfid = id;
        }
      }

      // We need to do geometry check to see if the selected keyframe is indeed spatially closer
      // No need to check if the kfid is the latest keyframe
      if (kfid != curr_kf_)
      {
        tracker_->SetRefKeyframe(kfmap->keyframes.Get(kfid));
        tracker_->SetWorldPose(kfmap->keyframes.Get(curr_kf_)->pose_wk * pose_kc_);
        if (!tracker_->TrackMatchGeoCheck(*live_frame_ptr_))
        {
          geo_checked = false;
          return curr_kf_;
        }

        Scalar desc_inlier_ratio = tracker_->GetDescInlierRatio();

        tracker_->SetRefKeyframe(kfmap->keyframes.Get(curr_kf_));
        tracker_->SetWorldPose(kfmap->keyframes.Get(curr_kf_)->pose_wk * pose_kc_);
        if (!tracker_->TrackMatchGeoCheck(*live_frame_ptr_))
        {
          LOG(WARNING) << "[DeepFactors<Scalar, CS>::SelectKeyframe] Camera tracking failed against the current keyframe";
        }

        if (tracker_->GetDescInlierRatio() >= opts_.tracking_ref_kf_select_ratio * desc_inlier_ratio)
        {
          VLOG(2) << "[DeepFactors<Scalar, CS>::SelectKeyframe] keyframe " << kfid << " is not closer than the current keyframe "
                  << curr_kf_ << " for tracking";
          geo_checked = true;
          return curr_kf_;
        }
        else
        {
          VLOG(2) << "[DeepFactors<Scalar, CS>::SelectKeyframe] keyframe " << kfid << " is closer than the current keyframe "
                  << curr_kf_ << " for tracking: curr inlier ratio: " << tracker_->GetDescInlierRatio() << " new: " << desc_inlier_ratio;
          geo_checked = false;
          return kfid;
        }
      }
    }
    else if (opts_.tracking_mode == DeepFactorsOptions::FIRST)
    {
      kfid = mapper_->GetMap()->keyframes.Ids()[0];
    }
    else
    {
      LOG(FATAL) << "[DeepFactors<Scalar, CS>::SelectKeyframe] Unhandled tracking mode";
    }

    return kfid;
  }

  // explicit instantiation
  template class DeepFactors<float, DF_CODE_SIZE>;

} // namespace df
