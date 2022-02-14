#include "loop_detector.h"

namespace df
{

  template <typename Scalar>
  LoopDetector<Scalar>::LoopDetector(std::string voc_path, MapPtr map, LoopDetectorConfig cfg)
      : voc_(voc_path, cfg.device), map_(map), cfg_(cfg), db_(voc_, false, 0)
  {
    // Set which device the descriptors will be stored in
    voc_.setDevice(cfg.device);
    local_tracker_ = std::make_shared<CameraTracker>(cfg.tracker_cfg, false);
    local_tracker_->SetName("local loop");
    global_tracker_ = std::make_shared<CameraTracker>(cfg.tracker_cfg, false);
    global_tracker_->SetName("global loop");
  }

  /* ************************************************************************* */
  template <typename Scalar>
  void LoopDetector<Scalar>::AddKeyframe(const KeyframePtr &kf)
  {
    using namespace torch::indexing;
    torch::NoGradGuard no_grad;
    // Only add keyframe if it is not in the dmap_ yet
    {
      {
        if (dmap_.count(kf->id) != 0)
        {
          return;
        }
      }

      long channel = kf->feat_desc.size(1);
      DBoW2::BowVector bow_vec;
      const at::Tensor features =
          kf->feat_desc.reshape({channel, -1}).index({Slice(), kf->valid_locations_1d}).to(cfg_.device);
      voc_.transform(features, bow_vec);

      db_.add(bow_vec);
      dmap_[kf->id] = bow_vec;
    }
  }

  template <typename Scalar>
  bool LoopDetector<Scalar>::CheckExist(const KeyframeId id)
  {
    return dmap_.count(id) != 0;
  }

  /* ************************************************************************* */
  template <typename Scalar>
  std::vector<typename LoopDetector<Scalar>::LoopInfo>
  LoopDetector<Scalar>::DetectLoop(const KeyframePtr &curr_kf)
  {
    using namespace torch::indexing;
    VLOG(3) << "[LoopDetector<Scalar>::DetectLoop] Finding maximum similar temporal neighbour for keyframe " << curr_kf->id;
    Scalar max_sim = 0;
    KeyframeId max_id = -1;

    auto conns = curr_kf->temporal_connections;
    for (auto c : conns)
    {
      auto sim = voc_.score(dmap_[c], dmap_[curr_kf->id]);
      VLOG(3) << "[LoopDetector<Scalar>::DetectLoop] Connection to " << c << " score = " << sim;
      if (sim > max_sim)
      {
        max_sim = sim;
        max_id = c;
      }
    }
    VLOG(2) << "[LoopDetector<Scalar>::DetectLoop] Lowest similarity to neighbours = " << max_sim;

    // Find several candidates
    DBoW2::QueryResults ret;
    // Here the bow of the current keyframe should already be in the database
    {
      db_.query(dmap_[curr_kf->id], ret, cfg_.max_candidates);
    }

    // Pre-filtering
    VLOG(2) << "[LoopDetector<Scalar>::DetectLoop] Checking " << ret.size() << " candidates from DBoW";
    std::vector<LoopInfo> global_loops;
    std::vector<KeyframeId> candidates;
    cv::Mat best_warp_image;
    for (auto &res : ret)
    {
      // Looks like keyframe id is one-based and res.Id is zero-based?
      auto kfid = res.Id + 1;

      // We probably shouldn't use the visited keyframe ids here because they may be temporally far away from each other
      if (labs(static_cast<long>(kfid) - static_cast<long>(curr_kf->id)) < cfg_.global_active_window)
      {
        VLOG(3) << "[LoopDetector<Scalar>::DetectLoop] Keyframe " << kfid << " within global active window";
        continue;
      }

      if (res.Score < cfg_.global_sim_ratio * max_sim)
      {
        VLOG(2) << "[LoopDetector<Scalar>::DetectLoop] Keyframe " << kfid << " similarity: " << res.Score << " smaller than " << max_sim << ", no need to check the rest of candidates";
        break;
      }

      if (map_->keyframes.LinkExists(curr_kf->id, kfid))
      {
        VLOG(2) << "[LoopDetector<Scalar>::DetectLoop] Link between " << curr_kf->id << " and " << kfid << " already exists";
        continue;
      }

      VLOG(2) << "[LoopDetector<Scalar>::DetectLoop] Global loop candidate keyframe " << kfid << " similarity: " << res.Score;
      candidates.push_back(kfid);
    }

    if (candidates.empty())
    {
      return global_loops;
    }

    // The matching needs to beat the best metric out of the temporal connection to be considered valid
    auto kf = map_->keyframes.Get(max_id);
    global_tracker_->SetRefKeyframe(kf);
    global_tracker_->SetWorldPose(curr_kf->pose_wk);

    if (!global_tracker_->TrackFrame(*curr_kf, true, cfg_.use_match_geom,
                                     false, false, false))
    {
      // Skip it if tracking failed (should mean no valid desc matches are found or no overlap with match geom disabled)
      return global_loops;
    }

    Scalar base_metric = cfg_.global_metric_ratio * global_tracker_->GetAreaRatio() * global_tracker_->GetInlierRatio();
    Scalar base_desc_ratio = cfg_.global_metric_ratio * global_tracker_->GetDescInlierRatio();

    VLOG(2) << "[LoopDetector<Scalar>::DetectLoop] Attenuated base overlap ratio, desc ratio: " << base_metric << " " << base_desc_ratio;

    // Sort the candidates so that it start from the earliest keyframe
    std::sort(candidates.begin(), candidates.end());
    // do geometry check
    VLOG(2) << "[LoopDetector<Scalar>::DetectLoop] Performing geometry checks on " << candidates.size() << " candidates: " << candidates;

    // Set the pose_wc in tracker_ to be the same as the query keyframe that looks for a loop closure candidate
    // Maybe close the one loop pair that has furthest spatial distance before tracking?
    bool detected = false;
    for (auto &id : candidates)
    {
      auto kf = map_->keyframes.Get(id);

      if (id != kf->id)
      {
        LOG(FATAL) << "[LoopDetector<Scalar>::DetectLoop] id is not equal to kf->id!";
      }

      VLOG(2) << "[LoopDetector<Scalar>::DetectLoop] Track current keyframe " << curr_kf->id << " against global loop candidate keyframe " << kf->id;

      global_tracker_->SetRefKeyframe(kf);
      global_tracker_->SetWorldPose(curr_kf->pose_wk);

      // Pre check on match geometry inlier ratio for computation speedup
      global_tracker_->MatchGeoCheck(*curr_kf);
      if (global_tracker_->GetDescInlierRatio() < cfg_.min_desc_inlier_ratio || global_tracker_->GetDescInlierRatio() < base_desc_ratio)
      {
        VLOG(2) << "[LoopDetector<Scalar>::DetectLoop] small desc inlier ratio -- Global loop candidate keyframe " << kf->id << " not accepted " << global_tracker_->GetDescInlierRatio();
        continue;
      }

      if (!global_tracker_->TrackFrame(*curr_kf, true, cfg_.use_match_geom,
                                       true, false, true))
      {
        // Skip it if tracking failed (should mean no valid desc matches are found or no overlap with match geom disabled)
        VLOG(2) << "[LoopDetector<Scalar>::DetectLoop] Geometry check on " << kf->id << " failed";
        continue;
      }

      if (global_tracker_->GetAreaRatio() < cfg_.min_area_ratio || global_tracker_->GetInlierRatio() < cfg_.min_inlier_ratio)
      {
        VLOG(2) << "[LoopDetector<Scalar>::DetectLoop] small scene overlap -- Global loop candidate keyframe " << kf->id << " not accepted " << global_tracker_->GetAreaRatio() << " " << global_tracker_->GetInlierRatio();
        continue;
      }

      const Scalar metric = global_tracker_->GetAreaRatio() * global_tracker_->GetInlierRatio();
      const Scalar desc_ratio = global_tracker_->GetDescInlierRatio();
      VLOG(2) << "[LoopDetector<Scalar>::DetectLoop] Before verification keyframe candidate overlap ratio: " << metric << " desc inlier ratio: " << desc_ratio;

      if (metric >= base_metric && desc_ratio >= base_desc_ratio)
      {
        detected = true;
        LoopInfo loop_info;
        loop_info.detected = true;
        loop_info.id_ref = id;
        SE3T temp_cur_ref_pose = global_tracker_->GetRelativePoseEstimate();
        auto temp_kf = global_tracker_->GetRefKeyframe();
        const Scalar temp_ref_scale = global_tracker_->GetRefScale();
        temp_cur_ref_pose.translation() = temp_cur_ref_pose.translation() * temp_kf->dpt_scale / temp_ref_scale;
        loop_info.pose_cur_ref = temp_cur_ref_pose;
        loop_info.query_scale = CorrectDepthScale(curr_kf, temp_kf, temp_cur_ref_pose, cfg_.dpt_eps);
        loop_info.ref_scale = temp_kf->dpt_scale;
        loop_info.desc_inlier_ratio = desc_ratio;
        warp_image_ = global_tracker_->GetWarpImage();
        global_loops.push_back(loop_info);
        VLOG(2) << "[LoopDetector<Scalar>::DetectLoop] Verified keyframe candidate overlap ratio: " << metric << " desc inlier ratio: " << desc_ratio;
      }
    }

    std::vector<LoopInfo> filtered_global_loops;
    if (detected)
    {
      // Take the top ranked specified number of global loop connections based on desc inlier ratio
      sort(global_loops.begin(), global_loops.end(), [](const LoopInfo &lhs, const LoopInfo &rhs)
           { return lhs.desc_inlier_ratio > rhs.desc_inlier_ratio; });

      filtered_global_loops.push_back(global_loops[0]);
      for (int i = 1; i < global_loops.size(); i++)
      {
        bool add_loop = true;
        for (const auto &loop : filtered_global_loops)
        {
          if (labs(static_cast<long>(loop.id_ref) - static_cast<long>(global_loops[i].id_ref)) < cfg_.global_redundant_range)
          {
            // Having redundant global loop
            add_loop = false;
            break;
          }
        }

        if (add_loop)
        {
          filtered_global_loops.push_back(global_loops[i]);
        }
      }
    }

    return filtered_global_loops;
  }

  /* ************************************************************************* */
  template <typename Scalar>
  typename LoopDetector<Scalar>::LoopInfo
  LoopDetector<Scalar>::DetectLocalLoop(const KeyframePtr &curr_kf,
                                        const std::vector<KeyframeId> &visited_keyframe_ids,
                                        typename std::vector<KeyframeId>::reverse_iterator it)
  {
    if (*it != curr_kf->id)
    {
      LOG(FATAL) << "current keyframe is not the same as the pointed id of visited_keyframe_ids";
    }

    double min_temporal_dist = std::numeric_limits<Scalar>::max();
    KeyframeId min_id = -1;
    for (auto conn : curr_kf->temporal_connections)
    {
      auto kf = map_->keyframes.Get(conn);
      auto dist = df::PoseDistance(curr_kf->pose_wk, kf->pose_wk, cfg_.trans_weight, cfg_.rot_weight);
      if (dist < min_temporal_dist)
      {
        min_temporal_dist = dist;
        min_id = kf->id;
      }
    }

    if (min_id == -1)
    {
      return LoopInfo{};
    }

    auto kf = map_->keyframes.Get(min_id);
    local_tracker_->SetRefKeyframe(kf);
    local_tracker_->SetWorldPose(curr_kf->pose_wk);

    if (!local_tracker_->TrackFrame(*curr_kf, true, cfg_.use_match_geom,
                                    false, false, false))
    {
      // Skip it if tracking failed (should mean no valid desc matches are found or no overlap with match geom disabled)
      return LoopInfo{};
    }

    KeyframeId best_id = -1;
    Scalar best_metric = cfg_.local_metric_ratio * local_tracker_->GetAreaRatio() * local_tracker_->GetInlierRatio();
    Scalar best_desc_ratio = cfg_.local_metric_ratio * local_tracker_->GetDescInlierRatio();
    Scalar best_sim = cfg_.local_metric_ratio * voc_.score(dmap_[min_id], dmap_[curr_kf->id]);
    Scalar best_motion = 1.0 / cfg_.local_metric_ratio * local_tracker_->GetAverageMotion();
    Scalar ref_dist = cfg_.local_dist_ratio * min_temporal_dist;
    // We should accept the local loop only if the pose distance between is smaller than the temporal connections
    for (long i = 0; i < cfg_.local_active_window; ++i)
    {
      if (it == visited_keyframe_ids.rend())
      {
        break;
      }

      auto id = *it;

      if (abs(id - curr_kf->id) <= cfg_.temporal_max_back_connections)
      {
        it++;
        continue;
      }

      if (map_->keyframes.LinkExists(curr_kf->id, id))
      {
        it++;
        continue;
      }

      auto kf = map_->keyframes.Get(id);
      auto dist = df::PoseDistance(curr_kf->pose_wk, kf->pose_wk, cfg_.trans_weight, cfg_.rot_weight);

      if (dist < ref_dist)
      {
        // Only check geometrically if within the temporal connection distance
        local_tracker_->SetRefKeyframe(kf);
        local_tracker_->SetWorldPose(curr_kf->pose_wk);

        // Pre check on match geometry inlier ratio for computation speedup
        local_tracker_->MatchGeoCheck(*curr_kf);
        if (local_tracker_->GetDescInlierRatio() < cfg_.min_desc_inlier_ratio || local_tracker_->GetDescInlierRatio() < best_desc_ratio)
        {
          VLOG(2) << "[LoopDetector<Scalar>::DetectLocalLoop] small desc inlier ratio -- Local loop candidate keyframe " << id << " not accepted " << local_tracker_->GetDescInlierRatio();
          it++;
          continue;
        }

        if (!local_tracker_->TrackFrame(*curr_kf, true, cfg_.use_match_geom,
                                        false, false, true))
        {
          // Skip it if tracking failed (should mean no valid desc matches are found or no overlap with match geom disabled)
          it++;
          continue;
        }

        if (local_tracker_->GetAreaRatio() < cfg_.min_area_ratio || local_tracker_->GetInlierRatio() < cfg_.min_inlier_ratio)
        {
          VLOG(2) << "[LoopDetector<Scalar>::DetectLocalLoop] small scene overlap -- Local loop candidate keyframe " << id << " not accepted " << local_tracker_->GetAreaRatio() << " " << local_tracker_->GetInlierRatio();
          it++;
          continue;
        }

        Scalar metric = local_tracker_->GetAreaRatio() * local_tracker_->GetInlierRatio();
        Scalar desc_ratio = local_tracker_->GetDescInlierRatio();
        Scalar sim = voc_.score(dmap_[kf->id], dmap_[curr_kf->id]);
        Scalar motion = local_tracker_->GetAverageMotion();

        // We select the one with the largest before-connecting distance to remove drift error
        if (metric > best_metric && desc_ratio > best_desc_ratio && motion < best_motion && sim > best_sim)
        {
          best_metric = metric;
          best_desc_ratio = desc_ratio;
          best_sim = sim;
          best_motion = motion;
          best_id = id;
        }
      }

      it++;
    }

    if (best_id != -1)
    {
      VLOG(1) << "[LoopDetector<Scalar>::DetectLocalLoop] Local loop is detected: keyframe " << best_id << " is the loop candidate to keyframe " << curr_kf->id;
      LoopInfo info;
      info.detected = true;
      info.id_ref = best_id;
      return info;
    }
    else
    {
      return LoopInfo{};
    }
  }

  /* ************************************************************************* */
  template <typename Scalar>
  void LoopDetector<Scalar>::Reset()
  {
    db_.clear();
    dmap_.clear();
  }

  /* ************************************************************************* */
  template class LoopDetector<float>;

} // namespace df
