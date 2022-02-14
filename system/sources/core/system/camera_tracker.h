#ifndef DF_CAMERA_TRACKER_H_
#define DF_CAMERA_TRACKER_H_

#include <cstddef>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/register/point.hpp>
#include <boost/algorithm/string.hpp>
#include <torch/torch.h>
#include <sophus/se3.hpp>
#include <teaser/registration.h>

#include "mapping_utils.h"
#include "keyframe.h"
#include "frame.h"
#include "camera_pyramid.h"
#include "geometric_factor_kernels.h"
#include "match_geometry_factor_kernels.h"
#include "photometric_factor_kernels.h"
#include "reprojection_factor_kernels.h"

// forward declarations
namespace cv
{
  class Mat;
}

namespace df
{
  struct Point
  {
    float x, y;
  };

  /**
 * Tracks against a specified keyframe
 */
  // Only photometric factor in the camera tracker for now. Only relative camera pose is optimized.
  // Use LM optimization instead of the 2nd-order one in the original implementation for robustness
  class CameraTracker
  {
  public:
    typedef df::Keyframe<float> KeyframeT;
    typedef df::Frame<float> FrameT;
    typedef typename KeyframeT::IdType KeyframeId;

    struct TrackerConfig
    {
      long cuda_id;
      long max_num_iters;
      float min_grad_thresh;
      float min_param_inc_thresh;
      float min_damp;
      float max_damp;
      float init_damp;
      float damp_dec_factor;
      float damp_inc_factor;
      float dpt_eps;
      float jac_update_err_inc_threshold;
      std::vector<float> photo_factor_weights;
      // feature matching related
      long desc_num_samples;
      float desc_cyc_consis_thresh;

      // match geometry
      float match_geom_factor_weight;
      float match_geom_loss_param_factor;

      // reproj
      float reproj_factor_weight;
      float reproj_loss_param_factor;

      // TEASER++ related
      double teaser_max_clique_time_limit;
      double teaser_kcore_heuristic_threshold;
      size_t teaser_rotation_max_iterations;
      double teaser_rotation_cost_threshold;
      double teaser_rotation_gnc_factor;
      std::string teaser_rotation_estimation_algorithm;
      std::string teaser_rotation_tim_graph;
      std::string teaser_inlier_selection_mode;
      double teaser_noise_bound_multiplier;

      // resolution
      std::vector<long> net_output_size;
    };

    CameraTracker() = delete;
    CameraTracker(const TrackerConfig &config, const bool display = true);
    virtual ~CameraTracker();

    bool MatchGeoCheck(FrameT &frame_to_track);
    bool TrackMatchGeoCheck(FrameT &frame_to_track);
    bool TrackFrame(FrameT &frame_to_track, bool use_photo, bool use_match_geom, bool display_image,
                    bool update_frame, bool match_geom_pre_checked);
    bool TrackNewFrame(FrameT &frame_to_track, bool use_photo, bool use_reproj, 
    bool match_geom_pre_checked, bool update_frame, bool display_image);
    void Reset();

    void SetRefKeyframe(std::shared_ptr<KeyframeT> kf);
    void SetWorldPose(const Sophus::SE3f &pose_wc);
    void SetConfig(const TrackerConfig &new_cfg);
    void SetName(const std::string name)
    {
      tracker_name_ = name;
    }

    std::shared_ptr<KeyframeT> GetRefKeyframe() { return kf_; }
    Sophus::SE3f GetRelativePoseEstimate();
    Sophus::SE3f GetWorldPoseEstimate();
    float GetAreaRatio() { return warp_area_ratio_; }
    float GetInlierRatio() { return inlier_ratio_; }
    float GetAverageMotion() { return average_motion_; }
    float GetDescInlierRatio() { return desc_match_inlier_ratio_; }
    float GetRelDescInlierRatio() { return relative_desc_match_inlier_ratio_; }
    float GetError() { return error_; }
    float GetQueryScale() { return guess_scale_; }
    float GetRefScale() { return ref_kf_scale_; }

    const std::tuple<KeyframeId, KeyframeId, at::Tensor, at::Tensor, at::Tensor, at::Tensor, float> GetLoopInfo()
    {
      using namespace torch::indexing;

      const auto cam = (*(kf_->camera_pyramid_ptr))[0];
      float width = cam.width();
      float fx = cam.fx();
      float fy = cam.fy();
      float cx = cam.u0();
      float cy = cam.v0();

      at::Tensor inlier_keypoint_locations_1d_0 =
          ((inlier_keypoint_locations_homo_0_.index({Slice(), 0}) * fx + cx) +
           (inlier_keypoint_locations_homo_0_.index({Slice(), 1}) * fy + cy) * width)
              .to(torch::kLong);

      at::Tensor matched_locations_1d_1 =
          ((matched_locations_homo_1_.index({Slice(), 0}) * fx + cx) +
           (matched_locations_homo_1_.index({Slice(), 1}) * fy + cy) * width)
              .to(torch::kLong);

      return std::make_tuple(frame_to_track_id_, kf_->id, inlier_keypoint_locations_1d_0, matched_locations_1d_1,
                             inlier_keypoint_locations_homo_0_, matched_locations_homo_1_, desc_match_inlier_ratio_);
    }

    cv::Mat &GetWarpImage()
    {
      std::shared_lock<std::shared_mutex> lock(warp_image_mutex_);
      return warp_image_;
    }

  private:
    void FeatureMatchingGeo(const FrameT &frame_to_track, const at::Tensor sampled_locations_1d_0,
                            const at::Tensor sampled_locations_homo_0, const at::Tensor sampled_dpts_0,
                            const at::Tensor dpt_map_1, const PinholeCamera<float> camera,
                            at::Tensor &inlier_keypoint_locations_homo_0, at::Tensor &inlier_keypoint_dpts_0,
                            at::Tensor &matched_locations_homo_1, at::Tensor &matched_dpts_1,
                            at::Tensor &guess_rotation, at::Tensor &guess_translation, float &guess_scale,
                            float &relative_desc_match_inlier_ratio, float &desc_match_inlier_ratio,
                            float &inlier_multiplier);
    void FeatureMatchingGeo(const at::Tensor feature_desc_0, const at::Tensor feature_desc_1,
                            const at::Tensor valid_locations_1d_0,
                            const at::Tensor valid_locations_homo_0, const at::Tensor valid_dpts_0,
                            const at::Tensor dpt_map_1, const PinholeCamera<float> camera,
                            at::Tensor &inlier_keypoint_locations_homo_0, at::Tensor &inlier_keypoint_dpts_0,
                            at::Tensor &matched_locations_2d_1,
                            float &relative_desc_match_inlier_ratio, float &desc_match_inlier_ratio,
                            float &inlier_multiplier);

    void ComputeAreaInlierRatio(const at::Tensor sampled_dpts_0, const at::Tensor sampled_locations_homo_0,
                                const at::Tensor guess_rotation, const at::Tensor guess_translation,
                                const df::PinholeCamera<float> &camera,
                                const at::Tensor video_mask,
                                float &area_ratio, float &inlier_ratio, float &average_motion);

    void ComputeError(const FrameT &frame_to_track, const at::Tensor cat_photo_features_0,
                      const at::Tensor photo_dpts_0, const at::Tensor photo_locations_homo_0,
                      const at::Tensor keypoint_dpts_0, const at::Tensor keypoint_locations_homo_0,
                      const at::Tensor matched_locations_2d_1,
                      const at::Tensor guess_rotation, const at::Tensor guess_translation, 
                      const bool use_photo, const bool use_reproj, float &error);
    void ComputeError(const FrameT &reference_frame, const at::Tensor cat_sampled_features_0,
                      const at::Tensor unscaled_sampled_dpts_0, const at::Tensor sampled_locations_homo_0,
                      const at::Tensor unscaled_keypoint_dpts_0, const at::Tensor keypoint_locations_homo_0,
                      const at::Tensor matched_dpts_1, const at::Tensor matched_locations_homo_1,
                      const at::Tensor guess_rotation, const at::Tensor guess_translation, const float guess_scale_0,
                      const bool use_photo, const bool use_match_geom, float &error);

    void ComputePhotoError(const FrameT &reference_frame, const at::Tensor cat_sampled_features_0,
                           const at::Tensor sampled_dpts_0, const at::Tensor sampled_locations_homo_0,
                           const at::Tensor guess_rotation, const at::Tensor guess_translation,
                           float &error);
    void ComputeReprojError(const CameraTracker::FrameT &frame_to_track, const at::Tensor keypoint_dpts_0,
                            const at::Tensor keypoint_locations_homo_0, const at::Tensor matched_locations_2d_1,
                            const at::Tensor guess_rotation, const at::Tensor guess_translation,
                            float &error);
    void ComputeMatchGeomError(const FrameT &reference_frame,
                               const at::Tensor keypoint_dpts_0, const at::Tensor keypoint_locations_homo_0,
                               const at::Tensor matched_dpts_1, const at::Tensor matched_locations_homo_1,
                               const at::Tensor guess_rotation, const at::Tensor guess_translation,
                               float &error);

    void ComputeJacobianAndError(const FrameT &frame_1, const at::Tensor cat_photo_features_0,
                                 const at::Tensor photo_dpts_0, const at::Tensor photo_locations_homo_0,
                                 const at::Tensor keypoint_dpts_0, const at::Tensor keypoint_locations_homo_0,
                                 const at::Tensor matched_locations_2d_1,
                                 const at::Tensor guess_rotation, const at::Tensor guess_translation,
                                 const bool use_photo, const bool use_reproj, 
                                 const bool update_error, at::Tensor &AtA, at::Tensor &Atb, float &error);

    void ComputeJacobianAndError(const FrameT &reference_frame, const at::Tensor cat_sampled_features_0,
                                 const at::Tensor unscaled_sampled_dpts_0, const at::Tensor sampled_locations_homo_0,
                                 const at::Tensor unscaled_keypoint_dpts_0, const at::Tensor keypoint_locations_homo_0,
                                 const at::Tensor matched_dpts_1, const at::Tensor matched_locations_homo_1,
                                 const at::Tensor guess_rotation, const at::Tensor guess_translation,
                                 const float scale_0, const bool use_photo, const bool use_match_geom, const bool update_error,
                                 at::Tensor &AtA, at::Tensor &Atb, float &error);
    void ComputePhotoJacobianAndErrorWithScale(const FrameT &reference_frame, const at::Tensor cat_sampled_features_0,
                                               const at::Tensor unscaled_sampled_dpts_0, const at::Tensor sampled_locations_homo_0,
                                               const at::Tensor guess_rotation, const at::Tensor guess_translation,
                                               const float scale_0, at::Tensor &AtA, at::Tensor &Atb, float &error);
    void ComputePhotoJacobianAndError(const FrameT &frame_1, const at::Tensor cat_photo_features_0,
                                      const at::Tensor photo_dpts_0, const at::Tensor photo_locations_homo_0,
                                      const at::Tensor guess_rotation, const at::Tensor guess_translation,
                                      at::Tensor &AtA, at::Tensor &Atb, float &error);
    void ComputeReprojJacobianAndError(const FrameT &frame_1, const at::Tensor sampled_dpts_0,
                                       const at::Tensor sampled_locations_homo_0,
                                       const at::Tensor matched_locations_2d_1,
                                       const at::Tensor guess_rotation,
                                       const at::Tensor guess_translation, at::Tensor &AtA, at::Tensor &Atb,
                                       float &error);
    void ComputeMatchGeomJacobianAndErrorWithScale(const FrameT &reference_frame,
                                                   const at::Tensor unscaled_keypoint_dpts_0, const at::Tensor keypoint_locations_homo_0,
                                                   const at::Tensor matched_dpts_1, const at::Tensor matched_locations_homo_1,
                                                   const at::Tensor guess_rotation, const at::Tensor guess_translation,
                                                   const float scale_0,
                                                   at::Tensor &AtA, at::Tensor &Atb, float &error);

    void UpdateVariables(const Eigen::Matrix<float, 6, 1> &solution, const at::Tensor curr_rot_mat, const at::Tensor curr_trans_vec,
                         at::Tensor &updated_rot_mat, at::Tensor &updated_trans_vec);
    void UpdateVariables(const Eigen::Matrix<float, 7, 1> &solution, const at::Tensor curr_rot_mat, const at::Tensor curr_trans_vec,
                         const float curr_scale, at::Tensor &updated_rot_mat, at::Tensor &updated_trans_vec, float &updated_scale);
    bool LMConvergence(const at::Tensor guess_rotation, const at::Tensor guess_translation,
                       const at::Tensor Atb, const at::Tensor solution);
    bool LMConvergence(const at::Tensor guess_rotation, const at::Tensor guess_translation, const float guess_scale,
                       const at::Tensor Atb, const at::Tensor solution);

  public:
    std::shared_mutex warp_image_mutex_;

  private:
    TrackerConfig config_;
    bool display_;

    cv::Mat checkerboard_;

    // T^currframe_keyframe
    Sophus::SE3f pose_ck_;

    at::Tensor photo_weights_tensor_;

    teaser::RobustRegistrationSolver::Params teaser_params_;

    std::shared_ptr<KeyframeT> kf_;

    float input_area_;
    float reproj_loss_param_;

    cv::Mat warp_image_;

    at::Tensor inlier_keypoint_locations_homo_0_, inlier_keypoint_dpts_0_, matched_locations_2d_1_,
        matched_locations_homo_1_, matched_dpts_1_, unscaled_inlier_keypoint_dpts_0_, sampled_dpts_0_, unscaled_sampled_dpts_0_;

    std::string tracker_name_;

    float error_;
    float warp_area_ratio_;
    float inlier_ratio_;
    float average_motion_;
    float desc_match_inlier_ratio_;
    float relative_desc_match_inlier_ratio_;
    float inlier_multiplier_;

    at::Tensor guess_rotation_, guess_translation_;
    float guess_scale_;

    float ref_kf_scale_;

    KeyframeId frame_to_track_id_;
  };

} // namespace df

#endif // DF_CAMERA_TRACKER_H_