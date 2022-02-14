#ifndef DF_LOOP_DETECTOR_H_
#define DF_LOOP_DETECTOR_H_

#include <vector>
#include <memory>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <DBoW2/DBoW2.h>
#include <torch/torch.h>
#include <unordered_map>

#include "mapping_utils.h"
#include "keyframe.h"
#include "keyframe_map.h"
#include "camera_tracker.h"
#include "camera_pyramid.h"
#include "dl_descriptor.h"
#include "tensor_vocabulary.h"
#include "mapping_utils.h"

namespace df
{

  typedef df::TemplatedTensorVocabulary<FTensor> TensorVocabulary;
  typedef DBoW2::TemplatedDatabase<FTensor::TDescriptor, FTensor> TensorDatabase;

  struct LoopDetectorConfig
  {
    CameraTracker::TrackerConfig tracker_cfg;
    long temporal_max_back_connections;
    long max_candidates;
    float min_area_ratio;
    float min_inlier_ratio;
    float min_desc_inlier_ratio;
    long global_redundant_range;
    bool use_match_geom;
    long local_active_window;
    float local_dist_ratio;
    long global_active_window;
    float trans_weight;
    float rot_weight;
    float global_sim_ratio;
    float global_metric_ratio;
    float local_metric_ratio;
    float dpt_eps;
    
    c10::Device device = c10::Device(torch::kCUDA, 0);
  };

  template <typename Scalar>
  class LoopDetector
  {
  public:
    typedef Keyframe<Scalar> KeyframeT;
    typedef typename df::Keyframe<Scalar>::IdType KeyframeId;
    typedef typename KeyframeT::Ptr KeyframePtr;
    typedef typename Map<Scalar>::Ptr MapPtr;
    typedef std::shared_ptr<CameraTracker> TrackerPtr;
    typedef CameraTracker::TrackerConfig TrackerConfig;
    typedef CameraPyramid<Scalar> CameraPyr;
    typedef std::unordered_map<KeyframeId, DBoW2::BowVector> DescriptionMap;
    typedef Sophus::SE3<Scalar> SE3T;

    struct LoopInfo
    {
      KeyframeId id_ref = 0;
      SE3T pose_cur_ref = SE3T();
      std::tuple<KeyframeId, KeyframeId, at::Tensor, at::Tensor, at::Tensor, at::Tensor, Scalar> info;
      bool detected = false;
      Scalar query_scale;
      Scalar ref_scale;
      Scalar desc_inlier_ratio;
    };

    LoopDetector(std::string voc_path, MapPtr map, LoopDetectorConfig cfg);
    ~LoopDetector()
    {
      VLOG(3) << "[LoopDetector<Scalar>::~LoopDetector] deconstructor called";
    }

    void AddKeyframe(const KeyframePtr &kf);
    bool CheckExist(const KeyframeId id);
    LoopInfo DetectLocalLoop(const KeyframePtr &curr_kf, const std::vector<KeyframeId> &visited_keyframe_ids,
                             typename std::vector<KeyframeId>::reverse_iterator it);
    std::vector<LoopInfo> DetectLoop(const KeyframePtr &kf);

    void Reset();
    TrackerPtr GetGlobalTracker() { return global_tracker_; }

    cv::Mat &GetWarpImage()
    {
      std::shared_lock<std::shared_mutex> lock(global_tracker_->warp_image_mutex_);
      return warp_image_;
    }

  private:
    TensorVocabulary voc_;

    MapPtr map_;

    LoopDetectorConfig cfg_;

    TensorDatabase db_;

    TrackerPtr local_tracker_;
    TrackerPtr global_tracker_;

    DescriptionMap dmap_;

    cv::Mat warp_image_;
  };

} // namespace df

#endif // DF_LOOP_DETECTOR_H_
