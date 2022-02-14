#ifndef DF_DEEPFACTORS_H_
#define DF_DEEPFACTORS_H_

#include <cstddef>
#include <memory>
#include <vector>
#include <Eigen/Core>
#include <torch/torch.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <future>
#include <algorithm>
#include <unordered_map>

#include "display_utils.h"
#include "keyframe_map.h"
#include "timing.h"
#include "tum_io.h"
#include "mapper.h"
#include "camera_tracker.h"
#include "pinhole_camera.h"
#include "frame.h"
#include "keyframe.h"
#include "deepfactors_options.h"
#include "loop_detector.h"
#include "logutils.h"

namespace cv
{
  class Mat;
}

namespace df
{

  struct DeepFactorsStatistics
  {
    float inlier_ratio;
    float area_ratio;
    float pose_distance;
    float tracker_error;
    std::unordered_map<long, bool> relin_info;
  };

  template <typename Scalar, int CS>
  class DeepFactors
  {
  public:
    typedef Scalar ScalarT;
    typedef Sophus::SE3<Scalar> SE3T;
    typedef df::Frame<Scalar> FrameT;
    typedef df::Keyframe<Scalar> KeyframeT;
    typedef typename df::Keyframe<Scalar>::IdType KeyframeId;
    typedef df::Map<Scalar> MapT;
    typedef typename MapT::Ptr MapPtr;
    typedef df::Mapper<Scalar, CS> MapperT;
    typedef LoopDetector<Scalar> LoopDetectorT;

    // callback types
    typedef std::function<void(MapPtr)> MapCallbackT;
    typedef std::function<void(const SE3T &)> PoseCallbackT;
    typedef std::function<void(const DeepFactorsStatistics &)> StatsCallbackT;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  public:
    DeepFactors();
    virtual ~DeepFactors();

    /* Do not create copies of this object */
    DeepFactors(const DeepFactors &other) = delete;

    /*
   * Logic
   */
    void Init(const df::PinholeCamera<Scalar> &cam, const cv::Mat &video_mask, const DeepFactorsOptions &opts);
    void Reset();
    void ProcessFrame(double timestamp, const cv::Mat &frame);
    bool RefineMapping();
    /*
   * Initialize the system with two images and optimize them
   */
    // void BootstrapTwoFrames(double ts1, double ts2, const cv::Mat& img0, const cv::Mat& img1);

    /*
   * Initialize the system with a single image (decodes zero-code only)
   */
    void BootstrapOneFrame(double timestamp, const cv::Mat &img);

    void ForceKeyframe();
    void ForceFrame();

    /*
   * Getters
   */
    MapPtr GetMap() { return mapper_->GetMap(); }
    DeepFactorsStatistics GetStatistics() { return stats_; }

    /*
   * Setters
   */
    void SetMapCallback(MapCallbackT cb)
    {
      map_callback_ = cb;
      mapper_->SetMapCallback(map_callback_);
    }
    void SetPoseCallback(PoseCallbackT cb) { pose_callback_ = cb; }
    void SetStatsCallback(StatsCallbackT cb) { stats_callback_ = cb; }
    void SetOptions(DeepFactorsOptions opts);

    /*
   * Notifications
   */
    void NotifyPoseObservers(const SE3T &pose_wc);
    void NotifyMapObservers();
    void NotifyStatsObservers();

    /*
   * Debugging
   */
    void SaveInfo(std::string dir);
    void SaveResults(std::string dir);
    void SaveKeyframes(std::string dir);

    void JoinMappingThreads();
    void JoinLoopThreads();

  private:
    void TrackFrame();

    bool NewKeyframeRequired();

    KeyframeId SelectKeyframe(bool &geo_checked);
    bool CheckTrackingLost();

    void LoopClosurePoseScaleEstimate(
        const std::map<std::pair<KeyframeId, KeyframeId>, std::pair<Scalar, Scalar>> &pre_global_loops,
        const std::vector<std::tuple<KeyframeId, KeyframeId, SE3T, Scalar, Scalar>> &new_global_loop_vec);

    void LoopClosurePoseEstimate(
        const std::set<std::pair<KeyframeId, KeyframeId>> &pre_global_loops,
        const std::tuple<KeyframeId, KeyframeId, SE3T, Scalar, Scalar> &cur_global_loop);

    void LoopClosurePoseScaleMGEstimate(
        const std::set<std::pair<KeyframeId, KeyframeId>> &pre_global_loops,
        const std::tuple<KeyframeId, KeyframeId, at::Tensor, at::Tensor, at::Tensor, at::Tensor, Scalar> &cur_global_loop);

    static void *MappingBackendWrapper(void *object)
    {
      reinterpret_cast<DeepFactors<Scalar, CS> *>(object)->MappingBackend();
      return 0;
    }

    void MappingBackend();

    static void *GlobalLoopDetectBackendWrapper(void *object)
    {
      reinterpret_cast<DeepFactors<Scalar, CS> *>(object)->GlobalLoopDetectBackend();
      return 0;
    }

    static void *LocalLoopDetectBackendWrapper(void *object)
    {
      reinterpret_cast<DeepFactors<Scalar, CS> *>(object)->LocalLoopDetectBackend();
      return 0;
    }

    void GlobalLoopDetectBackend();
    void LocalLoopDetectBackend();

  private:
    bool force_keyframe_;
    bool bootstrapped_;
    bool tracking_lost_;
    KeyframeId curr_kf_;
    bool enable_loop_;
    bool quit_loop_;
    bool enable_mapping_;
    bool quit_mapping_;

    SE3T pose_kc_;
    DeepFactorsOptions opts_;
    DeepFactorsStatistics stats_;

    std::shared_ptr<df::CodeDepthNetwork> depth_network_ptr_;
    std::shared_ptr<df::FeatureNetwork> feature_network_ptr_;

    std::shared_ptr<CameraTracker> tracker_;
    std::unique_ptr<MapperT> mapper_;
    std::unique_ptr<LoopDetectorT> loop_detector_;
    std::shared_ptr<FrameT> live_frame_ptr_;

    df::CameraPyramid<Scalar> output_cam_pyr_;
    df::PinholeCamera<Scalar> output_cam_;

    MapCallbackT map_callback_;
    PoseCallbackT pose_callback_;
    StatsCallbackT stats_callback_;

    std::vector<std::pair<KeyframeId, KeyframeId>> loop_links;

    std::vector<KeyframeId> visited_keyframe_ids_;

    std::shared_mutex kf_id_mutex_;
    std::shared_mutex visited_keyframes_mutex_;
    std::shared_mutex stats_mutex_;
    std::shared_mutex global_loop_mutex_;

    pthread_t global_loop_detect_thread_;
    pthread_t local_loop_detect_thread_;
    pthread_t mapping_thread_;

    // std::set<std::pair<KeyframeId, KeyframeId>> global_loops_info_;
    std::map<std::pair<KeyframeId, KeyframeId>, std::pair<Scalar, Scalar>> global_loops_info_;

    std::vector<KeyframeId> local_loop_prev_keyframe_ids_;
    std::vector<KeyframeId> local_loop_curr_keyframe_ids_;
    std::vector<KeyframeId> global_loop_curr_keyframe_ids_;
    std::shared_mutex global_loop_ids_mutex_;

    struct DebugImages
    {
      cv::Mat reprojection_errors;
      cv::Mat photometric_errors;
    };
    std::list<DebugImages> debug_buffer_;

    int global_loop_count_;
  };

} // namespace df

#endif // DF_H_
