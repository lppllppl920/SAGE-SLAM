#ifndef DF_MAPPER_H_
#define DF_MAPPER_H_

#include <memory>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/base/Vector.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <shared_mutex>
#include <boost/optional/optional.hpp>
#include <boost/algorithm/string.hpp>
#include <unordered_map>
#include <atomic>

#include "logutils.h"
#include "mapping_utils.h"
#include "display_utils.h"
#include "timing.h"
#include "work_manager.h"
#include "keyframe_map.h"
#include "code_depth_network.h"
#include "feature_network.h"
#include "df_work.h"
#include "pinhole_camera.h"
#include "reprojection_factor.h"

namespace df
{

  struct MapperOptions
  {
    std::string log_dir;
    long cuda_id;
    std::vector<long> net_input_size, net_output_size;

    float init_pose_prior_weight;
    float init_scale_prior_weight;
    float dpt_eps;

    /* photometric */
    bool use_photometric;
    int pho_num_samples;
    int factor_iters;
    std::vector<float> photo_factor_weights;

    /* reprojection */
    bool use_reprojection;
    int desc_num_keypoints;
    float desc_cyc_consis_thresh;

    /* geometric */
    bool use_geometric;
    float geo_factor_weight;
    float geo_loss_param_factor;

    // match geometry factor
    float match_geom_factor_weight;
    float match_geom_loss_param_factor;

    // reprojection factor
    float reproj_factor_weight;
    float reproj_loss_param_factor;

    // code factor
    float code_factor_weight;

    /* keyframe connections */
    int temporal_max_back_connections;

    /* ISAM2 */
    gtsam::ISAM2Params isam_params;

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
  };

  /*
 * Incremental mapping with ISAM2
 * Contains the keyframe map
 */
  template <typename Scalar, int CS>
  class Mapper
  {
  public:
    typedef gtsam::Vector CodeT;
    typedef Sophus::SE3<Scalar> SE3T;
    typedef df::Map<Scalar> MapT;
    typedef df::Keyframe<Scalar> KeyframeT;
    typedef df::Frame<Scalar> FrameT;
    typedef typename KeyframeT::Ptr KeyframePtr;
    typedef typename KeyframeT::IdType KeyframeId;
    typedef typename FrameT::Ptr FramePtr;
    typedef typename MapT::Ptr MapPtr;
    typedef typename MapT::FrameId FrameId;
    typedef std::function<void(MapPtr)> MapCallbackT;
    typedef std::vector<int> IterList;
    typedef Sophus::SE3<Scalar> PoseT;

    Mapper(const MapperOptions &opts,
           const cv::Mat &video_mask,
           const df::CameraPyramid<Scalar> &out_cam_pyr,
           const df::CodeDepthNetwork::Ptr &depth_network,
           const df::FeatureNetwork::Ptr &feature_network);

    ~Mapper()
    {
      VLOG(2) << "[Mapper<Scalar, CS>::~Mapper] deconstructor called";
    }

    //   /*!
    //  * \brief Add a new one-way frame to the current keyframe
    //  * \param img
    //  * \param col
    //  * \param pose_init
    //  * \param kf_id
    //  */
    // void EnqueueFrame(double timestamp, const cv::Mat &color, const SE3T &pose_init, FrameId kf_id);

    // void EnqueueFrame(const FramePtr &fr, FrameId kf_id);

    KeyframePtr EnqueueKeyframe(const FramePtr &frame_ptr,
                                const std::vector<FrameId> &conns);

    void EnqueueLink(FrameId id0, FrameId id, bool photo, bool match_geom, bool geo, bool global_loop);

    //   /*!
    //  * \brief Marginalize frames that are connected to keyframes
    //  * \param frames
    //  */
    //   void MarginalizeFrames(const std::vector<FrameId> &frames);

    // void MappingUpdate(const std::vector<KeyframeId> &visited_kf_ids);
    /*!
   * \brief Perform a single mapping step
   */
    void MappingStep(const std::vector<KeyframeId> &visited_kf_ids, const bool force_relinearize = false);

    //   /*!
    //  * \brief Initialize the mapper with two images
    //  */
    //   void InitTwoFrames(const cv::Mat &color0, const cv::Mat &color1,
    //                      double ts0, double ts1);

    /*!
   * \brief Initialize the mapper with a single image
   * Relies on the network prediction (decoding a zero code)
   */
    void InitOneFrame(double timestamp, const cv::Mat &color);

    /*!
   * \brief Reset the mapper
   */
    void Reset();

    /*!
   * \brief Saves current factor graph and isam2 bayes tree as .dot files
   * \param Filepath + filename prefix e.g. ~/s
   */
    void SaveGraphs(const std::string &save_dir, const std::string &prefix);

    /*!
   * \brief Prints all factors in the graph to stdout
   */
    void PrintFactors();

    /*!
   * \brief Prints debug info about the state of the graph
   */
    void PrintDebugInfo();

    /*!
   * \brief Display sparse keypoint matches between keyframes (if any)
   */
    void DisplayMatches();

    /*!
   * \brief Display loop closure reprojection errors
   */
    void DisplayLoopClosures(std::vector<std::pair<long, long>> &link, int N = 5);

    cv::Mat DisplayMatchGeometryErrors(int N = 3);
    cv::Mat DisplayReprojectionErrors(int N = 3);
    cv::Mat DisplayPhotometricErrors(int N = 3);

    void SavePhotometricDebug(std::string save_dir);
    void SaveReprojectionDebug(std::string save_dir);
    void SaveMatchGeometryDebug(std::string save_dir);
    void SaveGeometricDebug(std::string save_dir);

    /*!
   * \brief Set new mapper options
   */
    void SetOptions(const MapperOptions &new_opts);

    // std::vector<FrameId> NonmarginalizedFrames();
    std::unordered_map<KeyframeId, bool> KeyframeRelinearization();

    /*
   * Setters/getters
   */
    void AddGlobalLoopCount() { ++global_loop_count_; }
    void SetMapCallback(MapCallbackT cb) { map_callback_ = cb; }
    MapPtr GetMap() { return map_; }
    const cv::Mat &GetCheckerboard() { return checkerboard_; }
    gtsam::ISAM2Result GetResults() { return isam_res_; }
    const cv::Mat &GetDisplayKeyframes() { return keyframes_display_; }
    const std::shared_ptr<gtsam::ISAM2> GetISAMGraph() { return isam_graph_; }
    // const std::set<KeyframeId>& GetBookKeepedIds() { return bookkeeped_kf_ids_; }
    long NumKeyframes() const { return map_->NumKeyframes(); }
    bool HasWork();

    /*!
   * \brief Construct a new frame
   * \return Pointer to the new frame
   */
    FramePtr BuildFrame(double timestamp, const cv::Mat &color, const SE3T &pose_init);

    KeyframePtr BuildKeyframe(const FramePtr &frame_ptr);

    /*!
   * \brief Constructs a new keyframe
   */
    KeyframePtr BuildKeyframe(double timestamp, const cv::Mat &color, const SE3T &pose_init);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  private:
    /*!
   * \brief Track all changes to the optimization process
   * Determines new factors to be added to or remove from the graph
   * (e.g. when progressing to a new level in coarse to fine)
   *
   * \param New factors to be added to the graph
   * \param Indices of factors to be removed (in isam2 graph)
   * \param New variable initialization (if any)
   */
    void Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                     gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                     gtsam::Values &var_init,
                     gtsam::Values &var_update);

    //   /*!
    //  * \brief Create a list of backward connections of a keyframe based on the mapper options
    //  */
    // std::vector<FrameId> BuildBackConnections();

    /*!
   * \brief Update the keyframe map with new isam2 estimate
   * \param Estimated values for poses and codes of keyframes
   */
    void UpdateMap(const gtsam::Values &vals, const gtsam::VectorValues &delta, boost::optional<const FrameId> curr_kf_id = boost::none);

    /*!
   * \brief Call all callbacks to inform about new map estimate
   */
    void NotifyMapObservers();

    void GenerateGaussianPyramid(const at::Tensor feat_map,
                                 const int &num_levels,
                                 std::vector<at::Tensor> &feat_map_pyramid);

    void GenerateGaussianPyramidWithGrad(const at::Tensor feat_map,
                                         const int &num_levels,
                                         at::Tensor &feat_map_pyramid,
                                         at::Tensor &feat_map_grad_pyramid);

    void CorrectDepthScale(const KeyframePtr &kf_to_scale, const KeyframePtr &reference_kf);

  public:
    std::shared_ptr<std::vector<at::Tensor>> output_mask_pyramid_ptr_;
    std::shared_ptr<at::Tensor> output_mask_ptr_;
    std::shared_mutex new_factor_mutex_;
    std::shared_mutex new_keyframe_mutex_;
    std::shared_mutex new_links_mutex_;
    std::shared_mutex global_loop_mutex_;

  private:
    work::WorkManager *work_manager_;
    gtsam::Values estimate_;
    MapPtr map_;
    std::shared_ptr<gtsam::ISAM2> isam_graph_;
    std::shared_ptr<df::CodeDepthNetwork> depth_network_;
    std::shared_ptr<df::FeatureNetwork> feature_network_;

    at::Tensor input_video_mask_, output_video_mask_, feat_video_mask_;
    std::shared_ptr<at::Tensor> valid_normalized_locations_2d_ptr_;
    at::Tensor valid_locations_1d_;
    at::Tensor valid_locations_homo_;
    std::shared_ptr<at::Tensor> level_offsets_ptr_;

    at::Tensor gauss_kernel_;

    torch::nn::functional::Conv2dFuncOptions gauss_conv_options_;
    torch::nn::functional::InterpolateFuncOptions mask_interp_options_;

    Scalar pose_prior_sigma_;
    Scalar rep_loss_param_;

    // cuda device id
    int cuda_id_;

    // camera pyramid pointer
    std::shared_ptr<df::CameraPyramid<Scalar>> output_camera_pyramid_ptr_;

    // mapping parameters
    MapperOptions opts_;
    MapCallbackT map_callback_;

    // displaying matches
    std::vector<cv::Mat> match_imgs_;
    bool new_match_imgs_;

    // last result of the ISAM2 algorithm step
    gtsam::ISAM2Result isam_res_;

    cv::Mat checkerboard_;

    teaser::RobustRegistrationSolver::Params teaser_params_;

    cv::Mat keyframes_display_;

    std::atomic<int> global_loop_count_;
    // std::set<KeyframeId> bookkeeped_kf_ids_;
  };

} // namespace df

#endif // DF_MAPPER_H_
