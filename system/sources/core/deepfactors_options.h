#ifndef DF_OPTIONS_H_
#define DF_OPTIONS_H_

#include <vector>
#include <string>
#include <torch/torch.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "mapper.h"

namespace df
{

  struct DeepFactorsOptions
  {
    enum KeyframeMode
    {
      AUTO = 0,
      NEVER
    };
    enum TrackingMode
    {
      CLOSEST = 0,
      LAST,
      FIRST
    };

    bool enable_gui;
    
    std::string log_dir;
    long cuda_id;

    // network path
    std::string depth_network_path;
    std::string feature_network_path;

    // vocabulary path
    std::string vocabulary_path;

    /* camera tracking */
    TrackingMode tracking_mode;
    int tracking_max_num_iters;
    float tracking_min_grad_thresh;
    float tracking_min_param_inc_thresh;
    float tracking_init_damp;
    std::vector<float> tracking_min_max_damp;
    std::vector<float> tracking_damp_dec_inc_factor;
    float tracking_jac_update_err_inc_threshold;
    float tracking_ref_kf_select_ratio;

    // Camera tracking related metrics: area ratio and inlier ratio
    float tracking_lost_min_error;
    float tracking_lost_max_area_ratio;
    float tracking_lost_max_inlier_ratio;

    float new_kf_max_area_ratio;
    float new_kf_max_inlier_ratio;
    float new_kf_max_desc_inlier_ratio;
    float new_kf_min_average_motion;

    float pose_dist_trans_weight;
    float pose_dist_rot_weight;

    /* keyframe connection */
    long temporal_max_back_connections;

    /* keyframe adding */
    KeyframeMode keyframe_mode;

    /* loop closure */
    bool use_global_loop;
    bool use_local_loop;
    int loop_global_active_window;
    int loop_local_active_window;
    float loop_local_dist_ratio;
    int loop_max_candidates;
    float loop_min_area_ratio;
    float loop_min_inlier_ratio;
    float loop_min_desc_inlier_ratio;
    int loop_tracking_max_num_iters;
    float loop_tracking_min_grad_thresh;
    float loop_tracking_min_param_inc_thresh;
    std::vector<float> loop_tracking_damp_dec_inc_factor;
    float loop_global_sim_ratio;
    float loop_global_metric_ratio;
    float loop_local_metric_ratio;
    float loop_detection_frequency;
    long global_redundant_range;
    bool loop_use_match_geom;
    float loop_pose_linearize_threshold;
    float loop_scale_linearize_threshold;

    // temporal connection
    float temporal_min_desc_inlier_ratio;

    long refine_mapping_iters;
    long refine_mapping_num_no_linearize;

    float mapping_update_frequency;

    // epsilons
    float warp_dpt_eps;

    // network spatial size and video mask
    std::vector<long> net_input_size, net_output_size;

    // variable prior
    float init_pose_prior_weight;
    float init_scale_prior_weight;

    // feature matching related
    int desc_num_keypoints;
    int tracking_desc_num_keypoints;
    float desc_cyc_consis_thresh;

    /* photometric error */
    int factor_iters;
    bool use_photometric;
    bool use_tracker_photometric;
    int pho_num_samples;
    std::vector<float> photo_factor_weights;
    long num_pyramid_levels;

    /* geometric error */
    bool use_geometric;
    float geo_loss_param_factor;
    float geo_factor_weight;

    // match geometry factor
    float match_geom_loss_param_factor;
    float match_geom_factor_weight;
    float tracker_match_geom_factor_weight;

    // reprojection factor
    bool use_reprojection;
    bool use_tracker_reprojection;
    float reproj_factor_weight;
    float tracker_reproj_factor_weight;
    float reproj_loss_param_factor;

    // code factor
    float code_factor_weight;

    // ISAM2 related
    int isam_relinearize_skip;
    bool isam_partial_relin_check;
    bool isam_enable_detailed_results;
    double isam_pose_lin_eps; // linearize threshold for each element of the different variables
    double isam_code_lin_eps;
    double isam_scale_lin_eps;
    double isam_wildfire_threshold;

    // pose graph related
    float pose_graph_local_link_weight;
    float pose_graph_global_link_weight;
    float pose_graph_scale_prior_weight;
    float pose_graph_rot_weight;
    float pose_graph_scale_weight;
    int pose_graph_max_iters;
    int pose_scale_graph_max_iters;
    int pose_graph_no_relin_max_iters;
    int pose_scale_graph_no_relin_max_iters;

    // TEASER++ related
    double teaser_max_clique_time_limit;
    double teaser_kcore_heuristic_threshold;
    size_t teaser_rotation_max_iterations;
    double teaser_rotation_cost_threshold;
    double teaser_rotation_gnc_factor;
    std::string teaser_rotation_estimation_algorithm;
    std::string teaser_rotation_tim_graph;
    std::string teaser_inlier_selection_mode;
    std::string teaser_tracker_inlier_selection_mode;
    double teaser_noise_bound_multiplier;

    static KeyframeMode KeyframeModeTranslator(const std::string &s);
    static std::string KeyframeModeTranslator(KeyframeMode mode);
    static TrackingMode TrackingModeTranslator(const std::string &s);
    static std::string TrackingModeTranslator(TrackingMode mode);
  };

} // namespace df

#endif // DF_OPTIONS_H_
