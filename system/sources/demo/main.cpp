#include <gflags/gflags.h>
#include <glog/logging.h>

#include <execinfo.h>
#include <signal.h>
#include <string.h>

#include <iostream>
#include <cstdlib>
#include <stdexcept>

#include "live_demo.h"
#include "logutils.h"
#include "camera_interface_factory.h"

void my_terminate(void);

namespace
{
	// invoke set_terminate as part of global constant initialization
	static const bool SET_TERMINATE = std::set_terminate(my_terminate);
}

// This structure mirrors the one found in /usr/include/asm/ucontext.h
typedef struct _sig_ucontext
{
	unsigned long uc_flags;
	struct ucontext *uc_link;
	stack_t uc_stack;
	struct sigcontext uc_mcontext;
	sigset_t uc_sigmask;
} sig_ucontext_t;

void crit_err_hdlr(int sig_num, siginfo_t *info, void *ucontext)
{
	// std::array<char const *, 4> lookup =
	// 		{{"The demangling operation succeeded",
	// 			"A memory allocation failure occurred",
	// 			"mangled_name is not a valid name under the C++ ABI mangling rules",
	// 			"One of the arguments is invalid"}};

	void *array[50];
	void *caller_address;
	char **messages;
	int size, i;
	sig_ucontext_t *uc;

	uc = (sig_ucontext_t *)ucontext;

	/* Get the address at the time the signal was raised */
#if defined(__i386__)														// gcc specific
	caller_address = (void *)uc->uc_mcontext.eip; // EIP: x86 specific
#elif defined(__x86_64__)												// gcc specific
	caller_address = (void *)uc->uc_mcontext.rip; // RIP: x86_64 specific
#elif defined(__aarch64__)
	caller_address = (void*)uc->uc_mcontext.fault_address;
#else
#error Unsupported architecture.
#endif

	fprintf(stderr, "signal %d (%s), address is %p from %p\n",
					sig_num, strsignal(sig_num), info->si_addr,
					(void *)caller_address);

	size = backtrace(array, 50);

	/* overwrite sigaction with caller's address */
	array[1] = caller_address;

	messages = backtrace_symbols(array, size);

	/* skip first stack frame (points here) */
	for (i = 1; i < size && messages != NULL; ++i)
	{
		fprintf(stderr, "[bt]: (%d) %s\n", i, messages[i]);
	}

	free(messages);

	exit(EXIT_FAILURE);
}

void my_terminate()
{
	static bool tried_throw = false;

	try
	{
		// try once to re-throw currently active exception
		if (!tried_throw)
		{
			tried_throw = true;
			throw;
		}
	}
	catch (const std::exception &e)
	{
		std::cerr << __FUNCTION__ << " caught unhandled exception. what(): "
							<< e.what() << std::endl;
	}
	catch (...)
	{
		std::cerr << __FUNCTION__ << " caught unknown/unhandled exception."
							<< std::endl;
	}

	void *array[50];
	int size = backtrace(array, 50);

	std::cerr << __FUNCTION__ << " backtrace returned "
						<< size << " frames\n\n";

	char **messages = backtrace_symbols(array, size);

	for (int i = 0; i < size && messages != NULL; ++i)
	{
		std::cerr << "[bt]: (" << i << ") " << messages[i] << std::endl;
	}
	std::cerr << std::endl;

	free(messages);

	abort();
}

/* demo options */
std::string source_url_help = "Image source URL" + df::drivers::CameraInterfaceFactory::Get()->GetUrlHelp();
DEFINE_string(source_url, "hdf5://", source_url_help.c_str());
DEFINE_bool(init_on_start, false,
						"Initialize the system on the first captured image");
DEFINE_bool(quit_on_finish, false,
						"Close the program after finishing all frames");
DEFINE_bool(enable_gui, false,
						"Whether to enable the GUI visualization");
DEFINE_string(init_type, "TWOFRAME",
							"How to initialize the slam system (ONEFRAME, TWOFRAME)");
DEFINE_string(record_input, "", "Path where to save input images");
DEFINE_bool(pause_step, false, "Pause after each frame");
DEFINE_string(run_log_dir, "",
							"Directory where the run logs will be saved. The runs are stored in separate dirs named using current timestamp. \n"
							"If not specified the logs will not be saved. This directory will be created if it doesn't exist.");
DEFINE_string(run_dir_name, "",
							"Force a specific run directory name. If empty, a timestamp is used");
DEFINE_bool(enable_timing, false,
						"Enable profiling of certain parts of the algorithm");
DEFINE_uint32(frame_limit, 0, "Limit processing to first <frame_limit> frames");
DEFINE_uint32(skip_frames, 0, "Skip first <skip_frames> frames");
DEFINE_bool(demo_mode, false, "Whether to hide GUI elements and show only reconstruction");
DEFINE_bool(record_video, false, "Whether to record SLAM run rendering");
DEFINE_int32(frame_interval, 1, "the interval of frame for SLAM to process");

// visualization
DEFINE_int32(num_visible_keyframes, 5, "number of visualized keyframes");
DEFINE_int32(num_visible_frustums, 25, "number of visualized frustrums");

// network paths, vocabulary path, device id
DEFINE_string(depth_network_path, "", "Path to ScriptModule file containing depth network");
DEFINE_string(feature_network_path, "", "Path to ScriptModule file containing feature network");
DEFINE_string(vocab_path, "", "Path to vocabulary storage file for DBoW2");
DEFINE_uint32(cuda_id, 0, "Which cuda device to use for SLAM");

// keyframe adding
DEFINE_string(keyframe_mode, "AUTO",
							"New keyframe initialization criteria (AUTO, NEVER)");

// connections
// DEFINE_string(connection_mode, "LAST_N",
// 							"How new keyframes will be connected to the others (FULL, LAST_N, FIRST, LAST)");
DEFINE_double(temporal_max_back_connections, 4, "How far to connect back new keyframes");

// epsilons
DEFINE_double(warp_dpt_eps, 1.0e-4, "depth epsilon of determining positive depth mask in image warping");
DEFINE_double(linearize_eps, 1.0e-5, "linearization epsilon for variable relinearization");

// pose distance
DEFINE_double(pose_dist_trans_weight, 1.0, "translation weight of pose distance");
DEFINE_double(pose_dist_rot_weight, 1.0, "rotation weight of pose distance");

/* tracking */
DEFINE_string(tracking_mode, "LAST", "How to select which keyframe to track against (CLOSEST, LAST, FIRST)");
DEFINE_uint32(tracking_max_num_iters, 50, "maximum number of iterations (tracking)");
DEFINE_double(tracking_min_grad_thresh, 1.0e-5, "minimum gradient threshold of LM convergence");
DEFINE_double(tracking_min_param_inc_thresh, 1.0e-5, "minimum param increment threshold of LM convergence");
DEFINE_double(tracking_init_damp, 1.0e-4, "initial damp value for LM optimization in tracking");
DEFINE_string(tracking_min_max_damp, "1.0e-6,1.0e6", "min max damp value");
DEFINE_string(tracking_damp_dec_inc_factor, "9.0,11.0", "multiplicative factors for damp value during decrease and increase");
DEFINE_double(tracking_jac_update_err_inc_threshold, 1.0e-3, "err increment ratio threshold for updating jacobian");
DEFINE_double(tracking_ref_kf_select_ratio, 0.8, "ratio to surpass to accept the keyframe as the reference one");

// tracking lost criteria
DEFINE_double(tracking_lost_min_error, 0.05, "minimum error to consider tracking lost in tracking");
DEFINE_double(tracking_lost_max_area_ratio, 0.3, "max area ratio to consider tracking lost");
DEFINE_double(tracking_lost_max_inlier_ratio, 0.3, "max inlier ratio to consider tracking lost");

/* loop closure */
DEFINE_bool(use_global_loop, true, "whether to use global loop detection backend");
DEFINE_bool(use_local_loop, true, "whether to use local loop detection backend");
DEFINE_int64(loop_local_active_window, 10, "Active window for local loop detection");
DEFINE_double(loop_local_dist_ratio, 2.0, "distance ratio to check for local loop");
DEFINE_int64(loop_global_active_window, 30, "no acceptance window for global loop detection");
DEFINE_uint32(loop_max_candidates, 3, "Maximum number of loop candidates to get from DBoW2");
DEFINE_double(loop_min_area_ratio, 0.8, "min area ratio to consider as a valid loop candidate");
DEFINE_double(loop_min_inlier_ratio, 0.8, "min inlier ratio to consider as a valid loop candidate");
DEFINE_double(loop_min_desc_inlier_ratio, 0.3, "min inlier ratio to consider as a valid loop candidate");
DEFINE_uint32(global_redundant_range, 5, "range to not consider another global loop connection");
DEFINE_bool(loop_use_match_geom, true, "whether to use match geom factor in loop geom verification");

DEFINE_uint32(loop_tracking_max_num_iters, 40, "maximum number of iterations (tracking)");
DEFINE_double(loop_tracking_min_grad_thresh, 1.0e-3, "minimum gradient threshold of LM convergence");
DEFINE_double(loop_tracking_min_param_inc_thresh, 1.0e-3, "minimum param increment threshold of LM convergence");
DEFINE_string(loop_tracking_damp_dec_inc_factor, "9.0,11.0", "multiplicative factors for damp value during decrease and increase");
DEFINE_double(loop_global_sim_ratio, 0.8, "similarity ratio between the candidate and the temporal connection");
DEFINE_double(loop_global_metric_ratio, 0.7, "metric ratio between the candidate and the global connection");
DEFINE_double(loop_local_metric_ratio, 0.7, "metric ratio between the candidate and the local connection");
DEFINE_double(loop_detection_frequency, 10.0, "frequency of backend loop detection");
DEFINE_double(loop_pose_linearize_threshold, 1.0e-3, "pose linearize threshold for global loop closure optimization");
DEFINE_double(loop_scale_linearize_threshold, 1.0e-2, "scale linearize threshold for global loop closure optimization");

// temporal connection
DEFINE_double(temporal_min_desc_inlier_ratio, 0.1, "min inlier ratio to consider as a valid loop candidate");
//

DEFINE_int64(refine_mapping_iters, 100, "number of refinements for final mapping");
DEFINE_int64(refine_mapping_num_no_linearize, 10, "number of refinements for final mapping");
// new (key)frame criteria
DEFINE_double(new_kf_max_area_ratio, 0.8, "max area ratio to create new keyframe");
DEFINE_double(new_kf_max_desc_inlier_ratio, 0.1, "max desc match inlier ratio to create new keyframe");
DEFINE_double(new_kf_max_inlier_ratio, 0.8, "max inlier ratio to create new keyframe");
DEFINE_double(new_kf_min_average_motion, 0.04, "min average motion to create new keyframe");

// network size and video mask path
DEFINE_string(net_input_size, "128,160", "input size of network");
DEFINE_string(net_output_size, "64,80", "output size of network");

// mapping
DEFINE_double(mapping_update_frequency, 5.0, "frequency of backend mapping");

// variable prior
DEFINE_double(init_pose_prior_weight, 10.0, "Noise of the prior on the first pose");
DEFINE_double(init_scale_prior_weight, 10.0, "Noise of the prior on the first scale");

// factors
DEFINE_uint32(factor_iters, 100, "number of iterations for factors");

/* photometric error */
DEFINE_bool(use_photometric, true,
						"Use photometric error in keyframe-keyframe links");
DEFINE_bool(use_tracker_photometric, true,
						"Use sparse photometric error for camera tracking");
DEFINE_int32(pho_num_samples, 3072, "number of samples for photometric factor");
DEFINE_string(photo_factor_weights, "2.0,1.8,1.7,1.6", "factor weights for photometric factor (from finest to coarsest)");

/* reprojection error */
DEFINE_bool(use_reprojection, true,
						"Use sparse reprojection error for mapping");
DEFINE_bool(use_tracker_reprojection, true,
						"Use sparse reprojection error for camera tracking");
// use_tracker_photometric
DEFINE_uint32(desc_num_keypoints, 512, "Number of keypoints to detect in a keyframe");
DEFINE_uint32(tracking_desc_num_keypoints, 256, "Number of keypoints to detect in a keyframe");
DEFINE_double(desc_cyc_consis_thresh, 2.0, "cycle consistency distance threshold of feature matching in reprojection error");
// DEFINE_double(reproj_loss_param_factor, 0.235, "loss param factor of reprojection error");
// DEFINE_double(reproj_factor_weight, 0.53, "reprojection factor weight");

/* geometric error */
DEFINE_bool(use_geometric, true, "Use sparse geometric error for mapping");
DEFINE_double(geo_loss_param_factor, 0.12, "loss param of geometric factor");
DEFINE_double(geo_factor_weight, 0.05, "geometric factor weight");

// match geometry factor
DEFINE_double(match_geom_factor_weight, 5.0, "match geometry factor weight");
DEFINE_double(match_geom_loss_param_factor, 0.1, "loss param of match geometry factor");
DEFINE_double(tracker_match_geom_factor_weight, 5.0, "tracker match geometry factor weight");

// reprojection factor
DEFINE_double(reproj_factor_weight, 5.0, "reprojection factor weight");
DEFINE_double(tracker_reproj_factor_weight, 5.0, "reprojection factor weight");
DEFINE_double(reproj_loss_param_factor, 0.1, "loss param of reprojection factor");

// code factor
DEFINE_double(code_factor_weight, 1.0e-3, "code factor weight");

// ISAM2 related
DEFINE_double(isam_pose_lin_eps, 1.0e-2, "linearize threshold for pose vector");
DEFINE_double(isam_code_lin_eps, 3.0e-2, "linearize threshold for code vector");
DEFINE_double(isam_scale_lin_eps, 3.0e-2, "linearize threshold for scale");
DEFINE_double(isam_relinearize_skip, 1, "ISAM2 relinearize skip");
DEFINE_bool(isam_partial_relin_check, true, "ISAM2 partial relinarization check");
DEFINE_bool(isam_enable_detailed_results, false, "whether or not to enable detailed isam results");
DEFINE_double(isam_wildfire_threshold, 1.0e-3, "wildfire threshold for Dogleg optimizer");

// pose graph related
DEFINE_double(pose_graph_local_link_weight, 1.0, "weight of local link relative pose factor");
DEFINE_double(pose_graph_global_link_weight, 10.0, "weight of global link relative pose factor");
DEFINE_double(pose_graph_scale_prior_weight, 1.0e-5, "weight of scale prior weight in pose graph");
DEFINE_double(pose_graph_rot_weight, 1.0, "rotation part weight of relative pose factor");
DEFINE_double(pose_graph_scale_weight, 1.0, "scale weight of relative pose scale factor");
DEFINE_int32(pose_graph_max_iters, 200, "maximum number of iterations for pose graph");
DEFINE_int32(pose_scale_graph_max_iters, 10, "maximum number of iterations for pose scale graph");
DEFINE_int32(pose_graph_no_relin_max_iters, 20, "maximum number of consecutive no-linear iterations for pose graph");
DEFINE_int32(pose_scale_graph_no_relin_max_iters, 10, "maximum number of consecutive no-linear iterations for pose scale graph");

// TEASER++ related
DEFINE_double(teaser_max_clique_time_limit, 0.2, "maximum time limit for max clique finding");
DEFINE_double(teaser_kcore_heuristic_threshold, 0.5, "heuristic threshold for kcore");
DEFINE_uint32(teaser_rotation_max_iterations, 50, "maximum number of iterations for rotation solver");
DEFINE_double(teaser_rotation_cost_threshold, 1.0e-6, "rotation cost threshold between consecutive iterations");
DEFINE_double(teaser_rotation_gnc_factor, 1.4, "gnc factor of rotation estimation"); // For GNC-TLS: the algorithm multiples the control parameter by the factor every iteration.
DEFINE_string(teaser_rotation_estimation_algorithm, "gnc_tls", "rotation estimation algorithm type");
DEFINE_string(teaser_rotation_tim_graph, "chain", "rotation tim graph type");
DEFINE_string(teaser_inlier_selection_mode, "pmc_exact", "inlier selection mode of teaser");
DEFINE_string(teaser_tracker_inlier_selection_mode, "pmc_exact", "inlier selection mode of teaser for camera tracker");
DEFINE_double(teaser_noise_bound_multiplier, 2.0, "multiplier of noise bound per point");

std::vector<float> splitf(const std::string &s, char delim)
{
	std::stringstream ss(s);
	std::string item;
	std::vector<float> elems;
	while (std::getline(ss, item, delim))
		elems.push_back(std::stof(item));
	return elems;
}

std::vector<long> splitl(const std::string &s, char delim)
{
	std::stringstream ss(s);
	std::string item;
	std::vector<long> elems;
	while (std::getline(ss, item, delim))
		elems.push_back(std::stol(item));
	return elems;
}

std::string getParentFolderName(std::string path)
{
	// Remove directory if present.
	// Do this before extension removal incase directory has a period character.
	size_t second_last_slash_idx = path.find_last_of("/", path.find_last_of("/") - 1);

	if (std::string::npos != second_last_slash_idx)
	{
		path.erase(0, second_last_slash_idx + 1);
		size_t pos = path.find_last_of("/");
		if (std::string::npos != pos)
		{
			path.erase(pos);
		}
	}

	return path;
}
int main(int argc, char *argv[])
{
	torch::NoGradGuard no_grad;
	// parse command line flags
	google::SetUsageMessage("SLAM Live Demo");
	google::ParseCommandLineFlags(&argc, &argv, true);
	// Init google logging
	FLAGS_alsologtostderr = true;
	google::InitGoogleLogging(argv[0]);

	// create a logging directory for this run
	// point glog to log there
	// save command line flags
	std::string run_dir;
	if (!FLAGS_run_log_dir.empty())
	{
		run_dir = df::CreateLogDirForRun(FLAGS_run_log_dir, FLAGS_run_dir_name + "_" + getParentFolderName(FLAGS_source_url) + "_" + df::GetTimeStamp());
		VLOG(1) << "Logging directory: " << run_dir;
		// save flags
		google::AppendFlagsIntoFile(run_dir + "/df.flags", nullptr);
	}

	// FLAGS_stderrthreshold = 0;
	// FLAGS_minloglevel = 0;
	// FLAGS_v = 0;

	google::SetLogDestination(google::INFO, (run_dir + std::string("/info.log")).c_str());

	/* demo options */
	df::LiveDemoOptions opts;
	opts.source_url = FLAGS_source_url;
	opts.init_on_start = FLAGS_init_on_start;
	opts.quit_on_finish = FLAGS_quit_on_finish;
	opts.init_type = df::LiveDemoOptions::InitTypeTranslator(FLAGS_init_type);
	opts.record_input = !FLAGS_record_input.empty();
	opts.record_path = FLAGS_record_input;
	opts.pause_step = FLAGS_pause_step;
	opts.log_dir = run_dir;
	opts.enable_timing = FLAGS_enable_timing;
	opts.frame_limit = FLAGS_frame_limit;
	opts.skip_frames = FLAGS_skip_frames;
	opts.demo_mode = FLAGS_demo_mode;
	opts.record_video = FLAGS_record_video;
	opts.frame_interval = FLAGS_frame_interval;
	opts.enable_gui = FLAGS_enable_gui;
	// visualization
	opts.num_visible_keyframes = FLAGS_num_visible_keyframes;
	opts.num_visible_frustums = FLAGS_num_visible_frustums;
	
	opts.df_opts.enable_gui = FLAGS_enable_gui;
	opts.df_opts.log_dir = run_dir;
	opts.df_opts.cuda_id = FLAGS_cuda_id;

	// network paths, vocabulary path, device id
	opts.df_opts.depth_network_path = FLAGS_depth_network_path;
	opts.df_opts.feature_network_path = FLAGS_feature_network_path;
	opts.df_opts.vocabulary_path = FLAGS_vocab_path;

	/* tracking */
	opts.df_opts.tracking_mode = df::DeepFactorsOptions::TrackingModeTranslator(
			FLAGS_tracking_mode);
	opts.df_opts.tracking_max_num_iters = FLAGS_tracking_max_num_iters;
	opts.df_opts.tracking_min_grad_thresh = FLAGS_tracking_min_grad_thresh;
	opts.df_opts.tracking_min_param_inc_thresh = FLAGS_tracking_min_param_inc_thresh;
	opts.df_opts.tracking_init_damp = FLAGS_tracking_init_damp;
	opts.df_opts.tracking_min_max_damp = splitf(FLAGS_tracking_min_max_damp, ',');
	opts.df_opts.tracking_damp_dec_inc_factor = splitf(FLAGS_tracking_damp_dec_inc_factor, ',');
	opts.df_opts.tracking_jac_update_err_inc_threshold = FLAGS_tracking_jac_update_err_inc_threshold;
	opts.df_opts.tracking_ref_kf_select_ratio = FLAGS_tracking_ref_kf_select_ratio;
	// tracking lost related
	opts.df_opts.tracking_lost_min_error = FLAGS_tracking_lost_min_error;
	opts.df_opts.tracking_lost_max_area_ratio = FLAGS_tracking_lost_max_area_ratio;
	opts.df_opts.tracking_lost_max_inlier_ratio = FLAGS_tracking_lost_max_inlier_ratio;

	// New (key)frames criteria
	opts.df_opts.new_kf_max_area_ratio = FLAGS_new_kf_max_area_ratio;
	opts.df_opts.new_kf_max_inlier_ratio = FLAGS_new_kf_max_inlier_ratio;
	opts.df_opts.new_kf_max_desc_inlier_ratio = FLAGS_new_kf_max_desc_inlier_ratio;
	opts.df_opts.new_kf_min_average_motion = FLAGS_new_kf_min_average_motion;

	// pose distance
	opts.df_opts.pose_dist_trans_weight = FLAGS_pose_dist_trans_weight;
	opts.df_opts.pose_dist_rot_weight = FLAGS_pose_dist_rot_weight;

	// connections
	opts.df_opts.temporal_max_back_connections = FLAGS_temporal_max_back_connections;

	// keyframe adding
	opts.df_opts.keyframe_mode = df::DeepFactorsOptions::KeyframeModeTranslator(
			FLAGS_keyframe_mode);

	/* loop closure */
	opts.df_opts.use_global_loop = FLAGS_use_global_loop;
	opts.df_opts.use_local_loop = FLAGS_use_local_loop;
	opts.df_opts.loop_local_active_window = FLAGS_loop_local_active_window;
	opts.df_opts.loop_local_dist_ratio = FLAGS_loop_local_dist_ratio;
	opts.df_opts.loop_global_active_window = FLAGS_loop_global_active_window;
	opts.df_opts.loop_max_candidates = FLAGS_loop_max_candidates;
	opts.df_opts.loop_min_area_ratio = FLAGS_loop_min_area_ratio;
	opts.df_opts.loop_min_inlier_ratio = FLAGS_loop_min_inlier_ratio;
	opts.df_opts.global_redundant_range = FLAGS_global_redundant_range;
	opts.df_opts.loop_use_match_geom = FLAGS_loop_use_match_geom;
	opts.df_opts.loop_min_desc_inlier_ratio = FLAGS_loop_min_desc_inlier_ratio;
	opts.df_opts.loop_tracking_max_num_iters = FLAGS_loop_tracking_max_num_iters;
	opts.df_opts.loop_tracking_min_grad_thresh = FLAGS_loop_tracking_min_grad_thresh;
	opts.df_opts.loop_tracking_min_param_inc_thresh = FLAGS_loop_tracking_min_param_inc_thresh;
	opts.df_opts.loop_tracking_damp_dec_inc_factor = splitf(FLAGS_loop_tracking_damp_dec_inc_factor, ',');
	opts.df_opts.loop_global_sim_ratio = FLAGS_loop_global_sim_ratio;
	opts.df_opts.loop_global_metric_ratio = FLAGS_loop_global_metric_ratio;
	opts.df_opts.loop_local_metric_ratio = FLAGS_loop_local_metric_ratio;
	opts.df_opts.loop_detection_frequency = FLAGS_loop_detection_frequency;
	opts.df_opts.loop_pose_linearize_threshold = FLAGS_loop_pose_linearize_threshold;
	opts.df_opts.loop_scale_linearize_threshold = FLAGS_loop_scale_linearize_threshold;

	// temporal connection
	opts.df_opts.temporal_min_desc_inlier_ratio = FLAGS_temporal_min_desc_inlier_ratio;

	/* mapping */
	opts.df_opts.mapping_update_frequency = FLAGS_mapping_update_frequency;
	opts.df_opts.refine_mapping_iters = FLAGS_refine_mapping_iters;
	opts.df_opts.refine_mapping_num_no_linearize = FLAGS_refine_mapping_num_no_linearize;

	// epsilons
	opts.df_opts.warp_dpt_eps = FLAGS_warp_dpt_eps;

	// network spatial size and video mask path
	opts.df_opts.net_input_size = splitl(FLAGS_net_input_size, ',');
	opts.df_opts.net_output_size = splitl(FLAGS_net_output_size, ',');

	// variable prior
	opts.df_opts.init_pose_prior_weight = FLAGS_init_pose_prior_weight;
	opts.df_opts.init_scale_prior_weight = FLAGS_init_scale_prior_weight;

	// factors
	opts.df_opts.factor_iters = FLAGS_factor_iters;

	/* photometric error */
	opts.df_opts.use_photometric = FLAGS_use_photometric;
	opts.df_opts.use_tracker_photometric = FLAGS_use_tracker_photometric;
	opts.df_opts.pho_num_samples = FLAGS_pho_num_samples;
	opts.df_opts.photo_factor_weights = splitf(FLAGS_photo_factor_weights, ',');
	opts.df_opts.num_pyramid_levels = opts.df_opts.photo_factor_weights.size();

	/* reprojection error */
	opts.df_opts.use_reprojection = FLAGS_use_reprojection;
	opts.df_opts.use_tracker_reprojection = FLAGS_use_tracker_reprojection;
	opts.df_opts.desc_num_keypoints = FLAGS_desc_num_keypoints;
	opts.df_opts.tracking_desc_num_keypoints = FLAGS_tracking_desc_num_keypoints;
	opts.df_opts.desc_cyc_consis_thresh = FLAGS_desc_cyc_consis_thresh;

	/* geometric error */
	opts.df_opts.use_geometric = FLAGS_use_geometric;
	opts.df_opts.geo_factor_weight = FLAGS_geo_factor_weight;
	opts.df_opts.geo_loss_param_factor = FLAGS_geo_loss_param_factor;

	// match geometry error
	opts.df_opts.match_geom_factor_weight = FLAGS_match_geom_factor_weight;
	opts.df_opts.match_geom_loss_param_factor = FLAGS_match_geom_loss_param_factor;
	opts.df_opts.tracker_match_geom_factor_weight = FLAGS_tracker_match_geom_factor_weight;

	// reprojection factor
	opts.df_opts.reproj_factor_weight = FLAGS_reproj_factor_weight;
	opts.df_opts.tracker_reproj_factor_weight = FLAGS_tracker_reproj_factor_weight;
	opts.df_opts.reproj_loss_param_factor = FLAGS_reproj_loss_param_factor;

	// code factor prior
	opts.df_opts.code_factor_weight = FLAGS_code_factor_weight;

	// ISAM2 related
	opts.df_opts.isam_relinearize_skip = FLAGS_isam_relinearize_skip;
	opts.df_opts.isam_partial_relin_check = FLAGS_isam_partial_relin_check;
	opts.df_opts.isam_enable_detailed_results = FLAGS_isam_enable_detailed_results;
	opts.df_opts.isam_pose_lin_eps = FLAGS_isam_pose_lin_eps;
	opts.df_opts.isam_code_lin_eps = FLAGS_isam_code_lin_eps;
	opts.df_opts.isam_scale_lin_eps = FLAGS_isam_scale_lin_eps;
	opts.df_opts.isam_wildfire_threshold = FLAGS_isam_wildfire_threshold;

	// pose graph related
	opts.df_opts.pose_graph_local_link_weight = FLAGS_pose_graph_local_link_weight;
	opts.df_opts.pose_graph_global_link_weight = FLAGS_pose_graph_global_link_weight;
	opts.df_opts.pose_graph_scale_prior_weight = FLAGS_pose_graph_scale_prior_weight;
	opts.df_opts.pose_graph_rot_weight = FLAGS_pose_graph_rot_weight;
	opts.df_opts.pose_graph_scale_weight = FLAGS_pose_graph_scale_weight;
	opts.df_opts.pose_graph_max_iters = FLAGS_pose_graph_max_iters;
	opts.df_opts.pose_scale_graph_max_iters = FLAGS_pose_scale_graph_max_iters;
	opts.df_opts.pose_graph_no_relin_max_iters = FLAGS_pose_graph_no_relin_max_iters;
	opts.df_opts.pose_scale_graph_no_relin_max_iters = FLAGS_pose_scale_graph_no_relin_max_iters;

	// TEASER++ related
	opts.df_opts.teaser_max_clique_time_limit = FLAGS_teaser_max_clique_time_limit;
	opts.df_opts.teaser_kcore_heuristic_threshold = FLAGS_teaser_kcore_heuristic_threshold;
	opts.df_opts.teaser_rotation_max_iterations = FLAGS_teaser_rotation_max_iterations;
	opts.df_opts.teaser_rotation_cost_threshold = FLAGS_teaser_rotation_cost_threshold;
	opts.df_opts.teaser_rotation_gnc_factor = FLAGS_teaser_rotation_gnc_factor;
	opts.df_opts.teaser_rotation_estimation_algorithm = FLAGS_teaser_rotation_estimation_algorithm;
	opts.df_opts.teaser_rotation_tim_graph = FLAGS_teaser_rotation_tim_graph;
	opts.df_opts.teaser_inlier_selection_mode = FLAGS_teaser_inlier_selection_mode;
	opts.df_opts.teaser_tracker_inlier_selection_mode = FLAGS_teaser_tracker_inlier_selection_mode;
	opts.df_opts.teaser_noise_bound_multiplier = FLAGS_teaser_noise_bound_multiplier;

	struct sigaction sigact;
	sigact.sa_sigaction = crit_err_hdlr;
	sigact.sa_flags = SA_RESTART | SA_SIGINFO;

	if (sigaction(SIGABRT, &sigact, (struct sigaction *)NULL) != 0)
	{
		fprintf(stderr, "error setting signal handler for %d (%s)\n",
						SIGABRT, strsignal(SIGABRT));
		exit(EXIT_FAILURE);
	}

	if (sigaction(SIGSEGV, &sigact, (struct sigaction *)NULL) != 0)
	{
		fprintf(stderr, "error setting signal handler for %d (%s)\n",
						SIGSEGV, strsignal(SIGSEGV));
		exit(EXIT_FAILURE);
	}

	// create and run the live demo
	df::LiveDemo<DF_CODE_SIZE> demo(opts);
	demo.Run();

	// cleanup
	google::ShutdownGoogleLogging();
	google::ShutDownCommandLineFlags();

	return 0;
}
