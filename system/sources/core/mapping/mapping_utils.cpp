#include <opengv/relative_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/sac/Ransac.hpp>

#include "mapping_utils.h"

namespace df
{

  void PruneMatchesEightPoint(at::Tensor &inlier_indexes,
                              const at::Tensor keypoint_homo_locations_0,
                              const at::Tensor matched_homo_locations_1,
                              double threshold,
                              int max_iterations,
                              double probability)
  {
    /*
   * The RANSAC paradigm contains three unspecified parameters:
   * (1) the error tolerance used to determine whether or
   *     not a point is compatible with a model,
   * (2) the number of subsets to try
   * (3) the threshold t, which is the number of compatible points
   *     used to imply that the correct model has been found.
   */
    using namespace torch::indexing;
    using opengv::relative_pose::CentralRelativeAdapter;
    using opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem;

    // bearing vectors here are normalized homogenous locations (x, y, 1) / length
    opengv::bearingVectors_t bearingVectors0, bearingVectors1;
    bearingVectors0.resize(keypoint_homo_locations_0.size(0));
    bearingVectors1.resize(matched_homo_locations_1.size(0));

    const at::Tensor normalized_keypoint_homo_locations_0 =
        (keypoint_homo_locations_0 / torch::norm(keypoint_homo_locations_0, 2, 1, true)).to(torch::kDouble);
    const at::Tensor normalized_matched_homo_locations_1 =
        (matched_homo_locations_1 / torch::norm(matched_homo_locations_1, 2, 1, true)).to(torch::kDouble);

    // Transfer torch tensor to bearing vectors
    std::memcpy(bearingVectors0.data(),
                normalized_keypoint_homo_locations_0.to(torch::kCPU).contiguous().data_ptr(),
                sizeof(double) * normalized_keypoint_homo_locations_0.numel());
    std::memcpy(bearingVectors1.data(),
                normalized_matched_homo_locations_1.to(torch::kCPU).contiguous().data_ptr(),
                sizeof(double) * normalized_matched_homo_locations_1.numel());

    // Set up a CentralRelativePoseSacProblem with opengv
    CentralRelativeAdapter adapter(bearingVectors0, bearingVectors1);
    opengv::sac::Ransac<CentralRelativePoseSacProblem> ransac;
    ransac.sac_model_ = std::make_shared<CentralRelativePoseSacProblem>(adapter, CentralRelativePoseSacProblem::EIGHTPT);
    // This threshold seems to be a 2*(1 - cosine distance) between the reprojection bearing vector and the corresponding one range [0, 4]
    ransac.threshold_ = threshold;
    ransac.max_iterations_ = max_iterations;
    ransac.probability_ = probability;

    // get the result
    ransac.computeModel();

    // opengv::transformation_t transform = ransac.model_coefficients_;
    // Eigen::MatrixXd R = transform.block<3, 3>(0, 0);
    // Eigen::MatrixXd t = transform.block<3, 1>(0, 3);

    inlier_indexes = torch::from_blob(ransac.inliers_.data(), {static_cast<long>(ransac.inliers_.size())},
                                      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU))
                         .clone()
                         .to(torch::kLong)
                         .to(keypoint_homo_locations_0.device());
    return;
  }

  /* ************************************************************************* */
  template <typename Scalar>
  cv::Mat DisplayKeyframes(typename df::Map<Scalar>::Ptr map, int num_keyframes)
  {
    torch::NoGradGuard no_grad;
    if (map->NumKeyframes() == 0)
    {
      return cv::Mat();
    }

    std::vector<cv::Mat> array;
    int start = map->keyframes.Ids().size() - 1;
    int end = std::max((int)map->keyframes.Ids().size() - num_keyframes, 0);

    auto kf = map->keyframes.Get(map->keyframes.Ids()[0]);
    const at::Tensor video_mask = *(kf->video_mask_ptr);
    for (int i = start; i >= end; --i)
    {
      kf = map->keyframes.Get(map->keyframes.Ids()[i]);
      at::Tensor dpt_map;
      {
        std::shared_lock<std::shared_mutex> lock(kf->mutex);
        dpt_map = kf->dpt_map * video_mask;
      }

      const at::Tensor max = torch::max(dpt_map);
      float max_dpt = max.item<float>();
      cv::Mat dpt_display = Tensor2Mat(torch::clamp_min(dpt_map / max_dpt, 0.0));
      array.push_back(kf->color_img);
      array.push_back(apply_colormap(dpt_display));
    }
    int ncols = 2;

    cv::Mat mosaic;
    const double ratio = 128.0 / array[0].rows;
    cv::resize(CreateMosaic(array, array.size() / ncols, ncols), mosaic, cv::Size(0, 0), ratio, ratio);

    return mosaic;
  }
  /* ************************************************************************* */
  template <typename Scalar>
  cv::Mat DisplaySE3Warp(const df::Keyframe<Scalar> &keyframe_0, const df::Frame<Scalar> &frame_1,
                         const at::Tensor rotation_10, const at::Tensor translation_10, const Scalar &dpt_eps,
                         const cv::Mat &checkerboard)
  {
    const df::PinholeCamera<Scalar> &cam = (*(keyframe_0.camera_pyramid_ptr))[0];
    std::vector<cv::Mat> array;
    const long height = cam.height();
    const long width = cam.width();
    at::Tensor warped_color_1, warped_valid_mask_1;

    const at::Tensor video_mask = (*(keyframe_0.video_mask_ptr)).reshape({1, 1, height, width});
    cv::Mat resized_color_1, resized_color_0;
    cv::resize(frame_1.color_img, resized_color_1, cv::Size2l(width, height));
    cv::resize(keyframe_0.color_img, resized_color_0, cv::Size2l(width, height));

    const at::Tensor input_color_1 = torch::from_blob(static_cast<unsigned char *>(resized_color_1.data), {1, height, width, 3},
                                                      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                                         .to(torch::kFloat32)
                                         .to(video_mask.device())
                                         .permute({0, 3, 1, 2})
                                         .clone();
    Sophus::SE3f relpose_10;
    Eigen::Matrix<float, 3, 3> rotation10_mat;
    TensorToEigenMatrix(rotation_10, rotation10_mat);
    relpose_10.so3() = Sophus::SO3f(rotation10_mat);
    TensorToEigenMatrix(translation_10, relpose_10.translation());

    at::Tensor dpt_map;
    {
      std::shared_lock<std::shared_mutex> lock(keyframe_0.mutex);
      dpt_map = keyframe_0.dpt_map * video_mask;
    }

    SE3ImageWarping(relpose_10, cam, input_color_1, dpt_map, video_mask, warped_color_1, warped_valid_mask_1, dpt_eps);

    cv::Mat warped_img_1 = Tensor2Mat(warped_color_1);
    cv::cvtColor(resized_color_0.clone(), resized_color_0, cv::COLOR_BGR2RGB);
    resized_color_0.convertTo(resized_color_0, CV_32FC3);
    cv::Mat rev_checkerboard;
    cv::subtract(cv::Scalar(1, 1, 1), checkerboard, rev_checkerboard);
    cv::Mat ori_warp_overlay = resized_color_0.mul(checkerboard) + warped_img_1.mul(rev_checkerboard);
    ori_warp_overlay.convertTo(ori_warp_overlay, CV_8UC3);
    warped_img_1.convertTo(warped_img_1, CV_8UC3);
    resized_color_0.convertTo(resized_color_0, CV_8UC3);
    cv::cvtColor(resized_color_0.clone(), resized_color_0, cv::COLOR_RGB2BGR);

    const at::Tensor max = torch::max(dpt_map);
    float max_dpt_0 = max.item<float>();
    cv::Mat dpt_display_0 = Tensor2Mat(torch::clamp_min(dpt_map / max_dpt_0, 0.0));
    dpt_display_0 = apply_colormap(dpt_display_0);

    array.push_back(resized_color_0);
    array.push_back(warped_img_1);
    array.push_back(ori_warp_overlay);
    array.push_back(dpt_display_0);

    cv::Mat mosaic;
    const double ratio = 128.0 / array[0].rows;
    cv::resize(CreateMosaic(array, 1, 4), mosaic, cv::Size(0, 0), ratio, ratio);
    return mosaic;
  }

  /* ************************************************************************* */
  template <typename Scalar>
  cv::Mat DisplaySE3Warp(const df::Frame<Scalar> &frame_0, const df::Frame<Scalar> &frame_1,
                         const at::Tensor rotation_10, const at::Tensor translation_10, const float scale_0, const Scalar &dpt_eps,
                         const cv::Mat &checkerboard)
  {
    const df::PinholeCamera<Scalar> &cam = (*(frame_0.camera_pyramid_ptr))[0];
    std::vector<cv::Mat> array;
    const long height = cam.height();
    const long width = cam.width();
    at::Tensor warped_color_1, warped_valid_mask_1;

    const at::Tensor video_mask = (*(frame_0.video_mask_ptr)).reshape({1, 1, height, width});
    cv::Mat resized_color_1, resized_color_0;
    cv::resize(frame_1.color_img, resized_color_1, cv::Size2l(width, height));
    cv::resize(frame_0.color_img, resized_color_0, cv::Size2l(width, height));

    at::Tensor scaled_dpt_map_0;
    {
      std::shared_lock<std::shared_mutex> lock(frame_0.mutex);
      scaled_dpt_map_0 = frame_0.dpt_map / frame_0.dpt_scale * scale_0;
    }

    const at::Tensor input_color_1 = torch::from_blob(static_cast<unsigned char *>(resized_color_1.data), {1, height, width, 3},
                                                      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                                         .to(torch::kFloat32)
                                         .to(scaled_dpt_map_0.device())
                                         .permute({0, 3, 1, 2})
                                         .clone();
    Sophus::SE3f relpose_10;
    Eigen::Matrix<float, 3, 3> rotation10_mat;
    TensorToEigenMatrix(rotation_10, rotation10_mat);
    relpose_10.so3() = Sophus::SO3f(rotation10_mat);
    TensorToEigenMatrix(translation_10, relpose_10.translation());

    SE3ImageWarping(relpose_10, cam, input_color_1, scaled_dpt_map_0, video_mask, warped_color_1, warped_valid_mask_1, dpt_eps);

    cv::Mat warped_img_1 = Tensor2Mat(warped_color_1);
    cv::cvtColor(resized_color_0.clone(), resized_color_0, cv::COLOR_BGR2RGB);
    resized_color_0.convertTo(resized_color_0, CV_32FC3);
    cv::Mat rev_checkerboard;
    cv::subtract(cv::Scalar(1, 1, 1), checkerboard, rev_checkerboard);
    cv::Mat ori_warp_overlay = resized_color_0.mul(checkerboard) + warped_img_1.mul(rev_checkerboard);
    ori_warp_overlay.convertTo(ori_warp_overlay, CV_8UC3);
    warped_img_1.convertTo(warped_img_1, CV_8UC3);
    resized_color_0.convertTo(resized_color_0, CV_8UC3);
    cv::cvtColor(resized_color_0.clone(), resized_color_0, cv::COLOR_RGB2BGR);

    const at::Tensor dpt_map = scaled_dpt_map_0 * video_mask;
    const at::Tensor max = torch::max(dpt_map);
    float max_dpt_0 = max.item<float>();
    cv::Mat dpt_display_0 = Tensor2Mat(torch::clamp_min(dpt_map / max_dpt_0, 0.0));
    dpt_display_0 = apply_colormap(dpt_display_0);

    array.push_back(resized_color_0);
    array.push_back(warped_img_1);
    array.push_back(ori_warp_overlay);
    array.push_back(dpt_display_0);

    cv::Mat mosaic;
    const double ratio = 128.0 / array[0].rows;
    cv::resize(CreateMosaic(array, 1, 4), mosaic, cv::Size(0, 0), ratio, ratio);
    return mosaic;
  }

  /* ************************************************************************* */
  template <typename Scalar>
  cv::Mat DisplayPairs(const typename df::Map<Scalar>::Ptr &map,
                       const std::vector<std::pair<long, long>> &pairs,
                       const at::Tensor video_mask,
                       const df::PinholeCamera<Scalar> &cam, const Scalar &dpt_eps, int N,
                       const cv::Mat &checkerboard)
  {
    torch::NoGradGuard no_grad;
    if (pairs.empty())
    {
      return cv::Mat{};
    }

    std::vector<cv::Mat> array;
    const long height = cam.height();
    const long width = cam.width();
    at::Tensor warped_color_1, warped_valid_mask_1;
    at::Tensor input_color_0, input_color_1;

    for (auto it = std::make_reverse_iterator(pairs.end());
         it != std::make_reverse_iterator(pairs.begin());
         ++it)
    {
      auto pair = *it;
      auto kf0 = map->keyframes.Get(pair.first);
      auto kf1 = map->keyframes.Get(pair.second);
      Sophus::SE3<Scalar> relpose;
      df::RelativePose(kf1->pose_wk, kf0->pose_wk, relpose);

      cv::Mat resized_color_1, resized_color_0;
      cv::resize(kf1->color_img, resized_color_1, cv::Size2l(width, height));
      cv::resize(kf0->color_img, resized_color_0, cv::Size2l(width, height));

      input_color_1 = torch::from_blob(static_cast<unsigned char *>(resized_color_1.data), {1, height, width, 3},
                                       torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                          .to(torch::kFloat32)
                          .to(kf1->dpt_map.device())
                          .permute({0, 3, 1, 2})
                          .clone();
      at::Tensor dpt_map_0;
      {
        std::shared_lock<std::shared_mutex> lock(kf0->mutex);
        dpt_map_0 = kf0->dpt_map;
      }

      SE3ImageWarping(relpose, cam, input_color_1, dpt_map_0, video_mask.reshape({1, 1, height, width}), warped_color_1, warped_valid_mask_1, dpt_eps);
      const at::Tensor dpt_map = dpt_map_0 * video_mask;
      const at::Tensor max = torch::max(dpt_map);
      float max_dpt_0 = max.item<float>();
      cv::Mat dpt_display_0 = Tensor2Mat(torch::clamp_min(dpt_map / max_dpt_0, 0.0));
      dpt_display_0 = apply_colormap(dpt_display_0);

      cv::Mat warped_img_1 = Tensor2Mat(warped_color_1);
      cv::cvtColor(resized_color_0.clone(), resized_color_0, cv::COLOR_BGR2RGB);
      resized_color_0.convertTo(resized_color_0, CV_32FC3);
      cv::Mat rev_checkerboard;
      cv::subtract(cv::Scalar(1, 1, 1), checkerboard, rev_checkerboard);
      cv::Mat ori_warp_overlay = resized_color_0.mul(checkerboard) + warped_img_1.mul(rev_checkerboard);
      ori_warp_overlay.convertTo(ori_warp_overlay, CV_8UC3);
      warped_img_1.convertTo(warped_img_1, CV_8UC3);
      resized_color_0.convertTo(resized_color_0, CV_8UC3);
      cv::cvtColor(resized_color_0.clone(), resized_color_0, cv::COLOR_RGB2BGR);

      array.push_back(resized_color_0);
      array.push_back(warped_img_1);
      array.push_back(ori_warp_overlay);
      array.push_back(dpt_display_0);
      --N;

      if (N <= 0)
      {
        break;
      }
    }
    cv::Mat mosaic;
    const double ratio = 128.0 / array[0].rows;
    cv::resize(CreateMosaic(array, array.size() / 4, 4), mosaic, cv::Size(0, 0), ratio, ratio);
    return mosaic;
  }

  void GenerateMaskPyramid(const at::Tensor valid_mask,
                           const int &num_levels,
                           const std::shared_ptr<std::vector<at::Tensor>> &valid_mask_pyramid_ptr)
  {
    torch::NoGradGuard no_grad;
    namespace F = torch::nn::functional;

    long height = valid_mask.size(2);
    long width = valid_mask.size(3);

    at::Tensor curr_valid_mask = valid_mask;
    valid_mask_pyramid_ptr->push_back(curr_valid_mask);

    for (int i = 0; i < num_levels - 1; ++i)
    {
      height /= 2;
      width /= 2;
      curr_valid_mask = F::interpolate(curr_valid_mask, F::InterpolateFuncOptions().mode(torch::kNearest).size(std::vector<int64_t>{height, width}));
      valid_mask_pyramid_ptr->push_back(curr_valid_mask);
    }
    return;
  }

  void GenerateCheckerboard(cv::Mat &checkerboard, const std::vector<long> &image_size)
  {
    checkerboard = cv::Mat::zeros(image_size[0], image_size[1], CV_8UC3);
    const int grid_num = 7;
    const float grid_size_x = (float)image_size[1] / grid_num;
    const float grid_size_y = (float)image_size[0] / grid_num;

    // Draw a rectangle ( 5th argument is not -ve)
    for (int y = 0; y <= (grid_num - 1) / 2; y++)
    {
      for (int x = 0; x <= (grid_num - 1) / 2; x++)
      {
        cv::rectangle(checkerboard, cv::Point(grid_size_x * 2 * x, grid_size_y * 2 * y),
                      cv::Point(grid_size_x * (2 * x + 1), grid_size_y * (2 * y + 1)), cv::Scalar(1, 1, 1), -1, 4);
      }
    }

    for (int y = 0; y <= (grid_num - 1) / 2; y++)
    {
      for (int x = 0; x <= (grid_num - 1) / 2; x++)
      {
        cv::rectangle(checkerboard, cv::Point(grid_size_x * (2 * x + 1), grid_size_y * (2 * y + 1)),
                      cv::Point(grid_size_x * 2 * (x + 1), grid_size_y * 2 * (y + 1)), cv::Scalar(1, 1, 1), -1, 4);
      }
    }

    checkerboard.convertTo(checkerboard, CV_32FC3);
    return;
  }

  // explicit instantiation
  template cv::Mat DisplayKeyframes<float>(std::shared_ptr<Map<float>>, int);
  template cv::Mat DisplaySE3Warp<float>(const df::Keyframe<float> &, const df::Frame<float> &,
                                         const at::Tensor , const at::Tensor , const float &, const cv::Mat &);
  template cv::Mat DisplaySE3Warp<float>(const df::Frame<float> &keyframe_0, const df::Frame<float> &frame_1,
                                         const at::Tensor rotation_10, const at::Tensor translation_10, const float scale_0, const float &dpt_eps, const cv::Mat &);
  template cv::Mat DisplayPairs<float>(const typename Map<float>::Ptr &, const std::vector<std::pair<long, long>> &, const at::Tensor ,
                                       const PinholeCamera<float> &, const float &, int, const cv::Mat &);
}