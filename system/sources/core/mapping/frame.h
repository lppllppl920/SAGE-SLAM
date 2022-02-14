#ifndef DF_FRAME_H_
#define DF_FRAME_H_

#include <memory>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <atomic>

#include "camera_pyramid.h"

namespace df
{

  template <typename Scalar>
  class Frame
  {
  public:
    typedef Frame<Scalar> This;
    typedef std::shared_ptr<This> Ptr;
    typedef Sophus::SE3<Scalar> SE3T;
    typedef long IdType;

    Frame() : id(0),
              dpt_scale(1.0),
              avg_squared_dpt_bias(0),
              // scale_ratio_cur_ref(0),
              reinitialize_count(0)
    {
    }

    Frame(const Frame &other) : id(other.id),
                                pose_wk(other.pose_wk),
                                timestamp(other.timestamp),
                                color_img(other.color_img.clone()),
                                video_mask_ptr(other.video_mask_ptr),
                                feat_video_mask(other.feat_video_mask),
                                camera_pyramid_ptr(other.camera_pyramid_ptr),
                                feat_desc(other.feat_desc),
                                feat_map_pyramid(other.feat_map_pyramid),
                                feat_map_grad_pyramid(other.feat_map_grad_pyramid),
                                level_offsets_ptr(other.level_offsets_ptr),
                                dpt_map_bias(other.dpt_map_bias), dpt_jac_code(other.dpt_jac_code),
                                code(other.code), dpt_scale(other.dpt_scale), dpt_map(other.dpt_map),
                                avg_squared_dpt_bias(other.avg_squared_dpt_bias),
                                // scale_ratio_cur_ref(other.scale_ratio_cur_ref),
                                sampled_locations_homo(other.sampled_locations_homo),
                                sampled_locations_1d(other.sampled_locations_1d),
                                valid_locations_1d(other.valid_locations_1d),
                                valid_locations_homo(other.valid_locations_homo)
    {
      reinitialize_count = other.reinitialize_count.load(std::memory_order_relaxed);
    }

    virtual ~Frame() {}

    virtual Ptr Clone()
    {
      return std::make_shared<This>(*this);
    }

    virtual std::string Name() { return "fr" + std::to_string(id); }

    virtual bool IsKeyframe() { return false; }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    IdType id;

    SE3T pose_wk;

    // time of rgb image acquisition
    double timestamp;

    // color image for display
    cv::Mat color_img;

    std::shared_ptr<at::Tensor> video_mask_ptr;

    at::Tensor feat_video_mask;

    // pyramid of cameras
    std::shared_ptr<CameraPyramid<Scalar>> camera_pyramid_ptr;

    at::Tensor feat_desc;

    // C_feat x (N0 + N1 + ...)
    at::Tensor feat_map_pyramid;
    // 2 x C_feat x (N0 + N1 + ...)
    at::Tensor feat_map_grad_pyramid;

    std::shared_ptr<at::Tensor> level_offsets_ptr;

    std::shared_mutex weight_mutex;

    // avoid race condition in multi-thread environment
    mutable std::shared_mutex mutex;

    // depth map bias
    at::Tensor dpt_map_bias;
    // jacobian of unscaled depth map wrt depth code
    at::Tensor dpt_jac_code;
    // depth code
    // CS x 1
    at::Tensor code;
    // depth map scale
    Scalar dpt_scale;
    // scaled depth map
    at::Tensor dpt_map;
    Scalar avg_squared_dpt_bias;

    // Scalar scale_ratio_cur_ref;

    // sampled homogeneous 2d locations N x 3
    at::Tensor sampled_locations_homo;
    // sampled 1d locations N
    at::Tensor sampled_locations_1d;

    // all 1d locations within valid mask
    at::Tensor valid_locations_1d;
    at::Tensor valid_locations_homo;

    std::atomic<int> reinitialize_count;
  };

} // namespace df

#endif // DF_FRAME_H_
