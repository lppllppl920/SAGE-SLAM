#ifndef DF_KEYFRAME_RENDERER_H_
#define DF_KEYFRAME_RENDERER_H_

#include <sophus/se3.hpp>
#include <pangolin/pangolin.h>
#include <VisionCore/Buffers/Image2D.hpp>
#include <VisionCore/Buffers/Buffer1D.hpp>
#include <glog/logging.h>
#include <torch/torch.h>

#include "pinhole_camera.h"

namespace df
{

class KeyframeRenderer
{
public:
  struct DisplayData
  {

    DisplayData(int width, int height) {}
    cv::Mat color_img;
    at::Tensor dpt;
    cv::Mat vld;
    Sophus::SE3f pose_wk;
  };

  ~KeyframeRenderer() 
  {
    VLOG(3) << "[KeyframeRenderer::~KeyframeRenderer] deconstructor called";
  }

  void Init(const df::PinholeCamera<float>& cam);
  void RenderKeyframe(const pangolin::OpenGlMatrix& vp, const DisplayData& data);

  void SetPhong(bool enabled);
  void SetLightPos(float x, float y, float z);

private:
  std::size_t width_;
  std::size_t height_;
  df::PinholeCamera<float> cam_;

  // options
  bool phong_enabled_;
  float3 light_pos_;

  pangolin::GlSlProgram shader_;
  pangolin::GlTexture col_tex_;
  pangolin::GlTexture dpt_tex_;
  pangolin::GlTexture val_tex_;
};

} // namespace df

#endif // DF_KEYFRAME_RENDERER_H_
