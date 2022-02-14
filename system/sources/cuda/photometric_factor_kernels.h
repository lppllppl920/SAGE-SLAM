#ifndef PHOTOMETRIC_FACTOR_KERNELS_H_
#define PHOTOMETRIC_FACTOR_KERNELS_H_

#include <torch/torch.h>
#include "camera_pyramid.h"

namespace df
{
  template <int FS>
  float photometric_error_calculate(const at::Tensor rotation, const at::Tensor translation,
                                    const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
                                    const at::Tensor code_0, const at::Tensor valid_mask_1,
                                    const at::Tensor sampled_locations_1d_0,
                                    const at::Tensor sampled_locations_homo_0,
                                    const at::Tensor feat_map_pyramid_0, const at::Tensor feat_map_pyramid_1,
                                    const at::Tensor level_offsets,
                                    const float scale_0, const CameraPyramid<float> &camera_pyramid,
                                    const float eps, const at::Tensor weights_tensor);

  template <int CS, int FS>
  void photometric_jac_error_calculate(at::Tensor &AtA, at::Tensor &Atb, float &error,
                                       const at::Tensor rotation10, const at::Tensor translation10,
                                       const at::Tensor rotation0, const at::Tensor translation0,
                                       const at::Tensor rotation1, const at::Tensor translation1,
                                       const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
                                       const at::Tensor code_0, const at::Tensor valid_mask_1,
                                       const at::Tensor sampled_locations_1d_0,
                                       const at::Tensor sampled_locations_homo_0,
                                       const at::Tensor feat_map_pyramid_0, const at::Tensor feat_map_pyramid_1,
                                       const at::Tensor feat_map_grad_pyramid_1,
                                       const at::Tensor level_offsets,
                                       const float scale_0, const CameraPyramid<float> &camera_pyramid,
                                       const float eps, const at::Tensor weights_tensor);

  template <int FS>
  void tracker_photo_jac_error_calculate(at::Tensor &AtA, at::Tensor &Atb, float &error,
                                         const at::Tensor rotation, const at::Tensor translation,
                                         const at::Tensor valid_mask_1,
                                         const at::Tensor sampled_dpts_0,
                                         const at::Tensor sampled_locations_homo_0,
                                         const at::Tensor sampled_features_0,
                                         const at::Tensor feat_map_pyramid_1,
                                         const at::Tensor feat_map_grad_pyramid_1,
                                         const at::Tensor level_offsets,
                                         const CameraPyramid<float> &camera_pyramid,
                                         const float eps, const at::Tensor weights_tensor);

  template <int FS>
  void tracker_photo_jac_error_calculate_with_scale(at::Tensor &AtA, at::Tensor &Atb, float &error,
                                                    const at::Tensor rotation, const at::Tensor translation,
                                                    const at::Tensor valid_mask_1,
                                                    const at::Tensor sampled_dpts_0,
                                                    const at::Tensor sampled_locations_homo_0,
                                                    const at::Tensor sampled_features_0,
                                                    const at::Tensor feat_map_pyramid_1,
                                                    const at::Tensor feat_map_grad_pyramid_1,
                                                    const at::Tensor level_offsets,
                                                    const CameraPyramid<float> &camera_pyramid,
                                                    const float scale_0, const float eps, const at::Tensor weights_tensor);

  template <int FS>
  float tracker_photo_error_calculate(const at::Tensor rotation, const at::Tensor translation,
                                      const at::Tensor valid_mask_1,
                                      const at::Tensor sampled_dpts_0,
                                      const at::Tensor sampled_locations_homo_0,
                                      const at::Tensor sampled_features_0,
                                      const at::Tensor feat_map_pyramid_1,
                                      const at::Tensor level_offsets,
                                      const CameraPyramid<float> &camera_pyramid,
                                      const float eps, const at::Tensor weights_tensor);
}

#endif // PHOTOMETRIC_FACTOR_KERNELS_H_