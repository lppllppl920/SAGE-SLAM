#ifndef REPROJECTION_FACTOR_KERNELS_H_
#define REPROJECTION_FACTOR_KERNELS_H_

#include <torch/torch.h>
#include "camera_pyramid.h"

namespace df
{

  void tracker_reproj_jac_error_calculate(at::Tensor &AtA, at::Tensor &Atb, float &error,
                                          const at::Tensor rotation, const at::Tensor translation,
                                          const at::Tensor sampled_dpts_0,
                                          const at::Tensor sampled_locations_homo_0,
                                          const at::Tensor matched_locations_2d_1,
                                          const PinholeCamera<float> &camera,
                                          const float eps, const float loss_param, const float weight);

  float tracker_reproj_error_calculate(const at::Tensor rotation, const at::Tensor translation,
                                       const at::Tensor sampled_dpts_0,
                                       const at::Tensor sampled_locations_homo_0,
                                       const at::Tensor matched_locations_2d_1,
                                       const PinholeCamera<float> &camera,
                                       const float eps, const float loss_param, const float weight);

  template <int CS>
  void reprojection_jac_error_calculate(at::Tensor &AtA, at::Tensor &Atb, float &error,
                                        const at::Tensor rotation10, const at::Tensor translation10,
                                        const at::Tensor rotation0, const at::Tensor translation0,
                                        const at::Tensor rotation1, const at::Tensor translation1,
                                        const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
                                        const at::Tensor code_0, const at::Tensor sampled_locations_1d_0,
                                        const at::Tensor sampled_locations_homo_0,
                                        const at::Tensor matched_locations_2d_1,
                                        const float scale_0, const PinholeCamera<float> &camera,
                                        const float eps, const float loss_param, const float weight);

  template <int CS>
  float reprojection_error_calculate(const at::Tensor rotation10, const at::Tensor translation10,
                                     const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
                                     const at::Tensor code_0, const at::Tensor sampled_locations_1d_0,
                                     const at::Tensor sampled_locations_homo_0,
                                     const at::Tensor matched_locations_2d_1,
                                     const float scale_0, const PinholeCamera<float> &camera,
                                     const float eps, const float loss_param, const float weight);
}

#endif // REPROJECTION_FACTOR_KERNELS_H_