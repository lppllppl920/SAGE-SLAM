
#ifndef GEOMETRIC_FACTOR_KERNELS_H_
#define GEOMETRIC_FACTOR_KERNELS_H_

#include <torch/torch.h>
#include "camera_pyramid.h"

namespace df
{
  template <int CS>
  float geometric_error_calculate_unbiased(const at::Tensor rotation, const at::Tensor translation,
                                           const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
                                           const at::Tensor code_0, const at::Tensor unscaled_dpt_map_1, const at::Tensor valid_mask_1,
                                           const at::Tensor sampled_locations_1d_0, const at::Tensor sampled_locations_homo_0,
                                           const float scale_0, const float scale_1, const PinholeCamera<float> &camera,
                                           const float eps, const float loss_param, const float weight);

  template <int CS>
  float geometric_error_calculate(const at::Tensor rotation, const at::Tensor translation,
                                  const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
                                  const at::Tensor code_0, const at::Tensor dpt_map_1, const at::Tensor valid_mask_1,
                                  const at::Tensor sampled_locations_1d_0, const at::Tensor sampled_locations_homo_0,
                                  const float scale_0, const PinholeCamera<float> &camera,
                                  const float eps, const float loss_param, const float weight);

  template <int CS>
  void geometric_jac_error_calculate_unbiased(at::Tensor &AtA, at::Tensor &Atb, float &error,
                                              const at::Tensor rotation10, const at::Tensor translation10,
                                              const at::Tensor rotation0, const at::Tensor translation0,
                                              const at::Tensor rotation1, const at::Tensor translation1,
                                              const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
                                              const at::Tensor code_0, const at::Tensor unscaled_dpt_map_1,
                                              const at::Tensor unscaled_dpt_map_grad_1, const at::Tensor dpt_jac_code_1, const at::Tensor valid_mask_1,
                                              const at::Tensor sampled_locations_1d_0, const at::Tensor sampled_locations_homo_0,
                                              const float scale_0, const float scale_1, const PinholeCamera<float> &camera,
                                              const float eps, const float loss_param, const float weight);

  template <int CS>
  void geometric_jac_error_calculate(at::Tensor &AtA, at::Tensor &Atb, float &error,
                                     const at::Tensor rotation10, const at::Tensor translation10,
                                     const at::Tensor rotation0, const at::Tensor translation0,
                                     const at::Tensor rotation1, const at::Tensor translation1,
                                     const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
                                     const at::Tensor code_0, const at::Tensor dpt_map_1,
                                     const at::Tensor dpt_map_grad_1, const at::Tensor dpt_jac_code_1, const at::Tensor valid_mask_1,
                                     const at::Tensor sampled_locations_1d_0, const at::Tensor sampled_locations_homo_0,
                                     const float scale_0, const float scale_1, const PinholeCamera<float> &camera,
                                     const float eps, const float loss_param, const float weight);
}

#endif // GEOMETRIC_FACTOR_KERNELS_H_