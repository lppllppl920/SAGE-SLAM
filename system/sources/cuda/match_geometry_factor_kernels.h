#ifndef MATCH_GEOMETRY_FACTOR_KERNELS_H_
#define MATCH_GEOMETRY_FACTOR_KERNELS_H_

#include <torch/torch.h>
#include "camera_pyramid.h"

namespace df
{
  float tracker_match_geom_error_calculate(const at::Tensor rotation, const at::Tensor translation,
                                           const at::Tensor sampled_dpts_0, const at::Tensor matched_dpts_1,
                                           const at::Tensor sampled_locations_homo_0, const at::Tensor matched_locations_homo_1,
                                           const float loss_param, const float weight);

  void tracker_match_geom_jac_error_calculate(at::Tensor &AtA, at::Tensor &Atb, float &error,
                                              const at::Tensor rotation, const at::Tensor translation,
                                              const at::Tensor sampled_dpts_0, const at::Tensor matched_dpts_1,
                                              const at::Tensor sampled_locations_homo_0, const at::Tensor matched_locations_homo_1,
                                              const float loss_param, const float weight);

  void tracker_match_geom_jac_error_calculate_with_scale(at::Tensor &AtA, at::Tensor &Atb, float &error,
                                                         const at::Tensor rotation, const at::Tensor translation,
                                                         const at::Tensor sampled_dpts_0, const at::Tensor matched_dpts_1,
                                                         const at::Tensor sampled_locations_homo_0, const at::Tensor matched_locations_homo_1,
                                                         const float scale_0, const float loss_param, const float weight);
  template <int CS>
  float match_geometry_error_calculate(const at::Tensor rotation, const at::Tensor translation,
                                       const at::Tensor flatten_dpt_map_bias_0,
                                       const at::Tensor flatten_dpt_map_bias_1,
                                       const at::Tensor flatten_dpt_jac_code_0,
                                       const at::Tensor flatten_dpt_jac_code_1,
                                       const at::Tensor code_0, const at::Tensor code_1,
                                       const at::Tensor sampled_locations_homo_0,
                                       const at::Tensor matched_locations_homo_1,
                                       const at::Tensor sampled_locations_1d_0,
                                       const at::Tensor matched_locations_1d_1,
                                       const float scale_0, const float scale_1,
                                       const float loss_param, const float weight,
                                       const std::string robust_loss_type);

  template <int CS>
  void match_geometry_jac_error_calculate(at::Tensor &AtA, at::Tensor &Atb, float &error,
                                          const at::Tensor rotation10, const at::Tensor translation10,
                                          const at::Tensor rotation0, const at::Tensor translation0,
                                          const at::Tensor rotation1, const at::Tensor translation1,
                                          const at::Tensor flatten_dpt_map_bias_0,
                                          const at::Tensor flatten_dpt_map_bias_1,
                                          const at::Tensor flatten_dpt_jac_code_0,
                                          const at::Tensor flatten_dpt_jac_code_1,
                                          const at::Tensor code_0, const at::Tensor code_1,
                                          const at::Tensor sampled_locations_homo_0,
                                          const at::Tensor matched_locations_homo_1,
                                          const at::Tensor sampled_locations_1d_0,
                                          const at::Tensor matched_locations_1d_1,
                                          const float scale_0, const float scale_1,
                                          const float loss_param, const float weight,
                                          const std::string robust_loss_type);

  float loop_mg_error_calculate(const at::Tensor rotation, const at::Tensor translation,
                                const at::Tensor sampled_unscaled_dpts_0,
                                const at::Tensor matched_unscaled_dpts_1,
                                const at::Tensor sampled_locations_homo_0,
                                const at::Tensor matched_locations_homo_1,
                                const float scale_0, const float scale_1,
                                const float loss_param, const float weight);

  void loop_mg_jac_error_calculate(at::Tensor &AtA, at::Tensor &Atb, float &error,
                                   const at::Tensor rotation10, const at::Tensor translation10,
                                   const at::Tensor rotation0, const at::Tensor translation0,
                                   const at::Tensor rotation1, const at::Tensor translation1,
                                   const at::Tensor sampled_unscaled_dpts_0,
                                   const at::Tensor matched_unscaled_dpts_1,
                                   const at::Tensor sampled_locations_homo_0,
                                   const at::Tensor matched_locations_homo_1,
                                   const float scale_0, const float scale_1,
                                   const float loss_param, const float weight);
}

#endif // MATCH_GEOMETRY_FACTOR_KERNELS_H_