
#include "match_geometry_factor_kernels.h"

namespace df
{
#define MAX_THREADS_PER_BLOCK 512

#define WITHIN_BOUNDS(x, y, W, H) (x >= 0 && x < W && y >= 0 && y < H)

#define gpuErrchk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
  inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
  {
    if (code != cudaSuccess)
    {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort)
        exit(code);
    }
  }

  //   template <typename Scalar>
  //   __launch_bounds__(MAX_THREADS_PER_BLOCK)
  //       __global__ void tracker_match_geom_error_calculate_kernel_unbiased(
  //           torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
  //           const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation,
  //           const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation,
  //           const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_unscaled_dpts_0,
  //           const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> matched_unscaled_dpts_1,
  //           const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
  //           const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> matched_locations_homo_1,
  //           const Scalar scale_0, const Scalar scale_1, const Scalar loss_param)
  //   {
  //     const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  //     if (idx < sampled_unscaled_dpts_0.size(0))
  //     {
  //       const Scalar sum_scale = scale_0 + scale_1;
  //       const Scalar dpt_0 = sampled_unscaled_dpts_0[idx] * scale_0 / sum_scale;
  //       const Scalar dpt_1 = matched_unscaled_dpts_1[idx] * scale_1 / sum_scale;

  //       Scalar rotated_location_homo_0_in_1[3];
  // #pragma unroll_completely
  //       for (int i = 0; i < 3; ++i)
  //       {
  //         rotated_location_homo_0_in_1[i] = rotation[i][0] * sampled_locations_homo_0[idx][0] +
  //                                           rotation[i][1] * sampled_locations_homo_0[idx][1] +
  //                                           rotation[i][2] * sampled_locations_homo_0[idx][2];
  //       }

  //       Scalar location_3d_0_in_1[3];
  // #pragma unroll_completely
  //       for (int i = 0; i < 3; ++i)
  //       {
  //         location_3d_0_in_1[i] = dpt_0 * rotated_location_homo_0_in_1[i] + translation[i];
  //       }

  //       const Scalar matched_location_3d_1[3] = {dpt_1 * matched_locations_homo_1[idx][0],
  //                                                dpt_1 * matched_locations_homo_1[idx][1],
  //                                                dpt_1 * matched_locations_homo_1[idx][2]};

  //       // fair loss
  //       const Scalar sqrt_loss_param = sqrt(loss_param);
  //       const Scalar location_3d_diff[3] = {matched_location_3d_1[0] - location_3d_0_in_1[0],
  //                                           matched_location_3d_1[1] - location_3d_0_in_1[1],
  //                                           matched_location_3d_1[2] - location_3d_0_in_1[2]};
  //       const Scalar location_3d_diff_normalized[3] = {fabs(location_3d_diff[0]) / sqrt_loss_param,
  //                                                      fabs(location_3d_diff[1]) / sqrt_loss_param,
  //                                                      fabs(location_3d_diff[2]) / sqrt_loss_param};

  //       sampled_error[idx] =
  //           2 * (location_3d_diff_normalized[0] + location_3d_diff_normalized[1] + location_3d_diff_normalized[2] -
  //                log(1 + location_3d_diff_normalized[0]) - log(1 + location_3d_diff_normalized[1]) - log(1 + location_3d_diff_normalized[2]));
  //     }

  //     return;
  //   }

  template <typename Scalar>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void tracker_match_geom_error_calculate_kernel(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_dpts_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> matched_dpts_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> matched_locations_homo_1,
          const Scalar loss_param)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sampled_dpts_0.size(0))
    {
      Scalar rotated_location_homo_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        rotated_location_homo_0_in_1[i] = rotation[i][0] * sampled_locations_homo_0[idx][0] +
                                          rotation[i][1] * sampled_locations_homo_0[idx][1] +
                                          rotation[i][2] * sampled_locations_homo_0[idx][2];
      }

      Scalar location_3d_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        location_3d_0_in_1[i] = sampled_dpts_0[idx] * rotated_location_homo_0_in_1[i] + translation[i];
      }

      const Scalar matched_location_3d_1[3] = {matched_dpts_1[idx] * matched_locations_homo_1[idx][0],
                                               matched_dpts_1[idx] * matched_locations_homo_1[idx][1],
                                               matched_dpts_1[idx] * matched_locations_homo_1[idx][2]};

      // fair loss
      const Scalar sqrt_loss_param = sqrt(loss_param);
      const Scalar location_3d_diff[3] = {matched_location_3d_1[0] - location_3d_0_in_1[0],
                                          matched_location_3d_1[1] - location_3d_0_in_1[1],
                                          matched_location_3d_1[2] - location_3d_0_in_1[2]};
      const Scalar location_3d_diff_normalized[3] = {fabs(location_3d_diff[0]) / sqrt_loss_param,
                                                     fabs(location_3d_diff[1]) / sqrt_loss_param,
                                                     fabs(location_3d_diff[2]) / sqrt_loss_param};

      sampled_error[idx] =
          2 * (location_3d_diff_normalized[0] + location_3d_diff_normalized[1] + location_3d_diff_normalized[2] -
               log(1 + location_3d_diff_normalized[0]) - log(1 + location_3d_diff_normalized[1]) - log(1 + location_3d_diff_normalized[2]));
    }

    return;
  }

  template <typename Scalar>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void tracker_match_geom_jac_error_calculate_kernel(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> fair_sampled_3d_loc_diff_jac_rel_pose,
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> fair_sampled_3d_loc_diff,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_dpts_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> matched_dpts_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> matched_locations_homo_1,
          const Scalar loss_param)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sampled_dpts_0.size(0))
    {
      Scalar rotated_location_homo_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        rotated_location_homo_0_in_1[i] = rotation[i][0] * sampled_locations_homo_0[idx][0] +
                                          rotation[i][1] * sampled_locations_homo_0[idx][1] +
                                          rotation[i][2] * sampled_locations_homo_0[idx][2];
      }

      Scalar location_3d_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        location_3d_0_in_1[i] = sampled_dpts_0[idx] * rotated_location_homo_0_in_1[i] + translation[i];
      }

      const Scalar matched_location_3d_1[3] = {matched_dpts_1[idx] * matched_locations_homo_1[idx][0],
                                               matched_dpts_1[idx] * matched_locations_homo_1[idx][1],
                                               matched_dpts_1[idx] * matched_locations_homo_1[idx][2]};

      // fair loss
      const Scalar sqrt_loss_param = sqrt(loss_param);
      const Scalar location_3d_diff[3] = {matched_location_3d_1[0] - location_3d_0_in_1[0],
                                          matched_location_3d_1[1] - location_3d_0_in_1[1],
                                          matched_location_3d_1[2] - location_3d_0_in_1[2]};
      const Scalar location_3d_diff_normalized[3] = {fabs(location_3d_diff[0]) / sqrt_loss_param,
                                                     fabs(location_3d_diff[1]) / sqrt_loss_param,
                                                     fabs(location_3d_diff[2]) / sqrt_loss_param};

      sampled_error[idx] =
          2 * (location_3d_diff_normalized[0] + location_3d_diff_normalized[1] + location_3d_diff_normalized[2] -
               log(1 + location_3d_diff_normalized[0]) - log(1 + location_3d_diff_normalized[1]) - log(1 + location_3d_diff_normalized[2]));

      const Scalar sqrt_fair_weight[3] = {
          sqrt(1 / (loss_param * (1 + location_3d_diff_normalized[0]))),
          sqrt(1 / (loss_param * (1 + location_3d_diff_normalized[1]))),
          sqrt(1 / (loss_param * (1 + location_3d_diff_normalized[2])))};

#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        fair_sampled_3d_loc_diff[idx][i] = sqrt_fair_weight[i] * location_3d_diff[i];
      }

      const Scalar loc_3d_diff_jac_rel_pose[3][6] = {{1, 0, 0, 0, location_3d_0_in_1[2], -location_3d_0_in_1[1]},
                                                     {0, 1, 0, -location_3d_0_in_1[2], 0, location_3d_0_in_1[0]},
                                                     {0, 0, 1, location_3d_0_in_1[1], -location_3d_0_in_1[0], 0}};

#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          fair_sampled_3d_loc_diff_jac_rel_pose[idx][i][j] = sqrt_fair_weight[i] * loc_3d_diff_jac_rel_pose[i][j];
        }
      }
    }

    return;
  }

  template <typename Scalar>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void tracker_match_geom_jac_error_calculate_with_scale_kernel(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> fair_sampled_3d_loc_diff_jac_rel_pose_scale_0,
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> fair_sampled_3d_loc_diff,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_dpts_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> matched_dpts_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> matched_locations_homo_1,
          const Scalar scale_0, const Scalar loss_param)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sampled_dpts_0.size(0))
    {
      Scalar rotated_location_homo_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        rotated_location_homo_0_in_1[i] = rotation[i][0] * sampled_locations_homo_0[idx][0] +
                                          rotation[i][1] * sampled_locations_homo_0[idx][1] +
                                          rotation[i][2] * sampled_locations_homo_0[idx][2];
      }

      Scalar location_3d_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        location_3d_0_in_1[i] = sampled_dpts_0[idx] * rotated_location_homo_0_in_1[i] + translation[i];
      }

      const Scalar matched_location_3d_1[3] = {matched_dpts_1[idx] * matched_locations_homo_1[idx][0],
                                               matched_dpts_1[idx] * matched_locations_homo_1[idx][1],
                                               matched_dpts_1[idx] * matched_locations_homo_1[idx][2]};

      // fair loss
      const Scalar sqrt_loss_param = sqrt(loss_param);
      const Scalar location_3d_diff[3] = {matched_location_3d_1[0] - location_3d_0_in_1[0],
                                          matched_location_3d_1[1] - location_3d_0_in_1[1],
                                          matched_location_3d_1[2] - location_3d_0_in_1[2]};
      const Scalar location_3d_diff_normalized[3] = {fabs(location_3d_diff[0]) / sqrt_loss_param,
                                                     fabs(location_3d_diff[1]) / sqrt_loss_param,
                                                     fabs(location_3d_diff[2]) / sqrt_loss_param};

      sampled_error[idx] =
          2 * (location_3d_diff_normalized[0] + location_3d_diff_normalized[1] + location_3d_diff_normalized[2] -
               log(1 + location_3d_diff_normalized[0]) - log(1 + location_3d_diff_normalized[1]) - log(1 + location_3d_diff_normalized[2]));

      const Scalar sqrt_fair_weight[3] = {
          sqrt(1 / (loss_param * (1 + location_3d_diff_normalized[0]))),
          sqrt(1 / (loss_param * (1 + location_3d_diff_normalized[1]))),
          sqrt(1 / (loss_param * (1 + location_3d_diff_normalized[2])))};

#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        fair_sampled_3d_loc_diff[idx][i] = sqrt_fair_weight[i] * location_3d_diff[i];
      }

      const Scalar loc_3d_diff_jac_rel_pose[3][6] = {{1, 0, 0, 0, location_3d_0_in_1[2], -location_3d_0_in_1[1]},
                                                     {0, 1, 0, -location_3d_0_in_1[2], 0, location_3d_0_in_1[0]},
                                                     {0, 0, 1, location_3d_0_in_1[1], -location_3d_0_in_1[0], 0}};

#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          fair_sampled_3d_loc_diff_jac_rel_pose_scale_0[idx][i][j] = sqrt_fair_weight[i] * loc_3d_diff_jac_rel_pose[i][j];
        }

        fair_sampled_3d_loc_diff_jac_rel_pose_scale_0[idx][i][6] = sqrt_fair_weight[i] * (rotated_location_homo_0_in_1[i] * sampled_dpts_0[idx] / scale_0);
      }
    }

    return;
  }

  template <typename Scalar>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void loop_mg_jac_error_calculate_kernel(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> fair_sampled_3d_loc_diff_jac_pose_scale,
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> fair_sampled_3d_loc_diff,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation10,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation10,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_unscaled_dpts_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> matched_unscaled_dpts_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> matched_locations_homo_1,
          const Scalar scale_0, const Scalar scale_1, const Scalar loss_param)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sampled_locations_homo_0.size(0))
    {
      Scalar dpt_0 = sampled_unscaled_dpts_0[idx] * scale_0;
      Scalar dpt_1 = matched_unscaled_dpts_1[idx] * scale_1;

      Scalar rotated_location_homo_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        rotated_location_homo_0_in_1[i] = rotation10[i][0] * sampled_locations_homo_0[idx][0] +
                                          rotation10[i][1] * sampled_locations_homo_0[idx][1] +
                                          rotation10[i][2] * sampled_locations_homo_0[idx][2];
      }

      Scalar location_3d_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        location_3d_0_in_1[i] = dpt_0 * rotated_location_homo_0_in_1[i] + translation10[i];
      }

      const Scalar matched_location_3d_1[3] = {dpt_1 * matched_locations_homo_1[idx][0],
                                               dpt_1 * matched_locations_homo_1[idx][1],
                                               dpt_1 * matched_locations_homo_1[idx][2]};

      // fair loss
      const Scalar sqrt_loss_param = sqrt(loss_param);
      const Scalar location_3d_diff[3] = {matched_location_3d_1[0] - location_3d_0_in_1[0],
                                          matched_location_3d_1[1] - location_3d_0_in_1[1],
                                          matched_location_3d_1[2] - location_3d_0_in_1[2]};
      const Scalar location_3d_diff_normalized[3] = {fabs(location_3d_diff[0]) / sqrt_loss_param,
                                                     fabs(location_3d_diff[1]) / sqrt_loss_param,
                                                     fabs(location_3d_diff[2]) / sqrt_loss_param};

      sampled_error[idx] =
          2 * (location_3d_diff_normalized[0] + location_3d_diff_normalized[1] + location_3d_diff_normalized[2] -
               log(1 + location_3d_diff_normalized[0]) - log(1 + location_3d_diff_normalized[1]) - log(1 + location_3d_diff_normalized[2]));

      const Scalar sqrt_fair_weight[3] = {
          sqrt(1 / (loss_param * (1 + location_3d_diff_normalized[0]))),
          sqrt(1 / (loss_param * (1 + location_3d_diff_normalized[1]))),
          sqrt(1 / (loss_param * (1 + location_3d_diff_normalized[2])))};
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        fair_sampled_3d_loc_diff[idx][i] = sqrt_fair_weight[i] * location_3d_diff[i];
      }

      Scalar location_3d_0_in_world[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        location_3d_0_in_world[i] = dpt_0 * (rotation0[i][0] * sampled_locations_homo_0[idx][0] +
                                             rotation0[i][1] * sampled_locations_homo_0[idx][1] +
                                             rotation0[i][2] * sampled_locations_homo_0[idx][2]) +
                                    translation0[i];
      }

      // 10.3.6 in "tutorial on SE3"
      Scalar loc_3d_diff_jac_pose_1[3][6];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        loc_3d_diff_jac_pose_1[i][0] = -rotation1[0][i];
        loc_3d_diff_jac_pose_1[i][1] = -rotation1[1][i];
        loc_3d_diff_jac_pose_1[i][2] = -rotation1[2][i];
        loc_3d_diff_jac_pose_1[i][3] = rotation1[1][i] * location_3d_0_in_world[2] - rotation1[2][i] * location_3d_0_in_world[1];
        loc_3d_diff_jac_pose_1[i][4] = -rotation1[0][i] * location_3d_0_in_world[2] + rotation1[2][i] * location_3d_0_in_world[0];
        loc_3d_diff_jac_pose_1[i][5] = rotation1[0][i] * location_3d_0_in_world[1] - rotation1[1][i] * location_3d_0_in_world[0];
      }

      // 10.3.8 in "tutorial on SE3" which includes 10.3.5
      const Scalar loc_3d_diff_jac_pose_0_temp[3][6] = {{1, 0, 0, 0, location_3d_0_in_world[2], -location_3d_0_in_world[1]},
                                                        {0, 1, 0, -location_3d_0_in_world[2], 0, location_3d_0_in_world[0]},
                                                        {0, 0, 1, location_3d_0_in_world[1], -location_3d_0_in_world[0], 0}};
      Scalar loc_3d_diff_jac_pose_0[3][6];

#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          // inverse rotation 1 here
          loc_3d_diff_jac_pose_0[i][j] = rotation1[0][i] * loc_3d_diff_jac_pose_0_temp[0][j] +
                                         rotation1[1][i] * loc_3d_diff_jac_pose_0_temp[1][j] +
                                         rotation1[2][i] * loc_3d_diff_jac_pose_0_temp[2][j];
        }
      }

#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          fair_sampled_3d_loc_diff_jac_pose_scale[idx][i][j] = sqrt_fair_weight[i] * loc_3d_diff_jac_pose_0[i][j];
          fair_sampled_3d_loc_diff_jac_pose_scale[idx][i][6 + j] = sqrt_fair_weight[i] * loc_3d_diff_jac_pose_1[i][j];
        }
        fair_sampled_3d_loc_diff_jac_pose_scale[idx][i][12] = sqrt_fair_weight[i] * (rotated_location_homo_0_in_1[i] * sampled_unscaled_dpts_0[idx]);
        fair_sampled_3d_loc_diff_jac_pose_scale[idx][i][13] = sqrt_fair_weight[i] * (-matched_locations_homo_1[idx][i] * matched_unscaled_dpts_1[idx]);
      }
    }

    return;
  }

  template <typename Scalar, int CS>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void match_geometry_jac_error_calculate_kernel_unbiased(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> fair_sampled_3d_loc_diff_jac_pose_code_scale,
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> fair_sampled_3d_loc_diff,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation10,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation10,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> matched_locations_homo_1,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> sampled_locations_1d_0,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> matched_locations_1d_1,
          const Scalar scale_0, const Scalar scale_1, const Scalar loss_param)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sampled_locations_1d_0.size(0))
    {
      const Scalar sum_scale = scale_0 + scale_1;
      Scalar dpt_0 = flatten_dpt_map_bias_0[sampled_locations_1d_0[idx]];

      for (int i = 0; i < CS; ++i)
      {
        dpt_0 += flatten_dpt_jac_code_0[sampled_locations_1d_0[idx]][i] * code_0[i];
      }
      dpt_0 = dpt_0 * scale_0 / sum_scale;

      Scalar dpt_1 = flatten_dpt_map_bias_1[matched_locations_1d_1[idx]];

      for (int i = 0; i < CS; ++i)
      {
        dpt_1 += flatten_dpt_jac_code_1[matched_locations_1d_1[idx]][i] * code_1[i];
      }
      dpt_1 = dpt_1 * scale_1 / sum_scale;

      Scalar rotated_location_homo_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        rotated_location_homo_0_in_1[i] = rotation10[i][0] * sampled_locations_homo_0[idx][0] +
                                          rotation10[i][1] * sampled_locations_homo_0[idx][1] +
                                          rotation10[i][2] * sampled_locations_homo_0[idx][2];
      }

      Scalar location_3d_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        location_3d_0_in_1[i] = dpt_0 * rotated_location_homo_0_in_1[i] + translation10[i];
      }

      const Scalar matched_location_3d_1[3] = {dpt_1 * matched_locations_homo_1[idx][0],
                                               dpt_1 * matched_locations_homo_1[idx][1],
                                               dpt_1 * matched_locations_homo_1[idx][2]};

      // fair loss
      const Scalar sqrt_loss_param = sqrt(loss_param);
      const Scalar location_3d_diff[3] = {matched_location_3d_1[0] - location_3d_0_in_1[0],
                                          matched_location_3d_1[1] - location_3d_0_in_1[1],
                                          matched_location_3d_1[2] - location_3d_0_in_1[2]};
      const Scalar location_3d_diff_normalized[3] = {fabs(location_3d_diff[0]) / sqrt_loss_param,
                                                     fabs(location_3d_diff[1]) / sqrt_loss_param,
                                                     fabs(location_3d_diff[2]) / sqrt_loss_param};

      sampled_error[idx] =
          2 * (location_3d_diff_normalized[0] + location_3d_diff_normalized[1] + location_3d_diff_normalized[2] -
               log(1 + location_3d_diff_normalized[0]) - log(1 + location_3d_diff_normalized[1]) - log(1 + location_3d_diff_normalized[2]));

      const Scalar sqrt_fair_weight[3] = {
          sqrt(1 / (loss_param * (1 + location_3d_diff_normalized[0]))),
          sqrt(1 / (loss_param * (1 + location_3d_diff_normalized[1]))),
          sqrt(1 / (loss_param * (1 + location_3d_diff_normalized[2])))};
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        fair_sampled_3d_loc_diff[idx][i] = sqrt_fair_weight[i] * location_3d_diff[i];
      }

      Scalar location_3d_0_in_world[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        location_3d_0_in_world[i] = dpt_0 * (rotation0[i][0] * sampled_locations_homo_0[idx][0] +
                                             rotation0[i][1] * sampled_locations_homo_0[idx][1] +
                                             rotation0[i][2] * sampled_locations_homo_0[idx][2]) +
                                    translation0[i];
      }

      // 10.3.6 in "tutorial on SE3"
      Scalar loc_3d_diff_jac_pose_1[3][6];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        loc_3d_diff_jac_pose_1[i][0] = -rotation1[0][i];
        loc_3d_diff_jac_pose_1[i][1] = -rotation1[1][i];
        loc_3d_diff_jac_pose_1[i][2] = -rotation1[2][i];
        loc_3d_diff_jac_pose_1[i][3] = rotation1[1][i] * location_3d_0_in_world[2] - rotation1[2][i] * location_3d_0_in_world[1];
        loc_3d_diff_jac_pose_1[i][4] = -rotation1[0][i] * location_3d_0_in_world[2] + rotation1[2][i] * location_3d_0_in_world[0];
        loc_3d_diff_jac_pose_1[i][5] = rotation1[0][i] * location_3d_0_in_world[1] - rotation1[1][i] * location_3d_0_in_world[0];
      }

      // 10.3.8 in "tutorial on SE3" which includes 10.3.5
      const Scalar loc_3d_diff_jac_pose_0_temp[3][6] = {{1, 0, 0, 0, location_3d_0_in_world[2], -location_3d_0_in_world[1]},
                                                        {0, 1, 0, -location_3d_0_in_world[2], 0, location_3d_0_in_world[0]},
                                                        {0, 0, 1, location_3d_0_in_world[1], -location_3d_0_in_world[0], 0}};
      Scalar loc_3d_diff_jac_pose_0[3][6];

#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          // inverse rotation 1 here
          loc_3d_diff_jac_pose_0[i][j] = rotation1[0][i] * loc_3d_diff_jac_pose_0_temp[0][j] +
                                         rotation1[1][i] * loc_3d_diff_jac_pose_0_temp[1][j] +
                                         rotation1[2][i] * loc_3d_diff_jac_pose_0_temp[2][j];
        }
      }

#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          fair_sampled_3d_loc_diff_jac_pose_code_scale[idx][i][j] = sqrt_fair_weight[i] * loc_3d_diff_jac_pose_0[i][j];
          fair_sampled_3d_loc_diff_jac_pose_code_scale[idx][i][6 + j] = sqrt_fair_weight[i] * loc_3d_diff_jac_pose_1[i][j];
        }

        for (int j = 0; j < CS; ++j)
        {
          fair_sampled_3d_loc_diff_jac_pose_code_scale[idx][i][12 + j] =
              sqrt_fair_weight[i] * (rotated_location_homo_0_in_1[i] * flatten_dpt_jac_code_0[sampled_locations_1d_0[idx]][j] * scale_0 / sum_scale);
          fair_sampled_3d_loc_diff_jac_pose_code_scale[idx][i][12 + CS + j] =
              sqrt_fair_weight[i] * (-matched_locations_homo_1[idx][i] * flatten_dpt_jac_code_1[matched_locations_1d_1[idx]][j] * scale_1 / sum_scale);
        }

        fair_sampled_3d_loc_diff_jac_pose_code_scale[idx][i][12 + 2 * CS] =
            sqrt_fair_weight[i] * (rotated_location_homo_0_in_1[i] * dpt_0 * scale_1 / (scale_0 * sum_scale) + matched_locations_homo_1[idx][i] * dpt_1 / sum_scale);
        fair_sampled_3d_loc_diff_jac_pose_code_scale[idx][i][13 + 2 * CS] =
            sqrt_fair_weight[i] * (-rotated_location_homo_0_in_1[i] * dpt_0 / sum_scale - matched_locations_homo_1[idx][i] * dpt_1 * scale_0 / (scale_1 * sum_scale));
      }
    }

    return;
  }

  template <typename Scalar, int CS>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void match_geometry_jac_error_calculate_kernel_fair(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> fair_sampled_3d_loc_diff_jac_pose_code_scale,
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> fair_sampled_3d_loc_diff,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation10,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation10,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> matched_locations_homo_1,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> sampled_locations_1d_0,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> matched_locations_1d_1,
          const Scalar scale_0, const Scalar scale_1, const Scalar loss_param)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sampled_locations_1d_0.size(0))
    {
      Scalar dpt_0 = flatten_dpt_map_bias_0[sampled_locations_1d_0[idx]];

      for (int i = 0; i < CS; ++i)
      {
        dpt_0 += flatten_dpt_jac_code_0[sampled_locations_1d_0[idx]][i] * code_0[i];
      }
      dpt_0 *= scale_0;

      Scalar dpt_1 = flatten_dpt_map_bias_1[matched_locations_1d_1[idx]];

      for (int i = 0; i < CS; ++i)
      {
        dpt_1 += flatten_dpt_jac_code_1[matched_locations_1d_1[idx]][i] * code_1[i];
      }
      dpt_1 *= scale_1;

      Scalar rotated_location_homo_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        rotated_location_homo_0_in_1[i] = rotation10[i][0] * sampled_locations_homo_0[idx][0] +
                                          rotation10[i][1] * sampled_locations_homo_0[idx][1] +
                                          rotation10[i][2] * sampled_locations_homo_0[idx][2];
      }

      Scalar location_3d_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        location_3d_0_in_1[i] = dpt_0 * rotated_location_homo_0_in_1[i] + translation10[i];
      }

      const Scalar matched_location_3d_1[3] = {dpt_1 * matched_locations_homo_1[idx][0],
                                               dpt_1 * matched_locations_homo_1[idx][1],
                                               dpt_1 * matched_locations_homo_1[idx][2]};

      // fair loss
      const Scalar sqrt_loss_param = sqrt(loss_param);
      const Scalar location_3d_diff[3] = {matched_location_3d_1[0] - location_3d_0_in_1[0],
                                          matched_location_3d_1[1] - location_3d_0_in_1[1],
                                          matched_location_3d_1[2] - location_3d_0_in_1[2]};
      const Scalar location_3d_diff_normalized[3] = {fabs(location_3d_diff[0]) / sqrt_loss_param,
                                                     fabs(location_3d_diff[1]) / sqrt_loss_param,
                                                     fabs(location_3d_diff[2]) / sqrt_loss_param};

      sampled_error[idx] =
          2 * (location_3d_diff_normalized[0] + location_3d_diff_normalized[1] + location_3d_diff_normalized[2] -
               log(1 + location_3d_diff_normalized[0]) - log(1 + location_3d_diff_normalized[1]) - log(1 + location_3d_diff_normalized[2]));

      const Scalar sqrt_fair_weight[3] = {
          sqrt(1 / (loss_param * (1 + location_3d_diff_normalized[0]))),
          sqrt(1 / (loss_param * (1 + location_3d_diff_normalized[1]))),
          sqrt(1 / (loss_param * (1 + location_3d_diff_normalized[2])))};
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        fair_sampled_3d_loc_diff[idx][i] = sqrt_fair_weight[i] * location_3d_diff[i];
      }

      Scalar location_3d_0_in_world[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        location_3d_0_in_world[i] = dpt_0 * (rotation0[i][0] * sampled_locations_homo_0[idx][0] +
                                             rotation0[i][1] * sampled_locations_homo_0[idx][1] +
                                             rotation0[i][2] * sampled_locations_homo_0[idx][2]) +
                                    translation0[i];
      }

      // 10.3.6 in "tutorial on SE3"
      Scalar loc_3d_diff_jac_pose_1[3][6];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        loc_3d_diff_jac_pose_1[i][0] = -rotation1[0][i];
        loc_3d_diff_jac_pose_1[i][1] = -rotation1[1][i];
        loc_3d_diff_jac_pose_1[i][2] = -rotation1[2][i];
        loc_3d_diff_jac_pose_1[i][3] = rotation1[1][i] * location_3d_0_in_world[2] - rotation1[2][i] * location_3d_0_in_world[1];
        loc_3d_diff_jac_pose_1[i][4] = -rotation1[0][i] * location_3d_0_in_world[2] + rotation1[2][i] * location_3d_0_in_world[0];
        loc_3d_diff_jac_pose_1[i][5] = rotation1[0][i] * location_3d_0_in_world[1] - rotation1[1][i] * location_3d_0_in_world[0];
      }

      // 10.3.8 in "tutorial on SE3" which includes 10.3.5
      const Scalar loc_3d_diff_jac_pose_0_temp[3][6] = {{1, 0, 0, 0, location_3d_0_in_world[2], -location_3d_0_in_world[1]},
                                                        {0, 1, 0, -location_3d_0_in_world[2], 0, location_3d_0_in_world[0]},
                                                        {0, 0, 1, location_3d_0_in_world[1], -location_3d_0_in_world[0], 0}};
      Scalar loc_3d_diff_jac_pose_0[3][6];

#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          // inverse rotation 1 here
          loc_3d_diff_jac_pose_0[i][j] = rotation1[0][i] * loc_3d_diff_jac_pose_0_temp[0][j] +
                                         rotation1[1][i] * loc_3d_diff_jac_pose_0_temp[1][j] +
                                         rotation1[2][i] * loc_3d_diff_jac_pose_0_temp[2][j];
        }
      }

#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          fair_sampled_3d_loc_diff_jac_pose_code_scale[idx][i][j] = sqrt_fair_weight[i] * loc_3d_diff_jac_pose_0[i][j];
          fair_sampled_3d_loc_diff_jac_pose_code_scale[idx][i][6 + j] = sqrt_fair_weight[i] * loc_3d_diff_jac_pose_1[i][j];
        }

        for (int j = 0; j < CS; ++j)
        {
          fair_sampled_3d_loc_diff_jac_pose_code_scale[idx][i][12 + j] =
              sqrt_fair_weight[i] * (rotated_location_homo_0_in_1[i] * scale_0 * flatten_dpt_jac_code_0[sampled_locations_1d_0[idx]][j]);
          fair_sampled_3d_loc_diff_jac_pose_code_scale[idx][i][12 + CS + j] =
              sqrt_fair_weight[i] * (-matched_locations_homo_1[idx][i] * scale_1 * flatten_dpt_jac_code_1[matched_locations_1d_1[idx]][j]);
        }

        fair_sampled_3d_loc_diff_jac_pose_code_scale[idx][i][12 + 2 * CS] = sqrt_fair_weight[i] * (rotated_location_homo_0_in_1[i] * dpt_0 / scale_0);
        fair_sampled_3d_loc_diff_jac_pose_code_scale[idx][i][13 + 2 * CS] = sqrt_fair_weight[i] * (-matched_locations_homo_1[idx][i] * dpt_1 / scale_1);
      }
    }

    return;
  }

  template <typename Scalar, int CS>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void match_geometry_jac_error_calculate_kernel_l2(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> sampled_3d_loc_diff_jac_pose_code_scale,
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_3d_loc_diff,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation10,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation10,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> matched_locations_homo_1,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> sampled_locations_1d_0,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> matched_locations_1d_1,
          const Scalar scale_0, const Scalar scale_1)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sampled_locations_1d_0.size(0))
    {
      Scalar dpt_0 = flatten_dpt_map_bias_0[sampled_locations_1d_0[idx]];

      for (int i = 0; i < CS; ++i)
      {
        dpt_0 += flatten_dpt_jac_code_0[sampled_locations_1d_0[idx]][i] * code_0[i];
      }
      dpt_0 *= scale_0;

      Scalar dpt_1 = flatten_dpt_map_bias_1[matched_locations_1d_1[idx]];

      for (int i = 0; i < CS; ++i)
      {
        dpt_1 += flatten_dpt_jac_code_1[matched_locations_1d_1[idx]][i] * code_1[i];
      }
      dpt_1 *= scale_1;

      Scalar rotated_location_homo_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        rotated_location_homo_0_in_1[i] = rotation10[i][0] * sampled_locations_homo_0[idx][0] +
                                          rotation10[i][1] * sampled_locations_homo_0[idx][1] +
                                          rotation10[i][2] * sampled_locations_homo_0[idx][2];
      }

      Scalar location_3d_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        location_3d_0_in_1[i] = dpt_0 * rotated_location_homo_0_in_1[i] + translation10[i];
      }

      const Scalar matched_location_3d_1[3] = {dpt_1 * matched_locations_homo_1[idx][0],
                                               dpt_1 * matched_locations_homo_1[idx][1],
                                               dpt_1 * matched_locations_homo_1[idx][2]};

      // L2 loss
      const Scalar location_3d_diff[3] = {matched_location_3d_1[0] - location_3d_0_in_1[0],
                                          matched_location_3d_1[1] - location_3d_0_in_1[1],
                                          matched_location_3d_1[2] - location_3d_0_in_1[2]};

      sampled_error[idx] = (location_3d_diff[0] * location_3d_diff[0] + location_3d_diff[1] * location_3d_diff[1] + location_3d_diff[2] * location_3d_diff[2]);
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        sampled_3d_loc_diff[idx][i] = location_3d_diff[i];
      }

      Scalar location_3d_0_in_world[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        location_3d_0_in_world[i] = dpt_0 * (rotation0[i][0] * sampled_locations_homo_0[idx][0] +
                                             rotation0[i][1] * sampled_locations_homo_0[idx][1] +
                                             rotation0[i][2] * sampled_locations_homo_0[idx][2]) +
                                    translation0[i];
      }

      // 10.3.6 in "tutorial on SE3"
      Scalar loc_3d_diff_jac_pose_1[3][6];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        loc_3d_diff_jac_pose_1[i][0] = -rotation1[0][i];
        loc_3d_diff_jac_pose_1[i][1] = -rotation1[1][i];
        loc_3d_diff_jac_pose_1[i][2] = -rotation1[2][i];
        loc_3d_diff_jac_pose_1[i][3] = rotation1[1][i] * location_3d_0_in_world[2] - rotation1[2][i] * location_3d_0_in_world[1];
        loc_3d_diff_jac_pose_1[i][4] = -rotation1[0][i] * location_3d_0_in_world[2] + rotation1[2][i] * location_3d_0_in_world[0];
        loc_3d_diff_jac_pose_1[i][5] = rotation1[0][i] * location_3d_0_in_world[1] - rotation1[1][i] * location_3d_0_in_world[0];
      }

      // 10.3.8 in "tutorial on SE3"
      const Scalar loc_3d_diff_jac_pose_0_temp[3][6] = {{1, 0, 0, 0, location_3d_0_in_world[2], -location_3d_0_in_world[1]},
                                                        {0, 1, 0, -location_3d_0_in_world[2], 0, location_3d_0_in_world[0]},
                                                        {0, 0, 1, location_3d_0_in_world[1], -location_3d_0_in_world[0], 0}};
      Scalar loc_3d_diff_jac_pose_0[3][6];

#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          // inverse rotation 1 here
          loc_3d_diff_jac_pose_0[i][j] = rotation1[0][i] * loc_3d_diff_jac_pose_0_temp[0][j] +
                                         rotation1[1][i] * loc_3d_diff_jac_pose_0_temp[1][j] +
                                         rotation1[2][i] * loc_3d_diff_jac_pose_0_temp[2][j];
        }
      }

#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          sampled_3d_loc_diff_jac_pose_code_scale[idx][i][j] = loc_3d_diff_jac_pose_0[i][j];
          sampled_3d_loc_diff_jac_pose_code_scale[idx][i][6 + j] = loc_3d_diff_jac_pose_1[i][j];
        }

        for (int j = 0; j < CS; ++j)
        {
          sampled_3d_loc_diff_jac_pose_code_scale[idx][i][12 + j] =
              (rotated_location_homo_0_in_1[i] * scale_0 * flatten_dpt_jac_code_0[sampled_locations_1d_0[idx]][j]);
          sampled_3d_loc_diff_jac_pose_code_scale[idx][i][12 + CS + j] =
              (-matched_locations_homo_1[idx][i] * scale_1 * flatten_dpt_jac_code_1[matched_locations_1d_1[idx]][j]);
        }

        sampled_3d_loc_diff_jac_pose_code_scale[idx][i][12 + 2 * CS] = (rotated_location_homo_0_in_1[i] * dpt_0 / scale_0);
        sampled_3d_loc_diff_jac_pose_code_scale[idx][i][13 + 2 * CS] = (-matched_locations_homo_1[idx][i] * dpt_1 / scale_1);
      }
    }

    return;
  }

  template <typename Scalar, int CS>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void match_geometry_jac_error_calculate_kernel_huber(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> sampled_3d_loc_diff_jac_pose_code_scale,
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_3d_loc_diff,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation10,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation10,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> matched_locations_homo_1,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> sampled_locations_1d_0,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> matched_locations_1d_1,
          const Scalar scale_0, const Scalar scale_1, const Scalar loss_param)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sampled_locations_1d_0.size(0))
    {
      Scalar dpt_0 = flatten_dpt_map_bias_0[sampled_locations_1d_0[idx]];

      for (int i = 0; i < CS; ++i)
      {
        dpt_0 += flatten_dpt_jac_code_0[sampled_locations_1d_0[idx]][i] * code_0[i];
      }
      dpt_0 *= scale_0;

      Scalar dpt_1 = flatten_dpt_map_bias_1[matched_locations_1d_1[idx]];

      for (int i = 0; i < CS; ++i)
      {
        dpt_1 += flatten_dpt_jac_code_1[matched_locations_1d_1[idx]][i] * code_1[i];
      }
      dpt_1 *= scale_1;

      Scalar rotated_location_homo_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        rotated_location_homo_0_in_1[i] = rotation10[i][0] * sampled_locations_homo_0[idx][0] +
                                          rotation10[i][1] * sampled_locations_homo_0[idx][1] +
                                          rotation10[i][2] * sampled_locations_homo_0[idx][2];
      }

      Scalar location_3d_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        location_3d_0_in_1[i] = dpt_0 * rotated_location_homo_0_in_1[i] + translation10[i];
      }

      const Scalar matched_location_3d_1[3] = {dpt_1 * matched_locations_homo_1[idx][0],
                                               dpt_1 * matched_locations_homo_1[idx][1],
                                               dpt_1 * matched_locations_homo_1[idx][2]};

      // fair loss
      const Scalar sqrt_loss_param = sqrt(loss_param);
      const Scalar location_3d_diff[3] = {matched_location_3d_1[0] - location_3d_0_in_1[0],
                                          matched_location_3d_1[1] - location_3d_0_in_1[1],
                                          matched_location_3d_1[2] - location_3d_0_in_1[2]};
      const Scalar sq_location_3d_diff[3] = {location_3d_diff[0] * location_3d_diff[0],
                                             location_3d_diff[1] * location_3d_diff[1],
                                             location_3d_diff[2] * location_3d_diff[2]};

      sampled_error[idx] = 0;
      sampled_error[idx] =
          ((sq_location_3d_diff[0] <=
            loss_param)
               ? sq_location_3d_diff[0]
               : (2.0 * sqrt(loss_param * sq_location_3d_diff[0]) - loss_param)) +
          ((sq_location_3d_diff[1] <=
            loss_param)
               ? sq_location_3d_diff[1]
               : (2.0 * sqrt(loss_param * sq_location_3d_diff[1]) - loss_param)) +
          ((sq_location_3d_diff[2] <=
            loss_param)
               ? sq_location_3d_diff[2]
               : (2.0 * sqrt(loss_param * sq_location_3d_diff[2]) - loss_param));

      const Scalar sqrt_huber_weight[3] = {min(1.0f, sqrt(loss_param / sq_location_3d_diff[0])),
                                           min(1.0f, sqrt(loss_param / sq_location_3d_diff[1])),
                                           min(1.0f, sqrt(loss_param / sq_location_3d_diff[2]))};
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        sampled_3d_loc_diff[idx][i] = sqrt_huber_weight[i] * location_3d_diff[i];
      }

      Scalar location_3d_0_in_world[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        location_3d_0_in_world[i] = dpt_0 * (rotation0[i][0] * sampled_locations_homo_0[idx][0] +
                                             rotation0[i][1] * sampled_locations_homo_0[idx][1] +
                                             rotation0[i][2] * sampled_locations_homo_0[idx][2]) +
                                    translation0[i];
      }

      // 10.3.6 in "tutorial on SE3"
      Scalar loc_3d_diff_jac_pose_1[3][6];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        loc_3d_diff_jac_pose_1[i][0] = -rotation1[0][i];
        loc_3d_diff_jac_pose_1[i][1] = -rotation1[1][i];
        loc_3d_diff_jac_pose_1[i][2] = -rotation1[2][i];
        loc_3d_diff_jac_pose_1[i][3] = rotation1[1][i] * location_3d_0_in_world[2] - rotation1[2][i] * location_3d_0_in_world[1];
        loc_3d_diff_jac_pose_1[i][4] = -rotation1[0][i] * location_3d_0_in_world[2] + rotation1[2][i] * location_3d_0_in_world[0];
        loc_3d_diff_jac_pose_1[i][5] = rotation1[0][i] * location_3d_0_in_world[1] - rotation1[1][i] * location_3d_0_in_world[0];
      }

      // 10.3.8 in "tutorial on SE3"
      const Scalar loc_3d_diff_jac_pose_0_temp[3][6] = {{1, 0, 0, 0, location_3d_0_in_world[2], -location_3d_0_in_world[1]},
                                                        {0, 1, 0, -location_3d_0_in_world[2], 0, location_3d_0_in_world[0]},
                                                        {0, 0, 1, location_3d_0_in_world[1], -location_3d_0_in_world[0], 0}};
      Scalar loc_3d_diff_jac_pose_0[3][6];

#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          // inverse rotation 1 here
          loc_3d_diff_jac_pose_0[i][j] = rotation1[0][i] * loc_3d_diff_jac_pose_0_temp[0][j] +
                                         rotation1[1][i] * loc_3d_diff_jac_pose_0_temp[1][j] +
                                         rotation1[2][i] * loc_3d_diff_jac_pose_0_temp[2][j];
        }
      }

#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          sampled_3d_loc_diff_jac_pose_code_scale[idx][i][j] = sqrt_huber_weight[i] * loc_3d_diff_jac_pose_0[i][j];
          sampled_3d_loc_diff_jac_pose_code_scale[idx][i][6 + j] = sqrt_huber_weight[i] * loc_3d_diff_jac_pose_1[i][j];
        }

        for (int j = 0; j < CS; ++j)
        {
          sampled_3d_loc_diff_jac_pose_code_scale[idx][i][12 + j] =
              sqrt_huber_weight[i] * (rotated_location_homo_0_in_1[i] * scale_0 * flatten_dpt_jac_code_0[sampled_locations_1d_0[idx]][j]);
          sampled_3d_loc_diff_jac_pose_code_scale[idx][i][12 + CS + j] =
              sqrt_huber_weight[i] * (-matched_locations_homo_1[idx][i] * scale_1 * flatten_dpt_jac_code_1[matched_locations_1d_1[idx]][j]);
        }

        sampled_3d_loc_diff_jac_pose_code_scale[idx][i][12 + 2 * CS] = sqrt_huber_weight[i] * (rotated_location_homo_0_in_1[i] * dpt_0 / scale_0);
        sampled_3d_loc_diff_jac_pose_code_scale[idx][i][13 + 2 * CS] = sqrt_huber_weight[i] * (-matched_locations_homo_1[idx][i] * dpt_1 / scale_1);
      }
    }

    return;
  }

  /* sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
    rotation.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
    translation.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
    sampled_unscaled_dpts_0.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
    matched_unscaled_dpts_1.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
    sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
    matched_locations_homo_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
    scale_0, scale_1, loss_param */
  template <typename Scalar>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void loop_mg_error_calculate_kernel(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_unscaled_dpts_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> matched_unscaled_dpts_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> matched_locations_homo_1,
          const Scalar scale_0, const Scalar scale_1, const Scalar loss_param)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sampled_locations_homo_0.size(0))
    {
      Scalar dpt_0 = sampled_unscaled_dpts_0[idx] * scale_0;
      Scalar dpt_1 = matched_unscaled_dpts_1[idx] * scale_1;

      Scalar location_3d_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        location_3d_0_in_1[i] = dpt_0 * (rotation[i][0] * sampled_locations_homo_0[idx][0] +
                                         rotation[i][1] * sampled_locations_homo_0[idx][1] +
                                         rotation[i][2] * sampled_locations_homo_0[idx][2]) +
                                translation[i];
      }
      const Scalar matched_location_3d_1[3] = {dpt_1 * matched_locations_homo_1[idx][0],
                                               dpt_1 * matched_locations_homo_1[idx][1],
                                               dpt_1 * matched_locations_homo_1[idx][2]};

      // fair loss
      const Scalar sqrt_loss_param = sqrt(loss_param);
      const Scalar location_3d_diff[3] = {matched_location_3d_1[0] - location_3d_0_in_1[0],
                                          matched_location_3d_1[1] - location_3d_0_in_1[1],
                                          matched_location_3d_1[2] - location_3d_0_in_1[2]};
      const Scalar location_3d_diff_normalized[3] = {fabs(location_3d_diff[0]) / sqrt_loss_param,
                                                     fabs(location_3d_diff[1]) / sqrt_loss_param,
                                                     fabs(location_3d_diff[2]) / sqrt_loss_param};

      sampled_error[idx] =
          2 * (location_3d_diff_normalized[0] + location_3d_diff_normalized[1] + location_3d_diff_normalized[2] -
               log(1 + location_3d_diff_normalized[0]) - log(1 + location_3d_diff_normalized[1]) - log(1 + location_3d_diff_normalized[2]));
    }
    return;
  }

  template <typename Scalar, int CS>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void match_geometry_error_calculate_kernel_unbiased(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> matched_locations_homo_1,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> sampled_locations_1d_0,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> matched_locations_1d_1,
          const Scalar scale_0, const Scalar scale_1, const Scalar loss_param)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sampled_locations_1d_0.size(0))
    {
      const Scalar sum_scale = scale_0 + scale_1;
      Scalar dpt_0 = flatten_dpt_map_bias_0[sampled_locations_1d_0[idx]];

      for (int i = 0; i < CS; ++i)
      {
        dpt_0 += flatten_dpt_jac_code_0[sampled_locations_1d_0[idx]][i] * code_0[i];
      }
      dpt_0 = dpt_0 * scale_0 / sum_scale;

      Scalar dpt_1 = flatten_dpt_map_bias_1[matched_locations_1d_1[idx]];

      for (int i = 0; i < CS; ++i)
      {
        dpt_1 += flatten_dpt_jac_code_1[matched_locations_1d_1[idx]][i] * code_1[i];
      }
      dpt_1 = dpt_1 * scale_1 / sum_scale;

      Scalar location_3d_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        location_3d_0_in_1[i] = dpt_0 * (rotation[i][0] * sampled_locations_homo_0[idx][0] +
                                         rotation[i][1] * sampled_locations_homo_0[idx][1] +
                                         rotation[i][2] * sampled_locations_homo_0[idx][2]) +
                                translation[i];
      }
      const Scalar matched_location_3d_1[3] = {dpt_1 * matched_locations_homo_1[idx][0],
                                               dpt_1 * matched_locations_homo_1[idx][1],
                                               dpt_1 * matched_locations_homo_1[idx][2]};

      // fair loss
      const Scalar sqrt_loss_param = sqrt(loss_param);
      const Scalar location_3d_diff[3] = {matched_location_3d_1[0] - location_3d_0_in_1[0],
                                          matched_location_3d_1[1] - location_3d_0_in_1[1],
                                          matched_location_3d_1[2] - location_3d_0_in_1[2]};
      const Scalar location_3d_diff_normalized[3] = {fabs(location_3d_diff[0]) / sqrt_loss_param,
                                                     fabs(location_3d_diff[1]) / sqrt_loss_param,
                                                     fabs(location_3d_diff[2]) / sqrt_loss_param};

      sampled_error[idx] =
          2 * (location_3d_diff_normalized[0] + location_3d_diff_normalized[1] + location_3d_diff_normalized[2] -
               log(1 + location_3d_diff_normalized[0]) - log(1 + location_3d_diff_normalized[1]) - log(1 + location_3d_diff_normalized[2]));
    }

    return;
  }

  template <typename Scalar, int CS>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void match_geometry_error_calculate_kernel_fair(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> matched_locations_homo_1,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> sampled_locations_1d_0,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> matched_locations_1d_1,
          const Scalar scale_0, const Scalar scale_1, const Scalar loss_param)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sampled_locations_1d_0.size(0))
    {
      Scalar dpt_0 = flatten_dpt_map_bias_0[sampled_locations_1d_0[idx]];

      for (int i = 0; i < CS; ++i)
      {
        dpt_0 += flatten_dpt_jac_code_0[sampled_locations_1d_0[idx]][i] * code_0[i];
      }
      dpt_0 *= scale_0;

      Scalar dpt_1 = flatten_dpt_map_bias_1[matched_locations_1d_1[idx]];

      for (int i = 0; i < CS; ++i)
      {
        dpt_1 += flatten_dpt_jac_code_1[matched_locations_1d_1[idx]][i] * code_1[i];
      }
      dpt_1 *= scale_1;

      Scalar location_3d_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        location_3d_0_in_1[i] = dpt_0 * (rotation[i][0] * sampled_locations_homo_0[idx][0] +
                                         rotation[i][1] * sampled_locations_homo_0[idx][1] +
                                         rotation[i][2] * sampled_locations_homo_0[idx][2]) +
                                translation[i];
      }
      const Scalar matched_location_3d_1[3] = {dpt_1 * matched_locations_homo_1[idx][0],
                                               dpt_1 * matched_locations_homo_1[idx][1],
                                               dpt_1 * matched_locations_homo_1[idx][2]};

      // fair loss
      const Scalar sqrt_loss_param = sqrt(loss_param);
      const Scalar location_3d_diff[3] = {matched_location_3d_1[0] - location_3d_0_in_1[0],
                                          matched_location_3d_1[1] - location_3d_0_in_1[1],
                                          matched_location_3d_1[2] - location_3d_0_in_1[2]};
      const Scalar location_3d_diff_normalized[3] = {fabs(location_3d_diff[0]) / sqrt_loss_param,
                                                     fabs(location_3d_diff[1]) / sqrt_loss_param,
                                                     fabs(location_3d_diff[2]) / sqrt_loss_param};

      sampled_error[idx] =
          2 * (location_3d_diff_normalized[0] + location_3d_diff_normalized[1] + location_3d_diff_normalized[2] -
               log(1 + location_3d_diff_normalized[0]) - log(1 + location_3d_diff_normalized[1]) - log(1 + location_3d_diff_normalized[2]));
    }

    return;
  }

  template <typename Scalar, int CS>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void match_geometry_error_calculate_kernel_l2(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> matched_locations_homo_1,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> sampled_locations_1d_0,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> matched_locations_1d_1,
          const Scalar scale_0, const Scalar scale_1)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sampled_locations_1d_0.size(0))
    {
      Scalar dpt_0 = flatten_dpt_map_bias_0[sampled_locations_1d_0[idx]];

      for (int i = 0; i < CS; ++i)
      {
        dpt_0 += flatten_dpt_jac_code_0[sampled_locations_1d_0[idx]][i] * code_0[i];
      }
      dpt_0 *= scale_0;

      Scalar dpt_1 = flatten_dpt_map_bias_1[matched_locations_1d_1[idx]];

      for (int i = 0; i < CS; ++i)
      {
        dpt_1 += flatten_dpt_jac_code_1[matched_locations_1d_1[idx]][i] * code_1[i];
      }
      dpt_1 *= scale_1;

      Scalar location_3d_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        location_3d_0_in_1[i] = dpt_0 * (rotation[i][0] * sampled_locations_homo_0[idx][0] +
                                         rotation[i][1] * sampled_locations_homo_0[idx][1] +
                                         rotation[i][2] * sampled_locations_homo_0[idx][2]) +
                                translation[i];
      }
      const Scalar matched_location_3d_1[3] = {dpt_1 * matched_locations_homo_1[idx][0],
                                               dpt_1 * matched_locations_homo_1[idx][1],
                                               dpt_1 * matched_locations_homo_1[idx][2]};

      // L2 loss
      const Scalar location_3d_diff[3] = {matched_location_3d_1[0] - location_3d_0_in_1[0],
                                          matched_location_3d_1[1] - location_3d_0_in_1[1],
                                          matched_location_3d_1[2] - location_3d_0_in_1[2]};

      sampled_error[idx] = (location_3d_diff[0] * location_3d_diff[0] + location_3d_diff[1] * location_3d_diff[1] + location_3d_diff[2] * location_3d_diff[2]);
    }

    return;
  }

  template <typename Scalar, int CS>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void match_geometry_error_calculate_kernel_huber(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> matched_locations_homo_1,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> sampled_locations_1d_0,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> matched_locations_1d_1,
          const Scalar scale_0, const Scalar scale_1, const Scalar loss_param)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sampled_locations_1d_0.size(0))
    {
      Scalar dpt_0 = flatten_dpt_map_bias_0[sampled_locations_1d_0[idx]];

      for (int i = 0; i < CS; ++i)
      {
        dpt_0 += flatten_dpt_jac_code_0[sampled_locations_1d_0[idx]][i] * code_0[i];
      }
      dpt_0 *= scale_0;

      Scalar dpt_1 = flatten_dpt_map_bias_1[matched_locations_1d_1[idx]];

      for (int i = 0; i < CS; ++i)
      {
        dpt_1 += flatten_dpt_jac_code_1[matched_locations_1d_1[idx]][i] * code_1[i];
      }
      dpt_1 *= scale_1;

      Scalar location_3d_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        location_3d_0_in_1[i] = dpt_0 * (rotation[i][0] * sampled_locations_homo_0[idx][0] +
                                         rotation[i][1] * sampled_locations_homo_0[idx][1] +
                                         rotation[i][2] * sampled_locations_homo_0[idx][2]) +
                                translation[i];
      }
      const Scalar matched_location_3d_1[3] = {dpt_1 * matched_locations_homo_1[idx][0],
                                               dpt_1 * matched_locations_homo_1[idx][1],
                                               dpt_1 * matched_locations_homo_1[idx][2]};

      // Huber loss
      const Scalar location_3d_diff[3] = {matched_location_3d_1[0] - location_3d_0_in_1[0],
                                          matched_location_3d_1[1] - location_3d_0_in_1[1],
                                          matched_location_3d_1[2] - location_3d_0_in_1[2]};

      const Scalar sq_location_3d_diff[3] = {location_3d_diff[0] * location_3d_diff[0],
                                             location_3d_diff[1] * location_3d_diff[1],
                                             location_3d_diff[2] * location_3d_diff[2]};

      sampled_error[idx] = 0;
      for (int i = 0; i < 3; ++i)
      {
        sampled_error[idx] = sampled_error[idx] +
                                     (sq_location_3d_diff[i] <=
                                      loss_param)
                                 ? sq_location_3d_diff[i]
                                 : (2.0 * sqrt(loss_param * sq_location_3d_diff[i]) - loss_param);
      }
    }

    return;
  }

  float tracker_match_geom_error_calculate(const at::Tensor rotation, const at::Tensor translation,
                                           const at::Tensor sampled_dpts_0, const at::Tensor matched_dpts_1,
                                           const at::Tensor sampled_locations_homo_0, const at::Tensor matched_locations_homo_1,
                                           const float loss_param, const float weight)
  {
    const int thread_per_block = MAX_THREADS_PER_BLOCK;
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)));
    const dim3 block_size(thread_per_block);

    at::Tensor sampled_error = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());

    AT_DISPATCH_FLOATING_TYPES(rotation.type(), "tracker_match_geom_error_calculate_kernel", ([&] {
                                 tracker_match_geom_error_calculate_kernel<float><<<grid_size, block_size>>>(
                                     sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     rotation.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_dpts_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     matched_dpts_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     matched_locations_homo_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     loss_param);
                               }));

    gpuErrchk(cudaGetLastError());

    return weight * torch::mean(sampled_error).item<float>();
  }

  void tracker_match_geom_jac_error_calculate(at::Tensor &AtA, at::Tensor &Atb, float &error,
                                              const at::Tensor rotation, const at::Tensor translation,
                                              const at::Tensor sampled_dpts_0, const at::Tensor matched_dpts_1,
                                              const at::Tensor sampled_locations_homo_0, const at::Tensor matched_locations_homo_1,
                                              const float loss_param, const float weight)
  {
    const int thread_per_block = MAX_THREADS_PER_BLOCK;
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)));
    const dim3 block_size(thread_per_block);

    at::Tensor sampled_error = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    // N x 3 x 6
    at::Tensor fair_sampled_3d_loc_diff_jac_rel_pose = torch::zeros({sampled_locations_homo_0.size(0), 3, 6}, sampled_locations_homo_0.options());
    // N x 3
    at::Tensor fair_sampled_3d_loc_diff = torch::zeros({sampled_locations_homo_0.size(0), 3}, sampled_locations_homo_0.options());

    AT_DISPATCH_FLOATING_TYPES(rotation.type(), "tracker_match_geom_jac_error_calculate_kernel", ([&] {
                                 tracker_match_geom_jac_error_calculate_kernel<float><<<grid_size, block_size>>>(
                                     sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     fair_sampled_3d_loc_diff_jac_rel_pose.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
                                     fair_sampled_3d_loc_diff.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     rotation.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_dpts_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     matched_dpts_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     matched_locations_homo_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     loss_param);
                               }));

    gpuErrchk(cudaGetLastError());

    error = weight * torch::mean(sampled_error).item<float>();

    fair_sampled_3d_loc_diff_jac_rel_pose = fair_sampled_3d_loc_diff_jac_rel_pose.reshape({sampled_error.size(0) * 3, 6});
    // 6 x 6
    AtA = (weight / (float)sampled_error.size(0)) *
          torch::matmul(fair_sampled_3d_loc_diff_jac_rel_pose.permute({1, 0}), fair_sampled_3d_loc_diff_jac_rel_pose);
    // 6 x 1
    Atb = (weight / (float)sampled_error.size(0)) *
          torch::matmul(fair_sampled_3d_loc_diff_jac_rel_pose.permute({1, 0}), fair_sampled_3d_loc_diff.reshape({-1, 1}));

    return;
  }

  void tracker_match_geom_jac_error_calculate_with_scale(at::Tensor &AtA, at::Tensor &Atb, float &error,
                                                         const at::Tensor rotation, const at::Tensor translation,
                                                         const at::Tensor sampled_dpts_0, const at::Tensor matched_dpts_1,
                                                         const at::Tensor sampled_locations_homo_0, const at::Tensor matched_locations_homo_1,
                                                         const float scale_0, const float loss_param, const float weight)
  {
    const int thread_per_block = MAX_THREADS_PER_BLOCK;
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)));
    const dim3 block_size(thread_per_block);

    at::Tensor sampled_error = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    // N x 3 x 7
    at::Tensor fair_sampled_3d_loc_diff_jac_rel_pose_scale_0 = torch::zeros({sampled_locations_homo_0.size(0), 3, 7}, sampled_locations_homo_0.options());
    // N x 3
    at::Tensor fair_sampled_3d_loc_diff = torch::zeros({sampled_locations_homo_0.size(0), 3}, sampled_locations_homo_0.options());

    AT_DISPATCH_FLOATING_TYPES(rotation.type(), "tracker_match_geom_jac_error_calculate_with_scale_kernel", ([&] {
                                 tracker_match_geom_jac_error_calculate_with_scale_kernel<float><<<grid_size, block_size>>>(
                                     sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     fair_sampled_3d_loc_diff_jac_rel_pose_scale_0.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
                                     fair_sampled_3d_loc_diff.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     rotation.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_dpts_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     matched_dpts_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     matched_locations_homo_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     scale_0, loss_param);
                               }));

    gpuErrchk(cudaGetLastError());

    error = weight * torch::mean(sampled_error).item<float>();

    fair_sampled_3d_loc_diff_jac_rel_pose_scale_0 = fair_sampled_3d_loc_diff_jac_rel_pose_scale_0.reshape({sampled_error.size(0) * 3, 7});
    // 7 x 7
    AtA = (weight / (float)sampled_error.size(0)) *
          torch::matmul(fair_sampled_3d_loc_diff_jac_rel_pose_scale_0.permute({1, 0}), fair_sampled_3d_loc_diff_jac_rel_pose_scale_0);
    // 7 x 1
    Atb = (weight / (float)sampled_error.size(0)) *
          torch::matmul(fair_sampled_3d_loc_diff_jac_rel_pose_scale_0.permute({1, 0}), fair_sampled_3d_loc_diff.reshape({-1, 1}));

    return;
  }

  float loop_mg_error_calculate(const at::Tensor rotation, const at::Tensor translation,
                                const at::Tensor sampled_unscaled_dpts_0,
                                const at::Tensor matched_unscaled_dpts_1,
                                const at::Tensor sampled_locations_homo_0,
                                const at::Tensor matched_locations_homo_1,
                                const float scale_0, const float scale_1,
                                const float loss_param, const float weight)
  {

    const int thread_per_block = MAX_THREADS_PER_BLOCK;
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)));
    const dim3 block_size(thread_per_block);

    at::Tensor sampled_error = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());

    AT_DISPATCH_FLOATING_TYPES(rotation.type(), "loop_mg_error_calculate_kernel", ([&] {
                                 loop_mg_error_calculate_kernel<float><<<grid_size, block_size>>>(
                                     sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     rotation.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_unscaled_dpts_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     matched_unscaled_dpts_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     matched_locations_homo_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     scale_0, scale_1, loss_param);
                               }));

    gpuErrchk(cudaGetLastError());

    return weight * torch::mean(sampled_error).item<float>();
  }

  void loop_mg_jac_error_calculate(at::Tensor &AtA, at::Tensor &Atb, float &error,
                                   const at::Tensor rotation10, const at::Tensor translation10,
                                   const at::Tensor rotation0, const at::Tensor translation0,
                                   const at::Tensor rotation1, const at::Tensor translation1,
                                   const at::Tensor sampled_unscaled_dpts_0,
                                   const at::Tensor matched_unscaled_dpts_1,
                                   const at::Tensor sampled_locations_homo_0,
                                   const at::Tensor matched_locations_homo_1,
                                   const float scale_0, const float scale_1,
                                   const float loss_param, const float weight)
  {

    const int thread_per_block = MAX_THREADS_PER_BLOCK;
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)));
    const dim3 block_size(thread_per_block);

    at::Tensor sampled_error = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    // N x 3 x 14
    at::Tensor sampled_3d_loc_diff_jac_pose_scale = torch::zeros({sampled_locations_homo_0.size(0), 3, 14}, sampled_locations_homo_0.options());
    // N x 3
    at::Tensor sampled_3d_loc_diff = torch::zeros({sampled_locations_homo_0.size(0), 3}, sampled_locations_homo_0.options());

    AT_DISPATCH_FLOATING_TYPES(rotation10.type(), "loop_mg_jac_error_calculate_kernel", ([&] {
                                 loop_mg_jac_error_calculate_kernel<float><<<grid_size, block_size>>>(
                                     sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_3d_loc_diff_jac_pose_scale.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
                                     sampled_3d_loc_diff.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     rotation10.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation10.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     rotation0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     rotation1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_unscaled_dpts_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     matched_unscaled_dpts_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     matched_locations_homo_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     scale_0, scale_1, loss_param);
                               }));

    gpuErrchk(cudaGetLastError());

    error = weight * torch::mean(sampled_error).item<float>();

    sampled_3d_loc_diff_jac_pose_scale = sampled_3d_loc_diff_jac_pose_scale.reshape({sampled_error.size(0) * 3, 14});
    // 14 x 14
    AtA = (weight / (float)sampled_error.size(0)) *
          torch::matmul(sampled_3d_loc_diff_jac_pose_scale.permute({1, 0}),
                        sampled_3d_loc_diff_jac_pose_scale);
    // 14 x 1
    Atb = (weight / (float)sampled_error.size(0)) *
          torch::matmul(sampled_3d_loc_diff_jac_pose_scale.permute({1, 0}), sampled_3d_loc_diff.reshape({-1, 1}));

    return;
  }

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
                                       const std::string robust_loss_type)
  {

    const int thread_per_block = MAX_THREADS_PER_BLOCK;
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)));
    const dim3 block_size(thread_per_block);

    at::Tensor sampled_error = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());

    if (robust_loss_type == "fair")
    {
      AT_DISPATCH_FLOATING_TYPES(rotation.type(), "match_geometry_error_calculate_kernel_fair", ([&] {
                                   match_geometry_error_calculate_kernel_fair<float, CS><<<grid_size, block_size>>>(
                                       sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       rotation.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       translation.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_map_bias_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_map_bias_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_jac_code_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       flatten_dpt_jac_code_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       code_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       code_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       matched_locations_homo_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       sampled_locations_1d_0.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                       matched_locations_1d_1.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                       scale_0, scale_1, loss_param);
                                 }));
    }
    else if (robust_loss_type == "L2")
    {
      AT_DISPATCH_FLOATING_TYPES(rotation.type(), "match_geometry_error_calculate_kernel_l2", ([&] {
                                   match_geometry_error_calculate_kernel_l2<float, CS><<<grid_size, block_size>>>(
                                       sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       rotation.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       translation.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_map_bias_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_map_bias_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_jac_code_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       flatten_dpt_jac_code_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       code_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       code_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       matched_locations_homo_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       sampled_locations_1d_0.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                       matched_locations_1d_1.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                       scale_0, scale_1);
                                 }));
    }
    else if (robust_loss_type == "huber")
    {
      AT_DISPATCH_FLOATING_TYPES(rotation.type(), "match_geometry_error_calculate_kernel_huber", ([&] {
                                   match_geometry_error_calculate_kernel_huber<float, CS><<<grid_size, block_size>>>(
                                       sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       rotation.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       translation.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_map_bias_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_map_bias_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_jac_code_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       flatten_dpt_jac_code_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       code_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       code_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       matched_locations_homo_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       sampled_locations_1d_0.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                       matched_locations_1d_1.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                       scale_0, scale_1, loss_param);
                                 }));
    }
    else if (robust_loss_type == "unbiased")
    {
      AT_DISPATCH_FLOATING_TYPES(rotation.type(), "match_geometry_error_calculate_kernel_unbiased", ([&] {
                                   match_geometry_error_calculate_kernel_unbiased<float, CS><<<grid_size, block_size>>>(
                                       sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       rotation.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       translation.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_map_bias_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_map_bias_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_jac_code_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       flatten_dpt_jac_code_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       code_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       code_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       matched_locations_homo_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       sampled_locations_1d_0.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                       matched_locations_1d_1.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                       scale_0, scale_1, loss_param);
                                 }));
    }

    gpuErrchk(cudaGetLastError());

    return weight * torch::mean(sampled_error).item<float>();
  }

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
                                          const std::string robust_loss_type)
  {

    const int thread_per_block = MAX_THREADS_PER_BLOCK;
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)));
    const dim3 block_size(thread_per_block);

    at::Tensor sampled_error = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    // N x 3 x (14 + 2 * CS)
    at::Tensor sampled_3d_loc_diff_jac_pose_code_scale = torch::zeros({sampled_locations_homo_0.size(0), 3, 14 + 2 * CS}, sampled_locations_homo_0.options());
    // N x 3
    at::Tensor sampled_3d_loc_diff = torch::zeros({sampled_locations_homo_0.size(0), 3}, sampled_locations_homo_0.options());

    if (robust_loss_type == "fair")
    {
      AT_DISPATCH_FLOATING_TYPES(rotation10.type(), "match_geometry_jac_error_calculate_kernel_fair", ([&] {
                                   match_geometry_jac_error_calculate_kernel_fair<float, CS><<<grid_size, block_size>>>(
                                       sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       sampled_3d_loc_diff_jac_pose_code_scale.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
                                       sampled_3d_loc_diff.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       rotation10.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       translation10.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       rotation0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       translation0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       rotation1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       translation1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_map_bias_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_map_bias_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_jac_code_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       flatten_dpt_jac_code_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       code_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       code_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       matched_locations_homo_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       sampled_locations_1d_0.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                       matched_locations_1d_1.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                       scale_0, scale_1, loss_param);
                                 }));
    }
    else if (robust_loss_type == "L2")
    {
      AT_DISPATCH_FLOATING_TYPES(rotation10.type(), "match_geometry_jac_error_calculate_kernel_l2", ([&] {
                                   match_geometry_jac_error_calculate_kernel_l2<float, CS><<<grid_size, block_size>>>(
                                       sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       sampled_3d_loc_diff_jac_pose_code_scale.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
                                       sampled_3d_loc_diff.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       rotation10.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       translation10.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       rotation0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       translation0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       rotation1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       translation1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_map_bias_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_map_bias_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_jac_code_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       flatten_dpt_jac_code_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       code_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       code_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       matched_locations_homo_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       sampled_locations_1d_0.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                       matched_locations_1d_1.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                       scale_0, scale_1);
                                 }));
    }
    else if (robust_loss_type == "huber")
    {
      AT_DISPATCH_FLOATING_TYPES(rotation10.type(), "match_geometry_jac_error_calculate_kernel_huber", ([&] {
                                   match_geometry_jac_error_calculate_kernel_huber<float, CS><<<grid_size, block_size>>>(
                                       sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       sampled_3d_loc_diff_jac_pose_code_scale.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
                                       sampled_3d_loc_diff.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       rotation10.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       translation10.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       rotation0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       translation0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       rotation1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       translation1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_map_bias_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_map_bias_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_jac_code_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       flatten_dpt_jac_code_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       code_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       code_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       matched_locations_homo_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       sampled_locations_1d_0.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                       matched_locations_1d_1.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                       scale_0, scale_1, loss_param);
                                 }));
    }
    else if (robust_loss_type == "unbiased")
    {
      AT_DISPATCH_FLOATING_TYPES(rotation10.type(), "match_geometry_jac_error_calculate_kernel_unbiased", ([&] {
                                   match_geometry_jac_error_calculate_kernel_unbiased<float, CS><<<grid_size, block_size>>>(
                                       sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       sampled_3d_loc_diff_jac_pose_code_scale.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
                                       sampled_3d_loc_diff.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       rotation10.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       translation10.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       rotation0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       translation0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       rotation1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       translation1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_map_bias_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_map_bias_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       flatten_dpt_jac_code_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       flatten_dpt_jac_code_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       code_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       code_1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                       sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       matched_locations_homo_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                       sampled_locations_1d_0.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                       matched_locations_1d_1.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                       scale_0, scale_1, loss_param);
                                 }));
    }

    gpuErrchk(cudaGetLastError());

    error = weight * torch::mean(sampled_error).item<float>();

    sampled_3d_loc_diff_jac_pose_code_scale = sampled_3d_loc_diff_jac_pose_code_scale.reshape({sampled_error.size(0) * 3, 14 + 2 * CS});
    // (14 + 2*CS) x (14 + 2*CS)
    AtA = (weight / (float)sampled_error.size(0)) *
          torch::matmul(sampled_3d_loc_diff_jac_pose_code_scale.permute({1, 0}),
                        sampled_3d_loc_diff_jac_pose_code_scale);
    // (14 + 2*CS) x 1
    Atb = (weight / (float)sampled_error.size(0)) *
          torch::matmul(sampled_3d_loc_diff_jac_pose_code_scale.permute({1, 0}), sampled_3d_loc_diff.reshape({-1, 1}));

    return;
  }

#undef WITHIN_BOUNDS

  template float match_geometry_error_calculate<DF_CODE_SIZE>(
      const at::Tensor rotation, const at::Tensor translation,
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

  template void match_geometry_jac_error_calculate<DF_CODE_SIZE>(
      at::Tensor &AtA, at::Tensor &Atb, float &error,
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
}