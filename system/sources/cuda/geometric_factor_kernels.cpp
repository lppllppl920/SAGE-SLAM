
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "geometric_factor_kernels.h"
#include "camera_pyramid.h"

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

  template <typename Scalar, int CS>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void geometric_error_calculate_kernel_unbiased(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_valid_mask_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> unscaled_dpt_map_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> valid_mask_1,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> sampled_locations_1d_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const Scalar scale_0, const Scalar scale_1, const PinholeCamera<Scalar> camera,
          const Scalar eps, const Scalar loss_param)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sampled_locations_1d_0.size(0))
    {
      const Scalar sum_scale = scale_0 + scale_1;
      const Scalar fx = camera.fx();
      const Scalar fy = camera.fy();
      const Scalar cx = camera.u0();
      const Scalar cy = camera.v0();
      const int height = static_cast<int>(camera.height());
      const int width = static_cast<int>(camera.width());

      const int location_1d = sampled_locations_1d_0[idx];

      Scalar dpt_0 = flatten_dpt_map_bias_0[location_1d];
      for (int i = 0; i < CS; ++i)
      {
        dpt_0 += flatten_dpt_jac_code_0[location_1d][i] * code_0[i];
      }
      dpt_0 = dpt_0 * scale_0 / sum_scale;

      Scalar rotated_location_homo_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        rotated_location_homo_0_in_1[i] = rotation[i][0] * sampled_locations_homo_0[idx][0] +
                                          rotation[i][1] * sampled_locations_homo_0[idx][1] +
                                          rotation[i][2] * sampled_locations_homo_0[idx][2];
      }

      const Scalar location_3d_in_1_x = dpt_0 * rotated_location_homo_0_in_1[0] + translation[0];
      const Scalar location_3d_in_1_y = dpt_0 * rotated_location_homo_0_in_1[1] + translation[1];
      Scalar location_3d_in_1_z = dpt_0 * rotated_location_homo_0_in_1[2] + translation[2];

      // Whether or not the transformed depth is below the positive depth threshold
      bool is_dpt_positive = location_3d_in_1_z > eps;

      const Scalar proj_2d_x = (location_3d_in_1_x / location_3d_in_1_z) * fx + cx;
      const Scalar proj_2d_y = (location_3d_in_1_y / location_3d_in_1_z) * fy + cy;

      const int proj_2d_x_floor = int(floor(proj_2d_x));
      const int proj_2d_y_floor = int(floor(proj_2d_y));
      const int proj_2d_x_ceil = proj_2d_x_floor + 1;
      const int proj_2d_y_ceil = proj_2d_y_floor + 1;

      const Scalar lower_x_weight = (Scalar)proj_2d_x_ceil - proj_2d_x;
      const Scalar lower_y_weight = (Scalar)proj_2d_y_ceil - proj_2d_y;
      const Scalar upper_x_weight = 1 - lower_x_weight;
      const Scalar upper_y_weight = 1 - lower_y_weight;

      const Scalar nw_weight = lower_x_weight * lower_y_weight;
      const Scalar se_weight = upper_x_weight * upper_y_weight;
      const Scalar sw_weight = lower_x_weight * upper_y_weight;
      const Scalar ne_weight = upper_x_weight * lower_y_weight;

      // Zero padding bilinear interpolation
      const Scalar dpt_00 = WITHIN_BOUNDS(proj_2d_x_floor, proj_2d_y_floor, width, height) ? unscaled_dpt_map_1[proj_2d_y_floor][proj_2d_x_floor] * nw_weight : 0;
      const Scalar dpt_11 = WITHIN_BOUNDS(proj_2d_x_ceil, proj_2d_y_ceil, width, height) ? unscaled_dpt_map_1[proj_2d_y_ceil][proj_2d_x_ceil] * se_weight : 0;
      const Scalar dpt_01 = WITHIN_BOUNDS(proj_2d_x_floor, proj_2d_y_ceil, width, height) ? unscaled_dpt_map_1[proj_2d_y_ceil][proj_2d_x_floor] * sw_weight : 0;
      const Scalar dpt_10 = WITHIN_BOUNDS(proj_2d_x_ceil, proj_2d_y_floor, width, height) ? unscaled_dpt_map_1[proj_2d_y_floor][proj_2d_x_ceil] * ne_weight : 0;
      const Scalar sampled_dpt_in_1 = (dpt_00 + dpt_11 + dpt_01 + dpt_10) * scale_1 / sum_scale;

      // Nearest interpolation
      const int proj_2d_x_round = int(round(proj_2d_x));
      const int proj_2d_y_round = int(round(proj_2d_y));

      const Scalar within_mask =
          WITHIN_BOUNDS(proj_2d_x_round, proj_2d_y_round, width, height) ? valid_mask_1[proj_2d_y_round][proj_2d_x_round] : 0;

      sampled_error[idx] = is_dpt_positive ? log(1 + pow(within_mask * (sampled_dpt_in_1 - location_3d_in_1_z), 2) / loss_param) : 0;
      sampled_valid_mask_1[idx] = is_dpt_positive ? within_mask : 0;
    }

    return;
  }

  template <typename Scalar, int CS>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void geometric_error_calculate_kernel(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_valid_mask_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> dpt_map_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> valid_mask_1,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> sampled_locations_1d_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const Scalar scale_0, const PinholeCamera<Scalar> camera,
          const Scalar eps, const Scalar loss_param)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sampled_locations_1d_0.size(0))
    {
      const Scalar fx = camera.fx();
      const Scalar fy = camera.fy();
      const Scalar cx = camera.u0();
      const Scalar cy = camera.v0();
      const int height = static_cast<int>(camera.height());
      const int width = static_cast<int>(camera.width());

      const int location_1d = sampled_locations_1d_0[idx];
      Scalar dpt_0 = flatten_dpt_map_bias_0[location_1d];

      for (int i = 0; i < CS; ++i)
      {
        dpt_0 += flatten_dpt_jac_code_0[location_1d][i] * code_0[i];
      }

      dpt_0 *= scale_0;

      Scalar rotated_location_homo_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        rotated_location_homo_0_in_1[i] = rotation[i][0] * sampled_locations_homo_0[idx][0] +
                                          rotation[i][1] * sampled_locations_homo_0[idx][1] +
                                          rotation[i][2] * sampled_locations_homo_0[idx][2];
      }

      const Scalar location_3d_in_1_x = dpt_0 * rotated_location_homo_0_in_1[0] + translation[0];
      const Scalar location_3d_in_1_y = dpt_0 * rotated_location_homo_0_in_1[1] + translation[1];
      Scalar location_3d_in_1_z = dpt_0 * rotated_location_homo_0_in_1[2] + translation[2];

      // Whether or not the transformed depth is below the positive depth threshold
      bool is_dpt_positive = location_3d_in_1_z > eps;

      const Scalar proj_2d_x = (location_3d_in_1_x / location_3d_in_1_z) * fx + cx;
      const Scalar proj_2d_y = (location_3d_in_1_y / location_3d_in_1_z) * fy + cy;

      const int proj_2d_x_floor = int(floor(proj_2d_x));
      const int proj_2d_y_floor = int(floor(proj_2d_y));
      const int proj_2d_x_ceil = proj_2d_x_floor + 1;
      const int proj_2d_y_ceil = proj_2d_y_floor + 1;

      const Scalar lower_x_weight = (Scalar)proj_2d_x_ceil - proj_2d_x;
      const Scalar lower_y_weight = (Scalar)proj_2d_y_ceil - proj_2d_y;
      const Scalar upper_x_weight = 1 - lower_x_weight;
      const Scalar upper_y_weight = 1 - lower_y_weight;

      const Scalar nw_weight = lower_x_weight * lower_y_weight;
      const Scalar se_weight = upper_x_weight * upper_y_weight;
      const Scalar sw_weight = lower_x_weight * upper_y_weight;
      const Scalar ne_weight = upper_x_weight * lower_y_weight;

      // Zero padding bilinear interpolation
      const Scalar dpt_00 = WITHIN_BOUNDS(proj_2d_x_floor, proj_2d_y_floor, width, height) ? dpt_map_1[proj_2d_y_floor][proj_2d_x_floor] * nw_weight : 0;
      const Scalar dpt_11 = WITHIN_BOUNDS(proj_2d_x_ceil, proj_2d_y_ceil, width, height) ? dpt_map_1[proj_2d_y_ceil][proj_2d_x_ceil] * se_weight : 0;
      const Scalar dpt_01 = WITHIN_BOUNDS(proj_2d_x_floor, proj_2d_y_ceil, width, height) ? dpt_map_1[proj_2d_y_ceil][proj_2d_x_floor] * sw_weight : 0;
      const Scalar dpt_10 = WITHIN_BOUNDS(proj_2d_x_ceil, proj_2d_y_floor, width, height) ? dpt_map_1[proj_2d_y_floor][proj_2d_x_ceil] * ne_weight : 0;
      const Scalar sampled_dpt_in_1 = dpt_00 + dpt_11 + dpt_01 + dpt_10;

      // Nearest interpolation
      const int proj_2d_x_round = int(round(proj_2d_x));
      const int proj_2d_y_round = int(round(proj_2d_y));

      const Scalar within_mask =
          WITHIN_BOUNDS(proj_2d_x_round, proj_2d_y_round, width, height) ? valid_mask_1[proj_2d_y_round][proj_2d_x_round] : 0;

      sampled_error[idx] = is_dpt_positive ? log(1 + pow(within_mask * (sampled_dpt_in_1 - location_3d_in_1_z), 2) / loss_param) : 0;
      sampled_valid_mask_1[idx] = is_dpt_positive ? within_mask : 0;
    }

    return;
  }

  template <typename Scalar, int CS>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void geometric_jac_error_calculate_kernel_unbiased(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_valid_mask_1,
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> cauchy_sampled_dpt_diff_jac_pose_code_scale,
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> cauchy_sampled_dpt_diff,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation10,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation10,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> unscaled_dpt_map_1,
          const torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> unscaled_dpt_map_grad_1,
          const torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> dpt_jac_code_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> valid_mask_1,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> sampled_locations_1d_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const Scalar scale_0, const Scalar scale_1, const PinholeCamera<Scalar> camera,
          const Scalar eps, const Scalar loss_param)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sampled_locations_1d_0.size(0))
    {
      const Scalar sum_scale = scale_0 + scale_1;
      const Scalar fx = camera.fx();
      const Scalar fy = camera.fy();
      const Scalar cx = camera.u0();
      const Scalar cy = camera.v0();
      const int height = static_cast<int>(camera.height());
      const int width = static_cast<int>(camera.width());

      Scalar sampled_dpt_grad_1[2];
      Scalar proj_2d_jac_dpt_0[2];
      Scalar sampled_dpt_jac_code_1[CS];
      Scalar dpt_diff_jac_code_0[CS];
      Scalar dpt_diff_jac_code_1[CS];

      const int location_1d = sampled_locations_1d_0[idx];
      Scalar dpt_0 = flatten_dpt_map_bias_0[location_1d];

      for (int i = 0; i < CS; ++i)
      {
        dpt_0 += flatten_dpt_jac_code_0[location_1d][i] * code_0[i];
      }
      dpt_0 = dpt_0 * scale_0 / sum_scale;

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

      // Whether or not the transformed depth is below the positive depth threshold
      // We shouldn't do depth clipping, this somehow leads to wrong warp area result previously!!!
      bool is_dpt_positive = location_3d_0_in_1[2] > eps;

      const Scalar proj_2d_x = (location_3d_0_in_1[0] / location_3d_0_in_1[2]) * fx + cx;
      const Scalar proj_2d_y = (location_3d_0_in_1[1] / location_3d_0_in_1[2]) * fy + cy;

      const int proj_2d_x_floor = int(floor(proj_2d_x));
      const int proj_2d_y_floor = int(floor(proj_2d_y));
      const int proj_2d_x_ceil = proj_2d_x_floor + 1;
      const int proj_2d_y_ceil = proj_2d_y_floor + 1;

      const Scalar lower_x_weight = (Scalar)proj_2d_x_ceil - proj_2d_x;
      const Scalar lower_y_weight = (Scalar)proj_2d_y_ceil - proj_2d_y;
      const Scalar upper_x_weight = 1 - lower_x_weight;
      const Scalar upper_y_weight = 1 - lower_y_weight;

      const Scalar nw_weight = lower_x_weight * lower_y_weight;
      const Scalar se_weight = upper_x_weight * upper_y_weight;
      const Scalar sw_weight = lower_x_weight * upper_y_weight;
      const Scalar ne_weight = upper_x_weight * lower_y_weight;

      // Zero padding bilinear interpolation
      const bool nw_bool = WITHIN_BOUNDS(proj_2d_x_floor, proj_2d_y_floor, width, height);
      const bool se_bool = WITHIN_BOUNDS(proj_2d_x_ceil, proj_2d_y_ceil, width, height);
      const bool sw_bool = WITHIN_BOUNDS(proj_2d_x_floor, proj_2d_y_ceil, width, height);
      const bool ne_bool = WITHIN_BOUNDS(proj_2d_x_ceil, proj_2d_y_floor, width, height);

      const Scalar dpt_00 = nw_bool ? unscaled_dpt_map_1[proj_2d_y_floor][proj_2d_x_floor] * nw_weight : 0;
      const Scalar dpt_11 = se_bool ? unscaled_dpt_map_1[proj_2d_y_ceil][proj_2d_x_ceil] * se_weight : 0;
      const Scalar dpt_01 = sw_bool ? unscaled_dpt_map_1[proj_2d_y_ceil][proj_2d_x_floor] * sw_weight : 0;
      const Scalar dpt_10 = ne_bool ? unscaled_dpt_map_1[proj_2d_y_floor][proj_2d_x_ceil] * ne_weight : 0;
      const Scalar sampled_dpt_1 = (dpt_00 + dpt_11 + dpt_01 + dpt_10) * scale_1 / sum_scale;

// dpt_map_grad_1: 2 x H x W
#pragma unroll_completely
      for (int i = 0; i < 2; ++i)
      {
        sampled_dpt_grad_1[i] =
            ((nw_bool ? unscaled_dpt_map_grad_1[i][proj_2d_y_floor][proj_2d_x_floor] * nw_weight : 0) +
             (se_bool ? unscaled_dpt_map_grad_1[i][proj_2d_y_ceil][proj_2d_x_ceil] * se_weight : 0) +
             (sw_bool ? unscaled_dpt_map_grad_1[i][proj_2d_y_ceil][proj_2d_x_floor] * sw_weight : 0) +
             (ne_bool ? unscaled_dpt_map_grad_1[i][proj_2d_y_floor][proj_2d_x_ceil] * ne_weight : 0)) *
            scale_1 / sum_scale;
      }

      // Nearest interpolation
      const int proj_2d_x_round = int(round(proj_2d_x));
      const int proj_2d_y_round = int(round(proj_2d_y));

      // dpt_jac_code_1: H x W x CS
#pragma unroll_completely
      for (int i = 0; i < CS; ++i)
      {
        sampled_dpt_jac_code_1[i] = (nw_bool ? dpt_jac_code_1[proj_2d_y_floor][proj_2d_x_floor][i] * nw_weight : 0) +
                                    (se_bool ? dpt_jac_code_1[proj_2d_y_ceil][proj_2d_x_ceil][i] * se_weight : 0) +
                                    (sw_bool ? dpt_jac_code_1[proj_2d_y_ceil][proj_2d_x_floor][i] * sw_weight : 0) +
                                    (ne_bool ? dpt_jac_code_1[proj_2d_y_floor][proj_2d_x_ceil][i] * ne_weight : 0);
      }

      const Scalar within_mask = WITHIN_BOUNDS(proj_2d_x_round, proj_2d_y_round, width, height) ? valid_mask_1[proj_2d_y_round][proj_2d_x_round] : 0;

      sampled_error[idx] = is_dpt_positive ? log(1 + pow(within_mask * (sampled_dpt_1 - location_3d_0_in_1[2]), 2) / loss_param) : 0;
      sampled_valid_mask_1[idx] = is_dpt_positive ? within_mask : 0;

      // A.3 in "tutorial on SE3"
      const Scalar inv_z = 1 / location_3d_0_in_1[2];
      const Scalar x_z = inv_z * location_3d_0_in_1[0];
      const Scalar y_z = inv_z * location_3d_0_in_1[1];
      const Scalar proj_2d_jac_loc_3d_0_in_1[2][3] = {{fx * inv_z, 0, -fx * x_z * inv_z},
                                                      {0, fy * inv_z, -fy * y_z * inv_z}};
      // 10.3.6
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
      Scalar loc_3d_0_in_1_jac_pose_1[3][6];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        loc_3d_0_in_1_jac_pose_1[i][0] = -rotation1[0][i];
        loc_3d_0_in_1_jac_pose_1[i][1] = -rotation1[1][i];
        loc_3d_0_in_1_jac_pose_1[i][2] = -rotation1[2][i];
        loc_3d_0_in_1_jac_pose_1[i][3] = rotation1[1][i] * location_3d_0_in_world[2] - rotation1[2][i] * location_3d_0_in_world[1];
        loc_3d_0_in_1_jac_pose_1[i][4] = -rotation1[0][i] * location_3d_0_in_world[2] + rotation1[2][i] * location_3d_0_in_world[0];
        loc_3d_0_in_1_jac_pose_1[i][5] = rotation1[0][i] * location_3d_0_in_world[1] - rotation1[1][i] * location_3d_0_in_world[0];
      }

      Scalar proj_2d_jac_pose_1[2][6];
      for (int i = 0; i < 2; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          proj_2d_jac_pose_1[i][j] = proj_2d_jac_loc_3d_0_in_1[i][0] * loc_3d_0_in_1_jac_pose_1[0][j] +
                                     proj_2d_jac_loc_3d_0_in_1[i][1] * loc_3d_0_in_1_jac_pose_1[1][j] +
                                     proj_2d_jac_loc_3d_0_in_1[i][2] * loc_3d_0_in_1_jac_pose_1[2][j];
        }
      }

      // 10.3.8 in "tutorial on SE3" which includes 10.3.5
      const Scalar loc_3d_0_in_1_jac_pose_0_temp[3][6] = {{1, 0, 0, 0, location_3d_0_in_world[2], -location_3d_0_in_world[1]},
                                                          {0, 1, 0, -location_3d_0_in_world[2], 0, location_3d_0_in_world[0]},
                                                          {0, 0, 1, location_3d_0_in_world[1], -location_3d_0_in_world[0], 0}};
      Scalar loc_3d_0_in_1_jac_pose_0[3][6];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          // inverse rotation 1 here
          loc_3d_0_in_1_jac_pose_0[i][j] = rotation1[0][i] * loc_3d_0_in_1_jac_pose_0_temp[0][j] +
                                           rotation1[1][i] * loc_3d_0_in_1_jac_pose_0_temp[1][j] +
                                           rotation1[2][i] * loc_3d_0_in_1_jac_pose_0_temp[2][j];
        }
      }
      Scalar proj_2d_jac_pose_0[2][6];
      for (int i = 0; i < 2; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          proj_2d_jac_pose_0[i][j] = proj_2d_jac_loc_3d_0_in_1[i][0] * loc_3d_0_in_1_jac_pose_0[0][j] +
                                     proj_2d_jac_loc_3d_0_in_1[i][1] * loc_3d_0_in_1_jac_pose_0[1][j] +
                                     proj_2d_jac_loc_3d_0_in_1[i][2] * loc_3d_0_in_1_jac_pose_0[2][j];
        }
      }

      Scalar dpt_diff_jac_pose_0[6], dpt_diff_jac_pose_1[6];
#pragma unroll_completely
      for (int i = 0; i < 6; ++i)
      {
        dpt_diff_jac_pose_0[i] = loc_3d_0_in_1_jac_pose_0[2][i] - (sampled_dpt_grad_1[0] * proj_2d_jac_pose_0[0][i] +
                                                                   sampled_dpt_grad_1[1] * proj_2d_jac_pose_0[1][i]);
        dpt_diff_jac_pose_1[i] = loc_3d_0_in_1_jac_pose_1[2][i] - (sampled_dpt_grad_1[0] * proj_2d_jac_pose_1[0][i] +
                                                                   sampled_dpt_grad_1[1] * proj_2d_jac_pose_1[1][i]);
      }

      proj_2d_jac_dpt_0[0] = fx * (rotated_location_homo_0_in_1[0] * inv_z - location_3d_0_in_1[0] * rotated_location_homo_0_in_1[2] * inv_z * inv_z);
      proj_2d_jac_dpt_0[1] = fy * (rotated_location_homo_0_in_1[1] * inv_z - location_3d_0_in_1[1] * rotated_location_homo_0_in_1[2] * inv_z * inv_z);

      const Scalar dpt_1_jac_dpt_0 = sampled_dpt_grad_1[0] * proj_2d_jac_dpt_0[0] + sampled_dpt_grad_1[1] * proj_2d_jac_dpt_0[1];

      const Scalar dpt_diff_jac_scale_0 =
          (rotated_location_homo_0_in_1[2] - dpt_1_jac_dpt_0) * dpt_0 * scale_1 / (scale_0 * sum_scale) + sampled_dpt_1 / sum_scale;
      const Scalar dpt_diff_jac_scale_1 = (-rotated_location_homo_0_in_1[2] + dpt_1_jac_dpt_0) * dpt_0 / sum_scale - sampled_dpt_1 * scale_0 / (scale_1 * sum_scale);

      const Scalar sqrt_cauchy_weight = is_dpt_positive ? within_mask * sqrt(1 / (pow(sampled_dpt_1 - location_3d_0_in_1[2], 2) + loss_param)) : 0;

#pragma unroll_completely
      for (int i = 0; i < CS; ++i)
      {
        dpt_diff_jac_code_0[i] = (rotated_location_homo_0_in_1[2] - dpt_1_jac_dpt_0) * flatten_dpt_jac_code_0[location_1d][i] * scale_0 / sum_scale;
        dpt_diff_jac_code_1[i] = -scale_1 / sum_scale * sampled_dpt_jac_code_1[i];
      }

      cauchy_sampled_dpt_diff[idx] = sqrt_cauchy_weight * (sampled_dpt_1 - location_3d_0_in_1[2]);

#pragma unroll_completely
      for (int i = 0; i < 6; ++i)
      {
        cauchy_sampled_dpt_diff_jac_pose_code_scale[idx][i] = sqrt_cauchy_weight * dpt_diff_jac_pose_0[i];
        cauchy_sampled_dpt_diff_jac_pose_code_scale[idx][6 + i] = sqrt_cauchy_weight * dpt_diff_jac_pose_1[i];
      }

#pragma unroll_completely
      for (int i = 0; i < CS; ++i)
      {
        cauchy_sampled_dpt_diff_jac_pose_code_scale[idx][12 + i] = sqrt_cauchy_weight * dpt_diff_jac_code_0[i];
        cauchy_sampled_dpt_diff_jac_pose_code_scale[idx][12 + CS + i] = sqrt_cauchy_weight * dpt_diff_jac_code_1[i];
      }

      cauchy_sampled_dpt_diff_jac_pose_code_scale[idx][12 + 2 * CS] = sqrt_cauchy_weight * dpt_diff_jac_scale_0;
      cauchy_sampled_dpt_diff_jac_pose_code_scale[idx][13 + 2 * CS] = sqrt_cauchy_weight * dpt_diff_jac_scale_1;
    }

    return;
  }

  template <typename Scalar, int CS>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void geometric_jac_error_calculate_kernel(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_valid_mask_1,
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> cauchy_sampled_dpt_diff_jac_pose_code_scale,
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> cauchy_sampled_dpt_diff,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation10,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation10,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> dpt_map_1,
          const torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> dpt_map_grad_1,
          const torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> dpt_jac_code_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> valid_mask_1,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> sampled_locations_1d_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const Scalar scale_0, const Scalar scale_1, const PinholeCamera<Scalar> camera,
          const Scalar eps, const Scalar loss_param)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sampled_locations_1d_0.size(0))
    {
      const Scalar fx = camera.fx();
      const Scalar fy = camera.fy();
      const Scalar cx = camera.u0();
      const Scalar cy = camera.v0();
      const int height = static_cast<int>(camera.height());
      const int width = static_cast<int>(camera.width());

      Scalar sampled_dpt_grad_in_1[2];
      Scalar proj_2d_jac_dpt_0[2];
      Scalar sampled_dpt_jac_code_1[CS];
      Scalar dpt_diff_jac_code_0[CS];
      Scalar dpt_diff_jac_code_1[CS];

      const int location_1d = sampled_locations_1d_0[idx];
      Scalar dpt_0 = flatten_dpt_map_bias_0[location_1d];

      for (int i = 0; i < CS; ++i)
      {
        dpt_0 += flatten_dpt_jac_code_0[location_1d][i] * code_0[i];
      }
      dpt_0 *= scale_0;

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

      // Whether or not the transformed depth is below the positive depth threshold
      // We shouldn't do depth clipping, this somehow leads to wrong warp area result previously!!!
      bool is_dpt_positive = location_3d_0_in_1[2] > eps;

      const Scalar proj_2d_x = (location_3d_0_in_1[0] / location_3d_0_in_1[2]) * fx + cx;
      const Scalar proj_2d_y = (location_3d_0_in_1[1] / location_3d_0_in_1[2]) * fy + cy;

      const int proj_2d_x_floor = int(floor(proj_2d_x));
      const int proj_2d_y_floor = int(floor(proj_2d_y));
      const int proj_2d_x_ceil = proj_2d_x_floor + 1;
      const int proj_2d_y_ceil = proj_2d_y_floor + 1;

      const Scalar lower_x_weight = (Scalar)proj_2d_x_ceil - proj_2d_x;
      const Scalar lower_y_weight = (Scalar)proj_2d_y_ceil - proj_2d_y;
      const Scalar upper_x_weight = 1 - lower_x_weight;
      const Scalar upper_y_weight = 1 - lower_y_weight;

      const Scalar nw_weight = lower_x_weight * lower_y_weight;
      const Scalar se_weight = upper_x_weight * upper_y_weight;
      const Scalar sw_weight = lower_x_weight * upper_y_weight;
      const Scalar ne_weight = upper_x_weight * lower_y_weight;

      // Zero padding bilinear interpolation
      const bool nw_bool = WITHIN_BOUNDS(proj_2d_x_floor, proj_2d_y_floor, width, height);
      const bool se_bool = WITHIN_BOUNDS(proj_2d_x_ceil, proj_2d_y_ceil, width, height);
      const bool sw_bool = WITHIN_BOUNDS(proj_2d_x_floor, proj_2d_y_ceil, width, height);
      const bool ne_bool = WITHIN_BOUNDS(proj_2d_x_ceil, proj_2d_y_floor, width, height);

      const Scalar dpt_00 = nw_bool ? dpt_map_1[proj_2d_y_floor][proj_2d_x_floor] * nw_weight : 0;
      const Scalar dpt_11 = se_bool ? dpt_map_1[proj_2d_y_ceil][proj_2d_x_ceil] * se_weight : 0;
      const Scalar dpt_01 = sw_bool ? dpt_map_1[proj_2d_y_ceil][proj_2d_x_floor] * sw_weight : 0;
      const Scalar dpt_10 = ne_bool ? dpt_map_1[proj_2d_y_floor][proj_2d_x_ceil] * ne_weight : 0;
      const Scalar sampled_dpt_in_1 = dpt_00 + dpt_11 + dpt_01 + dpt_10;

// dpt_map_grad_1: 2 x H x W
#pragma unroll_completely
      for (int i = 0; i < 2; ++i)
      {
        sampled_dpt_grad_in_1[i] =
            (nw_bool ? dpt_map_grad_1[i][proj_2d_y_floor][proj_2d_x_floor] * nw_weight : 0) +
            (se_bool ? dpt_map_grad_1[i][proj_2d_y_ceil][proj_2d_x_ceil] * se_weight : 0) +
            (sw_bool ? dpt_map_grad_1[i][proj_2d_y_ceil][proj_2d_x_floor] * sw_weight : 0) +
            (ne_bool ? dpt_map_grad_1[i][proj_2d_y_floor][proj_2d_x_ceil] * ne_weight : 0);
      }

      // Nearest interpolation
      const int proj_2d_x_round = int(round(proj_2d_x));
      const int proj_2d_y_round = int(round(proj_2d_y));

      // dpt_jac_code_1: H x W x CS
#pragma unroll_completely
      for (int i = 0; i < CS; ++i)
      {
        sampled_dpt_jac_code_1[i] = (nw_bool ? dpt_jac_code_1[proj_2d_y_floor][proj_2d_x_floor][i] * nw_weight : 0) +
                                    (se_bool ? dpt_jac_code_1[proj_2d_y_ceil][proj_2d_x_ceil][i] * se_weight : 0) +
                                    (sw_bool ? dpt_jac_code_1[proj_2d_y_ceil][proj_2d_x_floor][i] * sw_weight : 0) +
                                    (ne_bool ? dpt_jac_code_1[proj_2d_y_floor][proj_2d_x_ceil][i] * ne_weight : 0);
      }

      const Scalar within_mask = WITHIN_BOUNDS(proj_2d_x_round, proj_2d_y_round, width, height) ? valid_mask_1[proj_2d_y_round][proj_2d_x_round] : 0;

      sampled_error[idx] = is_dpt_positive ? log(1 + pow(within_mask * (sampled_dpt_in_1 - location_3d_0_in_1[2]), 2) / loss_param) : 0;
      sampled_valid_mask_1[idx] = is_dpt_positive ? within_mask : 0;

      // A.3 in "tutorial on SE3"
      const Scalar inv_z = 1 / location_3d_0_in_1[2];
      const Scalar x_z = inv_z * location_3d_0_in_1[0];
      const Scalar y_z = inv_z * location_3d_0_in_1[1];
      const Scalar proj_2d_jac_loc_3d_0_in_1[2][3] = {{fx * inv_z, 0, -fx * x_z * inv_z},
                                                      {0, fy * inv_z, -fy * y_z * inv_z}};
      // 10.3.6
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
      Scalar loc_3d_0_in_1_jac_pose_1[3][6];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        loc_3d_0_in_1_jac_pose_1[i][0] = -rotation1[0][i];
        loc_3d_0_in_1_jac_pose_1[i][1] = -rotation1[1][i];
        loc_3d_0_in_1_jac_pose_1[i][2] = -rotation1[2][i];
        loc_3d_0_in_1_jac_pose_1[i][3] = rotation1[1][i] * location_3d_0_in_world[2] - rotation1[2][i] * location_3d_0_in_world[1];
        loc_3d_0_in_1_jac_pose_1[i][4] = -rotation1[0][i] * location_3d_0_in_world[2] + rotation1[2][i] * location_3d_0_in_world[0];
        loc_3d_0_in_1_jac_pose_1[i][5] = rotation1[0][i] * location_3d_0_in_world[1] - rotation1[1][i] * location_3d_0_in_world[0];
      }

      Scalar proj_2d_jac_pose_1[2][6];
      for (int i = 0; i < 2; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          proj_2d_jac_pose_1[i][j] = proj_2d_jac_loc_3d_0_in_1[i][0] * loc_3d_0_in_1_jac_pose_1[0][j] +
                                     proj_2d_jac_loc_3d_0_in_1[i][1] * loc_3d_0_in_1_jac_pose_1[1][j] +
                                     proj_2d_jac_loc_3d_0_in_1[i][2] * loc_3d_0_in_1_jac_pose_1[2][j];
        }
      }

      // 10.3.8 in "tutorial on SE3" which includes 10.3.5
      const Scalar loc_3d_0_in_1_jac_pose_0_temp[3][6] = {{1, 0, 0, 0, location_3d_0_in_world[2], -location_3d_0_in_world[1]},
                                                          {0, 1, 0, -location_3d_0_in_world[2], 0, location_3d_0_in_world[0]},
                                                          {0, 0, 1, location_3d_0_in_world[1], -location_3d_0_in_world[0], 0}};
      Scalar loc_3d_0_in_1_jac_pose_0[3][6];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          // inverse rotation 1 here
          loc_3d_0_in_1_jac_pose_0[i][j] = rotation1[0][i] * loc_3d_0_in_1_jac_pose_0_temp[0][j] +
                                           rotation1[1][i] * loc_3d_0_in_1_jac_pose_0_temp[1][j] +
                                           rotation1[2][i] * loc_3d_0_in_1_jac_pose_0_temp[2][j];
        }
      }
      Scalar proj_2d_jac_pose_0[2][6];
      for (int i = 0; i < 2; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          proj_2d_jac_pose_0[i][j] = proj_2d_jac_loc_3d_0_in_1[i][0] * loc_3d_0_in_1_jac_pose_0[0][j] +
                                     proj_2d_jac_loc_3d_0_in_1[i][1] * loc_3d_0_in_1_jac_pose_0[1][j] +
                                     proj_2d_jac_loc_3d_0_in_1[i][2] * loc_3d_0_in_1_jac_pose_0[2][j];
        }
      }

      Scalar dpt_diff_jac_pose_0[6], dpt_diff_jac_pose_1[6];
#pragma unroll_completely
      for (int i = 0; i < 6; ++i)
      {
        dpt_diff_jac_pose_0[i] = loc_3d_0_in_1_jac_pose_0[2][i] - (sampled_dpt_grad_in_1[0] * proj_2d_jac_pose_0[0][i] +
                                                                   sampled_dpt_grad_in_1[1] * proj_2d_jac_pose_0[1][i]);
        dpt_diff_jac_pose_1[i] = loc_3d_0_in_1_jac_pose_1[2][i] - (sampled_dpt_grad_in_1[0] * proj_2d_jac_pose_1[0][i] +
                                                                   sampled_dpt_grad_in_1[1] * proj_2d_jac_pose_1[1][i]);
      }

      proj_2d_jac_dpt_0[0] = fx * (rotated_location_homo_0_in_1[0] * inv_z - location_3d_0_in_1[0] * rotated_location_homo_0_in_1[2] * inv_z * inv_z);
      proj_2d_jac_dpt_0[1] = fy * (rotated_location_homo_0_in_1[1] * inv_z - location_3d_0_in_1[1] * rotated_location_homo_0_in_1[2] * inv_z * inv_z);

      const Scalar dpt_1_jac_dpt_0 = sampled_dpt_grad_in_1[0] * proj_2d_jac_dpt_0[0] + sampled_dpt_grad_in_1[1] * proj_2d_jac_dpt_0[1];
      const Scalar scaled_rotated_location_homo_z_sub_dpt_1_jac_dpt_0 = (rotated_location_homo_0_in_1[2] - dpt_1_jac_dpt_0) * scale_0;

      const Scalar dpt_diff_jac_scale_0 = (rotated_location_homo_0_in_1[2] - dpt_1_jac_dpt_0) * dpt_0 / scale_0;
      const Scalar dpt_diff_jac_scale_1 = -sampled_dpt_in_1 / scale_1;

      const Scalar sqrt_cauchy_weight = is_dpt_positive ? within_mask * sqrt(1 / (pow(sampled_dpt_in_1 - location_3d_0_in_1[2], 2) + loss_param)) : 0;

#pragma unroll_completely
      for (int i = 0; i < CS; ++i)
      {
        dpt_diff_jac_code_0[i] = scaled_rotated_location_homo_z_sub_dpt_1_jac_dpt_0 * flatten_dpt_jac_code_0[location_1d][i];
        dpt_diff_jac_code_1[i] = -scale_1 * sampled_dpt_jac_code_1[i];
      }

      cauchy_sampled_dpt_diff[idx] = sqrt_cauchy_weight * (sampled_dpt_in_1 - location_3d_0_in_1[2]);

#pragma unroll_completely
      for (int i = 0; i < 6; ++i)
      {
        cauchy_sampled_dpt_diff_jac_pose_code_scale[idx][i] = sqrt_cauchy_weight * dpt_diff_jac_pose_0[i];
        cauchy_sampled_dpt_diff_jac_pose_code_scale[idx][6 + i] = sqrt_cauchy_weight * dpt_diff_jac_pose_1[i];
      }

#pragma unroll_completely
      for (int i = 0; i < CS; ++i)
      {
        cauchy_sampled_dpt_diff_jac_pose_code_scale[idx][12 + i] = sqrt_cauchy_weight * dpt_diff_jac_code_0[i];
        cauchy_sampled_dpt_diff_jac_pose_code_scale[idx][12 + CS + i] = sqrt_cauchy_weight * dpt_diff_jac_code_1[i];
      }

      cauchy_sampled_dpt_diff_jac_pose_code_scale[idx][12 + 2 * CS] = sqrt_cauchy_weight * dpt_diff_jac_scale_0;
      cauchy_sampled_dpt_diff_jac_pose_code_scale[idx][13 + 2 * CS] = sqrt_cauchy_weight * dpt_diff_jac_scale_1;
    }

    return;
  }

  template <int CS>
  float geometric_error_calculate_unbiased(const at::Tensor rotation, const at::Tensor translation,
                                           const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
                                           const at::Tensor code_0, const at::Tensor unscaled_dpt_map_1, const at::Tensor valid_mask_1,
                                           const at::Tensor sampled_locations_1d_0, const at::Tensor sampled_locations_homo_0,
                                           const float scale_0, const float scale_1, const PinholeCamera<float> &camera,
                                           const float eps, const float loss_param, const float weight)
  {
    const int thread_per_block = MAX_THREADS_PER_BLOCK;
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)));
    const dim3 block_size(thread_per_block);

    at::Tensor sampled_error = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor sampled_validness = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());

    AT_DISPATCH_FLOATING_TYPES(rotation.type(), "geometric_error_calculate_kernel_unbiased", ([&] {
                                 geometric_error_calculate_kernel_unbiased<float, CS><<<grid_size, block_size>>>(
                                     sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_validness.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     rotation.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     flatten_dpt_map_bias_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     flatten_dpt_jac_code_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     code_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     unscaled_dpt_map_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     valid_mask_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     sampled_locations_1d_0.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                     sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     scale_0, scale_1, camera, eps, loss_param);
                               }));

    gpuErrchk(cudaGetLastError());

    float num_inliers = torch::sum(sampled_validness).item<float>();

    if (num_inliers > 0)
    {
      return weight * torch::sum(sampled_error).item<float>() / num_inliers;
    }
    else
    {
      return weight * 10.0;
    }
  }

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
                                              const float eps, const float loss_param, const float weight)
  {
    const int thread_per_block = MAX_THREADS_PER_BLOCK;
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)));
    const dim3 block_size(thread_per_block);

    at::Tensor sampled_error = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor sampled_validness = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor cauchy_sampled_dpt_diff_jac_pose_code_scale =
        torch::zeros({sampled_locations_homo_0.size(0), 14 + 2 * CS}, sampled_locations_homo_0.options());
    at::Tensor cauchy_sampled_dpt_diff = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());

    AT_DISPATCH_FLOATING_TYPES(rotation10.type(), "geometric_jac_error_calculate_kernel_unbiased", ([&] {
                                 geometric_jac_error_calculate_kernel_unbiased<float, CS><<<grid_size, block_size>>>(
                                     sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_validness.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     cauchy_sampled_dpt_diff_jac_pose_code_scale.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     cauchy_sampled_dpt_diff.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     rotation10.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation10.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     rotation0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     rotation1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     flatten_dpt_map_bias_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     flatten_dpt_jac_code_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     code_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     unscaled_dpt_map_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     unscaled_dpt_map_grad_1.packed_accessor32<float, 3, at::RestrictPtrTraits>(), // 2 x H x W
                                     dpt_jac_code_1.packed_accessor32<float, 3, at::RestrictPtrTraits>(),          // H x W x CS
                                     valid_mask_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     sampled_locations_1d_0.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                     sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     scale_0, scale_1, camera, eps, loss_param);
                               }));

    gpuErrchk(cudaGetLastError());

    using namespace torch::indexing;
    float num_inliers = torch::sum(sampled_validness).item<float>();
    if (num_inliers > 0)
    {
      error = weight / num_inliers * torch::sum(sampled_error).item<float>();
      // (14 + 2 * CS) x (14 + 2 * CS)
      AtA = (weight / num_inliers) *
            torch::matmul(cauchy_sampled_dpt_diff_jac_pose_code_scale.permute({1, 0}), cauchy_sampled_dpt_diff_jac_pose_code_scale);
      // (14 + 2 * CS) x 1
      Atb = (weight / num_inliers) *
            torch::matmul(cauchy_sampled_dpt_diff_jac_pose_code_scale.permute({1, 0}), cauchy_sampled_dpt_diff.reshape({-1, 1}));
    }
    else
    {
      error = weight * 10.0;
      AtA = torch::zeros({14 + 2 * CS, 14 + 2 * CS}, sampled_error.options());
      Atb = torch::zeros({14 + 2 * CS, 1}, sampled_error.options());
    }

    return;
  }

  template <int CS>
  float geometric_error_calculate(const at::Tensor rotation, const at::Tensor translation,
                                  const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
                                  const at::Tensor code_0, const at::Tensor dpt_map_1, const at::Tensor valid_mask_1,
                                  const at::Tensor sampled_locations_1d_0, const at::Tensor sampled_locations_homo_0,
                                  const float scale_0, const PinholeCamera<float> &camera,
                                  const float eps, const float loss_param, const float weight)
  {
    const int thread_per_block = MAX_THREADS_PER_BLOCK;
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)));
    const dim3 block_size(thread_per_block);

    at::Tensor sampled_error = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor sampled_validness = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());

    AT_DISPATCH_FLOATING_TYPES(rotation.type(), "geometric_error_calculate_kernel", ([&] {
                                 geometric_error_calculate_kernel<float, CS><<<grid_size, block_size>>>(
                                     sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_validness.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     rotation.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     flatten_dpt_map_bias_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     flatten_dpt_jac_code_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     code_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     dpt_map_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     valid_mask_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     sampled_locations_1d_0.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                     sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     scale_0, camera, eps, loss_param);
                               }));

    gpuErrchk(cudaGetLastError());

    float num_inliers = torch::sum(sampled_validness).item<float>();

    if (num_inliers > 0)
    {
      return weight * torch::sum(sampled_error).item<float>() / num_inliers;
    }
    else
    {
      return weight * 10.0;
    }
  }

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
                                     const float eps, const float loss_param, const float weight)
  {
    const int thread_per_block = MAX_THREADS_PER_BLOCK;
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)));
    const dim3 block_size(thread_per_block);

    at::Tensor sampled_error = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor sampled_validness = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor cauchy_sampled_dpt_diff_jac_pose_code_scale =
        torch::zeros({sampled_locations_homo_0.size(0), 14 + 2 * CS}, sampled_locations_homo_0.options());
    at::Tensor cauchy_sampled_dpt_diff = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());

    AT_DISPATCH_FLOATING_TYPES(rotation10.type(), "geometric_jac_error_calculate_kernel", ([&] {
                                 geometric_jac_error_calculate_kernel<float, CS><<<grid_size, block_size>>>(
                                     sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_validness.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     cauchy_sampled_dpt_diff_jac_pose_code_scale.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     cauchy_sampled_dpt_diff.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     rotation10.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation10.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     rotation0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     rotation1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     flatten_dpt_map_bias_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     flatten_dpt_jac_code_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     code_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     dpt_map_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     dpt_map_grad_1.packed_accessor32<float, 3, at::RestrictPtrTraits>(), // 2 x H x W
                                     dpt_jac_code_1.packed_accessor32<float, 3, at::RestrictPtrTraits>(), // H x W x CS
                                     valid_mask_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     sampled_locations_1d_0.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                     sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     scale_0, scale_1, camera, eps, loss_param);
                               }));

    gpuErrchk(cudaGetLastError());

    using namespace torch::indexing;
    float num_inliers = torch::sum(sampled_validness).item<float>();
    if (num_inliers > 0)
    {
      error = weight / num_inliers * torch::sum(sampled_error).item<float>();
      // (14 + 2 * CS) x (14 + 2 * CS)
      AtA = (weight / num_inliers) *
            torch::matmul(cauchy_sampled_dpt_diff_jac_pose_code_scale.permute({1, 0}), cauchy_sampled_dpt_diff_jac_pose_code_scale);
      // (14 + 2 * CS) x 1
      Atb = (weight / num_inliers) *
            torch::matmul(cauchy_sampled_dpt_diff_jac_pose_code_scale.permute({1, 0}), cauchy_sampled_dpt_diff.reshape({-1, 1}));
    }
    else
    {
      error = weight * 10.0;
      AtA = torch::zeros({14 + 2 * CS, 14 + 2 * CS}, sampled_error.options());
      Atb = torch::zeros({14 + 2 * CS, 1}, sampled_error.options());
    }

    return;
  }

#undef WITHIN_BOUNDS

  template float geometric_error_calculate_unbiased<DF_CODE_SIZE>(
      const at::Tensor rotation, const at::Tensor translation,
      const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
      const at::Tensor code_0, const at::Tensor unscaled_dpt_map_1, const at::Tensor valid_mask_1,
      const at::Tensor sampled_locations_1d_0, const at::Tensor sampled_locations_homo_0,
      const float scale_0, const float scale_1, const PinholeCamera<float> &camera,
      const float eps, const float loss_param, const float weight);

  template float geometric_error_calculate<DF_CODE_SIZE>(
      const at::Tensor rotation, const at::Tensor translation,
      const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
      const at::Tensor code_0, const at::Tensor dpt_map_1, const at::Tensor valid_mask_1,
      const at::Tensor sampled_locations_1d_0, const at::Tensor sampled_locations_homo_0,
      const float scale_0, const PinholeCamera<float> &camera, const float eps, const float loss_param, const float weight);

  template void geometric_jac_error_calculate_unbiased<DF_CODE_SIZE>(
      at::Tensor &AtA, at::Tensor &Atb, float &error,
      const at::Tensor rotation10, const at::Tensor translation10,
      const at::Tensor rotation0, const at::Tensor translation0,
      const at::Tensor rotation1, const at::Tensor translation1,
      const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
      const at::Tensor code_0, const at::Tensor unscaled_dpt_map_1,
      const at::Tensor unscaled_dpt_map_grad_1, const at::Tensor dpt_jac_code_1, const at::Tensor valid_mask_1,
      const at::Tensor sampled_locations_1d_0, const at::Tensor sampled_locations_homo_0,
      const float scale_0, const float scale_1, const PinholeCamera<float> &camera,
      const float eps, const float loss_param, const float weight);

  template void geometric_jac_error_calculate<DF_CODE_SIZE>(
      at::Tensor &AtA, at::Tensor &Atb, float &error,
      const at::Tensor rotation10, const at::Tensor translation10,
      const at::Tensor rotation0, const at::Tensor translation0,
      const at::Tensor rotation1, const at::Tensor translation1,
      const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
      const at::Tensor code_0, const at::Tensor dpt_map_1,
      const at::Tensor dpt_map_grad_1, const at::Tensor dpt_jac_code_1, const at::Tensor valid_mask_1,
      const at::Tensor sampled_locations_1d_0, const at::Tensor sampled_locations_homo_0,
      const float scale_0, const float scale_1, const PinholeCamera<float> &camera, const float eps, const float loss_param, const float weight);
}
