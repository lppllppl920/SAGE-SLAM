#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "photometric_factor_kernels.h"
#include "camera_pyramid.h"

namespace df
{
#define MAX_THREADS_PER_BLOCK 512

#define WITHIN_BOUNDS(x, y, W, H) (x >= 0 && x < W && y >= 0 && y < H)

#define gpuErrchk(ans)                          \
  do                                            \
  {                                             \
    gpuAssert((ans), __FILE__, __LINE__, true); \
  } while (0);
  inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
  {
    if (code != cudaSuccess)
    {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort)
        exit(code);
    }
  }

  template <typename Scalar, int CS, int FS>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void photometric_jac_error_calculate_kernel(
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_valid_mask_1,
          torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> sampled_feat_diff,                     // L x N x C_feat
          torch::PackedTensorAccessor32<Scalar, 4, at::RestrictPtrTraits> sampled_feat_diff_jac_pose_code_scale, // L x N x C_feat x (13 + C_code)
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation10,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation10,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_dpts_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> valid_mask_1,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> sampled_locations_1d_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_2d_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> feat_map_0_pyr, // C_feat x ( N_0 + N_1 + ... + N_(L-1) )
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> feat_map_1_pyr,
          const torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> feat_map_1_grad_pyr, // 2 x C_feat x ( N_0 + N_1 + ... + N_(L-1) )
          const PinholeCamera<float> *__restrict__ cam_pyr,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> level_offsets,
          const Scalar scale_0,
          const Scalar eps)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int level = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < sampled_locations_homo_0.size(0))
    {
      const int ori_height = static_cast<int>(cam_pyr[0].height());
      const int ori_width = static_cast<int>(cam_pyr[0].width());
      const Scalar ori_fx = cam_pyr[0].fx();
      const Scalar ori_fy = cam_pyr[0].fy();
      const Scalar ori_cx = cam_pyr[0].u0();
      const Scalar ori_cy = cam_pyr[0].v0();

      const int height = static_cast<int>(cam_pyr[level].height());
      const int width = static_cast<int>(cam_pyr[level].width());
      const Scalar fx = cam_pyr[level].fx();
      const Scalar fy = cam_pyr[level].fy();

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
        location_3d_0_in_1[i] = sampled_dpts_0[idx] * rotated_location_homo_0_in_1[i] + translation10[i];
      }

      // Whether or not the transformed depth is below the positive depth threshold
      bool is_dpt_positive = location_3d_0_in_1[2] > eps;

      // const Scalar sampled_location_2d_0[2] = {(sampled_locations_2d_0[idx][0] + static_cast<Scalar>(0.5)) * fx / ori_fx - static_cast<Scalar>(0.5),
      //                                          (sampled_locations_2d_0[idx][1] + static_cast<Scalar>(0.5)) * fy / ori_fy - static_cast<Scalar>(0.5)};

      const Scalar sampled_location_2d_0[2] =
          {(sampled_locations_homo_0[idx][0] * ori_fx + ori_cx + static_cast<Scalar>(0.5)) * fx / ori_fx - static_cast<Scalar>(0.5),
           (sampled_locations_homo_0[idx][1] * ori_fy + ori_cy + static_cast<Scalar>(0.5)) * fy / ori_fy - static_cast<Scalar>(0.5)};

      // Bilinear sampling on feat_map_0
      const int sampled_location_2d_0_floor[2] = {int(floor(sampled_location_2d_0[0])), int(floor(sampled_location_2d_0[1]))};
      const int sampled_location_2d_0_ceil[2] = {sampled_location_2d_0_floor[0] + 1, sampled_location_2d_0_floor[1] + 1};

      const Scalar lower_weight_0[2] = {(Scalar)sampled_location_2d_0_ceil[0] - sampled_location_2d_0[0],
                                        (Scalar)sampled_location_2d_0_ceil[1] - sampled_location_2d_0[1]};
      const Scalar upper_weight_0[2] = {1 - lower_weight_0[0], 1 - lower_weight_0[1]};

      const Scalar nw_weight_0 = lower_weight_0[0] * lower_weight_0[1];
      const Scalar se_weight_0 = upper_weight_0[0] * upper_weight_0[1];
      const Scalar sw_weight_0 = lower_weight_0[0] * upper_weight_0[1];
      const Scalar ne_weight_0 = upper_weight_0[0] * lower_weight_0[1];

      Scalar sampled_feat_0[FS];

#pragma unroll_completely
      for (int i = 0; i < FS; ++i)
      {
        sampled_feat_0[i] = (WITHIN_BOUNDS(sampled_location_2d_0_floor[0],
                                           sampled_location_2d_0_floor[1], width, height)
                                 ? feat_map_0_pyr[i][level_offsets[level] + sampled_location_2d_0_floor[1] * width + sampled_location_2d_0_floor[0]] * nw_weight_0
                                 : 0) +
                            (WITHIN_BOUNDS(sampled_location_2d_0_ceil[0],
                                           sampled_location_2d_0_ceil[1], width, height)
                                 ? feat_map_0_pyr[i][level_offsets[level] + sampled_location_2d_0_ceil[1] * width + sampled_location_2d_0_ceil[0]] * se_weight_0
                                 : 0) +
                            (WITHIN_BOUNDS(sampled_location_2d_0_floor[0],
                                           sampled_location_2d_0_ceil[1], width, height)
                                 ? feat_map_0_pyr[i][level_offsets[level] + sampled_location_2d_0_ceil[1] * width + sampled_location_2d_0_floor[0]] * sw_weight_0
                                 : 0) +
                            (WITHIN_BOUNDS(sampled_location_2d_0_ceil[0],
                                           sampled_location_2d_0_floor[1], width, height)
                                 ? feat_map_0_pyr[i][level_offsets[level] + sampled_location_2d_0_floor[1] * width + sampled_location_2d_0_ceil[0]] * ne_weight_0
                                 : 0);
      }

      // Bilinear sampling on feat_map_1
      const Scalar proj_location_2d_1[2] =
          {((location_3d_0_in_1[0] / location_3d_0_in_1[2]) * ori_fx + ori_cx + static_cast<Scalar>(0.5)) * fx / ori_fx - static_cast<Scalar>(0.5),
           ((location_3d_0_in_1[1] / location_3d_0_in_1[2]) * ori_fy + ori_cy + static_cast<Scalar>(0.5)) * fy / ori_fy - static_cast<Scalar>(0.5)};

      const int proj_location_2d_1_floor[2] = {int(floor(proj_location_2d_1[0])), int(floor(proj_location_2d_1[1]))};
      const int proj_location_2d_1_ceil[2] = {proj_location_2d_1_floor[0] + 1, proj_location_2d_1_floor[1] + 1};

      const Scalar lower_weight_1[2] = {(Scalar)proj_location_2d_1_ceil[0] - proj_location_2d_1[0],
                                        (Scalar)proj_location_2d_1_ceil[1] - proj_location_2d_1[1]};
      const Scalar upper_weight_1[2] = {1 - lower_weight_1[0], 1 - lower_weight_1[1]};

      const Scalar nw_weight_1 = lower_weight_1[0] * lower_weight_1[1];
      const Scalar se_weight_1 = upper_weight_1[0] * upper_weight_1[1];
      const Scalar sw_weight_1 = lower_weight_1[0] * upper_weight_1[1];
      const Scalar ne_weight_1 = upper_weight_1[0] * lower_weight_1[1];

      // Nearest interpolation on fine-resolution mask
      const int mask_proj_location_2d_1_round[2] =
          {(int)round((location_3d_0_in_1[0] / location_3d_0_in_1[2]) * ori_fx + ori_cx),
           (int)round((location_3d_0_in_1[1] / location_3d_0_in_1[2]) * ori_fy + ori_cy)};
      const Scalar within_mask =
          WITHIN_BOUNDS(mask_proj_location_2d_1_round[0],
                        mask_proj_location_2d_1_round[1], ori_width, ori_height)
              ? valid_mask_1[mask_proj_location_2d_1_round[1]][mask_proj_location_2d_1_round[0]]
              : 0;

      Scalar sampled_feat_1[FS];

#pragma unroll_completely
      for (int i = 0; i < FS; ++i)
      {
        sampled_feat_1[i] = (WITHIN_BOUNDS(proj_location_2d_1_floor[0],
                                           proj_location_2d_1_floor[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_floor[1] * width + proj_location_2d_1_floor[0]] * nw_weight_1
                                 : 0) +
                            (WITHIN_BOUNDS(proj_location_2d_1_ceil[0],
                                           proj_location_2d_1_ceil[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_ceil[1] * width + proj_location_2d_1_ceil[0]] * se_weight_1
                                 : 0) +
                            (WITHIN_BOUNDS(proj_location_2d_1_floor[0],
                                           proj_location_2d_1_ceil[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_ceil[1] * width + proj_location_2d_1_floor[0]] * sw_weight_1
                                 : 0) +
                            (WITHIN_BOUNDS(proj_location_2d_1_ceil[0],
                                           proj_location_2d_1_floor[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_floor[1] * width + proj_location_2d_1_ceil[0]] * ne_weight_1
                                 : 0);
      }

      Scalar sampled_feat_grad_1[FS][2];
#pragma unroll_completely
      for (int i = 0; i < FS; ++i)
      {
        for (int j = 0; j < 2; ++j)
        {
          if (is_dpt_positive)
          {
            sampled_feat_grad_1[i][j] =
                within_mask * ((WITHIN_BOUNDS(proj_location_2d_1_floor[0],
                                              proj_location_2d_1_floor[1], width, height)
                                    ? feat_map_1_grad_pyr[j][i][level_offsets[level] + proj_location_2d_1_floor[1] * width + proj_location_2d_1_floor[0]] * nw_weight_1
                                    : 0) +
                               (WITHIN_BOUNDS(proj_location_2d_1_ceil[0],
                                              proj_location_2d_1_ceil[1], width, height)
                                    ? feat_map_1_grad_pyr[j][i][level_offsets[level] + proj_location_2d_1_ceil[1] * width + proj_location_2d_1_ceil[0]] * se_weight_1
                                    : 0) +
                               (WITHIN_BOUNDS(proj_location_2d_1_floor[0],
                                              proj_location_2d_1_ceil[1], width, height)
                                    ? feat_map_1_grad_pyr[j][i][level_offsets[level] + proj_location_2d_1_ceil[1] * width + proj_location_2d_1_floor[0]] * sw_weight_1
                                    : 0) +
                               (WITHIN_BOUNDS(proj_location_2d_1_ceil[0],
                                              proj_location_2d_1_floor[1], width, height)
                                    ? feat_map_1_grad_pyr[j][i][level_offsets[level] + proj_location_2d_1_floor[1] * width + proj_location_2d_1_ceil[0]] * ne_weight_1
                                    : 0));
          }
          else
          {
            sampled_feat_grad_1[i][j] = 0;
          }
        }
      }

      Scalar feat_error = 0;

      for (int i = 0; i < FS; ++i)
      {
        feat_error += is_dpt_positive ? within_mask * pow(sampled_feat_0[i] - sampled_feat_1[i], 2) : 0;
      }

#pragma unroll_completely
      for (int i = 0; i < FS; ++i)
      {
        sampled_feat_diff[level][idx][i] = is_dpt_positive ? within_mask * (sampled_feat_0[i] - sampled_feat_1[i]) : 0;
      }

      sampled_valid_mask_1[level][idx] = is_dpt_positive ? within_mask : 0;
      sampled_error[level][idx] = feat_error;

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
        location_3d_0_in_world[i] = sampled_dpts_0[idx] * (rotation0[i][0] * sampled_locations_homo_0[idx][0] +
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
#pragma unroll_completely
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
#pragma unroll_completely
      for (int i = 0; i < 2; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          proj_2d_jac_pose_0[i][j] = proj_2d_jac_loc_3d_0_in_1[i][0] * loc_3d_0_in_1_jac_pose_0[0][j] +
                                     proj_2d_jac_loc_3d_0_in_1[i][1] * loc_3d_0_in_1_jac_pose_0[1][j] +
                                     proj_2d_jac_loc_3d_0_in_1[i][2] * loc_3d_0_in_1_jac_pose_0[2][j];
        }
      }

      Scalar feat_jac_pose_0[FS][6];
      Scalar feat_jac_pose_1[FS][6];

#pragma unroll_completely
      for (int i = 0; i < FS; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          feat_jac_pose_0[i][j] = sampled_feat_grad_1[i][0] * proj_2d_jac_pose_0[0][j] + sampled_feat_grad_1[i][1] * proj_2d_jac_pose_0[1][j];
          feat_jac_pose_1[i][j] = sampled_feat_grad_1[i][0] * proj_2d_jac_pose_1[0][j] + sampled_feat_grad_1[i][1] * proj_2d_jac_pose_1[1][j];
        }
      }

      Scalar proj_2d_jac_dpt_0[2] = {fx * (rotated_location_homo_0_in_1[0] * inv_z - location_3d_0_in_1[0] * rotated_location_homo_0_in_1[2] * inv_z * inv_z),
                                     fy * (rotated_location_homo_0_in_1[1] * inv_z - location_3d_0_in_1[1] * rotated_location_homo_0_in_1[2] * inv_z * inv_z)};
      Scalar proj_2d_jac_code_0[2][CS];

#pragma unroll_completely
      for (int i = 0; i < CS; ++i)
      {
        proj_2d_jac_code_0[0][i] = proj_2d_jac_dpt_0[0] * scale_0 * flatten_dpt_jac_code_0[sampled_locations_1d_0[idx]][i];
        proj_2d_jac_code_0[1][i] = proj_2d_jac_dpt_0[1] * scale_0 * flatten_dpt_jac_code_0[sampled_locations_1d_0[idx]][i];
      }

      Scalar proj_2d_jac_scale_0[2] = {proj_2d_jac_dpt_0[0] * sampled_dpts_0[idx] / scale_0, proj_2d_jac_dpt_0[1] * sampled_dpts_0[idx] / scale_0};

      Scalar feat_jac_code_0[FS][CS];
      Scalar feat_jac_scale_0[FS];

#pragma unroll_completely
      for (int i = 0; i < FS; ++i)
      {
        for (int j = 0; j < CS; ++j)
        {
          feat_jac_code_0[i][j] = sampled_feat_grad_1[i][0] * proj_2d_jac_code_0[0][j] + sampled_feat_grad_1[i][1] * proj_2d_jac_code_0[1][j];
        }
        feat_jac_scale_0[i] = sampled_feat_grad_1[i][0] * proj_2d_jac_scale_0[0] + sampled_feat_grad_1[i][1] * proj_2d_jac_scale_0[1];
      }

#pragma unroll_completely
      for (int i = 0; i < FS; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          sampled_feat_diff_jac_pose_code_scale[level][idx][i][j] = feat_jac_pose_0[i][j];
          sampled_feat_diff_jac_pose_code_scale[level][idx][i][6 + j] = feat_jac_pose_1[i][j];
        }

        for (int j = 0; j < CS; ++j)
        {
          sampled_feat_diff_jac_pose_code_scale[level][idx][i][12 + j] = feat_jac_code_0[i][j];
        }
        sampled_feat_diff_jac_pose_code_scale[level][idx][i][12 + CS] = feat_jac_scale_0[i];
      }
    }

    return;
  }

  template <typename Scalar, int FS>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void photometric_error_calculate_kernel(
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_valid_mask_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_dpts_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> valid_mask_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_2d_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> feat_map_0_pyr, // C_feat x ( N_0 + N_1 + ... + N_(L-1) )
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> feat_map_1_pyr,
          const PinholeCamera<float> *__restrict__ cam_pyr,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> level_offsets,
          const Scalar eps)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int level = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < sampled_locations_homo_0.size(0))
    {
      const int ori_height = static_cast<int>(cam_pyr[0].height());
      const int ori_width = static_cast<int>(cam_pyr[0].width());
      const Scalar ori_fx = cam_pyr[0].fx();
      const Scalar ori_fy = cam_pyr[0].fy();
      const Scalar ori_cx = cam_pyr[0].u0();
      const Scalar ori_cy = cam_pyr[0].v0();

      const int height = static_cast<int>(cam_pyr[level].height());
      const int width = static_cast<int>(cam_pyr[level].width());
      const Scalar fx = cam_pyr[level].fx();
      const Scalar fy = cam_pyr[level].fy();

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
      // Whether or not the transformed depth is below the positive depth threshold
      bool is_dpt_positive = location_3d_0_in_1[2] > eps;

      const Scalar sampled_location_2d_0[2] = {(sampled_locations_2d_0[idx][0] + static_cast<Scalar>(0.5)) * fx / ori_fx - static_cast<Scalar>(0.5),
                                               (sampled_locations_2d_0[idx][1] + static_cast<Scalar>(0.5)) * fy / ori_fy - static_cast<Scalar>(0.5)};

      // Bilinear sampling on feat_map_0
      const int sampled_location_2d_0_floor[2] = {int(floor(sampled_location_2d_0[0])), int(floor(sampled_location_2d_0[1]))};
      const int sampled_location_2d_0_ceil[2] = {sampled_location_2d_0_floor[0] + 1, sampled_location_2d_0_floor[1] + 1};

      const Scalar lower_weight_0[2] = {(Scalar)sampled_location_2d_0_ceil[0] - sampled_location_2d_0[0],
                                        (Scalar)sampled_location_2d_0_ceil[1] - sampled_location_2d_0[1]};
      const Scalar upper_weight_0[2] = {1 - lower_weight_0[0], 1 - lower_weight_0[1]};

      const Scalar nw_weight_0 = lower_weight_0[0] * lower_weight_0[1];
      const Scalar se_weight_0 = upper_weight_0[0] * upper_weight_0[1];
      const Scalar sw_weight_0 = lower_weight_0[0] * upper_weight_0[1];
      const Scalar ne_weight_0 = upper_weight_0[0] * lower_weight_0[1];

      Scalar sampled_feat_0[FS];
#pragma unroll_completely
      for (int i = 0; i < FS; ++i)
      {
        sampled_feat_0[i] = (WITHIN_BOUNDS(sampled_location_2d_0_floor[0],
                                           sampled_location_2d_0_floor[1], width, height)
                                 ? feat_map_0_pyr[i][level_offsets[level] + sampled_location_2d_0_floor[1] * width + sampled_location_2d_0_floor[0]] * nw_weight_0
                                 : 0) +
                            (WITHIN_BOUNDS(sampled_location_2d_0_ceil[0],
                                           sampled_location_2d_0_ceil[1], width, height)
                                 ? feat_map_0_pyr[i][level_offsets[level] + sampled_location_2d_0_ceil[1] * width + sampled_location_2d_0_ceil[0]] * se_weight_0
                                 : 0) +
                            (WITHIN_BOUNDS(sampled_location_2d_0_floor[0],
                                           sampled_location_2d_0_ceil[1], width, height)
                                 ? feat_map_0_pyr[i][level_offsets[level] + sampled_location_2d_0_ceil[1] * width + sampled_location_2d_0_floor[0]] * sw_weight_0
                                 : 0) +
                            (WITHIN_BOUNDS(sampled_location_2d_0_ceil[0],
                                           sampled_location_2d_0_floor[1], width, height)
                                 ? feat_map_0_pyr[i][level_offsets[level] + sampled_location_2d_0_floor[1] * width + sampled_location_2d_0_ceil[0]] * ne_weight_0
                                 : 0);
      }

      // Bilinear sampling on feat_map_1
      const Scalar proj_location_2d_1[2] =
          {((location_3d_0_in_1[0] / location_3d_0_in_1[2]) * ori_fx + ori_cx + static_cast<Scalar>(0.5)) * fx / ori_fx - static_cast<Scalar>(0.5),
           ((location_3d_0_in_1[1] / location_3d_0_in_1[2]) * ori_fy + ori_cy + static_cast<Scalar>(0.5)) * fy / ori_fy - static_cast<Scalar>(0.5)};

      const int proj_location_2d_1_floor[2] = {int(floor(proj_location_2d_1[0])), int(floor(proj_location_2d_1[1]))};
      const int proj_location_2d_1_ceil[2] = {proj_location_2d_1_floor[0] + 1, proj_location_2d_1_floor[1] + 1};

      const Scalar lower_weight_1[2] = {(Scalar)proj_location_2d_1_ceil[0] - proj_location_2d_1[0],
                                        (Scalar)proj_location_2d_1_ceil[1] - proj_location_2d_1[1]};
      const Scalar upper_weight_1[2] = {1 - lower_weight_1[0], 1 - lower_weight_1[1]};

      const Scalar nw_weight_1 = lower_weight_1[0] * lower_weight_1[1];
      const Scalar se_weight_1 = upper_weight_1[0] * upper_weight_1[1];
      const Scalar sw_weight_1 = lower_weight_1[0] * upper_weight_1[1];
      const Scalar ne_weight_1 = upper_weight_1[0] * lower_weight_1[1];

      // Nearest interpolation on fine-resolution mask
      const int mask_proj_location_2d_1_round[2] =
          {(int)round((location_3d_0_in_1[0] / location_3d_0_in_1[2]) * ori_fx + ori_cx),
           (int)round((location_3d_0_in_1[1] / location_3d_0_in_1[2]) * ori_fy + ori_cy)};
      const Scalar within_mask =
          WITHIN_BOUNDS(mask_proj_location_2d_1_round[0],
                        mask_proj_location_2d_1_round[1], ori_width, ori_height)
              ? valid_mask_1[mask_proj_location_2d_1_round[1]][mask_proj_location_2d_1_round[0]]
              : 0;

      Scalar sampled_feat_1[FS];
#pragma unroll_completely
      for (int i = 0; i < FS; ++i)
      {
        sampled_feat_1[i] = (WITHIN_BOUNDS(proj_location_2d_1_floor[0],
                                           proj_location_2d_1_floor[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_floor[1] * width + proj_location_2d_1_floor[0]] * nw_weight_1
                                 : 0) +
                            (WITHIN_BOUNDS(proj_location_2d_1_ceil[0],
                                           proj_location_2d_1_ceil[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_ceil[1] * width + proj_location_2d_1_ceil[0]] * se_weight_1
                                 : 0) +
                            (WITHIN_BOUNDS(proj_location_2d_1_floor[0],
                                           proj_location_2d_1_ceil[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_ceil[1] * width + proj_location_2d_1_floor[0]] * sw_weight_1
                                 : 0) +
                            (WITHIN_BOUNDS(proj_location_2d_1_ceil[0],
                                           proj_location_2d_1_floor[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_floor[1] * width + proj_location_2d_1_ceil[0]] * ne_weight_1
                                 : 0);
      }

      Scalar feat_error = 0;

      for (int i = 0; i < FS; ++i)
      {
        feat_error += is_dpt_positive ? within_mask * pow(sampled_feat_1[i] - sampled_feat_0[i], 2) : 0;
      }

      sampled_valid_mask_1[level][idx] = is_dpt_positive ? within_mask : 0;
      sampled_error[level][idx] = feat_error;
    }

    return;
  }

  template <typename Scalar, int FS>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void tracker_photo_jac_error_calculate_kernel(
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_valid_mask_1,
          torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> sampled_feat_diff,              // L x N x C_feat
          torch::PackedTensorAccessor32<Scalar, 4, at::RestrictPtrTraits> sampled_feat_diff_jac_rel_pose, // L x N x C_feat x 6
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_dpts_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> valid_mask_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> cat_sampled_features_0, // L x N x C_feat
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> feat_map_1_pyr,         // C_feat x ( N_0 + N_1 + ... + N_(L-1) )
          const torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> feat_map_1_grad_pyr,    // 2 x C_feat x ( N_0 + N_1 + ... + N_(L-1) )
          const PinholeCamera<float> *__restrict__ cam_pyr,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> level_offsets,
          const Scalar eps)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int level = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < sampled_locations_homo_0.size(0))
    {
      const int ori_height = static_cast<int>(cam_pyr[0].height());
      const int ori_width = static_cast<int>(cam_pyr[0].width());
      const Scalar ori_fx = cam_pyr[0].fx();
      const Scalar ori_fy = cam_pyr[0].fy();
      const Scalar ori_cx = cam_pyr[0].u0();
      const Scalar ori_cy = cam_pyr[0].v0();

      const int height = static_cast<int>(cam_pyr[level].height());
      const int width = static_cast<int>(cam_pyr[level].width());
      const Scalar fx = cam_pyr[level].fx();
      const Scalar fy = cam_pyr[level].fy();

      const Scalar location_3d_in_1_x = sampled_dpts_0[idx] *
                                            (rotation[0][0] * sampled_locations_homo_0[idx][0] +
                                             rotation[0][1] * sampled_locations_homo_0[idx][1] +
                                             rotation[0][2] * sampled_locations_homo_0[idx][2]) +
                                        translation[0];
      const Scalar location_3d_in_1_y = sampled_dpts_0[idx] *
                                            (rotation[1][0] * sampled_locations_homo_0[idx][0] +
                                             rotation[1][1] * sampled_locations_homo_0[idx][1] +
                                             rotation[1][2] * sampled_locations_homo_0[idx][2]) +
                                        translation[1];
      Scalar location_3d_in_1_z = sampled_dpts_0[idx] *
                                      (rotation[2][0] * sampled_locations_homo_0[idx][0] +
                                       rotation[2][1] * sampled_locations_homo_0[idx][1] +
                                       rotation[2][2] * sampled_locations_homo_0[idx][2]) +
                                  translation[2];

      // Whether or not the transformed depth is below the positive depth threshold
      bool is_dpt_positive = location_3d_in_1_z > eps;

      // Bilinear sampling on feat_map_1
      const Scalar proj_location_2d_1[2] =
          {((location_3d_in_1_x / location_3d_in_1_z) * ori_fx + ori_cx + static_cast<Scalar>(0.5)) * fx / ori_fx - static_cast<Scalar>(0.5),
           ((location_3d_in_1_y / location_3d_in_1_z) * ori_fy + ori_cy + static_cast<Scalar>(0.5)) * fy / ori_fy - static_cast<Scalar>(0.5)};

      const int proj_location_2d_1_floor[2] = {int(floor(proj_location_2d_1[0])), int(floor(proj_location_2d_1[1]))};
      const int proj_location_2d_1_ceil[2] = {proj_location_2d_1_floor[0] + 1, proj_location_2d_1_floor[1] + 1};

      const Scalar lower_weight_1[2] = {(Scalar)proj_location_2d_1_ceil[0] - proj_location_2d_1[0],
                                        (Scalar)proj_location_2d_1_ceil[1] - proj_location_2d_1[1]};
      const Scalar upper_weight_1[2] = {1 - lower_weight_1[0], 1 - lower_weight_1[1]};

      const Scalar nw_weight_1 = lower_weight_1[0] * lower_weight_1[1];
      const Scalar se_weight_1 = upper_weight_1[0] * upper_weight_1[1];
      const Scalar sw_weight_1 = lower_weight_1[0] * upper_weight_1[1];
      const Scalar ne_weight_1 = upper_weight_1[0] * lower_weight_1[1];

      // Nearest interpolation on fine-resolution mask
      const int mask_proj_location_2d_1_round[2] =
          {(int)round((location_3d_in_1_x / location_3d_in_1_z) * ori_fx + ori_cx),
           (int)round((location_3d_in_1_y / location_3d_in_1_z) * ori_fy + ori_cy)};
      const Scalar within_mask =
          WITHIN_BOUNDS(mask_proj_location_2d_1_round[0],
                        mask_proj_location_2d_1_round[1], ori_width, ori_height)
              ? valid_mask_1[mask_proj_location_2d_1_round[1]][mask_proj_location_2d_1_round[0]]
              : 0;

      Scalar sampled_feat_1[FS];
#pragma unroll_completely
      for (int i = 0; i < FS; ++i)
      {
        sampled_feat_1[i] = (WITHIN_BOUNDS(proj_location_2d_1_floor[0],
                                           proj_location_2d_1_floor[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_floor[1] * width + proj_location_2d_1_floor[0]] * nw_weight_1
                                 : 0) +
                            (WITHIN_BOUNDS(proj_location_2d_1_ceil[0],
                                           proj_location_2d_1_ceil[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_ceil[1] * width + proj_location_2d_1_ceil[0]] * se_weight_1
                                 : 0) +
                            (WITHIN_BOUNDS(proj_location_2d_1_floor[0],
                                           proj_location_2d_1_ceil[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_ceil[1] * width + proj_location_2d_1_floor[0]] * sw_weight_1
                                 : 0) +
                            (WITHIN_BOUNDS(proj_location_2d_1_ceil[0],
                                           proj_location_2d_1_floor[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_floor[1] * width + proj_location_2d_1_ceil[0]] * ne_weight_1
                                 : 0);
      }

      Scalar sampled_feat_grad_1[FS][2];
#pragma unroll_completely
      for (int i = 0; i < FS; ++i)
      {
        for (int j = 0; j < 2; ++j)
        {
          if (is_dpt_positive)
          {
            sampled_feat_grad_1[i][j] =
                within_mask * ((WITHIN_BOUNDS(proj_location_2d_1_floor[0],
                                              proj_location_2d_1_floor[1], width, height)
                                    ? feat_map_1_grad_pyr[j][i][level_offsets[level] + proj_location_2d_1_floor[1] * width + proj_location_2d_1_floor[0]] * nw_weight_1
                                    : 0) +
                               (WITHIN_BOUNDS(proj_location_2d_1_ceil[0],
                                              proj_location_2d_1_ceil[1], width, height)
                                    ? feat_map_1_grad_pyr[j][i][level_offsets[level] + proj_location_2d_1_ceil[1] * width + proj_location_2d_1_ceil[0]] * se_weight_1
                                    : 0) +
                               (WITHIN_BOUNDS(proj_location_2d_1_floor[0],
                                              proj_location_2d_1_ceil[1], width, height)
                                    ? feat_map_1_grad_pyr[j][i][level_offsets[level] + proj_location_2d_1_ceil[1] * width + proj_location_2d_1_floor[0]] * sw_weight_1
                                    : 0) +
                               (WITHIN_BOUNDS(proj_location_2d_1_ceil[0],
                                              proj_location_2d_1_floor[1], width, height)
                                    ? feat_map_1_grad_pyr[j][i][level_offsets[level] + proj_location_2d_1_floor[1] * width + proj_location_2d_1_ceil[0]] * ne_weight_1
                                    : 0));
          }
          else
          {
            sampled_feat_grad_1[i][j] = 0;
          }
        }
      }

      Scalar feat_error = 0;

      for (int i = 0; i < FS; ++i)
      {
        feat_error += is_dpt_positive ? within_mask * pow(cat_sampled_features_0[level][idx][i] - sampled_feat_1[i], 2) : 0;
      }

#pragma unroll_completely
      for (int i = 0; i < FS; ++i)
      {
        sampled_feat_diff[level][idx][i] = is_dpt_positive ? within_mask * (cat_sampled_features_0[level][idx][i] - sampled_feat_1[i]) : 0;
      }

      sampled_valid_mask_1[level][idx] = is_dpt_positive ? within_mask : 0;
      sampled_error[level][idx] = feat_error;

      const Scalar inv_z = 1 / location_3d_in_1_z;
      const Scalar x_z = inv_z * location_3d_in_1_x;
      const Scalar y_z = inv_z * location_3d_in_1_y;
      const Scalar proj_2d_jac_rel_pose[2][6] = {{fx * inv_z, 0, -fx * x_z * inv_z, -fx * x_z * y_z, fx * (1 + x_z * x_z), -fx * y_z},
                                                 {0, fy * inv_z, -fy * y_z * inv_z, -fy * (1 + y_z * y_z), fy * x_z * y_z, fy * x_z}};

#pragma unroll_completely
      for (int i = 0; i < FS; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          sampled_feat_diff_jac_rel_pose[level][idx][i][j] =
              sampled_feat_grad_1[i][0] * proj_2d_jac_rel_pose[0][j] + sampled_feat_grad_1[i][1] * proj_2d_jac_rel_pose[1][j];
        }
      }
    }

    return;
  }

  template <typename Scalar, int FS>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void tracker_photo_jac_error_calculate_with_scale_kernel(
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_valid_mask_1,
          torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> sampled_feat_diff,                      // L x N x C_feat
          torch::PackedTensorAccessor32<Scalar, 4, at::RestrictPtrTraits> sampled_feat_diff_jac_rel_pose_scale_0, // L x N x C_feat x 7
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_dpts_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> valid_mask_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> cat_sampled_features_0, // L x N x C_feat
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> feat_map_1_pyr,         // C_feat x ( N_0 + N_1 + ... + N_(L-1) )
          const torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> feat_map_1_grad_pyr,    // 2 x C_feat x ( N_0 + N_1 + ... + N_(L-1) )
          const PinholeCamera<float> *__restrict__ cam_pyr,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> level_offsets,
          const Scalar scale_0,
          const Scalar eps)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int level = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < sampled_locations_homo_0.size(0))
    {
      const int ori_height = static_cast<int>(cam_pyr[0].height());
      const int ori_width = static_cast<int>(cam_pyr[0].width());
      const Scalar ori_fx = cam_pyr[0].fx();
      const Scalar ori_fy = cam_pyr[0].fy();
      const Scalar ori_cx = cam_pyr[0].u0();
      const Scalar ori_cy = cam_pyr[0].v0();

      const int height = static_cast<int>(cam_pyr[level].height());
      const int width = static_cast<int>(cam_pyr[level].width());
      const Scalar fx = cam_pyr[level].fx();
      const Scalar fy = cam_pyr[level].fy();

      Scalar rotated_location_homo_0_in_1[3];
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        rotated_location_homo_0_in_1[i] = rotation[i][0] * sampled_locations_homo_0[idx][0] +
                                          rotation[i][1] * sampled_locations_homo_0[idx][1] +
                                          rotation[i][2] * sampled_locations_homo_0[idx][2];
      }

      const Scalar location_3d_in_1_x = sampled_dpts_0[idx] * rotated_location_homo_0_in_1[0] + translation[0];
      const Scalar location_3d_in_1_y = sampled_dpts_0[idx] * rotated_location_homo_0_in_1[1] + translation[1];
      Scalar location_3d_in_1_z = sampled_dpts_0[idx] * rotated_location_homo_0_in_1[2] + translation[2];

      // Whether or not the transformed depth is below the positive depth threshold
      bool is_dpt_positive = location_3d_in_1_z > eps;

      // Bilinear sampling on feat_map_1
      const Scalar proj_location_2d_1[2] =
          {((location_3d_in_1_x / location_3d_in_1_z) * ori_fx + ori_cx + static_cast<Scalar>(0.5)) * fx / ori_fx - static_cast<Scalar>(0.5),
           ((location_3d_in_1_y / location_3d_in_1_z) * ori_fy + ori_cy + static_cast<Scalar>(0.5)) * fy / ori_fy - static_cast<Scalar>(0.5)};

      const int proj_location_2d_1_floor[2] = {int(floor(proj_location_2d_1[0])), int(floor(proj_location_2d_1[1]))};
      const int proj_location_2d_1_ceil[2] = {proj_location_2d_1_floor[0] + 1, proj_location_2d_1_floor[1] + 1};

      const Scalar lower_weight_1[2] = {(Scalar)proj_location_2d_1_ceil[0] - proj_location_2d_1[0],
                                        (Scalar)proj_location_2d_1_ceil[1] - proj_location_2d_1[1]};
      const Scalar upper_weight_1[2] = {1 - lower_weight_1[0], 1 - lower_weight_1[1]};

      const Scalar nw_weight_1 = lower_weight_1[0] * lower_weight_1[1];
      const Scalar se_weight_1 = upper_weight_1[0] * upper_weight_1[1];
      const Scalar sw_weight_1 = lower_weight_1[0] * upper_weight_1[1];
      const Scalar ne_weight_1 = upper_weight_1[0] * lower_weight_1[1];

      // Nearest interpolation on fine-resolution mask
      const int mask_proj_location_2d_1_round[2] =
          {(int)round((location_3d_in_1_x / location_3d_in_1_z) * ori_fx + ori_cx),
           (int)round((location_3d_in_1_y / location_3d_in_1_z) * ori_fy + ori_cy)};
      const Scalar within_mask =
          WITHIN_BOUNDS(mask_proj_location_2d_1_round[0],
                        mask_proj_location_2d_1_round[1], ori_width, ori_height)
              ? valid_mask_1[mask_proj_location_2d_1_round[1]][mask_proj_location_2d_1_round[0]]
              : 0;

      Scalar sampled_feat_1[FS];
#pragma unroll_completely
      for (int i = 0; i < FS; ++i)
      {
        sampled_feat_1[i] = (WITHIN_BOUNDS(proj_location_2d_1_floor[0],
                                           proj_location_2d_1_floor[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_floor[1] * width + proj_location_2d_1_floor[0]] * nw_weight_1
                                 : 0) +
                            (WITHIN_BOUNDS(proj_location_2d_1_ceil[0],
                                           proj_location_2d_1_ceil[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_ceil[1] * width + proj_location_2d_1_ceil[0]] * se_weight_1
                                 : 0) +
                            (WITHIN_BOUNDS(proj_location_2d_1_floor[0],
                                           proj_location_2d_1_ceil[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_ceil[1] * width + proj_location_2d_1_floor[0]] * sw_weight_1
                                 : 0) +
                            (WITHIN_BOUNDS(proj_location_2d_1_ceil[0],
                                           proj_location_2d_1_floor[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_floor[1] * width + proj_location_2d_1_ceil[0]] * ne_weight_1
                                 : 0);
      }

      Scalar sampled_feat_grad_1[FS][2];
#pragma unroll_completely
      for (int i = 0; i < FS; ++i)
      {
        for (int j = 0; j < 2; ++j)
        {
          if (is_dpt_positive)
          {
            sampled_feat_grad_1[i][j] =
                within_mask * ((WITHIN_BOUNDS(proj_location_2d_1_floor[0],
                                              proj_location_2d_1_floor[1], width, height)
                                    ? feat_map_1_grad_pyr[j][i][level_offsets[level] + proj_location_2d_1_floor[1] * width + proj_location_2d_1_floor[0]] * nw_weight_1
                                    : 0) +
                               (WITHIN_BOUNDS(proj_location_2d_1_ceil[0],
                                              proj_location_2d_1_ceil[1], width, height)
                                    ? feat_map_1_grad_pyr[j][i][level_offsets[level] + proj_location_2d_1_ceil[1] * width + proj_location_2d_1_ceil[0]] * se_weight_1
                                    : 0) +
                               (WITHIN_BOUNDS(proj_location_2d_1_floor[0],
                                              proj_location_2d_1_ceil[1], width, height)
                                    ? feat_map_1_grad_pyr[j][i][level_offsets[level] + proj_location_2d_1_ceil[1] * width + proj_location_2d_1_floor[0]] * sw_weight_1
                                    : 0) +
                               (WITHIN_BOUNDS(proj_location_2d_1_ceil[0],
                                              proj_location_2d_1_floor[1], width, height)
                                    ? feat_map_1_grad_pyr[j][i][level_offsets[level] + proj_location_2d_1_floor[1] * width + proj_location_2d_1_ceil[0]] * ne_weight_1
                                    : 0));
          }
          else
          {
            sampled_feat_grad_1[i][j] = 0;
          }
        }
      }

      Scalar feat_error = 0;

      for (int i = 0; i < FS; ++i)
      {
        feat_error += is_dpt_positive ? within_mask * pow(cat_sampled_features_0[level][idx][i] - sampled_feat_1[i], 2) : 0;
      }

#pragma unroll_completely
      for (int i = 0; i < FS; ++i)
      {
        sampled_feat_diff[level][idx][i] = is_dpt_positive ? within_mask * (cat_sampled_features_0[level][idx][i] - sampled_feat_1[i]) : 0;
      }

      sampled_valid_mask_1[level][idx] = is_dpt_positive ? within_mask : 0;
      sampled_error[level][idx] = feat_error;

      const Scalar inv_z = 1 / location_3d_in_1_z;
      const Scalar x_z = inv_z * location_3d_in_1_x;
      const Scalar y_z = inv_z * location_3d_in_1_y;
      const Scalar proj_2d_jac_rel_pose[2][6] = {{fx * inv_z, 0, -fx * x_z * inv_z, -fx * x_z * y_z, fx * (1 + x_z * x_z), -fx * y_z},
                                                 {0, fy * inv_z, -fy * y_z * inv_z, -fy * (1 + y_z * y_z), fy * x_z * y_z, fy * x_z}};

      Scalar proj_2d_jac_dpt_0[2] = {fx * (rotated_location_homo_0_in_1[0] * inv_z - location_3d_in_1_x * rotated_location_homo_0_in_1[2] * inv_z * inv_z),
                                     fy * (rotated_location_homo_0_in_1[1] * inv_z - location_3d_in_1_y * rotated_location_homo_0_in_1[2] * inv_z * inv_z)};
      Scalar proj_2d_jac_scale_0[2] = {proj_2d_jac_dpt_0[0] * sampled_dpts_0[idx] / scale_0, proj_2d_jac_dpt_0[1] * sampled_dpts_0[idx] / scale_0};

#pragma unroll_completely
      for (int i = 0; i < FS; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
          sampled_feat_diff_jac_rel_pose_scale_0[level][idx][i][j] =
              sampled_feat_grad_1[i][0] * proj_2d_jac_rel_pose[0][j] + sampled_feat_grad_1[i][1] * proj_2d_jac_rel_pose[1][j];
        }

        sampled_feat_diff_jac_rel_pose_scale_0[level][idx][i][6] =
            sampled_feat_grad_1[i][0] * proj_2d_jac_scale_0[0] + sampled_feat_grad_1[i][1] * proj_2d_jac_scale_0[1];
      }
    }

    return;
  }

  template <typename Scalar, int FS>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void tracker_photo_error_calculate_kernel(
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_valid_mask_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_dpts_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> valid_mask_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> cat_sampled_features_0, // L x N x C_feat
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> feat_map_1_pyr,         // C_feat x ( N_0 + N_1 + ... + N_(L-1) )
          const PinholeCamera<float> *__restrict__ cam_pyr,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> level_offsets,
          const Scalar eps)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int level = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < sampled_locations_homo_0.size(0))
    {
      const int ori_height = static_cast<int>(cam_pyr[0].height());
      const int ori_width = static_cast<int>(cam_pyr[0].width());
      const Scalar ori_fx = cam_pyr[0].fx();
      const Scalar ori_fy = cam_pyr[0].fy();
      const Scalar ori_cx = cam_pyr[0].u0();
      const Scalar ori_cy = cam_pyr[0].v0();

      const int height = static_cast<int>(cam_pyr[level].height());
      const int width = static_cast<int>(cam_pyr[level].width());
      const Scalar fx = cam_pyr[level].fx();
      const Scalar fy = cam_pyr[level].fy();

      const Scalar location_3d_in_1_x = sampled_dpts_0[idx] *
                                            (rotation[0][0] * sampled_locations_homo_0[idx][0] +
                                             rotation[0][1] * sampled_locations_homo_0[idx][1] +
                                             rotation[0][2] * sampled_locations_homo_0[idx][2]) +
                                        translation[0];
      const Scalar location_3d_in_1_y = sampled_dpts_0[idx] *
                                            (rotation[1][0] * sampled_locations_homo_0[idx][0] +
                                             rotation[1][1] * sampled_locations_homo_0[idx][1] +
                                             rotation[1][2] * sampled_locations_homo_0[idx][2]) +
                                        translation[1];
      Scalar location_3d_in_1_z = sampled_dpts_0[idx] *
                                      (rotation[2][0] * sampled_locations_homo_0[idx][0] +
                                       rotation[2][1] * sampled_locations_homo_0[idx][1] +
                                       rotation[2][2] * sampled_locations_homo_0[idx][2]) +
                                  translation[2];

      // Whether or not the transformed depth is below the positive depth threshold
      bool is_dpt_positive = location_3d_in_1_z > eps;

      // Bilinear sampling on feat_map_1
      const Scalar proj_location_2d_1[2] =
          {((location_3d_in_1_x / location_3d_in_1_z) * ori_fx + ori_cx + static_cast<Scalar>(0.5)) * fx / ori_fx - static_cast<Scalar>(0.5),
           ((location_3d_in_1_y / location_3d_in_1_z) * ori_fy + ori_cy + static_cast<Scalar>(0.5)) * fy / ori_fy - static_cast<Scalar>(0.5)};

      const int proj_location_2d_1_floor[2] = {int(floor(proj_location_2d_1[0])), int(floor(proj_location_2d_1[1]))};
      const int proj_location_2d_1_ceil[2] = {proj_location_2d_1_floor[0] + 1, proj_location_2d_1_floor[1] + 1};

      const Scalar lower_weight_1[2] = {(Scalar)proj_location_2d_1_ceil[0] - proj_location_2d_1[0],
                                        (Scalar)proj_location_2d_1_ceil[1] - proj_location_2d_1[1]};
      const Scalar upper_weight_1[2] = {1 - lower_weight_1[0], 1 - lower_weight_1[1]};

      const Scalar nw_weight_1 = lower_weight_1[0] * lower_weight_1[1];
      const Scalar se_weight_1 = upper_weight_1[0] * upper_weight_1[1];
      const Scalar sw_weight_1 = lower_weight_1[0] * upper_weight_1[1];
      const Scalar ne_weight_1 = upper_weight_1[0] * lower_weight_1[1];

      // Nearest interpolation on fine-resolution mask
      const int mask_proj_location_2d_1_round[2] =
          {(int)round((location_3d_in_1_x / location_3d_in_1_z) * ori_fx + ori_cx),
           (int)round((location_3d_in_1_y / location_3d_in_1_z) * ori_fy + ori_cy)};
      const Scalar within_mask =
          WITHIN_BOUNDS(mask_proj_location_2d_1_round[0],
                        mask_proj_location_2d_1_round[1], ori_width, ori_height)
              ? valid_mask_1[mask_proj_location_2d_1_round[1]][mask_proj_location_2d_1_round[0]]
              : 0;

      Scalar sampled_feat_1[FS];
#pragma unroll_completely
      for (int i = 0; i < FS; ++i)
      {
        sampled_feat_1[i] = (WITHIN_BOUNDS(proj_location_2d_1_floor[0],
                                           proj_location_2d_1_floor[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_floor[1] * width + proj_location_2d_1_floor[0]] * nw_weight_1
                                 : 0) +
                            (WITHIN_BOUNDS(proj_location_2d_1_ceil[0],
                                           proj_location_2d_1_ceil[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_ceil[1] * width + proj_location_2d_1_ceil[0]] * se_weight_1
                                 : 0) +
                            (WITHIN_BOUNDS(proj_location_2d_1_floor[0],
                                           proj_location_2d_1_ceil[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_ceil[1] * width + proj_location_2d_1_floor[0]] * sw_weight_1
                                 : 0) +
                            (WITHIN_BOUNDS(proj_location_2d_1_ceil[0],
                                           proj_location_2d_1_floor[1], width, height)
                                 ? feat_map_1_pyr[i][level_offsets[level] + proj_location_2d_1_floor[1] * width + proj_location_2d_1_ceil[0]] * ne_weight_1
                                 : 0);
      }

      Scalar feat_error = 0;

      for (int i = 0; i < FS; ++i)
      {
        feat_error += is_dpt_positive ? within_mask * pow(cat_sampled_features_0[level][idx][i] - sampled_feat_1[i], 2) : 0;
      }

      sampled_valid_mask_1[level][idx] = is_dpt_positive ? within_mask : 0;
      sampled_error[level][idx] = feat_error;
    }

    return;
  }

  template <int FS>
  float photometric_error_calculate(const at::Tensor rotation, const at::Tensor translation,
                                    const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
                                    const at::Tensor code_0, const at::Tensor valid_mask_1,
                                    const at::Tensor sampled_locations_1d_0,
                                    const at::Tensor sampled_locations_homo_0,
                                    const at::Tensor feat_map_pyramid_0, const at::Tensor feat_map_pyramid_1,
                                    const at::Tensor level_offsets,
                                    const float scale_0, const CameraPyramid<float> &camera_pyramid,
                                    const float eps, const at::Tensor weights)
  {
    using namespace torch::indexing;
    const int thread_per_block = MAX_THREADS_PER_BLOCK;
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)), camera_pyramid.Levels());
    const dim3 block_size(thread_per_block, 1);

    const long num_levels = camera_pyramid.Levels();
    at::Tensor sampled_error = torch::zeros({num_levels, sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor sampled_validness = torch::zeros({num_levels, sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());

    // Refer https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/GridSampler.cuh for coordinate transformation
    // N x 2
    const at::Tensor sampled_locations_2d_0 = torch::stack({torch::fmod(sampled_locations_1d_0.to(torch::kFloat32), camera_pyramid[0].width()),
                                                            torch::floor(sampled_locations_1d_0.to(torch::kFloat32) / camera_pyramid[0].width())},
                                                           1);
    // apply (2.0 * pix_coord + 1.0) / size - 1 to pixel location -> [-1, 1] -> other resolution pixel location with ((norm_coord + 1.f) * size - 1) / 2
    // Convert pixel location from fine to coarse -- (pix + 0.5) * cur_ori_ratio - 0.5
    const at::Tensor sampled_dpts_0 = scale_0 * (flatten_dpt_map_bias_0.index({sampled_locations_1d_0}) +
                                                 torch::matmul(flatten_dpt_jac_code_0.index({sampled_locations_1d_0, Slice()}), code_0));

    const at::Tensor weights_tensor = weights.to(sampled_locations_homo_0.device()).to(sampled_locations_homo_0.dtype());

    thrust::device_vector<PinholeCamera<float>> device_cameras(num_levels);

    for (int i = 0; i < num_levels; i++)
    {
      device_cameras[i] = camera_pyramid[i];
    }

    const PinholeCamera<float> *device_camera_pointer = thrust::raw_pointer_cast(device_cameras.data());
    AT_DISPATCH_FLOATING_TYPES(rotation.type(), "photometric_error_calculate_kernel", ([&]
                                                                                       { photometric_error_calculate_kernel<float, FS><<<grid_size, block_size>>>(
                                                                                             sampled_error.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                             sampled_validness.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                             rotation.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                             translation.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                                                                             sampled_dpts_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                                                                             code_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                                                                             valid_mask_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                             sampled_locations_2d_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                             sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                             feat_map_pyramid_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                             feat_map_pyramid_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                             device_camera_pointer,
                                                                                             level_offsets.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                                                                             eps); }));

    gpuErrchk(cudaGetLastError());

    float num_inliers = torch::sum(sampled_validness.index({0, Slice()})).item<float>(); // * num_levels * feat_map_pyramid_0.size(0);

    if (num_inliers > 0)
    {
      return torch::sum(weights_tensor * torch::sum(sampled_error, 1, false)).item<float>() / num_inliers;
    }
    else
    {
      return torch::sum(weights_tensor).item<float>() * 10.0; //1.0e8; //std::numeric_limits<float>::infinity();
    }
  }

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
                                       const float eps, const at::Tensor weights)
  {
    using namespace torch::indexing;
    const int thread_per_block = MAX_THREADS_PER_BLOCK;
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)), camera_pyramid.Levels());
    const dim3 block_size(thread_per_block, 1);

    const long num_levels = camera_pyramid.Levels();
    at::Tensor sampled_error = torch::zeros({num_levels, sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor sampled_validness = torch::zeros({num_levels, sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor sampled_feat_diff = torch::zeros({num_levels, sampled_locations_homo_0.size(0), FS}, sampled_locations_homo_0.options());
    at::Tensor sampled_feat_diff_jac_pose_code_scale = torch::zeros({num_levels, sampled_locations_homo_0.size(0), FS, 13 + CS}, sampled_locations_homo_0.options());

    // Refer https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/GridSampler.cuh for coordinate transformation
    // N x 2
    const at::Tensor sampled_locations_2d_0 = torch::stack({torch::fmod(sampled_locations_1d_0.to(torch::kFloat32), camera_pyramid[0].width()),
                                                            torch::floor(sampled_locations_1d_0.to(torch::kFloat32) / camera_pyramid[0].width())},
                                                           1);
    // apply (2.0 * pix_coord + 1.0) / size - 1 to pixel location -> [-1, 1] -> other resolution pixel location with ((norm_coord + 1.f) * size - 1) / 2
    // Convert pixel location from fine to coarse -- (pix + 0.5) * cur_ori_ratio - 0.5
    const at::Tensor sampled_dpts_0 = scale_0 * (flatten_dpt_map_bias_0.index({sampled_locations_1d_0}) +
                                                 torch::matmul(flatten_dpt_jac_code_0.index({sampled_locations_1d_0, Slice()}), code_0));

    const at::Tensor weights_tensor = weights.to(sampled_locations_homo_0.device()).to(sampled_locations_homo_0.dtype());

    thrust::device_vector<PinholeCamera<float>> device_cameras(num_levels);

    for (int i = 0; i < num_levels; i++)
    {
      device_cameras[i] = camera_pyramid[i];
    }

    const PinholeCamera<float> *device_camera_pointer = thrust::raw_pointer_cast(device_cameras.data());

    const at::Tensor int_sampled_locations_1d_0 = sampled_locations_1d_0.to(torch::kInt32);
    AT_DISPATCH_FLOATING_TYPES(rotation10.type(), "photometric_jac_error_calculate_kernel", ([&]
                                                                                             {
                                                                                               photometric_jac_error_calculate_kernel<float, CS, FS><<<grid_size, block_size>>>(
                                                                                                   sampled_error.packed_accessor32<float, 2, at::RestrictPtrTraits>(),                         // L x N
                                                                                                   sampled_validness.packed_accessor32<float, 2, at::RestrictPtrTraits>(),                     // L x N
                                                                                                   sampled_feat_diff.packed_accessor32<float, 3, at::RestrictPtrTraits>(),                     // L x N x C_feat
                                                                                                   sampled_feat_diff_jac_pose_code_scale.packed_accessor32<float, 4, at::RestrictPtrTraits>(), // L x N x C_feat x (13 + C_code)
                                                                                                   rotation10.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                                   translation10.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                                                                                   rotation0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                                   translation0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                                                                                   rotation1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                                   translation1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                                                                                   sampled_dpts_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                                                                                   flatten_dpt_jac_code_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                                   code_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                                                                                   valid_mask_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                                   int_sampled_locations_1d_0.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                                                                                   sampled_locations_2d_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                                   sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                                   feat_map_pyramid_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                                   feat_map_pyramid_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                                   feat_map_grad_pyramid_1.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
                                                                                                   device_camera_pointer,
                                                                                                   level_offsets.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                                                                                   scale_0, eps);
                                                                                             }));

    gpuErrchk(cudaGetLastError());

    float num_inliers = torch::sum(sampled_validness.index({0, Slice()})).item<float>(); //* num_levels * feat_map_pyramid_0.size(0);

    if (num_inliers > 0)
    {
      const at::Tensor weighted_sampled_feat_diff_jac_pose_code_scale =
          (weights_tensor.reshape({-1, 1, 1, 1}) * sampled_feat_diff_jac_pose_code_scale).reshape({-1, 13 + CS});

      // (13 + CS) x (13 + CS)
      AtA = (1.0 / num_inliers) *
            torch::matmul(weighted_sampled_feat_diff_jac_pose_code_scale.permute({1, 0}),
                          sampled_feat_diff_jac_pose_code_scale.reshape({-1, 13 + CS}));
      // (13 + CS) x 1
      Atb = (1.0 / num_inliers) *
            torch::matmul(weighted_sampled_feat_diff_jac_pose_code_scale.permute({1, 0}), sampled_feat_diff.reshape({-1, 1}));

      error = torch::sum(weights_tensor * torch::sum(sampled_error, 1, false)).item<float>() / num_inliers;
    }
    else
    {
      error = torch::sum(weights_tensor).item<float>() * 10.0; // 1.0e8; //std::numeric_limits<float>::infinity();
      AtA = torch::zeros({13 + CS, 13 + CS}, sampled_error.options());
      Atb = torch::zeros({13 + CS, 1}, sampled_error.options());
    }

    return;
  }

  template <int FS>
  void tracker_photo_jac_error_calculate(at::Tensor &AtA, at::Tensor &Atb, float &error,
                                         const at::Tensor rotation, const at::Tensor translation,
                                         const at::Tensor valid_mask_1,
                                         const at::Tensor sampled_dpts_0,           // N
                                         const at::Tensor sampled_locations_homo_0, // N x 2
                                         const at::Tensor sampled_features_0,       // L x N x C_feat
                                         const at::Tensor feat_map_pyramid_1,
                                         const at::Tensor feat_map_grad_pyramid_1,
                                         const at::Tensor level_offsets,
                                         const CameraPyramid<float> &camera_pyramid,
                                         const float eps, const at::Tensor weights_tensor)
  {
    using namespace torch::indexing;
    const int thread_per_block = MAX_THREADS_PER_BLOCK;

    const long num_levels = camera_pyramid.Levels();
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)), num_levels);
    const dim3 block_size(thread_per_block, 1);

    at::Tensor sampled_error = torch::zeros({num_levels, sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor sampled_validness = torch::zeros({num_levels, sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor sampled_feat_diff = torch::zeros({num_levels, sampled_locations_homo_0.size(0), FS}, sampled_locations_homo_0.options());
    at::Tensor sampled_feat_diff_jac_rel_pose = torch::zeros({num_levels, sampled_locations_homo_0.size(0), FS, 6}, sampled_locations_homo_0.options());

    thrust::device_vector<PinholeCamera<float>> device_cameras(num_levels);
    for (int i = 0; i < num_levels; i++)
    {
      device_cameras[i] = camera_pyramid[i];
    }
    const PinholeCamera<float> *device_camera_pointer = thrust::raw_pointer_cast(device_cameras.data());

    AT_DISPATCH_FLOATING_TYPES(rotation.type(), "tracker_photo_jac_error_calculate_kernel", ([&]
                                                                                             {
                                                                                               tracker_photo_jac_error_calculate_kernel<float, FS><<<grid_size, block_size>>>(
                                                                                                   sampled_error.packed_accessor32<float, 2, at::RestrictPtrTraits>(),                  // L x N
                                                                                                   sampled_validness.packed_accessor32<float, 2, at::RestrictPtrTraits>(),              // L x N
                                                                                                   sampled_feat_diff.packed_accessor32<float, 3, at::RestrictPtrTraits>(),              // L x N x C_feat
                                                                                                   sampled_feat_diff_jac_rel_pose.packed_accessor32<float, 4, at::RestrictPtrTraits>(), // L x N x C_feat x 6
                                                                                                   rotation.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                                   translation.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                                                                                   sampled_dpts_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                                                                                   valid_mask_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                                   sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                                   sampled_features_0.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
                                                                                                   feat_map_pyramid_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                                   feat_map_grad_pyramid_1.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
                                                                                                   device_camera_pointer,
                                                                                                   level_offsets.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                                                                                   eps);
                                                                                             }));

    gpuErrchk(cudaGetLastError());

    float num_inliers = torch::sum(sampled_validness.index({0, Slice()})).item<float>();

    if (num_inliers > 0)
    {
      error = torch::sum(weights_tensor * torch::sum(sampled_error, 1, false)).item<float>() / num_inliers;

      const at::Tensor weighted_sampled_feat_diff_jac_rel_pose =
          (weights_tensor.reshape({-1, 1, 1, 1}) * sampled_feat_diff_jac_rel_pose).reshape({-1, 6});
      // 6 x 6
      AtA = (1.0 / num_inliers) *
            torch::matmul(weighted_sampled_feat_diff_jac_rel_pose.permute({1, 0}),
                          sampled_feat_diff_jac_rel_pose.reshape({-1, 6}));
      // 6 x 1
      Atb = (1.0 / num_inliers) *
            torch::matmul(weighted_sampled_feat_diff_jac_rel_pose.permute({1, 0}), sampled_feat_diff.reshape({-1, 1}));
    }
    else
    {
      // Assign addable large error and zero AtA and Atb so that zero overlap pairs won't prohibit optimiztion from working
      error = torch::sum(weights_tensor).item<float>() * 10.0; //1.0e8; //std::numeric_limits<float>::infinity();
      AtA = torch::zeros({6, 6}, sampled_error.options());
      Atb = torch::zeros({6, 1}, sampled_error.options());
    }

    return;
  }

  template <int FS>
  void tracker_photo_jac_error_calculate_with_scale(at::Tensor &AtA, at::Tensor &Atb, float &error,
                                                    const at::Tensor rotation, const at::Tensor translation,
                                                    const at::Tensor valid_mask_1,
                                                    const at::Tensor sampled_dpts_0,           // N
                                                    const at::Tensor sampled_locations_homo_0, // N x 2
                                                    const at::Tensor sampled_features_0,       // L x N x C_feat
                                                    const at::Tensor feat_map_pyramid_1,
                                                    const at::Tensor feat_map_grad_pyramid_1,
                                                    const at::Tensor level_offsets,
                                                    const CameraPyramid<float> &camera_pyramid,
                                                    const float scale_0, const float eps, const at::Tensor weights_tensor)
  {
    using namespace torch::indexing;
    const int thread_per_block = MAX_THREADS_PER_BLOCK;

    const long num_levels = camera_pyramid.Levels();
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)), num_levels);
    const dim3 block_size(thread_per_block, 1);

    at::Tensor sampled_error = torch::zeros({num_levels, sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor sampled_validness = torch::zeros({num_levels, sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor sampled_feat_diff = torch::zeros({num_levels, sampled_locations_homo_0.size(0), FS}, sampled_locations_homo_0.options());
    at::Tensor sampled_feat_diff_jac_rel_pose_scale_0 = torch::zeros({num_levels, sampled_locations_homo_0.size(0), FS, 7}, sampled_locations_homo_0.options());

    thrust::device_vector<PinholeCamera<float>> device_cameras(num_levels);
    for (int i = 0; i < num_levels; i++)
    {
      device_cameras[i] = camera_pyramid[i];
    }
    const PinholeCamera<float> *device_camera_pointer = thrust::raw_pointer_cast(device_cameras.data());

    AT_DISPATCH_FLOATING_TYPES(rotation.type(), "tracker_photo_jac_error_calculate_kernel", ([&]
                                                                                             {
                                                                                               tracker_photo_jac_error_calculate_with_scale_kernel<float, FS><<<grid_size, block_size>>>(
                                                                                                   sampled_error.packed_accessor32<float, 2, at::RestrictPtrTraits>(),                          // L x N
                                                                                                   sampled_validness.packed_accessor32<float, 2, at::RestrictPtrTraits>(),                      // L x N
                                                                                                   sampled_feat_diff.packed_accessor32<float, 3, at::RestrictPtrTraits>(),                      // L x N x C_feat
                                                                                                   sampled_feat_diff_jac_rel_pose_scale_0.packed_accessor32<float, 4, at::RestrictPtrTraits>(), // L x N x C_feat x 7
                                                                                                   rotation.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                                   translation.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                                                                                   sampled_dpts_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                                                                                   valid_mask_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                                   sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                                   sampled_features_0.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
                                                                                                   feat_map_pyramid_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                                   feat_map_grad_pyramid_1.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
                                                                                                   device_camera_pointer,
                                                                                                   level_offsets.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                                                                                   scale_0, eps);
                                                                                             }));

    gpuErrchk(cudaGetLastError());

    float num_inliers = torch::sum(sampled_validness.index({0, Slice()})).item<float>();
    if (num_inliers > 0)
    {
      error = torch::sum(weights_tensor * torch::sum(sampled_error, 1, false)).item<float>() / num_inliers;

      const at::Tensor weighted_sampled_feat_diff_jac_rel_pose_scale_0 =
          (weights_tensor.reshape({-1, 1, 1, 1}) * sampled_feat_diff_jac_rel_pose_scale_0).reshape({-1, 7});
      // 7 x 7
      AtA = (1.0 / num_inliers) *
            torch::matmul(weighted_sampled_feat_diff_jac_rel_pose_scale_0.permute({1, 0}),
                          sampled_feat_diff_jac_rel_pose_scale_0.reshape({-1, 7}));
      // 7 x 1
      Atb = (1.0 / num_inliers) *
            torch::matmul(weighted_sampled_feat_diff_jac_rel_pose_scale_0.permute({1, 0}), sampled_feat_diff.reshape({-1, 1}));
    }
    else
    {
      // Assign addable large error and zero AtA and Atb so that zero overlap pairs won't prohibit optimiztion from working
      error = torch::sum(weights_tensor).item<float>() * 10.0; //1.0e8; //std::numeric_limits<float>::infinity();
      AtA = torch::zeros({7, 7}, sampled_error.options());
      Atb = torch::zeros({7, 1}, sampled_error.options());
    }

    return;
  }

  template <int FS>
  float tracker_photo_error_calculate(const at::Tensor rotation, const at::Tensor translation,
                                      const at::Tensor valid_mask_1,
                                      const at::Tensor sampled_dpts_0,           // N
                                      const at::Tensor sampled_locations_homo_0, // N x 2
                                      const at::Tensor sampled_features_0,       // L x N x C_feat
                                      const at::Tensor feat_map_pyramid_1,
                                      const at::Tensor level_offsets,
                                      const CameraPyramid<float> &camera_pyramid,
                                      const float eps, const at::Tensor weights_tensor)
  {
    using namespace torch::indexing;
    const int thread_per_block = MAX_THREADS_PER_BLOCK;

    const long num_levels = camera_pyramid.Levels();
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)), num_levels);
    const dim3 block_size(thread_per_block, 1);

    at::Tensor sampled_error = torch::zeros({num_levels, sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor sampled_validness = torch::zeros({num_levels, sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());

    thrust::device_vector<PinholeCamera<float>> device_cameras(num_levels);
    for (int i = 0; i < num_levels; i++)
    {
      device_cameras[i] = camera_pyramid[i];
    }
    const PinholeCamera<float> *device_camera_pointer = thrust::raw_pointer_cast(device_cameras.data());

    AT_DISPATCH_FLOATING_TYPES(rotation.type(), "tracker_photo_error_calculate_kernel", ([&]
                                                                                         {
                                                                                           tracker_photo_error_calculate_kernel<float, FS><<<grid_size, block_size>>>(
                                                                                               sampled_error.packed_accessor32<float, 2, at::RestrictPtrTraits>(),     // L x N
                                                                                               sampled_validness.packed_accessor32<float, 2, at::RestrictPtrTraits>(), // L x N
                                                                                               rotation.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                               translation.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                                                                               sampled_dpts_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                                                                               valid_mask_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                               sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                               sampled_features_0.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
                                                                                               feat_map_pyramid_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                                                                               device_camera_pointer,
                                                                                               level_offsets.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                                                                               eps);
                                                                                         }));

    gpuErrchk(cudaGetLastError());

    float num_inliers = torch::sum(sampled_validness.index({0, Slice()})).item<float>();
    if (num_inliers > 0)
    {
      return torch::sum(weights_tensor * torch::sum(sampled_error, 1, false)).item<float>() / num_inliers;
    }
    else
    {
      // Assign addable large error so that zero overlap pairs won't prohibit optimiztion from working
      return torch::sum(weights_tensor).item<float>() * 10.0; // 1.0e8; //std::numeric_limits<float>::infinity();
    }
  }

#undef WITHIN_BOUNDS

  template float photometric_error_calculate<DF_FEAT_SIZE>(
      const at::Tensor rotation, const at::Tensor translation,
      const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
      const at::Tensor code_0, const at::Tensor valid_mask_1,
      const at::Tensor sampled_locations_1d_0,
      const at::Tensor sampled_locations_homo_0,
      const at::Tensor feat_map_pyramid_0, const at::Tensor feat_map_pyramid_1,
      const at::Tensor level_offsets,
      const float scale_0, const CameraPyramid<float> &camera_pyramid,
      const float eps, const at::Tensor weights);

  template void photometric_jac_error_calculate<DF_CODE_SIZE, DF_FEAT_SIZE>(
      at::Tensor &AtA, at::Tensor &Atb, float &error,
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
      const float eps, const at::Tensor weights);

  template void tracker_photo_jac_error_calculate<DF_FEAT_SIZE>(
      at::Tensor &AtA, at::Tensor &Atb, float &error,
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

  template void tracker_photo_jac_error_calculate_with_scale<DF_FEAT_SIZE>(
      at::Tensor &AtA, at::Tensor &Atb, float &error,
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

  template float tracker_photo_error_calculate<DF_FEAT_SIZE>(
      const at::Tensor rotation, const at::Tensor translation,
      const at::Tensor valid_mask_1,
      const at::Tensor sampled_dpts_0,
      const at::Tensor sampled_locations_homo_0,
      const at::Tensor sampled_features_0,
      const at::Tensor feat_map_pyramid_1,
      const at::Tensor level_offsets,
      const CameraPyramid<float> &camera_pyramid,
      const float eps, const at::Tensor weights_tensor);

}