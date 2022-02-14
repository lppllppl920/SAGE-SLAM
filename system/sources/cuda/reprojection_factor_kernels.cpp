#include "reprojection_factor_kernels.h"

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

  template <typename Scalar, int CS>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void reprojection_jac_error_calculate_kernel(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_valid_mask_1,
          torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> fair_proj_2d_jac_pose_code_scale, // N x 2 x (13 + CS)
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> fair_match_reproj_location_diff,  // N x 2
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation10,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation10,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation1,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_0,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> sampled_locations_1d_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> matched_locations_2d_1,
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

      Scalar dpt_0 = flatten_dpt_map_bias_0[sampled_locations_1d_0[idx]];

      for (int i = 0; i < CS; ++i)
      {
        dpt_0 += flatten_dpt_jac_code_0[sampled_locations_1d_0[idx]][i] * code_0[i];
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
      bool is_dpt_positive = location_3d_0_in_1[2] > eps;

      const Scalar proj_2d_x = (location_3d_0_in_1[0] / location_3d_0_in_1[2]) * fx + cx;
      const Scalar proj_2d_y = (location_3d_0_in_1[1] / location_3d_0_in_1[2]) * fy + cy;

      const Scalar sqrt_loss_param = sqrt(loss_param);
      const Scalar match_proj_location_diff[2] = {matched_locations_2d_1[idx][0] - proj_2d_x, matched_locations_2d_1[idx][1] - proj_2d_y};
      const Scalar match_proj_location_diff_normalized[2] = {fabs(match_proj_location_diff[0]) / sqrt_loss_param, fabs(match_proj_location_diff[1]) / sqrt_loss_param};

      const Scalar sqrt_fair_weight[2] = {is_dpt_positive ? sqrt(1 / (loss_param * (1 + match_proj_location_diff_normalized[0]))) : 0,
                                          is_dpt_positive ? sqrt(1 / (loss_param * (1 + match_proj_location_diff_normalized[1]))) : 0};

      // We pre-add the 2 error here while jac is left to the outside for calculation
      sampled_error[idx] = is_dpt_positive ? 2 * (match_proj_location_diff_normalized[0] + match_proj_location_diff_normalized[1] -
                                                  log(1 + match_proj_location_diff_normalized[0]) - log(1 + match_proj_location_diff_normalized[1]))
                                           : 0;
      sampled_valid_mask_1[idx] = is_dpt_positive ? 1 : 0;

      const Scalar inv_z = 1 / location_3d_0_in_1[2];
      const Scalar x_z = inv_z * location_3d_0_in_1[0];
      const Scalar y_z = inv_z * location_3d_0_in_1[1];

      // A.3 in "tutorial on SE3"
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

      const Scalar proj_2d_jac_dpt_0[2] = {fx * (rotated_location_homo_0_in_1[0] * inv_z - location_3d_0_in_1[0] * rotated_location_homo_0_in_1[2] * inv_z * inv_z),
                                           fy * (rotated_location_homo_0_in_1[1] * inv_z - location_3d_0_in_1[1] * rotated_location_homo_0_in_1[2] * inv_z * inv_z)};
      Scalar proj_2d_jac_code_0[2][CS];

#pragma unroll_completely
      for (int i = 0; i < CS; ++i)
      {
        proj_2d_jac_code_0[0][i] = proj_2d_jac_dpt_0[0] * scale_0 * flatten_dpt_jac_code_0[sampled_locations_1d_0[idx]][i];
        proj_2d_jac_code_0[1][i] = proj_2d_jac_dpt_0[1] * scale_0 * flatten_dpt_jac_code_0[sampled_locations_1d_0[idx]][i];
      }

      const Scalar proj_2d_jac_scale_0[2] = {proj_2d_jac_dpt_0[0] * dpt_0 / scale_0, proj_2d_jac_dpt_0[1] * dpt_0 / scale_0};

      fair_match_reproj_location_diff[idx][0] = sqrt_fair_weight[0] * match_proj_location_diff[0];
      fair_match_reproj_location_diff[idx][1] = sqrt_fair_weight[1] * match_proj_location_diff[1];

#pragma unroll_completely
      for (int i = 0; i < 6; ++i)
      {
        fair_proj_2d_jac_pose_code_scale[idx][0][i] = sqrt_fair_weight[0] * proj_2d_jac_pose_0[0][i];
        fair_proj_2d_jac_pose_code_scale[idx][1][i] = sqrt_fair_weight[1] * proj_2d_jac_pose_0[1][i];
        fair_proj_2d_jac_pose_code_scale[idx][0][6 + i] = sqrt_fair_weight[0] * proj_2d_jac_pose_1[0][i];
        fair_proj_2d_jac_pose_code_scale[idx][1][6 + i] = sqrt_fair_weight[1] * proj_2d_jac_pose_1[1][i];
      }

#pragma unroll_completely
      for (int i = 0; i < CS; ++i)
      {
        fair_proj_2d_jac_pose_code_scale[idx][0][12 + i] = sqrt_fair_weight[0] * proj_2d_jac_code_0[0][i];
        fair_proj_2d_jac_pose_code_scale[idx][1][12 + i] = sqrt_fair_weight[1] * proj_2d_jac_code_0[1][i];
      }

      fair_proj_2d_jac_pose_code_scale[idx][0][12 + CS] = sqrt_fair_weight[0] * proj_2d_jac_scale_0[0];
      fair_proj_2d_jac_pose_code_scale[idx][1][12 + CS] = sqrt_fair_weight[1] * proj_2d_jac_scale_0[1];
    }

    return;
  }

  template <typename Scalar, int CS>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void reprojection_error_calculate_kernel(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_valid_mask_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation10,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation10,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> flatten_dpt_map_bias_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> flatten_dpt_jac_code_0,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> code_0,
          const torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> sampled_locations_1d_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> matched_locations_2d_1,
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

      const auto location_homo = sampled_locations_homo_0[idx];
      Scalar dpt_0 = flatten_dpt_map_bias_0[sampled_locations_1d_0[idx]];

      for (int i = 0; i < CS; ++i)
      {
        dpt_0 += flatten_dpt_jac_code_0[sampled_locations_1d_0[idx]][i] * code_0[i];
      }

      dpt_0 *= scale_0;

      const Scalar location_3d_in_1_x = dpt_0 *
                                            (rotation10[0][0] * location_homo[0] +
                                             rotation10[0][1] * location_homo[1] +
                                             rotation10[0][2] * location_homo[2]) +
                                        translation10[0];
      const Scalar location_3d_in_1_y = dpt_0 *
                                            (rotation10[1][0] * location_homo[0] +
                                             rotation10[1][1] * location_homo[1] +
                                             rotation10[1][2] * location_homo[2]) +
                                        translation10[1];
      const Scalar location_3d_in_1_z = dpt_0 *
                                            (rotation10[2][0] * location_homo[0] +
                                             rotation10[2][1] * location_homo[1] +
                                             rotation10[2][2] * location_homo[2]) +
                                        translation10[2];

      // Whether or not the transformed depth is below the positive depth threshold
      bool is_dpt_positive = location_3d_in_1_z > eps;
      // bool is_dpt_positive;
      // location_3d_in_1_z = (is_dpt_positive = (location_3d_in_1_z > eps)) ? location_3d_in_1_z : eps;

      const Scalar proj_2d_x = (location_3d_in_1_x / location_3d_in_1_z) * fx + cx;
      const Scalar proj_2d_y = (location_3d_in_1_y / location_3d_in_1_z) * fy + cy;

      const Scalar sqrt_loss_param = sqrt(loss_param);

      const Scalar match_proj_location_diff[2] = {matched_locations_2d_1[idx][0] - proj_2d_x, matched_locations_2d_1[idx][1] - proj_2d_y};
      const Scalar match_proj_location_diff_normalized[2] = {fabs(match_proj_location_diff[0]) / sqrt_loss_param, fabs(match_proj_location_diff[1]) / sqrt_loss_param};

      sampled_error[idx] = is_dpt_positive ? 2 * (match_proj_location_diff_normalized[0] + match_proj_location_diff_normalized[1] -
                                                  log(1 + match_proj_location_diff_normalized[0]) - log(1 + match_proj_location_diff_normalized[1]))
                                           : 0;
      sampled_valid_mask_1[idx] = is_dpt_positive ? 1 : 0;
    }

    return;
  }

  template <typename Scalar>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void tracker_reproj_jac_error_calculate_kernel(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_valid_mask_1,
          torch::PackedTensorAccessor32<Scalar, 3, at::RestrictPtrTraits> fair_proj_2d_jac_rel_pose,       // N x 2 x 6
          torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> fair_match_reproj_location_diff, // N x 2
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_dpts_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> matched_locations_2d_1,
          const PinholeCamera<Scalar> camera, const Scalar eps, const Scalar loss_param)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sampled_locations_homo_0.size(0))
    {
      const Scalar fx = camera.fx();
      const Scalar fy = camera.fy();
      const Scalar cx = camera.u0();
      const Scalar cy = camera.v0();

      Scalar rotated_location_homo[3];
// With this pragma, the programmer asserts that there are no loop-carried dependencies which would prevent consecutive
// iterations of the following loop from executing concurrently with SIMD (single instruction multiple data) instructions.
#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        rotated_location_homo[i] = rotation[i][0] * sampled_locations_homo_0[idx][0] +
                                   rotation[i][1] * sampled_locations_homo_0[idx][1] +
                                   rotation[i][2] * sampled_locations_homo_0[idx][2];
      }

      const Scalar location_3d_in_1_x = sampled_dpts_0[idx] * rotated_location_homo[0] + translation[0];
      const Scalar location_3d_in_1_y = sampled_dpts_0[idx] * rotated_location_homo[1] + translation[1];
      const Scalar location_3d_in_1_z = sampled_dpts_0[idx] * rotated_location_homo[2] + translation[2];

      // Whether or not the transformed depth is below the positive depth threshold
      bool is_dpt_positive = location_3d_in_1_z > eps;
      // location_3d_in_1_z = (is_dpt_positive = (location_3d_in_1_z > eps)) ? location_3d_in_1_z : eps;

      const Scalar proj_2d_x = (location_3d_in_1_x / location_3d_in_1_z) * fx + cx;
      const Scalar proj_2d_y = (location_3d_in_1_y / location_3d_in_1_z) * fy + cy;

      const Scalar sqrt_loss_param = sqrt(loss_param);

      const Scalar match_proj_location_diff[2] = {matched_locations_2d_1[idx][0] - proj_2d_x, matched_locations_2d_1[idx][1] - proj_2d_y};
      const Scalar match_proj_location_diff_normalized[2] = {fabs(match_proj_location_diff[0]) / sqrt_loss_param, fabs(match_proj_location_diff[1]) / sqrt_loss_param};

      // We pre-add the 2 error here while jac is left to the outside for calculation
      sampled_error[idx] = is_dpt_positive ? 2 * (match_proj_location_diff_normalized[0] + match_proj_location_diff_normalized[1] -
                                                  log(1 + match_proj_location_diff_normalized[0]) - log(1 + match_proj_location_diff_normalized[1]))
                                           : 0;
      sampled_valid_mask_1[idx] = is_dpt_positive ? 1 : 0;

      const Scalar sqrt_fair_weight[2] = {is_dpt_positive ? sqrt(1 / (loss_param * (1 + match_proj_location_diff_normalized[0]))) : 0,
                                          is_dpt_positive ? sqrt(1 / (loss_param * (1 + match_proj_location_diff_normalized[1]))) : 0};
      fair_match_reproj_location_diff[idx][0] = sqrt_fair_weight[0] * match_proj_location_diff[0];
      fair_match_reproj_location_diff[idx][1] = sqrt_fair_weight[1] * match_proj_location_diff[1];

      const Scalar inv_z = 1 / location_3d_in_1_z;
      const Scalar x_z = inv_z * location_3d_in_1_x;
      const Scalar y_z = inv_z * location_3d_in_1_y;
      const Scalar proj_2d_jac_rel_pose[2][6] = {{fx * inv_z, 0, -fx * x_z * inv_z, -fx * x_z * y_z, fx * (1 + x_z * x_z), -fx * y_z},
                                                 {0, fy * inv_z, -fy * y_z * inv_z, -fy * (1 + y_z * y_z), fy * x_z * y_z, fy * x_z}};
#pragma unroll_completely
      for (int j = 0; j < 2; ++j)
      {
        for (int i = 0; i < 6; ++i)
        {
          fair_proj_2d_jac_rel_pose[idx][j][i] = sqrt_fair_weight[j] * proj_2d_jac_rel_pose[j][i];
        }
      }
    }

    return;
  }

  template <typename Scalar>
  __launch_bounds__(MAX_THREADS_PER_BLOCK)
      __global__ void tracker_reproj_error_calculate_kernel(
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_error,
          torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_valid_mask_1,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> rotation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> translation,
          const torch::PackedTensorAccessor32<Scalar, 1, at::RestrictPtrTraits> sampled_dpts_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> sampled_locations_homo_0,
          const torch::PackedTensorAccessor32<Scalar, 2, at::RestrictPtrTraits> matched_locations_2d_1,
          const PinholeCamera<Scalar> camera, const Scalar eps, const Scalar loss_param)
  {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sampled_locations_homo_0.size(0))
    {
      const Scalar fx = camera.fx();
      const Scalar fy = camera.fy();
      const Scalar cx = camera.u0();
      const Scalar cy = camera.v0();

      Scalar rotated_location_homo[3];

#pragma unroll_completely
      for (int i = 0; i < 3; ++i)
      {
        rotated_location_homo[i] = rotation[i][0] * sampled_locations_homo_0[idx][0] +
                                   rotation[i][1] * sampled_locations_homo_0[idx][1] +
                                   rotation[i][2] * sampled_locations_homo_0[idx][2];
      }

      const Scalar location_3d_in_1_x = sampled_dpts_0[idx] * rotated_location_homo[0] + translation[0];
      const Scalar location_3d_in_1_y = sampled_dpts_0[idx] * rotated_location_homo[1] + translation[1];
      const Scalar location_3d_in_1_z = sampled_dpts_0[idx] * rotated_location_homo[2] + translation[2];

      // Whether or not the transformed depth is below the positive depth threshold
      bool is_dpt_positive = location_3d_in_1_z > eps;
      // location_3d_in_1_z = (is_dpt_positive = (location_3d_in_1_z > eps)) ? location_3d_in_1_z : eps;

      const Scalar proj_2d_x = (location_3d_in_1_x / location_3d_in_1_z) * fx + cx;
      const Scalar proj_2d_y = (location_3d_in_1_y / location_3d_in_1_z) * fy + cy;

      const Scalar sqrt_loss_param = sqrt(loss_param);
      const Scalar match_proj_location_diff[2] = {matched_locations_2d_1[idx][0] - proj_2d_x, matched_locations_2d_1[idx][1] - proj_2d_y};
      const Scalar match_proj_location_diff_normalized[2] = {fabs(match_proj_location_diff[0]) / sqrt_loss_param, fabs(match_proj_location_diff[1]) / sqrt_loss_param};

      // We pre-add the 2 error here while jac is left to the outside for calculation
      sampled_error[idx] = is_dpt_positive ? 2 * (match_proj_location_diff_normalized[0] + match_proj_location_diff_normalized[1] -
                                                  log(1 + match_proj_location_diff_normalized[0]) - log(1 + match_proj_location_diff_normalized[1]))
                                           : 0;
      sampled_valid_mask_1[idx] = is_dpt_positive ? 1 : 0;
    }

    return;
  }

  template <int CS>
  float reprojection_error_calculate(const at::Tensor rotation10, const at::Tensor translation10,
                                     const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
                                     const at::Tensor code_0, const at::Tensor sampled_locations_1d_0,
                                     const at::Tensor sampled_locations_homo_0,
                                     const at::Tensor matched_locations_2d_1,
                                     const float scale_0, const PinholeCamera<float> &camera,
                                     const float eps, const float loss_param, const float weight)
  {
    const int thread_per_block = MAX_THREADS_PER_BLOCK;
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)));
    const dim3 block_size(thread_per_block);

    at::Tensor sampled_error = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor sampled_validness = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());

    AT_DISPATCH_FLOATING_TYPES(rotation10.type(), "reprojection_error_calculate_kernel", ([&] {
                                 reprojection_error_calculate_kernel<float, CS><<<grid_size, block_size>>>(
                                     sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_validness.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     rotation10.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation10.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     flatten_dpt_map_bias_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     flatten_dpt_jac_code_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     code_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_locations_1d_0.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                     sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     matched_locations_2d_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
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
  void reprojection_jac_error_calculate(at::Tensor &AtA, at::Tensor &Atb, float &error,
                                        const at::Tensor rotation10, const at::Tensor translation10,
                                        const at::Tensor rotation0, const at::Tensor translation0,
                                        const at::Tensor rotation1, const at::Tensor translation1,
                                        const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
                                        const at::Tensor code_0, const at::Tensor sampled_locations_1d_0,
                                        const at::Tensor sampled_locations_homo_0,
                                        const at::Tensor matched_locations_2d_1,
                                        const float scale_0, const PinholeCamera<float> &camera,
                                        const float eps, const float loss_param, const float weight)
  {
    const int thread_per_block = MAX_THREADS_PER_BLOCK;
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)));
    const dim3 block_size(thread_per_block);

    at::Tensor sampled_error = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor sampled_validness = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor fair_proj_2d_jac_pose_code_scale =
        torch::zeros({sampled_locations_homo_0.size(0), 2, 13 + CS}, sampled_locations_homo_0.options());
    at::Tensor fair_match_reproj_location_diff = torch::zeros({sampled_locations_homo_0.size(0), 2}, sampled_locations_homo_0.options());

    // __restrict__ declaration here is to promise the compiler that these pointers will not overlap with each other and this enables compilation optimization
    AT_DISPATCH_FLOATING_TYPES(rotation10.type(), "reprojection_jac_error_calculate_kernel", ([&] {
                                 reprojection_jac_error_calculate_kernel<float, CS><<<grid_size, block_size>>>(
                                     sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_validness.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     fair_proj_2d_jac_pose_code_scale.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
                                     fair_match_reproj_location_diff.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     rotation10.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation10.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     rotation0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     rotation1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation1.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     flatten_dpt_map_bias_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     flatten_dpt_jac_code_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     code_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_locations_1d_0.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                                     sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     matched_locations_2d_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     scale_0, camera, eps, loss_param);
                               }));

    gpuErrchk(cudaGetLastError());

    // N*2 x (13 + CS)
    fair_proj_2d_jac_pose_code_scale = fair_proj_2d_jac_pose_code_scale.reshape({-1, 13 + CS});
    // N*2
    fair_match_reproj_location_diff = fair_match_reproj_location_diff.reshape({-1, 1});

    float num_inliers = torch::sum(sampled_validness).item<float>();

    if (num_inliers > 0)
    {
      error = (weight / num_inliers) * torch::sum(sampled_error).item<float>();
      // (13 + CS) x (13 + CS)
      AtA = (weight / num_inliers) *
            torch::matmul(fair_proj_2d_jac_pose_code_scale.permute({1, 0}), fair_proj_2d_jac_pose_code_scale);
      // (13 + CS) x 1
      Atb = (weight / num_inliers) *
            torch::matmul(fair_proj_2d_jac_pose_code_scale.permute({1, 0}), fair_match_reproj_location_diff);
    }
    else
    {
      error = weight * 10.0;
      AtA = torch::zeros({13 + CS, 13 + CS}, sampled_error.options());
      Atb = torch::zeros({13 + CS, 1}, sampled_error.options());
    }

    return;
  }

  void tracker_reproj_jac_error_calculate(at::Tensor &AtA, at::Tensor &Atb, float &error,
                                          const at::Tensor rotation, const at::Tensor translation,
                                          const at::Tensor sampled_dpts_0,
                                          const at::Tensor sampled_locations_homo_0,
                                          const at::Tensor matched_locations_2d_1,
                                          const PinholeCamera<float> &camera,
                                          const float eps, const float loss_param, const float weight)
  {
    using namespace torch::indexing;

    const int thread_per_block = MAX_THREADS_PER_BLOCK;
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)));
    const dim3 block_size(thread_per_block);

    at::Tensor sampled_error = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor sampled_validness = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor fair_match_reproj_loc_diff = torch::zeros({sampled_locations_homo_0.size(0), 2}, sampled_locations_homo_0.options());
    at::Tensor fair_proj_2d_jac_rel_pose = torch::zeros({sampled_locations_homo_0.size(0), 2, 6}, sampled_locations_homo_0.options());

    AT_DISPATCH_FLOATING_TYPES(rotation.type(), "tracker_reproj_jac_error_calculate_kernel", ([&] {
                                 tracker_reproj_jac_error_calculate_kernel<float><<<grid_size, block_size>>>(
                                     sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_validness.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     fair_proj_2d_jac_rel_pose.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
                                     fair_match_reproj_loc_diff.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     rotation.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_dpts_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     matched_locations_2d_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     camera, eps, loss_param);
                               }));

    gpuErrchk(cudaGetLastError());

    // N*2 x 6
    fair_proj_2d_jac_rel_pose = fair_proj_2d_jac_rel_pose.reshape({-1, 6});
    // N*2 x 1
    fair_match_reproj_loc_diff = fair_match_reproj_loc_diff.reshape({-1, 1});

    float num_inliers = torch::sum(sampled_validness).item<float>();

    if (num_inliers > 0)
    {
      error = (weight / num_inliers) * torch::sum(sampled_error).item<float>();
      // 6 x 6
      AtA = (weight / num_inliers) *
            torch::matmul(fair_proj_2d_jac_rel_pose.permute({1, 0}), fair_proj_2d_jac_rel_pose);
      // 6 x 1
      Atb = (weight / num_inliers) *
            torch::matmul(fair_proj_2d_jac_rel_pose.permute({1, 0}), fair_match_reproj_loc_diff);
    }
    else
    {
      error = weight * 10.0;
      AtA = torch::zeros({6, 6}, sampled_error.options());
      Atb = torch::zeros({6, 1}, sampled_error.options());
    }

    return;
  }

  float tracker_reproj_error_calculate(const at::Tensor rotation, const at::Tensor translation,
                                       const at::Tensor sampled_dpts_0,
                                       const at::Tensor sampled_locations_homo_0,
                                       const at::Tensor matched_locations_2d_1,
                                       const PinholeCamera<float> &camera,
                                       const float eps, const float loss_param, const float weight)
  {
    using namespace torch::indexing;

    const int thread_per_block = MAX_THREADS_PER_BLOCK;
    const dim3 grid_size(int(ceil((float)sampled_locations_homo_0.size(0) / (float)thread_per_block)));
    const dim3 block_size(thread_per_block);

    at::Tensor sampled_error = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());
    at::Tensor sampled_validness = torch::zeros({sampled_locations_homo_0.size(0)}, sampled_locations_homo_0.options());

    AT_DISPATCH_FLOATING_TYPES(rotation.type(), "tracker_reproj_error_calculate_kernel", ([&] {
                                 tracker_reproj_error_calculate_kernel<float><<<grid_size, block_size>>>(
                                     sampled_error.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_validness.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     rotation.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     translation.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_dpts_0.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                                     sampled_locations_homo_0.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     matched_locations_2d_1.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
                                     camera, eps, loss_param);
                               }));

    gpuErrchk(cudaGetLastError());

    float num_inliers = torch::sum(sampled_validness).item<float>();

    if (num_inliers > 0)
    {
      return (weight / num_inliers) * torch::sum(sampled_error).item<float>();
    }
    else
    {
      return weight * 10.0;
    }
  }

#undef WITHIN_BOUNDS
  template void reprojection_jac_error_calculate<DF_CODE_SIZE>(
      at::Tensor &AtA, at::Tensor &Atb, float &error,
      const at::Tensor rotation10, const at::Tensor translation10,
      const at::Tensor rotation0, const at::Tensor translation0,
      const at::Tensor rotation1, const at::Tensor translation1,
      const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
      const at::Tensor code_0, const at::Tensor sampled_locations_1d_0,
      const at::Tensor sampled_locations_homo_0,
      const at::Tensor matched_locations_2d_1,
      const float scale_0, const PinholeCamera<float> &camera,
      const float eps, const float loss_param, const float weight);

  template float reprojection_error_calculate<DF_CODE_SIZE>(
      const at::Tensor rotation10, const at::Tensor translation10,
      const at::Tensor flatten_dpt_map_bias_0, const at::Tensor flatten_dpt_jac_code_0,
      const at::Tensor code_0, const at::Tensor sampled_locations_1d_0,
      const at::Tensor sampled_locations_homo_0,
      const at::Tensor matched_locations_2d_1,
      const float scale_0, const PinholeCamera<float> &camera,
      const float eps, const float loss_param, const float weight);

}