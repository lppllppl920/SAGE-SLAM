import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
import torchgeometry as tgm
import cv2
import umap


import utils
import models
from utils import logger


class DiffBundleAdjustment(torch.nn.Module):
    def __init__(self, match_geom_param_factor, match_geom_term_weight, code_term_weight, geometry_cauchy_param_factor,
                 geometry_term_weight, scale_term_weight, photo_pow_factor, photo_weight, num_photo_level,
                 depth_eps, num_display_matches):
        super(DiffBundleAdjustment, self).__init__()

        self.photo_pow_factor = torch.nn.Parameter(photo_pow_factor * torch.ones(1, dtype=torch.float32),
                                                   requires_grad=True)
        self.photo_weight = torch.nn.Parameter(photo_weight * torch.ones(1, dtype=torch.float32),
                                               requires_grad=True)
        self.num_photo_level = num_photo_level
        self.geometry_cauchy_param_factor = torch.tensor(
            geometry_cauchy_param_factor)
        self.geometry_term_weight = torch.tensor(geometry_term_weight)
        self.match_geom_param_factor = torch.tensor(match_geom_param_factor)
        self.match_geom_term_weight = torch.tensor(match_geom_term_weight)
        self.scale_term_weight = torch.tensor(scale_term_weight)
        self.code_term_weight = torch.tensor(code_term_weight)

        self.chessboard = np.kron(
            [[1, 0] * 4, [0, 1] * 4] * 4, np.ones((100, 100)))

        self.gauss_kernel = torch.from_numpy(1.0 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])).float(). \
            reshape(1, 1, 3, 3).cuda()
        self.depth_eps = depth_eps
        self.num_display_matches = num_display_matches
        self.display_match_indexes = None

    @staticmethod
    def generate_gaussian_pyramid(feature_map, valid_mask, return_mask, num_photo_level, gauss_kernel):
        feature_map_pyramid = list()
        mask_pyramid = list()
        feature_map_pyramid.append(feature_map)
        mask_pyramid.append(valid_mask)

        _, feat_channel, height, width = feature_map.shape
        feature_map = feature_map.reshape(feat_channel, 1, height, width)
        for _ in range(num_photo_level - 1):
            raw_feature_map = F.conv2d(feature_map * valid_mask, weight=gauss_kernel, bias=None,
                                       stride=2, padding=1, dilation=1, groups=1)
            height, width = raw_feature_map.shape[2:]
            raw_mask = F.conv2d(valid_mask,
                                weight=gauss_kernel, bias=None, stride=2,
                                padding=1, dilation=1, groups=1)
            valid_mask = F.interpolate(
                valid_mask, size=raw_feature_map.shape[2:], mode='nearest')

            if return_mask:
                mask_pyramid.append(valid_mask.reshape(1, 1, height, width))
            # correct the feature value around the edges and corners
            feature_map = raw_feature_map / (raw_mask + 1.0e-8)
            feature_map_pyramid.append(
                feature_map.reshape(1, feat_channel, height, width))
        if return_mask:
            return list(reversed(feature_map_pyramid)), list(reversed(mask_pyramid))
        else:
            return list(reversed(feature_map_pyramid))

    @staticmethod
    def generate_hierarchy_sampled_depth_jac_code(depth_map_basis_list, sampled_normalized_2d_locations):
        sampled_depth_jac_code_hierarchy = list()
        for depth_map_basis in depth_map_basis_list:
            code_length = depth_map_basis.shape[1]
            num_photo_points = sampled_normalized_2d_locations.shape[2]
            # N x C_code
            sampled_depth_jac_code_hierarchy.append(F.grid_sample(input=depth_map_basis,
                                                                  grid=sampled_normalized_2d_locations,
                                                                  mode='bilinear',
                                                                  padding_mode='zeros',
                                                                  align_corners=False).reshape(
                code_length, num_photo_points).permute(1, 0))
        # N x C_code_total
        sampled_depth_jac_code_hierarchy = torch.cat(
            sampled_depth_jac_code_hierarchy, dim=1)
        return sampled_depth_jac_code_hierarchy

    def ba_iteration(self, sampled_src_features_pyramid, tgt_feature_map_pyramid,
                     tgt_feature_map_spatial_grad_pyramid, tgt_valid_mask_pyramid,
                     camera_intrinsics_pyramid, sampled_homo_2d_locations,
                     sampled_depth_bias, keypoint_depth_bias, tgt_depth_map,
                     mean_squared_tgt_depth_value,
                     tgt_depth_map_spatial_grad,
                     sampled_depth_jac_code_hierarchy, keypoint_depth_jac_code_hierarchy,
                     keypoint_homo_2d_locations,
                     match_homo_2d_locations, match_depths,
                     guess_rotation, guess_translation,
                     guess_code_hierarchy, guess_scale, damp,
                     gradient_checkpoint, scale_pyramid, init_scale):
        AtA = None
        Atb = None
        optimize_objective = None

        # Need to make sure factors contribute to the overall objective on a similar level
        # Photometric term
        logger.debug("--------------DEBUG--------------")
        for level in range(len(sampled_src_features_pyramid)):
            # Assign a higher impact factor to the low level feature maps
            weight = torch.abs(self.photo_weight * 10) * scale_pyramid[
                level] ** self.photo_pow_factor
            tgt_feature_map = tgt_feature_map_pyramid[level]
            sampled_src_features = sampled_src_features_pyramid[level]
            camera_intrinsics = camera_intrinsics_pyramid[level]
            tgt_valid_mask = tgt_valid_mask_pyramid[level]
            tgt_feature_map_spatial_grad = tgt_feature_map_spatial_grad_pyramid[level]

            # 1 x C_feat x H x W
            _, feat_channel, _, _ = tgt_feature_map.shape
            # C_feat x N
            _, num_points = sampled_src_features.shape

            if gradient_checkpoint:
                A, sampled_src_tgt_feature_differences, sampled_tgt_valid_masks = \
                    checkpoint(self.photo_term, tgt_feature_map,
                               tgt_feature_map_spatial_grad,
                               tgt_valid_mask, sampled_src_features,
                               sampled_depth_bias, sampled_depth_jac_code_hierarchy,
                               sampled_homo_2d_locations,
                               camera_intrinsics, guess_rotation, guess_translation,
                               guess_code_hierarchy, guess_scale)
            else:
                A, sampled_src_tgt_feature_differences, sampled_tgt_valid_masks = \
                    self.photo_term(tgt_feature_map=tgt_feature_map,
                                    tgt_feature_map_spatial_grad=tgt_feature_map_spatial_grad,
                                    tgt_valid_mask=tgt_valid_mask,
                                    sampled_src_features=sampled_src_features,
                                    sampled_depth_bias=sampled_depth_bias,
                                    sampled_depth_jac_code_hierarchy=sampled_depth_jac_code_hierarchy,
                                    sampled_homo_2d_locations=sampled_homo_2d_locations,
                                    camera_intrinsics=camera_intrinsics,
                                    guess_rotation=guess_rotation,
                                    guess_translation=guess_translation,
                                    guess_code_hierarchy=guess_code_hierarchy,
                                    guess_scale=guess_scale)

            num_samples = torch.sum(sampled_tgt_valid_masks)
            # We should discourage the optimization to go that null space by assigning largest error
            if num_samples <= 0:
                error = 1.0e8
                return None
            else:
                error = weight * \
                    torch.sum(sampled_src_tgt_feature_differences **
                              2) / num_samples

            inc_AtA = weight * torch.matmul(A.permute(1, 0), A) / num_samples
            inc_Atb = weight * torch.matmul(A.permute(1, 0),
                                            sampled_src_tgt_feature_differences.
                                            reshape(feat_channel, num_points).permute(1, 0).
                                            reshape(-1, 1)) / num_samples
            logger.debug(
                f"level-{level} err: {error.item()}")

            if AtA is None:
                # (7 + code) x (7 + code)
                AtA = inc_AtA
                # (7 + code) x 1
                Atb = inc_Atb
                with torch.no_grad():
                    optimize_objective = error
            else:
                # (7 + code) x (7 + code)
                AtA = AtA + inc_AtA
                # (7 + code) x 1
                Atb = Atb + inc_Atb
                with torch.no_grad():
                    optimize_objective = optimize_objective + error

        if self.use_match_geom:
            # match geometry term
            if gradient_checkpoint:
                A, loc_3d_diff, loc_3d_errors = \
                    checkpoint(self.match_geometry_term,
                               keypoint_depth_bias, keypoint_depth_jac_code_hierarchy,
                               keypoint_homo_2d_locations, match_homo_2d_locations,
                               match_depths, mean_squared_tgt_depth_value,
                               guess_rotation, guess_translation,
                               guess_code_hierarchy, guess_scale)
            else:
                A, loc_3d_diff, loc_3d_errors = \
                    self.match_geometry_term(keypoint_depth_bias=keypoint_depth_bias,
                                             keypoint_depth_jac_code_hierarchy=keypoint_depth_jac_code_hierarchy,
                                             keypoint_homo_2d_locations=keypoint_homo_2d_locations,
                                             match_homo_2d_locations=match_homo_2d_locations,
                                             match_depths=match_depths,
                                             mean_squared_tgt_depth_value=mean_squared_tgt_depth_value,
                                             guess_rotation=guess_rotation, guess_translation=guess_translation,
                                             guess_code_hierarchy=guess_code_hierarchy, guess_scale=guess_scale)

            error = torch.abs(self.match_geom_term_weight) * \
                torch.mean(loc_3d_errors)

            inc_AtA = torch.abs(self.match_geom_term_weight) * \
                torch.matmul(A.permute(1, 0), A) / loc_3d_errors.shape[0]
            inc_Atb = torch.abs(self.match_geom_term_weight) * torch.matmul(A.permute(1, 0), loc_3d_diff) \
                / loc_3d_errors.shape[0]
            # (7 + CS) x (7 + CS)
            AtA = AtA + inc_AtA
            # (7 + CS) x 1
            Atb = Atb + inc_Atb

            with torch.no_grad():
                optimize_objective = optimize_objective + error

            logger.debug(
                f"match geometry err: {error.item()}")

        # zero code prior
        A, code_diff = self.code_term(
            guess_code_hierarchy=guess_code_hierarchy)
        code_length = A.shape[0]
        inc_AtA = torch.abs(self.code_term_weight) * \
            torch.matmul(A.permute(1, 0), A) / code_length
        inc_Atb = torch.abs(self.code_term_weight) * \
            torch.matmul(A.permute(1, 0), code_diff) / code_length
        # (7 + code) x (7 + code)
        AtA = AtA + inc_AtA
        # (7 + code) x 1
        Atb = Atb + inc_Atb

        with torch.no_grad():
            error = torch.abs(self.code_term_weight) * \
                torch.mean(torch.square(code_diff))
            optimize_objective = optimize_objective + error

        logger.debug(
            f"code prior err, code: {error.item()} {guess_code_hierarchy.detach().cpu().numpy().reshape((-1,))}")

        if self.use_geom:
            # geometry term at finest resolution level
            if gradient_checkpoint:
                A, depth_differences, depth_errors, sampled_valid_mask = \
                    checkpoint(self.geometry_term,
                               tgt_valid_mask_pyramid[-1], tgt_depth_map, mean_squared_tgt_depth_value,
                               tgt_depth_map_spatial_grad,
                               sampled_depth_bias, sampled_depth_jac_code_hierarchy,
                               sampled_homo_2d_locations, camera_intrinsics_pyramid[-1],
                               guess_rotation, guess_translation,
                               guess_code_hierarchy, guess_scale)
            else:
                A, depth_differences, depth_errors, sampled_valid_mask = \
                    self.geometry_term(
                        tgt_valid_mask=tgt_valid_mask_pyramid[-1],
                        tgt_depth_map=tgt_depth_map,
                        mean_squared_tgt_depth_value=mean_squared_tgt_depth_value,
                        tgt_depth_map_spatial_grad=tgt_depth_map_spatial_grad,
                        sampled_depth_bias=sampled_depth_bias,
                        sampled_depth_jac_code_hierarchy=sampled_depth_jac_code_hierarchy,
                        sampled_homo_2d_locations=sampled_homo_2d_locations,
                        camera_intrinsics=camera_intrinsics_pyramid[-1], guess_rotation=guess_rotation,
                        guess_translation=guess_translation, guess_code_hierarchy=guess_code_hierarchy,
                        guess_scale=guess_scale)

            num_samples = torch.sum(sampled_valid_mask)
            if num_samples <= 0:
                error = 1.0e8
                return None
            else:
                error = torch.abs(self.geometry_term_weight) * \
                    torch.sum(depth_errors) / num_samples

            inc_AtA = torch.abs(self.geometry_term_weight) * \
                torch.matmul(A.permute(1, 0), A) / num_samples
            inc_Atb = torch.abs(self.geometry_term_weight) * torch.matmul(A.permute(1, 0),
                                                                          depth_differences) / num_samples
            # (7 + code) x (7 + code)
            AtA = AtA + inc_AtA
            # (7 + code) x 1
            Atb = Atb + inc_Atb

            with torch.no_grad():
                optimize_objective = optimize_objective + error

            logger.debug(
                f"geometry err: {error.item()}")

        # scale term
        A, scale_log_difference = \
            self.scale_term(guess_scale=guess_scale,
                            init_scale=init_scale,
                            code_length=code_length)
        inc_AtA = torch.abs(self.scale_term_weight) * \
            torch.matmul(A.permute(1, 0), A)
        inc_Atb = torch.abs(self.scale_term_weight) * \
            torch.matmul(A.permute(1, 0), scale_log_difference)
        # (7 + code) x (7 + code)
        AtA = AtA + inc_AtA
        # (7 + code) x 1
        Atb = Atb + inc_Atb

        with torch.no_grad():
            error = torch.abs(self.scale_term_weight) * \
                torch.mean(torch.square(scale_log_difference))
            optimize_objective = optimize_objective + error

        logger.debug(
            f"scale prior err, init and curr scale: {error.item()} {init_scale.item(), guess_scale.item()}")

        AtA_diag = torch.diag(torch.diag(AtA))
        try:
            with torch.no_grad():
                solution, _ = torch.solve(input=Atb, A=AtA + damp * AtA_diag)
            return AtA, Atb, AtA_diag, solution, optimize_objective
        except RuntimeError as err:
            logger.error("runtime error encountered {}".format(err))
            return None

    @staticmethod
    def jacobian_projected_2d_location_wrt_camera_pose(sampled_tgt_3d_locations, fx, fy, mode="wh"):
        # the linearization point of rotation and translation
        # is at the location of the transformed 3d locations (sampled_tgt_3d_locations)
        # sampled_tgt_3d_locations: 3 x N
        # N
        X = sampled_tgt_3d_locations[0, :]
        Y = sampled_tgt_3d_locations[1, :]
        Z = sampled_tgt_3d_locations[2, :]

        with torch.no_grad():
            zeros = torch.zeros_like(Z)

        fx = fx.reshape(-1)
        fy = fy.reshape(-1)

        inv_z = 1 / Z
        x_z = X * inv_z
        y_z = Y * inv_z
        x_2_z_2 = x_z * x_z
        y_2_z_2 = y_z * y_z
        xy_z2 = x_z * y_z

        row_0 = torch.stack([-fx * xy_z2, fx * (1.0 + x_2_z_2), -fx * y_z,
                             fx * inv_z, zeros, -fx * x_z * inv_z], dim=1)
        row_1 = torch.stack([-fy * (1 + y_2_z_2), fy * xy_z2, fy * x_z,
                             zeros, fy * inv_z, -fy * y_z * inv_z], dim=1)

        jacobian_projected_2d_wrt_pose = torch.stack([row_0, row_1], dim=1)

        if mode.lower() == "hw":
            jacobian_projected_2d_wrt_pose = torch.cat([jacobian_projected_2d_wrt_pose[:, 1:2, :],
                                                        jacobian_projected_2d_wrt_pose[:, 0:1, :]], dim=1)

        return jacobian_projected_2d_wrt_pose

    @staticmethod
    def jacobian_projected_2d_location_wrt_src_depth(rotated_sampled_src_homo_2d_locations,
                                                     sampled_tgt_3d_locations, fx, fy, mode="wh"):
        num_points = rotated_sampled_src_homo_2d_locations.shape[1]
        # N
        rx = rotated_sampled_src_homo_2d_locations[0, :]
        ry = rotated_sampled_src_homo_2d_locations[1, :]
        rz = rotated_sampled_src_homo_2d_locations[2, :]
        X = sampled_tgt_3d_locations[0, :]
        Y = sampled_tgt_3d_locations[1, :]
        Z = sampled_tgt_3d_locations[2, :]

        inv_z = 1 / Z
        inv_z_sq = inv_z * inv_z
        # 1
        fx = fx.reshape(-1)
        fy = fy.reshape(-1)

        item_0 = fx * (rx * inv_z - (X * rz) * inv_z_sq)
        item_1 = fy * (ry * inv_z - (Y * rz) * inv_z_sq)

        # N x 2 x 1
        jacobian_projected_2d_wrt_depth = torch.stack(
            [item_0, item_1], dim=1).reshape(num_points, 2, 1)

        if mode.lower() == "hw":
            jacobian_projected_2d_wrt_depth = torch.cat([jacobian_projected_2d_wrt_depth[:, 1:2, :],
                                                         jacobian_projected_2d_wrt_depth[:, 0:1, :]], dim=1)
        return jacobian_projected_2d_wrt_depth

    @staticmethod
    def jacobian_transformed_depth_wrt_camera_pose(sampled_tgt_3d_locations):
        # the linearization point of rotation and translation
        # is at the location of the transformed 3d locations (sampled_tgt_3d_locations)
        # sampled_tgt_3d_locations: 3 x N
        # N
        X = sampled_tgt_3d_locations[0, :]
        Y = sampled_tgt_3d_locations[1, :]
        Z = sampled_tgt_3d_locations[2, :]

        with torch.no_grad():
            zeros = torch.zeros_like(Z)
            ones = torch.ones_like(Z)

        # N x 1 x 6
        jacobian_transformed_depth_wrt_pose = \
            torch.stack([Y, -X, zeros, zeros, zeros, ones],
                        dim=1).reshape(-1, 1, 6)

        return jacobian_transformed_depth_wrt_pose

    @staticmethod
    def jacobian_transformed_depth_wrt_src_depth(rotated_sampled_src_homo_2d_locations):
        # N
        rz = rotated_sampled_src_homo_2d_locations[2, :]
        # N x 1 x 1
        return rz.reshape(-1, 1, 1)

    def cauchy_sqrt_jacobian_weight_and_error(self, square_error, weight, cauchy_param):
        # cauchy loss form: ln(1 + err / param)
        # cauchy gradient form: 1 / (err + param)
        return torch.sqrt(weight / (square_error + cauchy_param)), weight * torch.log(1 + square_error / cauchy_param)

    def ba_optimize(self, keypoint_2d_hw_locations, matched_2d_hw_locations, sampled_2d_hw_locations,
                    camera_intrinsics, src_feature_map, tgt_feature_map,
                    src_valid_mask, tgt_valid_mask,
                    refer_tgt_depth_map,
                    depth_map_bias, depth_map_basis_list,
                    init_rotation, init_translation,
                    init_code_list, init_scale, gradient_checkpoint,
                    max_num_iters, init_damp, damp_min_max_cap,
                    damp_inc_dec_scale, grad_thresh, param_thresh, max_cond,
                    src_input_image, tgt_input_image,
                    src_desc_feature_map, tgt_desc_feature_map,
                    visualize_lm, gt_translation,
                    gt_rotation, gt_flow_map, gt_flow_mask,
                    src_gt_depth_map, tgt_gt_depth_map,
                    use_match_geom, use_geom,
                    produce_video=False):

        self.use_match_geom = use_match_geom
        self.use_geom = use_geom

        color_src_feature_map = None
        color_tgt_feature_map = None
        color_src_desc_feature_map = None
        color_tgt_desc_feature_map = None

        height, width = depth_map_bias.shape[2:4]

        # 1 x N x 2
        _, num_photo_points, _ = sampled_2d_hw_locations.shape
        # 1 x M x 2
        _, num_reproj_points, _ = keypoint_2d_hw_locations.shape

        # N x 1
        sampled_2d_locations_h = sampled_2d_hw_locations[0, :, 0].reshape(
            -1, 1)
        sampled_2d_locations_w = sampled_2d_hw_locations[0, :, 1].reshape(
            -1, 1)

        # resolution-agnostic sampling convention
        # 1 x 1 x N x 2
        sampled_normalized_2d_locations = \
            torch.cat([(sampled_2d_locations_w + 0.5) * (2 / width) - 1.0,
                       (sampled_2d_locations_h + 0.5) * (2 / height) - 1.0], dim=1). \
            reshape(1, 1, num_photo_points, 2)

        # M x 1
        keypoint_2d_hw_locations_h = keypoint_2d_hw_locations[0, :, 0].reshape(
            -1, 1)
        keypoint_2d_hw_locations_w = keypoint_2d_hw_locations[0, :, 1].reshape(
            -1, 1)
        # 1 x 1 x M x 2
        keypoint_normalized_2d_locations = \
            torch.cat([(keypoint_2d_hw_locations_w + 0.5) * (2 / width) - 1.0,
                       (keypoint_2d_hw_locations_h + 0.5) * (2 / height) - 1.0], dim=1). \
            reshape(1, 1, num_reproj_points, 2)

        # N
        sampled_depth_bias = F.grid_sample(input=depth_map_bias,
                                           grid=sampled_normalized_2d_locations,
                                           mode='bilinear', padding_mode='zeros',
                                           align_corners=False).reshape(num_photo_points)
        # N x C_code_total
        sampled_depth_jac_code_hierarchy = \
            self.generate_hierarchy_sampled_depth_jac_code(depth_map_basis_list,
                                                           sampled_normalized_2d_locations)

        # M
        keypoint_depth_bias = F.grid_sample(input=depth_map_bias,
                                            grid=keypoint_normalized_2d_locations,
                                            mode='bilinear', padding_mode='zeros',
                                            align_corners=False).reshape(num_reproj_points)
        # M x C_code_total
        keypoint_depth_jac_code_hierarchy = \
            self.generate_hierarchy_sampled_depth_jac_code(depth_map_basis_list,
                                                           keypoint_normalized_2d_locations)

        # M x 1
        matched_2d_hw_locations_h = matched_2d_hw_locations[0, :, 0].reshape(
            -1, 1)
        matched_2d_hw_locations_w = matched_2d_hw_locations[0, :, 1].reshape(
            -1, 1)
        # 1 x 1 x M x 2
        match_normalized_2d_locations = \
            torch.cat([(matched_2d_hw_locations_w + 0.5) * (2 / width) - 1.0,
                       (matched_2d_hw_locations_h + 0.5) * (2 / height) - 1.0], dim=1). \
            reshape(1, 1, num_reproj_points, 2)
        # M
        match_depths = F.grid_sample(input=refer_tgt_depth_map,
                                     grid=match_normalized_2d_locations,
                                     mode='bilinear', padding_mode='zeros',
                                     align_corners=False).reshape(num_reproj_points)

        # sampled_homo_2d_locations: 3 x N
        sampled_homo_2d_locations = utils.compute_homogenous_2d_locations(
            sampled_2d_locations=sampled_2d_hw_locations,
            camera_intrinsics=camera_intrinsics). \
            reshape(3, num_photo_points)
        # 3 x M
        keypoint_homo_2d_locations = utils.compute_homogenous_2d_locations(
            sampled_2d_locations=keypoint_2d_hw_locations,
            camera_intrinsics=camera_intrinsics). \
            reshape(3, num_reproj_points)
        # 3 x M
        match_homo_2d_locations = utils.compute_homogenous_2d_locations(
            sampled_2d_locations=matched_2d_hw_locations,
            camera_intrinsics=camera_intrinsics). \
            reshape(3, num_reproj_points)

        refer_tgt_depth_map = refer_tgt_depth_map * tgt_valid_mask
        # Scalar
        mean_squared_tgt_depth_value = (
            torch.sum((src_valid_mask * depth_map_bias) ** 2) / torch.sum(src_valid_mask))
        # 1 x 2 x H x W (grad_x, grad_y)
        if refer_tgt_depth_map.requires_grad and gradient_checkpoint:
            tgt_depth_map_spatial_grad = checkpoint(
                utils.compute_spatial_grad, refer_tgt_depth_map)
            tgt_depth_map_spatial_grad = tgt_depth_map_spatial_grad.reshape(
                1, 2, height, width)
        else:
            tgt_depth_map_spatial_grad = utils.compute_spatial_grad(
                refer_tgt_depth_map).reshape(1, 2, height, width)

        # init_rotation: 3 x 3, init_translation: 3 x 1
        guess_rotation = init_rotation.reshape(3, 3)
        guess_translation = init_translation.reshape(3, 1)
        guess_scale = init_scale
        # C_code_total
        guess_code_hierarchy = torch.cat(init_code_list, dim=0)

        # Resample features in source and target frame for different resolutions
        #  and run optimization in a joint manner
        camera_intrinsics_pyramid = list()
        tgt_feature_map_spatial_grad_pyramid = list()
        sampled_src_features_pyramid = list()
        scale_pyramid = list()

        _, feat_channel, height, width = tgt_feature_map.shape
        tgt_feature_map_pyramid, tgt_valid_mask_pyramid = self.generate_gaussian_pyramid(tgt_feature_map,
                                                                                         tgt_valid_mask,
                                                                                         return_mask=True,
                                                                                         num_photo_level=self.num_photo_level,
                                                                                         gauss_kernel=self.gauss_kernel)
        src_feature_map_pyramid = self.generate_gaussian_pyramid(
            src_feature_map, src_valid_mask, return_mask=False, num_photo_level=self.num_photo_level,
            gauss_kernel=self.gauss_kernel)

        for level in range(self.num_photo_level):
            _, _, scaled_height, scaled_width = tgt_feature_map_pyramid[level].shape
            scale = height / scaled_height
            assert (height / scaled_height == width / scaled_width)

            scale_pyramid.append(scale)
            scaled_camera_intrinsics = camera_intrinsics / scale
            camera_intrinsics_pyramid.append(scaled_camera_intrinsics)

            # src_feature_map: 1 x C_feat x H' x W'
            feat_channel = src_feature_map.shape[1]
            # sampled_src_features: C_feat x N
            sampled_src_features = \
                F.grid_sample(input=src_feature_map_pyramid[level],
                              grid=sampled_normalized_2d_locations,
                              mode='bilinear', padding_mode='zeros',
                              align_corners=False).reshape(feat_channel, num_photo_points)
            sampled_src_features_pyramid.append(sampled_src_features)

            # 1 x 2*C_feat x H' x W' (grad_x, grad_y)
            if src_feature_map_pyramid[level].requires_grad and gradient_checkpoint:
                tgt_feature_map_spatial_grad = checkpoint(
                    utils.compute_spatial_grad, tgt_feature_map_pyramid[level])
                tgt_feature_map_spatial_grad = tgt_feature_map_spatial_grad. \
                    reshape(1, 2 * feat_channel, scaled_height, scaled_width)
            else:
                tgt_feature_map_spatial_grad = utils.compute_spatial_grad(
                    tgt_feature_map_pyramid[level])
            tgt_feature_map_spatial_grad_pyramid.append(
                tgt_feature_map_spatial_grad)

        result_list = list()
        depth_list = list()
        curr_damp = init_damp
        curr_iter = 0
        guess_src_depth_map = \
            utils.generate_depth_from_bias_and_basis(guess_scale, depth_map_bias,
                                                     depth_map_basis_list, guess_code_hierarchy)
        depth_list.append(guess_src_depth_map *
                          src_valid_mask.reshape(1, 1, height, width))

        if visualize_lm or produce_video:
            # GT pose with GT depth
            result, color_src_desc_feature_map, color_tgt_desc_feature_map, \
                color_src_feature_map, color_tgt_feature_map = \
                self.generate_visualization_result(
                    src_pred_depth_map=src_gt_depth_map,
                    tgt_pred_depth_map=tgt_gt_depth_map,
                    src_desc_feature_map=src_desc_feature_map, tgt_desc_feature_map=tgt_desc_feature_map,
                    src_feature_map=src_feature_map, tgt_feature_map=tgt_feature_map,
                    guess_rotation=gt_rotation, guess_translation=gt_translation,
                    src_valid_mask=src_valid_mask, tgt_valid_mask=tgt_valid_mask,
                    camera_intrinsics=camera_intrinsics, color_src_desc_feature_map=color_src_desc_feature_map,
                    color_tgt_desc_feature_map=color_tgt_desc_feature_map,
                    color_src_feature_map=color_src_feature_map,
                    color_tgt_feature_map=color_tgt_feature_map,
                    src_input_image=src_input_image,
                    tgt_input_image=tgt_input_image, src_gt_depth_map=src_gt_depth_map,
                    tgt_gt_depth_map=tgt_gt_depth_map,
                    keypoint_2d_hw_locations=keypoint_2d_hw_locations,
                    keypoint_homo_2d_locations=keypoint_homo_2d_locations,
                    keypoint_normalized_2d_locations=keypoint_normalized_2d_locations,
                    matched_2d_hw_locations=matched_2d_hw_locations,
                    gt_flow_map=gt_flow_map, gt_flow_mask=gt_flow_mask, generate_match_indexes=True,
                    return_feature_map=True
                )
            result_list.append(result)

            result = \
                self.generate_visualization_result(
                    src_pred_depth_map=depth_list[0],
                    tgt_pred_depth_map=refer_tgt_depth_map,
                    src_desc_feature_map=src_desc_feature_map, tgt_desc_feature_map=tgt_desc_feature_map,
                    src_feature_map=src_feature_map, tgt_feature_map=tgt_feature_map,
                    guess_rotation=guess_rotation, guess_translation=guess_translation,
                    src_valid_mask=src_valid_mask, tgt_valid_mask=tgt_valid_mask,
                    camera_intrinsics=camera_intrinsics, color_src_desc_feature_map=color_src_desc_feature_map,
                    color_tgt_desc_feature_map=color_tgt_desc_feature_map,
                    color_src_feature_map=color_src_feature_map,
                    color_tgt_feature_map=color_tgt_feature_map, src_input_image=src_input_image,
                    tgt_input_image=tgt_input_image, src_gt_depth_map=src_gt_depth_map,
                    tgt_gt_depth_map=tgt_gt_depth_map,
                    keypoint_2d_hw_locations=keypoint_2d_hw_locations,
                    keypoint_homo_2d_locations=keypoint_homo_2d_locations,
                    keypoint_normalized_2d_locations=keypoint_normalized_2d_locations,
                    matched_2d_hw_locations=matched_2d_hw_locations,
                    gt_flow_map=gt_flow_map, gt_flow_mask=gt_flow_mask, generate_match_indexes=False,
                    return_feature_map=False
                )
            result_list.append(result)

        if max_num_iters == 0:
            return result_list

        while True:
            if curr_iter >= max_num_iters:
                logger.debug("reaching maximum iterations")
                break

            x = self.ba_iteration(sampled_src_features_pyramid=sampled_src_features_pyramid,
                                  tgt_feature_map_pyramid=tgt_feature_map_pyramid,
                                  tgt_feature_map_spatial_grad_pyramid=tgt_feature_map_spatial_grad_pyramid,
                                  tgt_valid_mask_pyramid=tgt_valid_mask_pyramid,
                                  camera_intrinsics_pyramid=camera_intrinsics_pyramid,
                                  sampled_homo_2d_locations=sampled_homo_2d_locations,
                                  sampled_depth_bias=sampled_depth_bias,
                                  keypoint_depth_bias=keypoint_depth_bias,
                                  mean_squared_tgt_depth_value=mean_squared_tgt_depth_value,
                                  tgt_depth_map=refer_tgt_depth_map,
                                  tgt_depth_map_spatial_grad=tgt_depth_map_spatial_grad,
                                  sampled_depth_jac_code_hierarchy=sampled_depth_jac_code_hierarchy,
                                  keypoint_depth_jac_code_hierarchy=keypoint_depth_jac_code_hierarchy,
                                  keypoint_homo_2d_locations=keypoint_homo_2d_locations,
                                  guess_rotation=guess_rotation,
                                  guess_translation=guess_translation,
                                  guess_code_hierarchy=guess_code_hierarchy,
                                  guess_scale=guess_scale, damp=curr_damp, gradient_checkpoint=gradient_checkpoint,
                                  scale_pyramid=scale_pyramid, init_scale=init_scale, match_depths=match_depths,
                                  match_homo_2d_locations=match_homo_2d_locations)
            if x is not None:
                # solution has no gradient here
                AtA, Atb, AtA_diag, solution, residual_error = x
            else:
                break

            curr_iter += 1
            if self.lm_convergence(Atb, solution, guess_rotation, guess_translation, guess_scale, guess_code_hierarchy,
                                   grad_thresh, param_thresh, curr_iter, max_num_iters):
                break

            while True:
                with torch.no_grad():
                    temp = AtA + curr_damp * AtA_diag
                    cond = np.linalg.cond(temp.cpu().numpy())
                    logger.debug(f"solution cond number: {cond}")

                    candidate_rotation, candidate_translation, candidate_scale, candidate_code_hierarchy = \
                        self.update_variables(solution=solution, guess_rotation=guess_rotation,
                                              guess_translation=guess_translation,
                                              guess_scale=guess_scale,
                                              guess_code_hierarchy=guess_code_hierarchy)

                    #  Need to be careful
                    candidate_residual_error = \
                        self.compute_residual_error(
                            sampled_homo_2d_locations=sampled_homo_2d_locations,
                            sampled_depth_bias=sampled_depth_bias,
                            sampled_depth_jac_code_hierarchy=sampled_depth_jac_code_hierarchy,
                            keypoint_depth_bias=keypoint_depth_bias,
                            keypoint_depth_jac_code_hierarchy=keypoint_depth_jac_code_hierarchy,
                            sampled_src_features_pyramid=sampled_src_features_pyramid,
                            keypoint_homo_2d_locations=keypoint_homo_2d_locations,
                            match_depths=match_depths,
                            match_homo_2d_locations=match_homo_2d_locations,
                            tgt_depth_map=refer_tgt_depth_map,
                            mean_squared_tgt_depth_value=mean_squared_tgt_depth_value,
                            camera_intrinsics_pyramid=camera_intrinsics_pyramid,
                            scale_pyramid=scale_pyramid,
                            tgt_feature_map_pyramid=tgt_feature_map_pyramid,
                            tgt_valid_mask_pyramid=tgt_valid_mask_pyramid,
                            guess_rotation=candidate_rotation,
                            guess_translation=candidate_translation,
                            guess_scale=candidate_scale,
                            guess_code_hierarchy=candidate_code_hierarchy,
                            init_scale=init_scale)

                    # Accept the update if lookahead error is smaller
                    if candidate_residual_error < residual_error and cond < max_cond:
                        break
                    elif curr_damp < damp_min_max_cap[1]:
                        temp_damp = curr_damp * damp_inc_dec_scale[0]
                        curr_damp = np.minimum(np.maximum(
                            damp_min_max_cap[0], temp_damp), damp_min_max_cap[1])
                        solution, _ = torch.solve(
                            input=Atb, A=AtA + curr_damp * AtA_diag)
                        logger.debug(f"No better solution found, "
                                     f"candidate: {candidate_residual_error}, original: {residual_error}, "
                                     f"updated damp value: {curr_damp}")
                        continue
                    else:
                        logger.debug(
                            f"reaching maximum damp value, searching end")
                        break

            if (candidate_residual_error >= residual_error or cond >= max_cond) and curr_damp >= damp_min_max_cap[1]:
                break

            logger.debug("accepted: {}, original: {}, damp value: {}".
                         format(candidate_residual_error, residual_error, curr_damp))
            # If the updated residual error is accepted,
            # add the linear system solving to the computation graph for backpropagation
            solution, _ = torch.solve(input=Atb, A=AtA + curr_damp * AtA_diag)
            curr_damp = np.minimum(np.maximum(damp_min_max_cap[0],
                                              curr_damp / damp_inc_dec_scale[1]), damp_min_max_cap[1])
            guess_rotation, guess_translation, guess_scale, guess_code_hierarchy = \
                self.update_variables(solution=solution, guess_rotation=guess_rotation,
                                      guess_translation=guess_translation,
                                      guess_scale=guess_scale,
                                      guess_code_hierarchy=guess_code_hierarchy)
            guess_src_depth_map = \
                utils.generate_depth_from_bias_and_basis(guess_scale, depth_map_bias,
                                                         depth_map_basis_list, guess_code_hierarchy)
            depth_list.append(guess_src_depth_map *
                              src_valid_mask.reshape(1, 1, height, width))

            if produce_video:
                result = \
                    self.generate_visualization_result(
                        src_pred_depth_map=depth_list[-1],
                        tgt_pred_depth_map=refer_tgt_depth_map,
                        src_desc_feature_map=src_desc_feature_map, tgt_desc_feature_map=tgt_desc_feature_map,
                        src_feature_map=src_feature_map, tgt_feature_map=tgt_feature_map,
                        guess_rotation=guess_rotation, guess_translation=guess_translation,
                        src_valid_mask=src_valid_mask, tgt_valid_mask=tgt_valid_mask,
                        camera_intrinsics=camera_intrinsics, color_src_desc_feature_map=color_src_desc_feature_map,
                        color_tgt_desc_feature_map=color_tgt_desc_feature_map,
                        color_src_feature_map=color_src_feature_map,
                        color_tgt_feature_map=color_tgt_feature_map, src_input_image=src_input_image,
                        tgt_input_image=tgt_input_image, src_gt_depth_map=src_gt_depth_map,
                        tgt_gt_depth_map=tgt_gt_depth_map,
                        keypoint_2d_hw_locations=keypoint_2d_hw_locations,
                        keypoint_homo_2d_locations=keypoint_homo_2d_locations,
                        keypoint_normalized_2d_locations=keypoint_normalized_2d_locations,
                        matched_2d_hw_locations=matched_2d_hw_locations,
                        gt_flow_map=gt_flow_map, gt_flow_mask=gt_flow_mask, generate_match_indexes=False,
                        return_feature_map=False
                    )
                result_list.append(result)

        if visualize_lm and not produce_video:
            result = \
                self.generate_visualization_result(
                    src_pred_depth_map=depth_list[-1],
                    tgt_pred_depth_map=refer_tgt_depth_map,
                    src_desc_feature_map=src_desc_feature_map, tgt_desc_feature_map=tgt_desc_feature_map,
                    src_feature_map=src_feature_map, tgt_feature_map=tgt_feature_map,
                    guess_rotation=guess_rotation, guess_translation=guess_translation,
                    src_valid_mask=src_valid_mask, tgt_valid_mask=tgt_valid_mask,
                    camera_intrinsics=camera_intrinsics, color_src_desc_feature_map=color_src_desc_feature_map,
                    color_tgt_desc_feature_map=color_tgt_desc_feature_map,
                    color_src_feature_map=color_src_feature_map,
                    color_tgt_feature_map=color_tgt_feature_map, src_input_image=src_input_image,
                    tgt_input_image=tgt_input_image, src_gt_depth_map=src_gt_depth_map,
                    tgt_gt_depth_map=tgt_gt_depth_map,
                    keypoint_2d_hw_locations=keypoint_2d_hw_locations,
                    keypoint_homo_2d_locations=keypoint_homo_2d_locations,
                    keypoint_normalized_2d_locations=keypoint_normalized_2d_locations,
                    matched_2d_hw_locations=matched_2d_hw_locations,
                    gt_flow_map=gt_flow_map, gt_flow_mask=gt_flow_mask, generate_match_indexes=False,
                    return_feature_map=False
                )
            result_list.append(result)

        guess_flow_map, guess_flow_mask = utils.generate_dense_flow_map(depth_map=depth_list[-1],
                                                                         valid_mask=src_valid_mask,
                                                                         rotation=guess_rotation,
                                                                         translation=guess_translation,
                                                                         camera_intrinsics=camera_intrinsics,
                                                                         depth_eps=self.depth_eps)

        return guess_rotation, guess_translation, guess_scale, \
            guess_code_hierarchy, guess_flow_map, guess_flow_mask, curr_iter, depth_list, result_list

    @staticmethod
    def update_variables(solution, guess_rotation, guess_translation, guess_scale, guess_code_hierarchy):
        se3_matrix = utils.se3_exp(solution[:6, 0])
        dR = se3_matrix[:3, :3]
        dt = se3_matrix[:3, 3:4]
        updated_rotation = torch.matmul(dR, guess_rotation.reshape(3, 3))
        updated_translation = torch.matmul(
            dR, guess_translation.reshape(3, 1)) + dt
        updated_scale = guess_scale.reshape(-1) + solution[6:7].reshape(-1)
        updated_code_hierarchy = guess_code_hierarchy.reshape(
            -1) + solution[7:].reshape(-1)
        return updated_rotation, updated_translation, updated_scale, updated_code_hierarchy

    @staticmethod
    @torch.no_grad()
    def lm_convergence(Atb, solution, guess_rotation, guess_translation, guess_scale, guess_code_hierarchy,
                       grad_thresh, param_thresh, curr_iter, max_num_iters):
        # solution order: pose, scale, code
        max_grad = torch.max(torch.abs(Atb))
        guess_rotvec = utils.rotation_matrix_to_angle_axis(
            torch.cat([guess_rotation.reshape(1, 3, 3), torch.zeros(1, 3, 1, device=guess_rotation.device)], dim=2))
        max_param_inc = torch.max(torch.abs(solution.reshape(-1) /
                                            torch.cat(
                                                [torch.abs(guess_rotvec.reshape(-1)) + 1.0e-8,
                                                 torch.abs(
                                                     guess_translation.reshape(-1)) + 1.0e-8,
                                                 guess_scale.reshape(-1),
                                                 torch.abs(guess_code_hierarchy.reshape(-1)) + 1.0e-8],
                                                dim=0)))

        if curr_iter >= max_num_iters or max_grad <= grad_thresh or max_param_inc <= param_thresh:
            if curr_iter >= max_num_iters:
                logger.debug("reaching maximum iterations")
            if max_grad <= grad_thresh:
                logger.debug("reaching minimum gradient threshold")
            if max_param_inc <= param_thresh:
                logger.debug("reaching minimum param update threshold")
            return True
        else:
            return False

    def compute_loc_3d_diff_jac_rel_pose(self, keypoint_3d_locations_in_1):

        with torch.no_grad():
            zeros = torch.zeros_like(keypoint_3d_locations_in_1[2, :])

        # N x 3
        row_0 = torch.stack(
            [zeros, keypoint_3d_locations_in_1[2, :], -keypoint_3d_locations_in_1[1, :]], dim=1)
        row_1 = torch.stack([-keypoint_3d_locations_in_1[2, :],
                            zeros, keypoint_3d_locations_in_1[0, :]], dim=1)
        row_2 = torch.stack([-keypoint_3d_locations_in_1[1, :], -
                            keypoint_3d_locations_in_1[0, :], zeros], dim=1)
        # N x 3 x 3
        temp = torch.stack([row_0, row_1, row_2], dim=1)
        eyes = torch.eye(3, dtype=keypoint_3d_locations_in_1.dtype, device=keypoint_3d_locations_in_1.device). \
            reshape(1, 3, 3).expand(
                keypoint_3d_locations_in_1.shape[1], -1, -1)

        # N x 3 x 6
        return torch.cat([temp, eyes], dim=2)

    def match_geometry_term(self,
                            keypoint_depth_bias, keypoint_depth_jac_code_hierarchy,
                            keypoint_homo_2d_locations, match_homo_2d_locations,
                            match_depths, mean_squared_tgt_depth_value,
                            guess_rotation, guess_translation, guess_code_hierarchy,
                            guess_scale):
        # N x C_code_total
        num_points, code_length = keypoint_depth_jac_code_hierarchy.shape
        # N
        guess_keypoint_depths = guess_scale * (keypoint_depth_bias +
                                               torch.matmul(keypoint_depth_jac_code_hierarchy, guess_code_hierarchy))
        # 3 x N
        rotated_keypoint_homo_2d_locations = torch.matmul(guess_rotation.reshape(3, 3),
                                                          keypoint_homo_2d_locations.reshape(3, num_points))
        # 3 x N
        keypoint_3d_locations_in_1 = guess_keypoint_depths.reshape(1, num_points) * \
            rotated_keypoint_homo_2d_locations + \
            guess_translation.reshape(3, 1)

        # 3 x N
        match_3d_locations = match_depths.reshape(
            1, num_points) * match_homo_2d_locations
        locations_3d_diff = match_3d_locations - keypoint_3d_locations_in_1
        sqrt_loss_param = torch.sqrt(
            torch.abs(self.match_geom_param_factor * mean_squared_tgt_depth_value))
        locations_3d_diff_normalized = torch.abs(
            locations_3d_diff) / sqrt_loss_param

        # N
        fair_error = torch.sum(2 * (locations_3d_diff_normalized -
                                    torch.log(1.0 + locations_3d_diff_normalized)), dim=0)
        # 3 x N
        sqrt_fair_weights = 1.0 / sqrt_loss_param * \
            torch.sqrt(1.0 / (1 + locations_3d_diff_normalized))
        fair_locations_3d_diff = sqrt_fair_weights * locations_3d_diff
        # N x 3 x 6
        locations_3d_diff_jac_rel_pose = self.compute_loc_3d_diff_jac_rel_pose(
            keypoint_3d_locations_in_1)
        # N x 3 x 1
        locations_3d_diff_jac_scale = rotated_keypoint_homo_2d_locations.permute(1, 0).reshape(num_points, 3, 1) * \
            guess_keypoint_depths.reshape(num_points, 1, 1) / guess_scale
        # N x 3 x CS
        locations_3d_diff_jac_code = rotated_keypoint_homo_2d_locations.permute(1, 0).reshape(num_points, 3, 1) * \
            guess_scale * \
            keypoint_depth_jac_code_hierarchy.reshape(
                num_points, 1, code_length)
        # N x 3 x (6 + 1 + CS)
        fair_locations_3d_diff_jac_pose_scale_code = sqrt_fair_weights.permute(1, 0).reshape(num_points, 3, 1) * \
            torch.cat([locations_3d_diff_jac_rel_pose,
                       locations_3d_diff_jac_scale,
                       locations_3d_diff_jac_code], dim=2)

        # N*3 x (6 + 1 + CS)
        fair_locations_3d_diff_jac_pose_scale_code = \
            fair_locations_3d_diff_jac_pose_scale_code.reshape(
                num_points * 3, -1)
        # N*3 x 1
        fair_locations_3d_diff = fair_locations_3d_diff.permute(
            1, 0).reshape(-1, 1)
        return fair_locations_3d_diff_jac_pose_scale_code, fair_locations_3d_diff, \
            fair_error

    def photo_term(self, tgt_feature_map, tgt_feature_map_spatial_grad, tgt_valid_mask,
                   sampled_src_features, sampled_depth_bias, sampled_depth_jac_code_hierarchy,
                   sampled_homo_2d_locations, camera_intrinsics,
                   guess_rotation, guess_translation,
                   guess_code_hierarchy, guess_scale):
        # C_feat x N
        feat_channel, num_points = sampled_src_features.shape
        # 1 x C_feat x H x W
        _, feat_channel, height, width = tgt_feature_map.shape
        # N x C_code_total
        _, code_length = sampled_depth_jac_code_hierarchy.shape
        # N
        guess_sampled_depths = guess_scale * (sampled_depth_bias +
                                              torch.matmul(sampled_depth_jac_code_hierarchy, guess_code_hierarchy))
        # 3 x N
        rotated_sampled_src_homo_2d_locations = torch.matmul(guess_rotation.reshape(3, 3),
                                                             sampled_homo_2d_locations.reshape(3, num_points))
        # 3 x N
        sampled_tgt_3d_locations = guess_sampled_depths.reshape(1, num_points) * \
            rotated_sampled_src_homo_2d_locations + \
            guess_translation.reshape(3, 1)
        tgt_z = sampled_tgt_3d_locations[2:3, :]
        # 1 x N
        sampled_pos_depth_mask = (tgt_z >= self.depth_eps).float()
        tgt_z = torch.clamp_min(tgt_z, min=self.depth_eps)
        # 3 x N
        sampled_tgt_3d_locations = torch.cat(
            [sampled_tgt_3d_locations[:2], tgt_z], dim=0)

        # Scalar
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[0, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[0, 3]

        # Normalize the locations by depth values
        sampled_tgt_homo_2d_locations = sampled_tgt_3d_locations / \
            sampled_tgt_3d_locations[2:3, :]
        # 1 x N
        sampled_tgt_2d_locations_x = sampled_tgt_homo_2d_locations[0:1, :] * fx + cx
        sampled_tgt_2d_locations_y = sampled_tgt_homo_2d_locations[1:2, :] * fy + cy

        # 1 x 1 x N x 2 : Normalize to range [-1, 1]
        sampled_tgt_normalized_2d_locations = \
            torch.cat([(sampled_tgt_2d_locations_x.reshape(-1, 1) + 0.5) * (2 / width) - 1.0,
                       (sampled_tgt_2d_locations_y.reshape(-1, 1) + 0.5) * (2 / height) - 1.0], dim=1). \
            reshape(1, 1, num_points, 2)

        # 1 x C_feat x 1 x N -> C_feat x N
        sampled_tgt_features = F.grid_sample(input=tgt_feature_map,
                                             grid=sampled_tgt_normalized_2d_locations,
                                             mode='bilinear', padding_mode='zeros',
                                             align_corners=False). \
            reshape(feat_channel, num_points)
        # 1 x N
        sampled_tgt_valid_masks = \
            F.grid_sample(input=tgt_valid_mask,
                          grid=sampled_tgt_normalized_2d_locations,
                          mode='nearest', padding_mode='zeros', align_corners=False). \
            reshape(1, num_points) * \
            sampled_pos_depth_mask.reshape(1, num_points)

        # 2*C_feat x N (grad_x,grad_y)
        sampled_tgt_feature_spatial_grads = \
            F.grid_sample(input=tgt_feature_map_spatial_grad,
                          grid=sampled_tgt_normalized_2d_locations,
                          mode='bilinear', padding_mode='zeros',
                          align_corners=False).reshape(2 * feat_channel, num_points)
        # C_feat x N
        sampled_src_tgt_feature_differences = torch.mul(sampled_tgt_valid_masks,
                                                        (sampled_src_features - sampled_tgt_features))

        # N x 2 x 6
        jac_2d_loc_wrt_cam_pose = self.jacobian_projected_2d_location_wrt_camera_pose(
            sampled_tgt_3d_locations=sampled_tgt_3d_locations, fx=fx, fy=fy, mode="wh")
        # N x C_feat x 2
        jac_feature_wrt_2d_loc = \
            sampled_tgt_feature_spatial_grads.reshape(
                2, feat_channel, num_points).permute(2, 1, 0)
        # N x C_feat x 6 -> N*C_feat x 6
        jac_feature_wrt_pose = torch.bmm(jac_feature_wrt_2d_loc, jac_2d_loc_wrt_cam_pose). \
            reshape(num_points * feat_channel, 6)
        # N x 2 x 1
        jac_2d_loc_wrt_depth = \
            self.jacobian_projected_2d_location_wrt_src_depth(rotated_sampled_src_homo_2d_locations=rotated_sampled_src_homo_2d_locations,
                                                              sampled_tgt_3d_locations=sampled_tgt_3d_locations,
                                                              fx=fx, fy=fy, mode="wh")
        # N x 2 x C_code
        jac_2d_loc_wrt_code = torch.bmm(jac_2d_loc_wrt_depth, guess_scale *
                                        sampled_depth_jac_code_hierarchy.reshape(num_points, 1, code_length))
        # N x C_feat x C_code
        jac_feature_wrt_code = torch.bmm(
            jac_feature_wrt_2d_loc, jac_2d_loc_wrt_code)
        # N*C_feat x C_code
        jac_feature_wrt_code = jac_feature_wrt_code.reshape(
            num_points * feat_channel, code_length)

        # N x 2 x 1
        jac_2d_loc_wrt_scale = torch.bmm(jac_2d_loc_wrt_depth,
                                         guess_sampled_depths.reshape(num_points, 1, 1) / guess_scale)
        # N x C_feat x 1 -> N*C_feat x 1
        jac_feature_wrt_scale = torch.bmm(jac_feature_wrt_2d_loc, jac_2d_loc_wrt_scale). \
            reshape(num_points * feat_channel, 1)

        # N*C_feat x (6 + 1 + C_code)
        jac_feature_wrt_pose_scale_code = torch.cat([jac_feature_wrt_pose,
                                                     jac_feature_wrt_scale, jac_feature_wrt_code], dim=1)

        return jac_feature_wrt_pose_scale_code, sampled_src_tgt_feature_differences, sampled_tgt_valid_masks

    def reproj_term(self, keypoint_depth_bias, keypoint_depth_jac_code_hierarchy,
                    keypoint_homo_2d_locations, matched_2d_hw_locations,
                    camera_intrinsics, guess_rotation, guess_translation,
                    guess_code_hierarchy, guess_scale, width):
        # M x C_code_total
        num_points, code_length = keypoint_depth_jac_code_hierarchy.shape
        # M
        guess_keypoint_depths = guess_scale * (keypoint_depth_bias +
                                               torch.matmul(keypoint_depth_jac_code_hierarchy, guess_code_hierarchy))
        # Scalar
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[0, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[0, 3]

        # 3 x M
        rotated_keypoint_homo_2d_locations = torch.matmul(guess_rotation.reshape(3, 3),
                                                          keypoint_homo_2d_locations.reshape(3, num_points))
        # 3 x M
        corr_tgt_3d_locations = guess_keypoint_depths * \
            rotated_keypoint_homo_2d_locations + \
            guess_translation.reshape(3, 1)

        tgt_z = corr_tgt_3d_locations[2:3, :]
        sampled_valid_mask = (tgt_z >= self.depth_eps).float()  # 1 x M
        sampled_valid_mask.requires_grad = True
        tgt_z = torch.clamp_min(tgt_z, min=self.depth_eps)
        corr_tgt_3d_locations = torch.cat(
            [corr_tgt_3d_locations[:2], tgt_z], dim=0)
        corr_tgt_homo_2d_locations = corr_tgt_3d_locations / \
            corr_tgt_3d_locations[2:3, :]
        # M
        corr_tgt_2d_locations_x = corr_tgt_homo_2d_locations[0, :] * fx + cx
        corr_tgt_2d_locations_y = corr_tgt_homo_2d_locations[1, :] * fy + cy
        # M x 2
        corr_tgt_2d_hw_locations = torch.stack([corr_tgt_2d_locations_y,
                                                corr_tgt_2d_locations_x], dim=1)
        # M x 2 x 6
        jac_corr_2d_loc_wrt_cam_pose = self.jacobian_projected_2d_location_wrt_camera_pose(
            sampled_tgt_3d_locations=corr_tgt_3d_locations,
            fx=fx, fy=fy, mode='hw')

        # Mask out rows related to negative depth values
        jac_corr_2d_loc_wrt_cam_pose = (jac_corr_2d_loc_wrt_cam_pose *
                                        sampled_valid_mask.reshape(num_points, 1, 1)). \
            reshape(num_points * 2, 6)
        # M*2
        corr_tgt_2d_location_diff = (sampled_valid_mask.reshape(num_points, 1) *
                                     (matched_2d_hw_locations - corr_tgt_2d_hw_locations)).reshape(
            num_points * 2)
        # M*2
        sqrt_cauchy_weights, cauchy_corr_tgt_2d_location_error \
            = self.cauchy_sqrt_jacobian_weight_and_error(torch.square(corr_tgt_2d_location_diff),
                                                         weight=torch.abs(
                                                             self.reproj_term_weight),
                                                         cauchy_param=torch.abs(self.reproj_cauchy_param) * (
                width ** 2))

        # M*2 x 6
        cauchy_jac_corr_2d_loc_wrt_cam_pose = sqrt_cauchy_weights.reshape(
            -1, 1) * jac_corr_2d_loc_wrt_cam_pose
        # M*2 x 1
        cauchy_corr_tgt_2d_location_diff = (sqrt_cauchy_weights * corr_tgt_2d_location_diff). \
            reshape(2 * num_points, 1)

        # M x 2 x 1
        jac_corr_2d_loc_wrt_depth = \
            self.jacobian_projected_2d_location_wrt_src_depth(rotated_sampled_src_homo_2d_locations=rotated_keypoint_homo_2d_locations,
                                                              sampled_tgt_3d_locations=corr_tgt_3d_locations,
                                                              fx=fx, fy=fy, mode='hw')

        # M x 2 x C_code
        jac_corr_2d_loc_wrt_code = torch.bmm(jac_corr_2d_loc_wrt_depth,
                                             guess_scale * keypoint_depth_jac_code_hierarchy.
                                             reshape(num_points, 1, code_length))
        # M x 2 x 1
        jac_corr_2d_loc_wrt_scale = torch.bmm(jac_corr_2d_loc_wrt_depth,
                                              guess_keypoint_depths.reshape(num_points, 1, 1) / guess_scale)
        # M*2 x 1
        jac_corr_2d_loc_wrt_scale = (jac_corr_2d_loc_wrt_scale *
                                     sampled_valid_mask.reshape(num_points, 1, 1)).reshape(num_points * 2, 1)
        cauchy_jac_corr_2d_loc_wrt_scale = sqrt_cauchy_weights.reshape(
            -1, 1) * jac_corr_2d_loc_wrt_scale

        # Mask out rows related to negative depth values
        jac_corr_2d_loc_wrt_code = (jac_corr_2d_loc_wrt_code *
                                    sampled_valid_mask.reshape(num_points, 1, 1)). \
            reshape(num_points * 2, code_length)

        # M*2 x 6
        cauchy_jac_corr_2d_loc_wrt_code = sqrt_cauchy_weights.reshape(
            -1, 1) * jac_corr_2d_loc_wrt_code

        # M*2 x (6 + 1 + C_code)
        cauchy_jac_corr_2d_loc_wrt_pose_scale_code = \
            torch.cat([cauchy_jac_corr_2d_loc_wrt_cam_pose, cauchy_jac_corr_2d_loc_wrt_scale,
                       cauchy_jac_corr_2d_loc_wrt_code], dim=1)

        return cauchy_jac_corr_2d_loc_wrt_pose_scale_code, \
            cauchy_corr_tgt_2d_location_diff, cauchy_corr_tgt_2d_location_error, sampled_valid_mask

    def geometry_term(self, tgt_valid_mask, tgt_depth_map, mean_squared_tgt_depth_value,
                      tgt_depth_map_spatial_grad,
                      sampled_depth_bias, sampled_depth_jac_code_hierarchy,
                      sampled_homo_2d_locations,
                      camera_intrinsics, guess_rotation, guess_translation,
                      guess_code_hierarchy, guess_scale):
        # 1 x 1 x H x W
        _, _, height, width = tgt_valid_mask.shape
        # N x C_code_total
        num_points, code_length = sampled_depth_jac_code_hierarchy.shape
        # N
        guess_sampled_depths = guess_scale * (sampled_depth_bias +
                                              torch.matmul(sampled_depth_jac_code_hierarchy, guess_code_hierarchy))
        # 3 x N
        rotated_sampled_src_homo_2d_locations = torch.matmul(guess_rotation.reshape(3, 3),
                                                             sampled_homo_2d_locations.reshape(3, num_points))
        # 3 x N
        sampled_tgt_3d_locations = guess_sampled_depths.reshape(1, num_points) * \
            rotated_sampled_src_homo_2d_locations + \
            guess_translation.reshape(3, 1)
        # 1 x N
        tgt_z = sampled_tgt_3d_locations[2:3, :]
        sampled_pos_depth_mask = (tgt_z >= self.depth_eps).float()
        tgt_z = torch.clamp_min(tgt_z, min=self.depth_eps)
        # 3 x N
        sampled_tgt_3d_locations = torch.cat(
            [sampled_tgt_3d_locations[:2], tgt_z], dim=0)

        # Scalar
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[0, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[0, 3]

        # Normalize the locations by depth values
        sampled_tgt_homo_2d_locations = sampled_tgt_3d_locations / \
            sampled_tgt_3d_locations[2:3, :]
        # 1 x N
        sampled_tgt_2d_locations_x = sampled_tgt_homo_2d_locations[0:1, :] * fx + cx
        sampled_tgt_2d_locations_y = sampled_tgt_homo_2d_locations[1:2, :] * fy + cy

        # 1 x 1 x N x 2 : Normalize to range [-1, 1]
        sampled_tgt_normalized_2d_locations = \
            torch.cat([(sampled_tgt_2d_locations_x.reshape(-1, 1) + 0.5) * (2 / width) - 1.0,
                       (sampled_tgt_2d_locations_y.reshape(-1, 1) + 0.5) * (2 / height) - 1.0], dim=1). \
            reshape(1, 1, num_points, 2)
        # 1 x N
        sampled_tgt_valid_masks = \
            F.grid_sample(input=tgt_valid_mask,
                          grid=sampled_tgt_normalized_2d_locations,
                          mode='nearest', padding_mode='zeros', align_corners=False). \
            reshape(1, num_points) * \
            sampled_pos_depth_mask.reshape(1, num_points)

        # N x 1 x 2 (grad_x,grad_y)
        sampled_tgt_depth_spatial_grads = F.grid_sample(input=tgt_depth_map_spatial_grad,
                                                        grid=sampled_tgt_normalized_2d_locations,
                                                        mode='bilinear', padding_mode='zeros',
                                                        align_corners=False). \
            reshape(2, 1, num_points).permute(2, 1, 0)
        # 1 x 1 x 1 x N -> N
        sampled_tgt_depths = F.grid_sample(input=tgt_depth_map,
                                           grid=sampled_tgt_normalized_2d_locations,
                                           mode='bilinear', padding_mode='zeros',
                                           align_corners=False).reshape(num_points)
        # N x 2 x 6
        jac_2d_loc_wrt_cam_pose = self.jacobian_projected_2d_location_wrt_camera_pose(
            sampled_tgt_3d_locations=sampled_tgt_3d_locations, fx=fx, fy=fy)

        # N x 1 x 6
        jac_depth_wrt_pose = \
            self.jacobian_transformed_depth_wrt_camera_pose(sampled_tgt_3d_locations=sampled_tgt_3d_locations) - \
            torch.bmm(sampled_tgt_depth_spatial_grads, jac_2d_loc_wrt_cam_pose)

        # N x 2 x 1
        jac_2d_loc_wrt_src_depth = \
            self.jacobian_projected_2d_location_wrt_src_depth(
                rotated_sampled_src_homo_2d_locations=rotated_sampled_src_homo_2d_locations,
                sampled_tgt_3d_locations=sampled_tgt_3d_locations, fx=fx, fy=fy)

        # N x 1 x 1
        jac_transformed_depth_wrt_src_depth = \
            self.jacobian_transformed_depth_wrt_src_depth(
                rotated_sampled_src_homo_2d_locations)
        # N x 1 x C_code
        jac_depth_wrt_code_1 = torch.bmm(jac_transformed_depth_wrt_src_depth,
                                         guess_scale * sampled_depth_jac_code_hierarchy.reshape(
                                             num_points, 1, code_length))
        jac_depth_wrt_code_2 = \
            torch.bmm(torch.bmm(sampled_tgt_depth_spatial_grads, jac_2d_loc_wrt_src_depth),
                      guess_scale * sampled_depth_jac_code_hierarchy.reshape(num_points, 1, code_length))
        # N x 1 x C_code
        jac_depth_wrt_code = jac_depth_wrt_code_1 - jac_depth_wrt_code_2
        # N x 1 x 1
        jac_depth_wrt_scale_1 = torch.bmm(jac_transformed_depth_wrt_src_depth,
                                          guess_sampled_depths.reshape(num_points, 1, 1) / guess_scale)
        # N x 1 x 1
        jac_depth_wrt_scale_2 = torch.bmm(torch.bmm(sampled_tgt_depth_spatial_grads,
                                                    jac_2d_loc_wrt_src_depth),
                                          guess_sampled_depths.reshape(num_points, 1, 1) / guess_scale)
        jac_depth_wrt_scale = jac_depth_wrt_scale_1 - jac_depth_wrt_scale_2
        # N x 1 x (6 + 1 + C_code) and masking out invalid elements
        jac_depth_wrt_pose_scale_code = sampled_tgt_valid_masks.reshape(num_points, 1, 1) * \
            torch.cat([jac_depth_wrt_pose, jac_depth_wrt_scale,
                      jac_depth_wrt_code], dim=2)

        # N
        depth_difference = sampled_tgt_valid_masks.reshape(num_points) * \
            (sampled_tgt_depths.reshape(num_points) - tgt_z.reshape(num_points))
        # cauchy loss
        sqrt_cauchy_weights, cauchy_depth_error = \
            self.cauchy_sqrt_jacobian_weight_and_error(torch.square(depth_difference),
                                                       weight=1.0,
                                                       cauchy_param=torch.abs(self.geometry_cauchy_param_factor) *
                                                       mean_squared_tgt_depth_value)
        # N x (7 + Code)
        cauchy_jac_depth_wrt_pose_scale_code = sqrt_cauchy_weights.reshape(num_points, 1) * \
            jac_depth_wrt_pose_scale_code.reshape(num_points, 7 + code_length)
        # N x 1
        cauchy_depth_difference = (
            sqrt_cauchy_weights * depth_difference).reshape(num_points, 1)

        return cauchy_jac_depth_wrt_pose_scale_code, cauchy_depth_difference, cauchy_depth_error, \
            sampled_tgt_valid_masks

    def code_term(self, guess_code_hierarchy):
        guess_code_hierarchy = guess_code_hierarchy.reshape(-1, 1)
        code_length = guess_code_hierarchy.shape[0]
        # prior code is all-zero code
        code_diff = -guess_code_hierarchy
        # 6 + 1 + C_code_total
        jac_code_wrt_pose_scale_code = torch.cat([
            torch.zeros(code_length, 7, dtype=torch.float32,
                        device=guess_code_hierarchy.device),
            torch.eye(code_length, dtype=torch.float32, device=guess_code_hierarchy.device)], dim=1)
        return jac_code_wrt_pose_scale_code, code_diff

    def scale_term(self, guess_scale, init_scale, code_length):
        guess_scale = guess_scale.reshape(1, 1)
        init_scale = init_scale.reshape(1, 1)
        scale_log_diff = torch.log(init_scale) - torch.log(guess_scale)
        jac_scale_wrt_pose_scale_code = torch.cat([
            torch.zeros(1, 6, dtype=torch.float32).to(guess_scale.device),
            1.0 / guess_scale,
            torch.zeros(1, code_length, dtype=torch.float32).to(guess_scale.device)], dim=1)
        return jac_scale_wrt_pose_scale_code, scale_log_diff

    @staticmethod
    def colorize_feature_maps(src_feature_map, tgt_feature_map):
        fit = umap.UMAP(
            n_neighbors=5,
            min_dist=0.1,
            n_components=3,
            metric='euclidean'
        )
        _, feat_channel, height, width = src_feature_map.shape
        src_features = src_feature_map.reshape(feat_channel, height * width)
        tgt_features = tgt_feature_map.reshape(feat_channel, height * width)
        # 2*H*W x C_feat
        feats = torch.cat([src_features, tgt_features], dim=1).permute(
            1, 0).detach().cpu().numpy().astype(np.float32)

        color_coded_feats = fit.fit_transform(feats)
        color_max = np.amax(color_coded_feats, axis=0, keepdims=True)
        color_min = np.amin(color_coded_feats, axis=0, keepdims=True)
        color_coded_feats = (color_coded_feats - color_min) / \
            (color_max - color_min)
        color_coded_feats = color_coded_feats.astype(np.float64)

        src_color_coded_feats = color_coded_feats[:height * width]
        tgt_color_coded_feats = color_coded_feats[height * width:]

        # 1 x 3 x H x W
        src_color_coded_feats = np.moveaxis(src_color_coded_feats.reshape(1, height, width, 3), source=[0, 1, 2, 3],
                                            destination=[0, 2, 3, 1])
        tgt_color_coded_feats = np.moveaxis(tgt_color_coded_feats.reshape(1, height, width, 3), source=[0, 1, 2, 3],
                                            destination=[0, 2, 3, 1])

        return src_color_coded_feats, tgt_color_coded_feats

    @torch.no_grad()
    def warp_feature_map(self, guess_src_depth_map, guess_rotation, guess_translation,
                         tgt_feature_map, src_valid_mask, tgt_valid_mask, camera_intrinsics):
        nbatch, feat_channel, height, width = tgt_feature_map.shape
        assert (nbatch == 1)

        src_hw_2d_locations = torch.nonzero(
            src_valid_mask.reshape(height, width) >= 0.9)
        num_points = src_hw_2d_locations.shape[0]
        src_1d_locations = (
            width * src_hw_2d_locations[:, 0] + src_hw_2d_locations[:, 1]).reshape(-1).long()
        # 1 x M
        sampled_src_depths = torch.index_select(
            guess_src_depth_map.reshape(1, -1), dim=1, index=src_1d_locations)
        # 3 x M
        sampled_homo_2d_locations = utils.compute_homogenous_2d_locations(sampled_2d_locations=src_hw_2d_locations.
                                                                          reshape(
                                                                              1, num_points, 2),
                                                                          camera_intrinsics=camera_intrinsics). \
            reshape(3, num_points)

        # 3 x M
        rotated_sampled_src_homo_2d_locations = torch.matmul(guess_rotation.reshape(3, 3),
                                                             sampled_homo_2d_locations.reshape(3, num_points))
        # 3 x M
        sampled_tgt_3d_locations = sampled_src_depths.reshape(1, num_points) * \
            rotated_sampled_src_homo_2d_locations + \
            guess_translation.reshape(3, 1)

        tgt_z = sampled_tgt_3d_locations[2:3, :]
        sampled_pos_depth_mask = (tgt_z >= self.depth_eps).float()
        tgt_z = torch.clamp_min(tgt_z, min=self.depth_eps)
        sampled_tgt_3d_locations = torch.cat(
            [sampled_tgt_3d_locations[:2], tgt_z], dim=0)

        # fx fy cx cy: 1 x 1
        fx = camera_intrinsics[:, 0:1].reshape(1, 1)
        fy = camera_intrinsics[:, 1:2].reshape(1, 1)
        cx = camera_intrinsics[:, 2:3].reshape(1, 1)
        cy = camera_intrinsics[:, 3:4].reshape(1, 1)

        # Normalize the locations by depth values
        sampled_tgt_homo_2d_locations = sampled_tgt_3d_locations / \
            sampled_tgt_3d_locations[2:3, :]
        # obtain pixel locations in the target frame
        # 1 x M
        sampled_tgt_2d_locations_x = sampled_tgt_homo_2d_locations[0:1, :] * fx + cx
        sampled_tgt_2d_locations_y = sampled_tgt_homo_2d_locations[1:2, :] * fy + cy

        # 1 x 1 x N x 2 : Normalize to range [-1, 1]
        sampled_tgt_normalized_2d_locations = \
            torch.cat([(sampled_tgt_2d_locations_x.reshape(-1, 1) + 0.5) * (2 / width) - 1.0,
                       (sampled_tgt_2d_locations_y.reshape(-1, 1) + 0.5) * (2 / height) - 1.0], dim=1). \
            reshape(1, 1, num_points, 2)

        # 1 x C_feat x M
        sampled_tgt_features = F.grid_sample(input=tgt_feature_map,
                                             grid=sampled_tgt_normalized_2d_locations,
                                             mode='bilinear', padding_mode='zeros',
                                             align_corners=False). \
            reshape(1, feat_channel, num_points)

        # 1 x 1 x M
        sampled_tgt_valid_masks = \
            F.grid_sample(input=tgt_valid_mask,
                          grid=sampled_tgt_normalized_2d_locations,
                          mode='nearest', padding_mode='zeros', align_corners=False). \
            reshape(1, 1, num_points)

        sampled_tgt_valid_masks = sampled_pos_depth_mask.reshape(
            1, 1, num_points) * sampled_tgt_valid_masks

        warped_tgt_feature_map = torch.zeros_like(
            tgt_feature_map).reshape(feat_channel, height * width)
        warped_tgt_valid_mask = torch.zeros_like(
            tgt_valid_mask).reshape(1, height * width)

        warped_tgt_feature_map[:, src_1d_locations] = sampled_tgt_features.reshape(feat_channel, num_points) * \
            sampled_tgt_valid_masks.reshape(1, num_points)
        warped_tgt_valid_mask[:, src_1d_locations] = sampled_tgt_valid_masks.reshape(
            1, num_points)
        return warped_tgt_feature_map.reshape(feat_channel, height, width).permute(1, 2, 0).cpu().numpy(), \
            warped_tgt_valid_mask.reshape(
                1, height, width).permute(1, 2, 0).cpu().numpy()

    @torch.no_grad()
    def visualize_feature_map_warping(self, color_src_desc_feature_map, color_tgt_desc_feature_map,
                                      color_src_feature_map, color_tgt_feature_map,
                                      src_desc_feature_map, tgt_desc_feature_map,
                                      src_feature_map, tgt_feature_map,
                                      src_depth_map, guess_rotation, guess_translation, src_valid_mask,
                                      tgt_valid_mask, camera_intrinsics
                                      ):
        if color_src_feature_map is None:
            color_src_feature_map, color_tgt_feature_map = \
                self.colorize_feature_maps(src_feature_map, tgt_feature_map)
            color_src_desc_feature_map, color_tgt_desc_feature_map = \
                self.colorize_feature_maps(
                    src_desc_feature_map, tgt_desc_feature_map)

        _, _, curr_height, curr_width = color_src_feature_map.shape

        if isinstance(color_tgt_feature_map, np.ndarray):
            color_tgt_feature_map = torch.from_numpy(
                color_tgt_feature_map).float().cuda()
            color_tgt_desc_feature_map = torch.from_numpy(
                color_tgt_desc_feature_map).float().cuda()

        # H x W x 3
        warped_tgt_feature_map, _ = self.warp_feature_map(src_depth_map, guess_rotation,
                                                          guess_translation,
                                                          color_tgt_feature_map, src_valid_mask,
                                                          tgt_valid_mask,
                                                          camera_intrinsics)
        warped_tgt_desc_feature_map, _ = self.warp_feature_map(src_depth_map, guess_rotation,
                                                               guess_translation,
                                                               color_tgt_desc_feature_map,
                                                               src_valid_mask, tgt_valid_mask,
                                                               camera_intrinsics)

        display_src_feature_map = np.moveaxis(color_src_feature_map.
                                              reshape(
                                                  (3, curr_height, curr_width)),
                                              source=[0, 1, 2], destination=[2, 0, 1])
        display_tgt_feature_map = np.moveaxis(color_tgt_feature_map.data.cpu().numpy().
                                              reshape(
                                                  (3, curr_height, curr_width)),
                                              source=[0, 1, 2], destination=[2, 0, 1])

        display_src_desc_feature_map = np.moveaxis(color_src_desc_feature_map.
                                                   reshape(
                                                       (3, curr_height, curr_width)),
                                                   source=[0, 1, 2], destination=[2, 0, 1])
        display_tgt_desc_feature_map = np.moveaxis(color_tgt_desc_feature_map.data.cpu().numpy().
                                                   reshape(
                                                       (3, curr_height, curr_width)),
                                                   source=[0, 1, 2], destination=[2, 0, 1])

        display_src_feature_map = (
            255 * display_src_feature_map).astype(np.uint8)
        display_tgt_feature_map = (
            255 * display_tgt_feature_map).astype(np.uint8)

        display_src_desc_feature_map = (
            255 * display_src_desc_feature_map).astype(np.uint8)
        display_tgt_desc_feature_map = (
            255 * display_tgt_desc_feature_map).astype(np.uint8)

        warped_tgt_feature_map = (
            255 * warped_tgt_feature_map).astype(np.uint8)
        warped_tgt_desc_feature_map = (
            255 * warped_tgt_desc_feature_map).astype(np.uint8)

        return display_src_feature_map, display_tgt_feature_map, \
            display_src_desc_feature_map, display_tgt_desc_feature_map, \
            warped_tgt_feature_map, warped_tgt_desc_feature_map, \
            color_src_feature_map, color_tgt_feature_map, \
            color_src_desc_feature_map, color_tgt_desc_feature_map

    @torch.no_grad()
    def visualize_input_image_warping(self, src_depth_map,
                                      guess_rotation, guess_translation,
                                      src_valid_mask, tgt_valid_mask, camera_intrinsics,
                                      src_input_image, tgt_input_image
                                      ):
        _, _, height, width = src_depth_map.shape
        warped_tgt_input_image, warped_tgt_valid_mask = self.warp_feature_map(src_depth_map, guess_rotation,
                                                                              guess_translation,
                                                                              tgt_input_image, src_valid_mask,
                                                                              tgt_valid_mask,
                                                                              camera_intrinsics)
        warped_tgt_input_image = (
            255 * warped_tgt_input_image * warped_tgt_valid_mask).astype(np.uint8)

        src_input_image = src_input_image.reshape(3, height, width). \
            permute(1, 2, 0).data.cpu().numpy()
        src_input_image_display = \
            np.uint8(255 * src_input_image)
        tgt_input_image = tgt_input_image.reshape(3, height, width). \
            permute(1, 2, 0).data.cpu().numpy()
        tgt_input_image_display = \
            np.uint8(255 * tgt_input_image)

        reverse_src_input_image_display = cv2.cvtColor(
            src_input_image_display, cv2.COLOR_RGB2BGR)

        chessboard = warped_tgt_valid_mask * \
            cv2.resize(self.chessboard,
                       dsize=(width, height), interpolation=cv2.INTER_NEAREST).reshape((height, width, 1))
        src_warped_tgt_input_image_overlay = (chessboard * warped_tgt_input_image +
                                              (1.0 - chessboard) * reverse_src_input_image_display).astype(np.uint8)

        return src_warped_tgt_input_image_overlay, \
            src_input_image_display, tgt_input_image_display, \
            warped_tgt_input_image

    def visualize_feature_matches(self, src_image, tgt_image, src_keypoint_2d_hw_locations,
                                  tgt_keypoint_2d_hw_locations, corr_tgt_2d_locations, corr_tgt_valid_mask,
                                  generate_match_indexes=False):
        _, _, height, width = src_image.shape

        # do a downsampling of keypoints for quick visualization here
        _, num_corr_points, _ = src_keypoint_2d_hw_locations.shape
        # M x 2
        src_keypoint_2d_hw_locations = src_keypoint_2d_hw_locations.reshape(
            num_corr_points, 2)
        tgt_keypoint_2d_hw_locations = tgt_keypoint_2d_hw_locations.reshape(
            num_corr_points, 2)
        corr_tgt_2d_locations = corr_tgt_2d_locations.reshape(
            num_corr_points, 2)

        if self.display_match_indexes is None or generate_match_indexes:
            self.display_match_indexes = \
                np.random.choice(np.arange(start=0, stop=num_corr_points), size=min(num_corr_points,
                                                                                    self.num_display_matches),
                                 replace=False)

        src_keypoint_2d_hw_locations = src_keypoint_2d_hw_locations[self.display_match_indexes, :]
        tgt_keypoint_2d_hw_locations = tgt_keypoint_2d_hw_locations[self.display_match_indexes, :]
        corr_tgt_2d_locations = corr_tgt_2d_locations[self.display_match_indexes, :]

        corr_tgt_valid_mask = corr_tgt_valid_mask.reshape(-1)
        valid_subindexes_in_display_indexes = torch.nonzero(
            corr_tgt_valid_mask[self.display_match_indexes] > 0.5)

        # H x W x 3
        src_image_display = src_image.reshape(
            3, height, width).permute(1, 2, 0).cpu().numpy()
        tgt_image_display = tgt_image.reshape(
            3, height, width).permute(1, 2, 0).cpu().numpy()

        src_image_display = np.uint8(255 * src_image_display)
        tgt_image_display = np.uint8(255 * tgt_image_display)

        src_keypoints_list = list()
        tgt_keypoints_list = list()
        corr_tgt_keypoints_list = list()
        matches_list = list()
        for i in range(len(self.display_match_indexes)):
            src_2d_loc = src_keypoint_2d_hw_locations[i, :]
            keypoint = cv2.KeyPoint(
                x=src_2d_loc[1], y=src_2d_loc[0], _size=0.1)
            src_keypoints_list.append(keypoint)

            tgt_2d_loc = tgt_keypoint_2d_hw_locations[i, :]
            keypoint = cv2.KeyPoint(
                x=tgt_2d_loc[1], y=tgt_2d_loc[0], _size=0.1)
            tgt_keypoints_list.append(keypoint)

            corr_tgt_2d_loc = corr_tgt_2d_locations[i, :]
            keypoint = cv2.KeyPoint(
                x=corr_tgt_2d_loc[1], y=corr_tgt_2d_loc[0], _size=0.1)
            corr_tgt_keypoints_list.append(keypoint)

            matches_list.append(cv2.DMatch(
                _queryIdx=i, _trainIdx=i, _distance=0.0))

        feature_matches_desc = cv2.drawMatches(src_image_display, src_keypoints_list,
                                               tgt_image_display, tgt_keypoints_list,
                                               matches_list, flags=2, outImg=None)

        valid_subindexes_in_display_indexes = valid_subindexes_in_display_indexes.reshape(
            -1).cpu().numpy()

        temp = list()
        temp2 = list()
        for idx in valid_subindexes_in_display_indexes:
            temp.append(src_keypoints_list[idx])
            temp2.append(corr_tgt_keypoints_list[idx])

        feature_matches_est = cv2.drawMatches(src_image_display,
                                              temp,
                                              tgt_image_display,
                                              temp2,
                                              matches_list[:len(
                                                  valid_subindexes_in_display_indexes)], flags=2,
                                              outImg=None)

        return feature_matches_desc, feature_matches_est

    @torch.no_grad()
    def generate_visualization_result(self, src_pred_depth_map,
                                      tgt_pred_depth_map,
                                      src_desc_feature_map, tgt_desc_feature_map,
                                      src_feature_map, tgt_feature_map,
                                      guess_rotation, guess_translation, src_valid_mask,
                                      tgt_valid_mask, camera_intrinsics,
                                      color_src_desc_feature_map,
                                      color_tgt_desc_feature_map,
                                      color_src_feature_map, color_tgt_feature_map,
                                      src_input_image, tgt_input_image,
                                      src_gt_depth_map, tgt_gt_depth_map,
                                      keypoint_2d_hw_locations,
                                      keypoint_homo_2d_locations,
                                      keypoint_normalized_2d_locations,
                                      matched_2d_hw_locations,
                                      gt_flow_map, generate_match_indexes,
                                      gt_flow_mask, return_feature_map):
        # 1 x M
        src_keypoint_depths = F.grid_sample(input=src_pred_depth_map,
                                            grid=keypoint_normalized_2d_locations,
                                            mode='bilinear', padding_mode='zeros',
                                            align_corners=False).reshape(1, -1)

        display_src_feature_map, display_tgt_feature_map, \
            display_src_desc_feature_map, display_tgt_desc_feature_map, \
            warped_tgt_feature_map, warped_tgt_desc_feature_map, color_src_feature_map, color_tgt_feature_map, \
            color_src_desc_feature_map, color_tgt_desc_feature_map = \
            self.visualize_feature_map_warping(color_src_desc_feature_map, color_tgt_desc_feature_map,
                                               color_src_feature_map, color_tgt_feature_map,
                                               src_desc_feature_map, tgt_desc_feature_map,
                                               src_feature_map, tgt_feature_map,
                                               src_pred_depth_map, guess_rotation, guess_translation, src_valid_mask,
                                               tgt_valid_mask, camera_intrinsics
                                               )

        src_warped_tgt_input_image_overlay, \
            src_input_image_display, tgt_input_image_display, \
            warped_tgt_input_image = \
            self.visualize_input_image_warping(src_pred_depth_map,
                                               guess_rotation, guess_translation,
                                               src_valid_mask, tgt_valid_mask, camera_intrinsics,
                                               src_input_image, tgt_input_image)

        gt_src_depth_map_display, max_depth = \
            utils.visualize_depth_map(src_gt_depth_map, max_depth=None)

        gt_tgt_depth_map_display, max_depth = \
            utils.visualize_depth_map(tgt_gt_depth_map, max_depth=None)

        scale = torch.sum(tgt_gt_depth_map * tgt_valid_mask) / \
            torch.clamp_min(
                torch.sum(torch.abs(tgt_pred_depth_map) * tgt_valid_mask), min=1.0e-8)

        pred_src_depth_map_display, _ = \
            utils.visualize_depth_map(
                torch.abs(src_pred_depth_map), max_depth=max_depth / scale)
        pred_tgt_depth_map_display, _ = \
            utils.visualize_depth_map(
                torch.abs(tgt_pred_depth_map), max_depth=max_depth / scale)

        input_row = cv2.hconcat(
            [src_input_image_display, tgt_input_image_display])
        gt_depth_row = cv2.hconcat(
            [gt_src_depth_map_display, gt_tgt_depth_map_display])
        pred_depth_row = cv2.hconcat(
            [pred_src_depth_map_display, pred_tgt_depth_map_display])
        warp_input_row = cv2.hconcat(
            [src_warped_tgt_input_image_overlay, warped_tgt_input_image])

        feature_map_row = cv2.hconcat(
            [display_src_feature_map, display_tgt_feature_map])
        feature_desc_row = cv2.hconcat(
            [display_src_desc_feature_map, display_tgt_desc_feature_map])
        warped_feature_row = cv2.hconcat(
            [warped_tgt_feature_map, warped_tgt_desc_feature_map])

        corr_tgt_2d_locations, corr_tgt_valid_mask = \
            utils.transform_keypoint_locations_ba(src_keypoint_depths, keypoint_homo_2d_locations,
                                                  guess_rotation, guess_translation, camera_intrinsics, self.depth_eps)

        feat_match_desc_row, feature_match_est_row = self.visualize_feature_matches(
            src_input_image, tgt_input_image,
            keypoint_2d_hw_locations, matched_2d_hw_locations, corr_tgt_2d_locations, corr_tgt_valid_mask,
            generate_match_indexes)

        pred_flow_map, pred_flow_mask = utils.generate_dense_flow_map(depth_map=src_pred_depth_map, valid_mask=src_valid_mask,
                                                                      rotation=guess_rotation, translation=guess_translation,
                                                                      camera_intrinsics=camera_intrinsics,
                                                                      depth_eps=self.depth_eps)
        # visualize gt and pred flow map
        gt_flow_mask_numpy = gt_flow_mask.reshape(
            *gt_flow_mask.shape[2:], 1).data.cpu().numpy()
        pred_flow_mask_numpy = pred_flow_mask.reshape(
            *gt_flow_mask.shape[2:], 1).data.cpu().numpy()

        gt_flow_map_display, max_v = utils.draw_flow_map(
            flows=gt_flow_map, max_v=None)
        pred_flow_map_display, _ = utils.draw_flow_map(
            flows=pred_flow_map, max_v=max_v)
        flow_row = cv2.hconcat(
            src=[gt_flow_map_display * gt_flow_mask_numpy.astype(np.uint8),
                 pred_flow_map_display * pred_flow_mask_numpy.astype(np.uint8)])

        ratio = int(256 / input_row.shape[0])
        result_1 = cv2.vconcat(
            [input_row, gt_depth_row, pred_depth_row, warp_input_row, feature_match_est_row])
        result_1 = cv2.resize(result_1, dsize=(0, 0), fx=ratio, fy=ratio)

        result_2 = cv2.vconcat(
            [feature_map_row, feature_desc_row, warped_feature_row, flow_row, feat_match_desc_row])
        result_2 = cv2.resize(result_2, dsize=(0, 0), fx=ratio, fy=ratio)
        result = cv2.hconcat([result_1, result_2])

        if return_feature_map:
            return cv2.cvtColor(result, cv2.COLOR_BGR2RGB), color_src_desc_feature_map, color_tgt_desc_feature_map, \
                color_src_feature_map, color_tgt_feature_map
        else:
            return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    @torch.no_grad()
    def compute_residual_error(self, sampled_homo_2d_locations,
                               sampled_depth_bias,
                               sampled_depth_jac_code_hierarchy,
                               keypoint_depth_bias,
                               keypoint_depth_jac_code_hierarchy,
                               sampled_src_features_pyramid,
                               keypoint_homo_2d_locations,
                               match_homo_2d_locations,
                               match_depths, tgt_depth_map,
                               mean_squared_tgt_depth_value,
                               camera_intrinsics_pyramid, scale_pyramid,
                               tgt_feature_map_pyramid, tgt_valid_mask_pyramid,
                               guess_rotation, guess_translation,
                               guess_scale, guess_code_hierarchy, init_scale):

        match_geom_error = 0
        geometry_error = 0
        # N
        guess_sampled_depths = guess_scale * (sampled_depth_bias.reshape(-1) +
                                              torch.matmul(sampled_depth_jac_code_hierarchy,
                                                           guess_code_hierarchy.reshape(-1)))
        # M
        guess_keypoint_depths = guess_scale * (keypoint_depth_bias.reshape(-1) +
                                               torch.matmul(keypoint_depth_jac_code_hierarchy,
                                                            guess_code_hierarchy.reshape(-1)))

        photo_error = self.compute_photo_error(sampled_homo_2d_locations=sampled_homo_2d_locations,
                                               sampled_depths=guess_sampled_depths,
                                               sampled_src_features_pyramid=sampled_src_features_pyramid,
                                               camera_intrinsics_pyramid=camera_intrinsics_pyramid,
                                               scale_pyramid=scale_pyramid,
                                               tgt_feature_map_pyramid=tgt_feature_map_pyramid,
                                               tgt_valid_mask_pyramid=tgt_valid_mask_pyramid,
                                               guess_rotation=guess_rotation, guess_translation=guess_translation)
        optimize_objective = photo_error

        if self.use_match_geom:
            # match geometry term
            match_geom_error = self.compute_match_geom_error(keypoint_depths=guess_keypoint_depths,
                                                             keypoint_homo_2d_locations=keypoint_homo_2d_locations,
                                                             match_homo_2d_locations=match_homo_2d_locations,
                                                             match_depths=match_depths,
                                                             mean_squared_tgt_depth_value=mean_squared_tgt_depth_value,
                                                             guess_rotation=guess_rotation,
                                                             guess_translation=guess_translation)
            optimize_objective = optimize_objective + match_geom_error

        # code term
        code_error = torch.abs(self.code_term_weight) * torch.mean(
            torch.square(-guess_code_hierarchy))
        optimize_objective = optimize_objective + code_error

        if self.use_geom:
            # geometry term
            geometry_error = self.compute_geometry_error(tgt_valid_mask=tgt_valid_mask_pyramid[-1],
                                                         tgt_depth_map=tgt_depth_map,
                                                         mean_squared_tgt_depth_value=mean_squared_tgt_depth_value,
                                                         sampled_depths=guess_sampled_depths,
                                                         sampled_homo_2d_locations=sampled_homo_2d_locations,
                                                         camera_intrinsics=camera_intrinsics_pyramid[-1],
                                                         guess_rotation=guess_rotation,
                                                         guess_translation=guess_translation)
            optimize_objective = optimize_objective + geometry_error

        scale_error = torch.abs(self.scale_term_weight) * torch.mean(
            torch.square(torch.log(init_scale.reshape(1)) - torch.log(guess_scale.reshape(1))))
        optimize_objective = optimize_objective + scale_error

        logger.debug(f"error photo: {photo_error}, match geom: {match_geom_error}, "
                     f"geom: {geometry_error}, code: {code_error}, scale: {scale_error}")
        return optimize_objective

    @torch.no_grad()
    def compute_match_geom_error(self, keypoint_depths,
                                 keypoint_homo_2d_locations, match_homo_2d_locations,
                                 match_depths, mean_squared_tgt_depth_value,
                                 guess_rotation, guess_translation):
        # N
        num_points = keypoint_depths.shape[0]
        # 3 x N
        rotated_keypoint_homo_2d_locations = torch.matmul(guess_rotation.reshape(3, 3),
                                                          keypoint_homo_2d_locations.reshape(3, num_points))
        # 3 x N
        keypoint_3d_locations_in_1 = keypoint_depths.reshape(1, num_points) * \
            rotated_keypoint_homo_2d_locations + \
            guess_translation.reshape(3, 1)

        # 3 x N
        match_3d_locations = match_depths.reshape(
            1, num_points) * match_homo_2d_locations
        locations_3d_diff = match_3d_locations - keypoint_3d_locations_in_1
        sqrt_loss_param = torch.sqrt(
            torch.abs(self.match_geom_param_factor * mean_squared_tgt_depth_value))
        locations_3d_diff_normalized = torch.abs(
            locations_3d_diff) / sqrt_loss_param
        # N
        fair_error = torch.sum(2 * (locations_3d_diff_normalized -
                                    torch.log(1.0 + locations_3d_diff_normalized)), dim=0)
        fair_error = torch.abs(
            self.match_geom_term_weight) * torch.mean(fair_error)

        return fair_error

    @torch.no_grad()
    def compute_photo_error(self, sampled_homo_2d_locations, sampled_depths,
                            sampled_src_features_pyramid,
                            camera_intrinsics_pyramid, scale_pyramid,
                            tgt_feature_map_pyramid, tgt_valid_mask_pyramid,
                            guess_rotation, guess_translation):
        optimize_objective = None
        for level in range(len(sampled_src_features_pyramid)):
            weight = torch.abs(self.photo_weight * 10) * \
                scale_pyramid[level] ** self.photo_pow_factor
            tgt_feature_map = tgt_feature_map_pyramid[level]
            sampled_src_features = sampled_src_features_pyramid[level]
            camera_intrinsics = camera_intrinsics_pyramid[level]
            tgt_valid_mask = tgt_valid_mask_pyramid[level]

            # 1 x C_feat x H x W
            _, feat_channel, height, width = tgt_feature_map.shape
            # C_feat x N
            _, num_points = sampled_src_features.shape
            # 3 x N
            rotated_sampled_src_homo_2d_locations = torch.matmul(guess_rotation.reshape(3, 3),
                                                                 sampled_homo_2d_locations.reshape(3, num_points))
            # 3 x N
            sampled_tgt_3d_locations = sampled_depths.reshape(1, num_points) * \
                rotated_sampled_src_homo_2d_locations + \
                guess_translation.reshape(3, 1)

            tgt_z = sampled_tgt_3d_locations[2:3, :]
            sampled_pos_depth_mask = (tgt_z >= self.depth_eps).float()
            tgt_z = torch.clamp_min(tgt_z, min=self.depth_eps)
            sampled_tgt_3d_locations = torch.cat(
                [sampled_tgt_3d_locations[:2], tgt_z], dim=0)

            # Scalar
            fx = camera_intrinsics[0, 0]
            fy = camera_intrinsics[0, 1]
            cx = camera_intrinsics[0, 2]
            cy = camera_intrinsics[0, 3]

            # Normalize the locations by depth values
            sampled_tgt_homo_2d_locations = sampled_tgt_3d_locations / \
                sampled_tgt_3d_locations[2:3, :]
            # process with camera intrinsics (assuming the camera intrinsics are the same for both tgt and src images)
            # 1 x N
            sampled_tgt_2d_locations_x = sampled_tgt_homo_2d_locations[0:1, :] * fx + cx
            sampled_tgt_2d_locations_y = sampled_tgt_homo_2d_locations[1:2, :] * fy + cy

            # 1 x 1 x N x 2 : Normalize to range [-1, 1]
            sampled_tgt_normalized_2d_locations = \
                torch.cat([(sampled_tgt_2d_locations_x.reshape(-1, 1) + 0.5) * (2 / width) - 1.0,
                           (sampled_tgt_2d_locations_y.reshape(-1, 1) + 0.5) * (2 / height) - 1.0], dim=1). \
                reshape(1, 1, num_points, 2)

            # 1 x C_feat x 1 x N -> C_feat x N
            sampled_tgt_features = F.grid_sample(input=tgt_feature_map,
                                                 grid=sampled_tgt_normalized_2d_locations,
                                                 mode='bilinear', padding_mode='zeros',
                                                 align_corners=False).reshape(feat_channel,
                                                                              num_points)
            # 1 x N
            sampled_tgt_valid_masks = \
                F.grid_sample(input=tgt_valid_mask,
                              grid=sampled_tgt_normalized_2d_locations,
                              mode='nearest', padding_mode='zeros', align_corners=False). \
                reshape(1, num_points) * \
                sampled_pos_depth_mask.reshape(1, num_points)

            # C_feat x N
            sampled_src_tgt_feature_differences = torch.mul(sampled_tgt_valid_masks,
                                                            (sampled_src_features -
                                                             sampled_tgt_features))

            num_samples = torch.sum(sampled_tgt_valid_masks)

            if num_samples == 0:
                error = 1.0e8
            else:
                error = weight * \
                    torch.sum(sampled_src_tgt_feature_differences **
                              2) / num_samples

            if optimize_objective is None:
                optimize_objective = error
            else:
                optimize_objective = optimize_objective + error

        return optimize_objective

    @torch.no_grad()
    def compute_reproj_error(self, keypoint_depths, keypoint_homo_2d_locations,
                             matched_2d_hw_locations, camera_intrinsics,
                             guess_rotation, guess_translation, width):
        # M x 2
        matched_2d_hw_locations = matched_2d_hw_locations.reshape(-1, 2)

        # 3 x M
        rotated_src_keypoint_homo_2d_locations = torch.matmul(guess_rotation.reshape(3, 3),
                                                              keypoint_homo_2d_locations.reshape(3, -1))
        # 3 x M
        corr_tgt_3d_locations = keypoint_depths.reshape(1, -1) * \
            rotated_src_keypoint_homo_2d_locations + \
            guess_translation.reshape(3, 1)
        # Scalar
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[0, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[0, 3]

        tgt_z = corr_tgt_3d_locations[2:3, :]
        sampled_pos_depth_mask = (tgt_z >= self.depth_eps).float()
        tgt_z = torch.clamp_min(tgt_z, min=self.depth_eps)
        corr_tgt_3d_locations = torch.cat(
            [corr_tgt_3d_locations[:2], tgt_z], dim=0)

        corr_tgt_homo_2d_locations = corr_tgt_3d_locations / \
            corr_tgt_3d_locations[2:3, :]
        # M
        corr_tgt_2d_locations_x = corr_tgt_homo_2d_locations[0, :] * fx + cx
        corr_tgt_2d_locations_y = corr_tgt_homo_2d_locations[1, :] * fy + cy
        # M x 2
        corr_tgt_2d_hw_locations = torch.stack([corr_tgt_2d_locations_y,
                                                corr_tgt_2d_locations_x], dim=1)
        # M x 2 -> M*2
        corr_tgt_2d_location_diff = (sampled_pos_depth_mask.reshape(-1, 1) *
                                     (matched_2d_hw_locations - corr_tgt_2d_hw_locations)).reshape(-1)
        # M*2
        _, cauchy_corr_tgt_2d_location_error \
            = self.cauchy_sqrt_jacobian_weight_and_error(torch.square(corr_tgt_2d_location_diff),
                                                         weight=torch.abs(
                                                             self.reproj_term_weight),
                                                         cauchy_param=torch.abs(self.reproj_cauchy_param) * (
                width ** 2))

        num_samples = torch.sum(sampled_pos_depth_mask)
        if num_samples <= 0:
            optimize_objective = 1.0e8
        else:
            optimize_objective = torch.sum(
                cauchy_corr_tgt_2d_location_error) / num_samples

        return optimize_objective

    @torch.no_grad()
    def compute_geometry_error(self, tgt_valid_mask, tgt_depth_map,
                               mean_squared_tgt_depth_value,
                               sampled_depths, sampled_homo_2d_locations,
                               camera_intrinsics, guess_rotation, guess_translation):
        # 1 x 1 x H x W
        _, _, height, width = tgt_valid_mask.shape
        # 3 x N
        rotated_sampled_src_homo_2d_locations = torch.matmul(guess_rotation.reshape(3, 3),
                                                             sampled_homo_2d_locations.reshape(3, -1))
        # 3 x N
        sampled_tgt_3d_locations = sampled_depths.reshape(1, -1) * rotated_sampled_src_homo_2d_locations + \
            guess_translation.reshape(3, 1)
        # 1 x N
        tgt_z = sampled_tgt_3d_locations[2:3, :]
        sampled_pos_depth_mask = (tgt_z >= self.depth_eps).float()
        tgt_z = torch.clamp_min(tgt_z, min=self.depth_eps)
        sampled_tgt_3d_locations = torch.cat(
            [sampled_tgt_3d_locations[:2], tgt_z], dim=0)

        # fx fy cx cy: 1 x 1
        fx = camera_intrinsics[:, 0:1].reshape(1, 1)
        fy = camera_intrinsics[:, 1:2].reshape(1, 1)
        cx = camera_intrinsics[:, 2:3].reshape(1, 1)
        cy = camera_intrinsics[:, 3:4].reshape(1, 1)

        # Normalize the locations by depth values
        sampled_tgt_homo_2d_locations = sampled_tgt_3d_locations / \
            sampled_tgt_3d_locations[2:3, :]
        # process with camera intrinsics (assuming the camera intrinsics are the same for both tgt and src images)
        # N
        sampled_tgt_2d_locations_x = sampled_tgt_homo_2d_locations[0, :] * fx + cx
        sampled_tgt_2d_locations_y = sampled_tgt_homo_2d_locations[1, :] * fy + cy

        # 1 x 1 x N x 2 : Normalize to range [-1, 1]
        sampled_tgt_normalized_2d_locations = \
            torch.stack([(sampled_tgt_2d_locations_x.reshape(-1, 1) + 0.5) * (2 / width) - 1.0,
                         (sampled_tgt_2d_locations_y.reshape(-1, 1) + 0.5) * (2 / height) - 1.0], dim=1). \
            reshape(1, 1, -1, 2)

        # N
        sampled_tgt_valid_masks = \
            F.grid_sample(input=tgt_valid_mask,
                          grid=sampled_tgt_normalized_2d_locations,
                          mode='nearest', padding_mode='zeros', align_corners=False). \
            reshape(-1) * sampled_pos_depth_mask.reshape(-1)

        # N
        sampled_tgt_depths = F.grid_sample(input=tgt_depth_map,
                                           grid=sampled_tgt_normalized_2d_locations,
                                           mode='bilinear', padding_mode='zeros',
                                           align_corners=False).reshape(-1)
        # N
        depth_difference = sampled_tgt_valid_masks * \
            (sampled_tgt_depths - tgt_z.reshape(-1))
        # use cauchy loss for this term
        _, cauchy_depth_error = \
            self.cauchy_sqrt_jacobian_weight_and_error(torch.square(depth_difference),
                                                       weight=1.0,
                                                       cauchy_param=torch.abs(self.geometry_cauchy_param_factor) *
                                                       mean_squared_tgt_depth_value)
        num_samples = torch.sum(sampled_tgt_valid_masks)

        if num_samples == 0:
            error = 1.0e8
        else:
            error = torch.abs(self.geometry_term_weight) * \
                torch.sum(cauchy_depth_error) / num_samples

        return error
