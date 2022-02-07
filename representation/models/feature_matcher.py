import torch
from utils import logger


class DiffFeatureMatcher(torch.nn.Module):
    def __init__(self, input_image_size: tuple, response_sigma: float, cycle_consis_threshold: float,
                 loss_eps=1.0e-10, norm_eps=1.0e-8):
        super(DiffFeatureMatcher, self).__init__()
        self.init_response_sigma = response_sigma
        self.response_sigma = torch.nn.Parameter(self.init_response_sigma *
                                                 torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.input_image_size = input_image_size
        self.loss_eps = loss_eps
        self.norm_eps = norm_eps
        self.cycle_consis_threshold = cycle_consis_threshold
        # H x W, tuple of size 2
        self.mesh_grid = torch.meshgrid(
            [torch.arange(start=0, end=input_image_size[0]).long().cuda(),
             torch.arange(start=0, end=input_image_size[1]).long().cuda()])

    def calculate_no_match_loss(self, x, return_response=False):
        src_feature_map, tgt_feature_map, no_match_2d_hw_locations = x
        _, feat_channel, height, width = src_feature_map.shape
        _, num_corr_points, _ = no_match_2d_hw_locations.shape

        # M x 1
        no_match_2d_hw_locations = torch.round(no_match_2d_hw_locations)
        # M x 1
        no_match_2d_hw_locations_h = torch.clamp(no_match_2d_hw_locations[:, :, 0].reshape(-1, 1),
                                                 min=0, max=height - 1)
        no_match_2d_hw_locations_w = torch.clamp(no_match_2d_hw_locations[:, :, 1].reshape(-1, 1),
                                                 min=0, max=width - 1)
        # M
        no_match_1d_locations = (width * no_match_2d_hw_locations_h + no_match_2d_hw_locations_w). \
            long().reshape(num_corr_points)
        src_feature_map = src_feature_map.reshape(
            1, feat_channel, height * width)
        # 1 x C_feat x M
        src_keypoint_features = src_feature_map[:, :, no_match_1d_locations]
        # 1 x M x H*W
        feature_response = torch.sum(torch.square(src_keypoint_features.
                                                  reshape(1, feat_channel, num_corr_points, 1) -
                                                  tgt_feature_map.
                                                  reshape(1, feat_channel, 1, height * width)),
                                     dim=1, keepdim=False)
        feature_response = torch.exp(-self.response_sigma.reshape(1,
                                     1, 1) * feature_response)
        feature_response = feature_response / \
            torch.sum(feature_response, dim=2, keepdim=True)

        loss = torch.mean(torch.sum(torch.square(1.0 / (height * width) - feature_response), dim=(0, 2),
                                    keepdim=False))
        if not return_response:
            return loss
        else:
            return loss, feature_response[0, 0, :].reshape(1, 1, height, width)

    def calculate_rr_loss(self, x, return_response=False):
        src_feature_map, tgt_feature_map, \
            src_keypoint_2d_hw_locations, gt_tgt_keypoint_2d_hw_locations = x

        _, feat_channel, height, width = src_feature_map.shape
        _, num_corr_points, _ = src_keypoint_2d_hw_locations.shape

        # M x 1
        src_keypoint_2d_hw_locations = torch.round(
            src_keypoint_2d_hw_locations)
        # M x 1
        src_keypoint_2d_hw_locations_h = torch.clamp(src_keypoint_2d_hw_locations[:, :, 0].reshape(-1, 1),
                                                     min=0, max=height - 1)
        src_keypoint_2d_hw_locations_w = torch.clamp(src_keypoint_2d_hw_locations[:, :, 1].reshape(-1, 1),
                                                     min=0, max=width - 1)
        # M
        src_keypoint_1d_locations = (width * src_keypoint_2d_hw_locations_h + src_keypoint_2d_hw_locations_w). \
            long().reshape(num_corr_points)
        src_feature_map = src_feature_map.reshape(
            1, feat_channel, height * width)
        # 1 x C_feat x M
        src_keypoint_features = src_feature_map[:,
                                                :, src_keypoint_1d_locations]
        # 1 x M x H*W
        feature_response = torch.sum(torch.square(src_keypoint_features.
                                                  reshape(1, feat_channel, num_corr_points, 1) -
                                                  tgt_feature_map.
                                                  reshape(1, feat_channel, 1, height * width)),
                                     dim=1, keepdim=False)
        feature_response = torch.exp(-self.response_sigma.reshape(1,
                                     1, 1) * feature_response)
        feature_response = feature_response / \
            torch.sum(feature_response, dim=2, keepdim=True)

        gt_tgt_keypoint_2d_hw_locations = torch.round(
            gt_tgt_keypoint_2d_hw_locations)
        # M x 1
        gt_tgt_keypoint_2d_hw_locations_h = torch.clamp(gt_tgt_keypoint_2d_hw_locations[:, :, 0].reshape(-1, 1),
                                                        min=0, max=height - 1)
        gt_tgt_keypoint_2d_hw_locations_w = torch.clamp(gt_tgt_keypoint_2d_hw_locations[:, :, 1].reshape(-1, 1),
                                                        min=0, max=width - 1)
        gt_tgt_keypoint_1d_locations = gt_tgt_keypoint_2d_hw_locations_h * width + \
            gt_tgt_keypoint_2d_hw_locations_w

        # 1 x M x 1
        sampled_response = torch.gather(feature_response.view(1, num_corr_points, height * width), 2,
                                        gt_tgt_keypoint_1d_locations.view(1, num_corr_points, 1).long())

        loss = torch.mean(-torch.log(self.loss_eps + sampled_response))

        if not return_response:
            return loss
        else:
            return loss, feature_response[0, 0, :].reshape(1, 1, height, width)

    def matching_location_estimation_cycle_consis(self, src_feature_map, tgt_feature_map,
                                                  src_keypoint_2d_hw_locations):
        _, feat_channel, height, width = src_feature_map.shape
        _, num_corr_points, _ = src_keypoint_2d_hw_locations.shape

        # M x 1
        src_keypoint_2d_hw_locations_h = torch.floor(
            src_keypoint_2d_hw_locations[:, :, 0].reshape(-1, 1))
        src_keypoint_2d_hw_locations_w = torch.floor(
            src_keypoint_2d_hw_locations[:, :, 1].reshape(-1, 1))
        # M
        src_keypoint_1d_locations = (width * src_keypoint_2d_hw_locations_h + src_keypoint_2d_hw_locations_w). \
            long().reshape(num_corr_points)
        src_feature_map = src_feature_map.reshape(
            1, feat_channel, height * width)
        # 1 x C_feat x M
        src_keypoint_features = src_feature_map[:,
                                                :, src_keypoint_1d_locations]
        # 1 x M x H*W
        tgt_feature_response = -torch.sum(torch.square(src_keypoint_features.
                                                       reshape(1, feat_channel, num_corr_points, 1) -
                                                       tgt_feature_map.reshape(1, feat_channel, 1, height * width)),
                                          dim=1, keepdim=False)

        _, detected_tgt_1d_locations = torch.max(
            tgt_feature_response, dim=2, keepdim=False)
        detected_tgt_1d_locations = detected_tgt_1d_locations.reshape(-1)
        tgt_feature_map = tgt_feature_map.reshape(
            1, feat_channel, height * width)
        # 1 x C_feat x M
        detected_tgt_features = tgt_feature_map[:,
                                                :, detected_tgt_1d_locations.long()]
        # 1 x M x H*W
        src_feature_response = -torch.sum(torch.square(detected_tgt_features.
                                                       reshape(1, feat_channel, num_corr_points, 1) -
                                                       src_feature_map.reshape(1, feat_channel, 1, height * width)),
                                          dim=1, keepdim=False)
        _, cycle_detected_src_1d_locations = torch.max(
            src_feature_response, dim=2, keepdim=False)
        cycle_detected_src_1d_locations = cycle_detected_src_1d_locations.reshape(
            -1)

        # M x 2
        cycle_detected_src_2d_hw_locations = torch.stack(
            [torch.floor(cycle_detected_src_1d_locations / width),
             torch.fmod(cycle_detected_src_1d_locations, width)],
            dim=1).float()

        # M
        cycle_consis_distances = torch.norm(src_keypoint_2d_hw_locations.reshape(num_corr_points, 2) -
                                            cycle_detected_src_2d_hw_locations, dim=1, p=2, keepdim=False)
        inlier_src_keypoint_indexes = torch.nonzero(cycle_consis_distances <= self.cycle_consis_threshold).view(
            -1)

        if inlier_src_keypoint_indexes.shape[0] == 0:
            logger.debug("No feature matching candidates found")
            return None

        inlier_detected_tgt_1d_locations = detected_tgt_1d_locations[inlier_src_keypoint_indexes]
        inlier_detected_tgt_2d_hw_locations = torch.stack(
            [torch.floor(inlier_detected_tgt_1d_locations / width),
             torch.fmod(inlier_detected_tgt_1d_locations, width)],
            dim=1).float()

        inlier_keypoint_2d_hw_locations = src_keypoint_2d_hw_locations[0,
                                                                       inlier_src_keypoint_indexes, :]
        inlier_detected_tgt_2d_hw_locations = inlier_detected_tgt_2d_hw_locations.reshape(
            -1, 2)

        return inlier_keypoint_2d_hw_locations.reshape(1, -1, 2), \
            inlier_detected_tgt_2d_hw_locations.reshape(1, -1, 2)
