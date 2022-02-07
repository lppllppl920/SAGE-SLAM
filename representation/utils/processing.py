import torch
import torch.nn.functional as F
import numpy as np
import torchgeometry as tgm
import scipy


def generate_dense_flow_map(depth_map, valid_mask, rotation, translation, camera_intrinsics, depth_eps):
    # Only generating flows for tgt depth larger than zero
    _, _, height, width = depth_map.shape
    src_hw_2d_locations = torch.nonzero(
        valid_mask.reshape(height, width) >= 0.9)
    num_points = src_hw_2d_locations.shape[0]
    src_1d_locations = (
        width * src_hw_2d_locations[:, 0] + src_hw_2d_locations[:, 1]).reshape(-1).long()
    # 1 x N
    sampled_src_depths = torch.index_select(
        depth_map.reshape(1, -1), dim=1, index=src_1d_locations)
    # 3 x N
    sampled_homo_2d_locations = compute_homogenous_2d_locations(sampled_2d_locations=src_hw_2d_locations.
                                                                reshape(
                                                                    1, num_points, 2),
                                                                camera_intrinsics=camera_intrinsics). \
        reshape(3, num_points)

    # 3 x N
    rotated_sampled_src_homo_2d_locations = torch.matmul(rotation.reshape(3, 3),
                                                         sampled_homo_2d_locations.reshape(3, num_points))
    # 3 x N
    sampled_tgt_3d_locations = sampled_src_depths.reshape(1, num_points) * \
        rotated_sampled_src_homo_2d_locations + translation.reshape(3, 1)

    tgt_z = sampled_tgt_3d_locations[2:3, :]
    sampled_valid_mask = (tgt_z >= depth_eps).float()
    tgt_z = torch.clamp_min(tgt_z, min=depth_eps)
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
    # 1 x N
    sampled_tgt_2d_locations_x = sampled_tgt_homo_2d_locations[0:1, :] * fx + cx
    sampled_tgt_2d_locations_y = sampled_tgt_homo_2d_locations[1:2, :] * fy + cy

    # N x 2
    sampled_tgt_2d_locations = torch.cat([sampled_tgt_2d_locations_x.reshape(-1, 1),
                                          sampled_tgt_2d_locations_y.reshape(-1, 1)], dim=1)

    dense_flow_map = torch.zeros(
        height * width, 2, dtype=torch.float32).to(depth_map.device)
    dense_flow_map[src_1d_locations, :] = sampled_valid_mask.reshape(-1, 1) * (
        sampled_tgt_2d_locations - torch.cat([src_hw_2d_locations[:, 1:2],
                                              src_hw_2d_locations[:, 0:1]], dim=1))

    valid_mask = torch.zeros(
        height * width, 1, dtype=torch.float32).to(depth_map.device)
    valid_mask[src_1d_locations, :] = sampled_valid_mask.reshape(-1, 1)

    return dense_flow_map.reshape(height, width, 2).permute(2, 0, 1).unsqueeze(dim=0), \
        valid_mask.reshape(height, width, 1).permute(2, 0, 1).unsqueeze(dim=0)


def compute_homogenous_2d_locations(sampled_2d_locations, camera_intrinsics):
    # camera_intrinsics: 1 x 4
    # fx fy cx cy: 1 x 1
    fx = camera_intrinsics[:, 0:1].reshape(1, 1)
    fy = camera_intrinsics[:, 1:2].reshape(1, 1)
    cx = camera_intrinsics[:, 2:3].reshape(1, 1)
    cy = camera_intrinsics[:, 3:4].reshape(1, 1)

    num_points = sampled_2d_locations.shape[1]
    # The sampled_2d_locations are in the order of height and width
    x = ((sampled_2d_locations[:, :, 1] - cx) / fx).reshape(1, num_points)
    y = ((sampled_2d_locations[:, :, 0] - cy) / fy).reshape(1, num_points)
    sampled_homo_2d_locations = torch.cat([x, y, torch.ones_like(x)], dim=0)

    # 3 x N
    return sampled_homo_2d_locations


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2).
                                     expand(-1, flattened_tensor.shape[1], -1)). \
        reshape(*tensor.shape[:2], *indices.shape[2:])
    return output


def images_rotation_coordinates_calculate(thetas, image_h, image_w):
    # B x 1 x 1
    thetas = thetas.cpu()
    cos_theta = torch.cos(thetas).view(-1, 1, 1)
    sin_theta = torch.sin(thetas).view(-1, 1, 1)

    image_center_h = torch.tensor(np.floor(image_h / 2.0)).float()
    image_center_w = torch.tensor(np.floor(image_w / 2.0)).float()

    h_grid, w_grid = torch.meshgrid(
        [torch.arange(start=0, end=image_h, dtype=torch.float32),
         torch.arange(start=0, end=image_w, dtype=torch.float32)])

    # 1 x H x W
    h_grid = h_grid.view(1, image_h, image_w)
    w_grid = w_grid.view(1, image_h, image_w)

    # B x H x W
    source_coord_w = cos_theta * (w_grid - image_center_w) + \
        sin_theta * (h_grid - image_center_h) + image_center_w
    source_coord_h = -sin_theta * (w_grid - image_center_w) + \
        cos_theta * (h_grid - image_center_h) + image_center_h

    return source_coord_h, source_coord_w


def images_warping(im, x, y, mode, padding_mode):
    num_batch, _, height, width = im.shape

    grid = torch.cat([(x.reshape(num_batch, height, width, 1) + 0.5) * (2 / width) - 1.0,
                      (y.reshape(num_batch, height, width, 1) + 0.5) * (2 / height) - 1.0], dim=3). \
        reshape(num_batch, height, width, 2).to(im.device)

    return F.grid_sample(input=im, grid=grid, mode=mode,
                         padding_mode=padding_mode, align_corners=False)


def diff_rotation_aug(color_list, mask_list, rotation_limit, mode):
    # Differentiable spatial augmentation
    aug_color_list = list()
    aug_mask_list = list()
    rotation_angles = torch.tensor(
        np.random.uniform(low=-rotation_limit, high=rotation_limit,
                          size=color_list[0].shape[0])).float()

    for i in range(len(color_list)):
        source_coord_h, source_coord_w = \
            images_rotation_coordinates_calculate(thetas=rotation_angles,
                                                  image_h=color_list[i].shape[2],
                                                  image_w=color_list[i].shape[3])
        aug_color_list.append(images_warping(color_list[i], source_coord_w,
                                             source_coord_h, padding_mode="zeros", mode=mode))
    for i in range(len(mask_list)):
        source_coord_h, source_coord_w = \
            images_rotation_coordinates_calculate(thetas=rotation_angles,
                                                  image_h=mask_list[i].shape[2],
                                                  image_w=mask_list[i].shape[3])
        aug_mask_list.append(images_warping(mask_list[i], source_coord_w,
                                            source_coord_h, padding_mode="zeros", mode="nearest"))

    return aug_color_list, aug_mask_list, rotation_angles, source_coord_w, source_coord_h


def diff_rotation_aug_reverse(prediction_list, mask_list, rotation_angles, mode):
    aug_prediction_list = list()
    aug_mask_list = list()

    for i, prediction in enumerate(prediction_list):
        inverse_source_coord_h, inverse_source_coord_w = images_rotation_coordinates_calculate(
            thetas=-rotation_angles, image_h=prediction_list[i].shape[2], image_w=prediction_list[i].shape[3])
        prediction = images_warping(prediction,
                                    inverse_source_coord_w,
                                    inverse_source_coord_h,
                                    padding_mode="zeros", mode=mode)
        aug_prediction_list.append(prediction)

    # Reverse augment the data
    for i in range(len(mask_list)):
        inverse_source_coord_h, inverse_source_coord_w = images_rotation_coordinates_calculate(
            thetas=-rotation_angles, image_h=mask_list[i].shape[2], image_w=mask_list[i].shape[3])
        mask = images_warping(mask_list[i],
                              inverse_source_coord_w,
                              inverse_source_coord_h,
                              padding_mode="zeros", mode="nearest")
        aug_mask_list.append(mask)

    return aug_prediction_list, aug_mask_list


def diff_1d_histogram_generation(feat_map, mask, num_bins):
    # 1 x C x H x W
    channel, height, width = feat_map.shape[1:]
    # value range should be within [-1, 1]
    bin_size = 2.0 / num_bins
    bandwidth = bin_size / 2.5
    # K
    bin_indexes = torch.arange(start=0, end=num_bins).float().cuda()
    bin_center_values = -1.0 + bin_size * (bin_indexes + 0.5)

    temp = feat_map.reshape(1, channel, height * width) - \
        bin_center_values.reshape(num_bins, 1, 1)
    # K x C x H*W
    soft_bin_assignment = torch.sigmoid((temp + bin_size / 2.0) / bandwidth) - \
        torch.sigmoid((temp - bin_size / 2.0) / bandwidth)
    # 1 x 1 x H*W
    mask = mask.reshape(1, 1, height * width)
    # K x C x 1
    # We remove the contributions from the out-of-boundary regions
    histograms = torch.sum(soft_bin_assignment * mask, dim=2,
                           keepdim=True) / torch.sum(mask, dim=2, keepdim=True)
    # Normalize the histogram to make sure summation of all bin values equal to 1 for one channel
    histograms = histograms / torch.sum(histograms, dim=0, keepdim=True)
    # cumulative density function of histograms for earth-mover loss calculation
    cdf_histograms = torch.cumsum(histograms, dim=0)

    return histograms, cdf_histograms


def sample_2d_locations_in_mask(mask, num_points):
    # M x 2
    mask_2d_locations = np.transpose(np.nonzero(mask >= 0.9))

    selected_indexes = np.random.choice(np.arange(mask_2d_locations.shape[0]),
                                        size=min(num_points, mask_2d_locations.shape[0]), replace=False)
    # min(N, M) x 2
    selected_2d_locations = mask_2d_locations[selected_indexes]

    return torch.from_numpy(selected_2d_locations).float()


def rotation_matrix_to_angle_axis(rotation_matrix):
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return tgm.quaternion_to_angle_axis(quaternion)

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~(mask_d0_d1)
    mask_c2 = ~(mask_d2) * mask_d0_nd1
    mask_c3 = ~(mask_d2) * ~(mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q

def generate_random_rotation(gt_rotation, max_rot_dir_rad, max_rot_angle_rad):
    gt_rotvec = rotation_matrix_to_angle_axis(torch.cat([gt_rotation.reshape(1, 3, 3),
                                                             torch.zeros(1, 3, 1).float()], dim=2))
    # 1 x 1
    gt_rot_angle = torch.norm(gt_rotvec, dim=1, keepdim=True)
    if gt_rot_angle > 0:
        gt_rot_direction = gt_rotvec / gt_rot_angle
    else:
        gt_rot_direction = 2.0 * torch.rand(1, 3).float() - 1.0
        gt_rot_direction = gt_rot_direction / \
            torch.norm(gt_rot_direction, dim=1, keepdim=True)

    temp = 2.0 * torch.rand(1, 3).float() - 1.0
    norm = torch.norm(temp, dim=1, keepdim=True)
    assert (norm.item() > 0)
    temp = temp / norm
    assert (torch.sum((gt_rot_direction - temp) ** 2) > 0)

    # do cross product to find perp vector
    perp_vector = torch.cross(gt_rot_direction, temp, dim=1)
    perp_vector = perp_vector / torch.norm(perp_vector, dim=1, keepdim=True)

    random_rot_direction = \
        torch.tensor(np.tan(np.random.random() * max_rot_dir_rad)). \
        reshape(1, 1).float() * perp_vector
    random_rot_direction = random_rot_direction / \
        torch.norm(random_rot_direction, dim=1, keepdim=True)

    random_rot_angle = (
        2.0 * torch.rand(gt_rot_angle.size()) - 1.0) * max_rot_angle_rad
    random_rot_vec = random_rot_angle * random_rot_direction

    random_rotation = tgm.angle_axis_to_rotation_matrix(
        random_rot_vec.reshape(1, 3))
    random_rotation = random_rotation[0, :3, :3]

    return random_rotation.reshape(3, 3)


def generate_random_translation(gt_translation, max_trans_dir_rad, max_trans_dist_offset):
    gt_translation = gt_translation.reshape(1, 3)
    gt_trans_dist = torch.norm(gt_translation, dim=1, keepdim=True)
    if gt_trans_dist > 0:
        gt_trans_dir = gt_translation / gt_trans_dist
    else:
        gt_trans_dir = 2.0 * torch.rand(1, 3).float() - 1.0
        gt_trans_dir = gt_trans_dir / \
            torch.norm(gt_trans_dir, dim=1, keepdim=True)

    temp = 2.0 * torch.rand(1, 3).float() - 1.0
    norm = torch.norm(temp, dim=1, keepdim=True)
    assert (norm.item() > 0)
    temp = temp / norm
    assert (torch.sum((gt_trans_dir - temp) ** 2) > 0)

    perp_vector = torch.cross(gt_trans_dir, temp, dim=1)
    perp_vector = perp_vector / torch.norm(perp_vector, dim=1, keepdim=True)

    random_trans_dir = \
        torch.tensor(np.tan(np.random.random() * max_trans_dir_rad)). \
        reshape(1, 1).float() * perp_vector
    random_trans_dir = random_trans_dir / \
        torch.norm(random_trans_dir, dim=1, keepdim=True)

    random_translation = (gt_trans_dist + torch.rand(1, 1).float()
                          * max_trans_dist_offset * random_trans_dir)

    return random_translation.reshape(3, 1)


def compute_scene_overlap(init_rotation, init_translation, src_depth_map,
                          src_valid_mask, tgt_valid_mask, camera_intrinsics, image_size, depth_eps):
    src_hw_2d_locations = torch.nonzero(
        src_valid_mask.reshape(image_size) >= 0.9)
    num_points = src_hw_2d_locations.shape[0]
    src_1d_locations = (image_size[1] *
                        src_hw_2d_locations[:, 0] + src_hw_2d_locations[:, 1]).reshape(-1).long()
    # 1 x M
    sampled_src_depths = torch.index_select(
        src_depth_map.reshape(1, -1), dim=1, index=src_1d_locations)
    # 3 x M
    sampled_homo_2d_locations = compute_homogenous_2d_locations(sampled_2d_locations=src_hw_2d_locations.
                                                                reshape(
                                                                    1, num_points, 2),
                                                                camera_intrinsics=camera_intrinsics). \
        reshape(3, num_points)

    # 3 x M
    rotated_sampled_src_homo_2d_locations = torch.matmul(init_rotation.reshape(3, 3),
                                                         sampled_homo_2d_locations.reshape(3, num_points))
    # 3 x M
    sampled_tgt_3d_locations = sampled_src_depths.reshape(1, num_points) * \
        rotated_sampled_src_homo_2d_locations + \
        init_translation.reshape(3, 1)
    tgt_z = sampled_tgt_3d_locations[2:3, :]
    tgt_z = torch.clamp_min(tgt_z, min=depth_eps)
    sampled_tgt_3d_locations = torch.cat(
        [sampled_tgt_3d_locations[:2], tgt_z], dim=0)

    # fx fy cx cy: 1 x 1
    fx = camera_intrinsics[:, 0]
    fy = camera_intrinsics[:, 1]
    cx = camera_intrinsics[:, 2]
    cy = camera_intrinsics[:, 3]

    # Normalize the locations by depth values
    sampled_tgt_homo_2d_locations = sampled_tgt_3d_locations / \
        sampled_tgt_3d_locations[2:3, :]
    # obtain pixel locations in the target frame
    # M
    sampled_tgt_2d_locations_x = sampled_tgt_homo_2d_locations[0, :] * fx + cx
    sampled_tgt_2d_locations_y = sampled_tgt_homo_2d_locations[1, :] * fy + cy
    # 1 x 1 x N x 2 : Normalize to range [-1, 1]
    sampled_tgt_normalized_2d_locations = \
        torch.stack([(sampled_tgt_2d_locations_x + 0.5) * (2 / image_size[1]) - 1.0,
                     (sampled_tgt_2d_locations_y + 0.5) * (2 / image_size[0]) - 1.0], dim=1). \
        reshape(1, 1, num_points, 2)

    # 1 x 1 x M
    sampled_tgt_valid_masks = \
        F.grid_sample(input=tgt_valid_mask,
                      grid=sampled_tgt_normalized_2d_locations,
                      mode='nearest', padding_mode='zeros', align_corners=False). \
        reshape(1, 1, num_points)

    ori_hull = scipy.spatial.ConvexHull(src_hw_2d_locations.cpu().numpy())
    ori_area = ori_hull.area

    sampled_tgt_2d_locations = \
        torch.stack([sampled_tgt_2d_locations_y,
                    sampled_tgt_2d_locations_x], dim=1).cpu().numpy()
    warped_hull = scipy.spatial.ConvexHull(sampled_tgt_2d_locations)
    warped_area = warped_hull.area

    area_ratio = min(warped_area / ori_area, 1)

    # the point ratio here is only smaller than 1 for top down view. bottom up it will be always equal to one
    return torch.sum(sampled_tgt_valid_masks) / num_points, area_ratio


def transform_keypoint_locations_ba(src_keypoint_depths, src_keypoint_homo_2d_locations,
                                    guess_rotation, guess_translation, camera_intrinsics, depth_eps):
    # 1 x M
    src_keypoint_depths = src_keypoint_depths.reshape(1, -1)
    num_corr_points = src_keypoint_depths.shape[1]
    # 3 x M
    rotated_src_keypoint_homo_2d_locations = torch.matmul(guess_rotation.reshape(3, 3),
                                                          src_keypoint_homo_2d_locations.reshape(3,
                                                                                                 num_corr_points))
    # 3 x M
    corr_tgt_3d_locations = src_keypoint_depths * \
        rotated_src_keypoint_homo_2d_locations + \
        guess_translation.reshape(3, 1)
    # fx fy cx cy: 1 x 1
    fx = camera_intrinsics[:, 0:1].reshape(1, 1)
    fy = camera_intrinsics[:, 1:2].reshape(1, 1)
    cx = camera_intrinsics[:, 2:3].reshape(1, 1)
    cy = camera_intrinsics[:, 3:4].reshape(1, 1)

    tgt_z = corr_tgt_3d_locations[2:3, :]
    sampled_pos_depth_mask = (tgt_z >= depth_eps).float()
    tgt_z = torch.clamp_min(tgt_z, min=depth_eps)
    corr_tgt_3d_locations = torch.cat(
        [corr_tgt_3d_locations[:2], tgt_z], dim=0)

    corr_tgt_homo_2d_locations = corr_tgt_3d_locations / \
        corr_tgt_3d_locations[2:3, :]
    # 1 x M
    corr_tgt_2d_locations_x = corr_tgt_homo_2d_locations[0:1, :] * fx + cx
    corr_tgt_2d_locations_y = corr_tgt_homo_2d_locations[1:2, :] * fy + cy
    # M x 2
    corr_tgt_2d_hw_locations = torch.cat([corr_tgt_2d_locations_y.reshape(num_corr_points, 1),
                                          corr_tgt_2d_locations_x.reshape(num_corr_points, 1)], dim=1)

    return corr_tgt_2d_hw_locations, sampled_pos_depth_mask.reshape(num_corr_points, 1)


def transform_keypoint_locations(src_depth_map, tgt_valid_mask,
                                 src_keypoint_2d_hw_locations,
                                 rotation, translation, camera_intrinsics, depth_eps):
    height, width = src_depth_map.shape[2:4]
    src_keypoint_2d_hw_locations = src_keypoint_2d_hw_locations.reshape(
        -1, 2)
    num_corr_points = src_keypoint_2d_hw_locations.shape[0]

    src_keypoint_homo_2d_locations = \
        compute_homogenous_2d_locations(
            src_keypoint_2d_hw_locations.reshape(1, -1, 2), camera_intrinsics)
    # N x 1
    src_keypoint_2d_hw_locations_h = src_keypoint_2d_hw_locations[:, 0].reshape(
        -1, 1)
    src_keypoint_2d_hw_locations_w = src_keypoint_2d_hw_locations[:, 1].reshape(
        -1, 1)
    # 1 x 1 x N x 2
    src_keypoint_normalized_2d_locations = \
        torch.cat([(src_keypoint_2d_hw_locations_w + 0.5) * (2 / width) - 1.0,
                   (src_keypoint_2d_hw_locations_h + 0.5) * (2 / height) - 1.0], dim=1). \
        reshape(1, 1, num_corr_points, 2)

    # 1 x M
    src_keypoint_depths = torch.nn.functional.grid_sample(input=src_depth_map,
                                                          grid=src_keypoint_normalized_2d_locations,
                                                          mode='bilinear', padding_mode='zeros',
                                                          align_corners=False).reshape(1, num_corr_points)

    # 1 x M
    src_keypoint_depths = src_keypoint_depths.reshape(1, -1)
    num_corr_points = src_keypoint_depths.shape[1]
    # 3 x M
    rotated_src_keypoint_homo_2d_locations = torch.matmul(rotation.reshape(3, 3),
                                                          src_keypoint_homo_2d_locations.reshape(3,
                                                                                                 num_corr_points))
    # 3 x M
    corr_tgt_3d_locations = src_keypoint_depths * \
        rotated_src_keypoint_homo_2d_locations + translation.reshape(3, 1)
    # fx fy cx cy: 1 x 1
    fx = camera_intrinsics[:, 0:1].reshape(1, 1)
    fy = camera_intrinsics[:, 1:2].reshape(1, 1)
    cx = camera_intrinsics[:, 2:3].reshape(1, 1)
    cy = camera_intrinsics[:, 3:4].reshape(1, 1)

    tgt_z = corr_tgt_3d_locations[2:3, :]
    sampled_pos_depth_mask = (tgt_z >= depth_eps).float()  # 1 x M
    tgt_z = torch.clamp_min(tgt_z, min=depth_eps)
    corr_tgt_3d_locations = torch.cat(
        [corr_tgt_3d_locations[:2], tgt_z], dim=0)

    # Normalize the locations by depth values
    corr_tgt_homo_2d_locations = corr_tgt_3d_locations / \
        corr_tgt_3d_locations[2:3, :]
    # process with camera intrinsics (assuming the camera intrinsics are the same for both tgt and src images)
    # 1 x M
    corr_tgt_2d_locations_x = corr_tgt_homo_2d_locations[0:1, :] * fx + cx
    corr_tgt_2d_locations_y = corr_tgt_homo_2d_locations[1:2, :] * fy + cy

    # 1 x 1 x M x 2 : Normalize to range [-1, 1]
    sampled_tgt_normalized_2d_locations = \
        torch.cat([(corr_tgt_2d_locations_x.reshape(-1, 1) + 0.5) * (2 / width) - 1.0,
                   (corr_tgt_2d_locations_y.reshape(-1, 1) + 0.5) * (2 / height) - 1.0], dim=1). \
        reshape(1, 1, num_corr_points, 2)

    # 1 x 1 x M
    sampled_tgt_valid_masks = \
        torch.nn.functional.grid_sample(input=tgt_valid_mask,
                                        grid=sampled_tgt_normalized_2d_locations,
                                        mode='nearest', padding_mode='zeros', align_corners=False). \
        reshape(1, 1, num_corr_points)

    sampled_tgt_valid_masks = \
        sampled_pos_depth_mask.reshape(
            1, 1, num_corr_points) * sampled_tgt_valid_masks

    # M x 2
    corr_tgt_2d_hw_locations = torch.cat([corr_tgt_2d_locations_y.reshape(num_corr_points, 1),
                                          corr_tgt_2d_locations_x.reshape(num_corr_points, 1)], dim=1)

    valid_indexes = torch.nonzero(sampled_tgt_valid_masks.reshape(-1))
    corr_tgt_2d_hw_locations = corr_tgt_2d_hw_locations[valid_indexes, :]

    src_keypoint_2d_hw_locations = src_keypoint_2d_hw_locations.reshape(
        -1, 2)
    src_keypoint_2d_hw_locations = src_keypoint_2d_hw_locations[valid_indexes, :]
    return src_keypoint_2d_hw_locations.reshape(-1, 2), corr_tgt_2d_hw_locations.reshape(-1, 2)


# coarse to fine order
def generate_depth_from_bias_and_basis(scale, depth_bias, depth_basis_list, code_hierarchy):
    final_depth_map = depth_bias
    height, width = depth_bias.shape[2:]
    offset = 0
    for i in range(len(depth_basis_list)):
        code_channel = depth_basis_list[i].shape[1]
        resized_depth_basis = F.interpolate(input=depth_basis_list[i],
                                            size=(
            height, width), mode='bilinear',
            align_corners=False)
        final_depth_map = final_depth_map + torch.sum(resized_depth_basis *
                                                      code_hierarchy[offset:offset + code_channel].
                                                      reshape(1, -1, 1, 1), dim=1, keepdim=True)
        offset += code_channel

    return scale * final_depth_map


def compute_spatial_grad(feature_map):
    _, _, height, width = feature_map.shape
    # Pad both the spatial dimensions with 1 pixel on both sides
    paded_feature_map = F.pad(
        feature_map, (1, 1, 1, 1), mode='replicate')

    # Central difference to mimic the gradient at the center location
    gradx = 0.5 * (paded_feature_map[:, :, 1:height + 1, 2:width + 2] -
                   paded_feature_map[:, :, 1:height + 1, 0:width])
    grady = 0.5 * (paded_feature_map[:, :, 2:height + 2, 1:width + 1] -
                   paded_feature_map[:, :, 0:height, 1:width + 1])
    # Correct the boundary gradient estimates with replicate padding mode (x2)
    gradx = torch.cat([2.0 * gradx[:, :, :, 0:1], gradx[:,
                                                        :, :, 1:-1], 2.0 * gradx[:, :, :, -1:]], dim=3)
    grady = torch.cat([2.0 * grady[:, :, 0:1, :], grady[:,
                                                        :, 1:-1, :], 2.0 * grady[:, :, -1:, :]], dim=2)

    # 1 x C_feat x H x W
    return torch.cat([gradx, grady], dim=1)


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """Computes the exponential map for the coordinate-vector xi.
    Returns a 3 x 4 SE(3) matrix.
    """
    # 3
    omega = xi[:3]
    v = xi[3:6]

    theta = omega.norm()
    sintheta = theta.sin()
    costheta = theta.cos()

    # Here we need to handle the nan case
    if theta > 0:
        normalized_omega = omega / theta
    else:
        # If the angle is zero, randomize a rotation vector
        normalized_omega = torch.rand(3).float().to(omega.device)

    normalized_omega_hat = so3_hat(normalized_omega)
    normalized_omega_hat_sq = normalized_omega_hat.mm(normalized_omega_hat)

    A = sintheta
    B = 1 - costheta
    C = theta - sintheta

    identity = torch.eye(3, 3).type(omega.dtype).to(omega.device)
    R = identity \
        + A * normalized_omega_hat \
        + B * normalized_omega_hat_sq

    V = identity \
        + (B / theta) * normalized_omega_hat \
        + (C / theta) * normalized_omega_hat_sq

    t = torch.mm(V, v.view(3, 1))

    return torch.cat([R, t], dim=1).reshape(3, 4)


def so3_hat(omega: torch.Tensor) -> torch.Tensor:
    """Implements the hat operator for SO(3), given an input axis-angle
    vector omega.

    """
    assert torch.is_tensor(omega), "Input must be of type torch.tensor."

    omega_hat = torch.zeros(3, 3).type(omega.dtype).to(omega.device)
    omega_hat[0, 1] = -omega[2]
    omega_hat[0, 2] = omega[1]

    omega_hat[1, 0] = omega[2]
    omega_hat[1, 2] = -omega[0]

    omega_hat[2, 0] = -omega[1]
    omega_hat[2, 1] = omega[0]

    return omega_hat