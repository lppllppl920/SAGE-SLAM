from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import numpy as np
from pathlib import Path
import cv2
import h5py

import utils


class EndoscopyDataset(Dataset):
    def __init__(self, data_root: Path, patient_id_list, frame_interval, input_image_size, output_map_size,
                 max_rot_dir_rad, max_rot_angle_rad,
                 max_trans_dir_rad, max_trans_dist_offset,
                 num_iter_per_epoch, num_photo_samples, num_reproj_samples,
                 aug_rot_limit, hdf5_pattern, far_frame_interval,
                 tgt_overlap_ratio, far_overlap_ratio, random_overlap_ratio, args, depth_eps=1.0e-2):

        super(EndoscopyDataset, self).__init__()
        self.data_root = data_root
        self.frame_interval = frame_interval
        self.max_rot_dir_rad = max_rot_dir_rad
        self.max_rot_angle_rad = max_rot_angle_rad
        self.max_trans_dir_rad = max_trans_dir_rad
        self.max_trans_dist_offset = max_trans_dist_offset
        self.min_init_overlap_ratio = random_overlap_ratio
        self.min_gt_overlap_ratio = tgt_overlap_ratio
        self.hdf5_pattern = hdf5_pattern
        self.num_photo_samples = num_photo_samples
        self.num_reproj_samples = num_reproj_samples
        # H, W
        self.image_size = output_map_size
        self.num_iter_per_epoch = num_iter_per_epoch
        self.sequence_hdf5_path_list = sorted(
            list(self.data_root.rglob(hdf5_pattern)))
        self.sequence_probability = None
        # load it when it is in the get item for each thread separately
        self.sequence_hdf5_dict = None
        self.depth_eps = depth_eps
        self.input_image_size = input_image_size
        self.output_map_size = output_map_size
        self.aug_rot_limit = aug_rot_limit
        self.patient_id_list = patient_id_list
        self.far_frame_interval = far_frame_interval
        self.overlap_ratio_dict = {"target": tgt_overlap_ratio, "far": far_overlap_ratio,
                                   "random": random_overlap_ratio}
        self.fast = cv2.FastFeatureDetector_create(threshold=args.fast_threshold, nonmaxSuppression=True,
                                                   type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    def __len__(self):
        return self.num_iter_per_epoch
        
    @staticmethod
    def extract_keypoints(descriptor, init_rotation, init_translation, fine_src_color, src_depth_map,
                          src_valid_mask, tgt_valid_mask, camera_intrinsics,
                          fine_image_size, coarse_image_size, depth_eps):

        src_valid_mask = (255 * src_valid_mask.reshape(*coarse_image_size).cpu().numpy()).astype(
            np.uint8)
        fine_src_valid_mask = cv2.resize(src_valid_mask,
                                         dsize=(
                                             fine_image_size[1], fine_image_size[0]),
                                         interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((3, 3), np.uint8)
        fine_src_valid_mask = cv2.erode(
            fine_src_valid_mask, kernel, iterations=6)

        keypoints = descriptor.detect((255 * fine_src_color.reshape(3, *fine_image_size).
                                       permute(1, 2, 0).cpu().numpy()).astype(np.uint8), fine_src_valid_mask)
        keypoint_hw_2d_locations = np.zeros((len(keypoints), 2))

        ratio = fine_image_size[0] / coarse_image_size[0]
        for j, kp in enumerate(keypoints):
            keypoint_hw_2d_locations[j, 1] = np.round(kp.pt[0] / ratio)
            keypoint_hw_2d_locations[j, 0] = np.round(kp.pt[1] / ratio)
        keypoint_hw_2d_locations = np.unique(keypoint_hw_2d_locations, axis=0)

        keypoint_hw_2d_locations = torch.from_numpy(
            keypoint_hw_2d_locations).float()

        num_points = keypoint_hw_2d_locations.shape[0]
        src_1d_locations = (coarse_image_size[1] *
                            keypoint_hw_2d_locations[:, 0] + keypoint_hw_2d_locations[:, 1]).reshape(-1).long()
        # 1 x M
        sampled_src_depths = torch.index_select(
            src_depth_map.reshape(1, -1), dim=1, index=src_1d_locations)
        # 3 x M
        sampled_homo_2d_locations = \
            utils.compute_homogenous_2d_locations(sampled_2d_locations=keypoint_hw_2d_locations.
                                            reshape(1, num_points, 2),
                                            camera_intrinsics=camera_intrinsics). \
            reshape(3, num_points)

        # 3 x M
        rotated_sampled_src_homo_2d_locations = torch.matmul(init_rotation.reshape(3, 3),
                                                             sampled_homo_2d_locations.reshape(3, num_points))
        # 3 x M
        sampled_tgt_3d_locations = sampled_src_depths.reshape(1, num_points) * \
            rotated_sampled_src_homo_2d_locations + \
            init_translation.reshape(3, 1)
        # 1 x M
        pos_depth_mask = sampled_tgt_3d_locations[2:3, :] > depth_eps

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
            torch.stack([(sampled_tgt_2d_locations_x + 0.5) * (2 / coarse_image_size[1]) - 1.0,
                        (sampled_tgt_2d_locations_y + 0.5) * (2 / coarse_image_size[0]) - 1.0], dim=1). \
            reshape(1, 1, num_points, 2)

        # 1 x M
        sampled_tgt_valid_masks = \
            F.grid_sample(input=tgt_valid_mask,
                          grid=sampled_tgt_normalized_2d_locations,
                          mode='nearest', padding_mode='zeros', align_corners=False). \
            reshape(1, num_points)

        return keypoint_hw_2d_locations[
            (sampled_tgt_valid_masks.reshape(-1) > 0.5) * (pos_depth_mask.reshape(-1) > 0.5), :], \
            keypoint_hw_2d_locations[
            (sampled_tgt_valid_masks.reshape(-1) < 0.5) * (pos_depth_mask.reshape(-1) > 0.5), :]

    @staticmethod
    def get_tensor_from_hdf5(hdf5_file, dataset_name, frame_idx, spatial_size, scale, channel,
                             erode, interp):

        img = hdf5_file[dataset_name][frame_idx]
        if erode:
            kernel = np.ones((3, 3), np.uint8)
            img = cv2.erode(img, kernel, iterations=3)

        img = torch.from_numpy(cv2.resize(img,
                                          dsize=(
                                              spatial_size[1], spatial_size[0]),
                                          interpolation=interp) / scale). \
            reshape(*spatial_size, channel).permute(2, 0, 1).float()

        return img

    @staticmethod
    def augmentation_computing(fine_image_dict, coarse_image_dict, frame_keys, aug_rot_limit):

        aug_image_list_dict = dict()
        aug_video_mask_list_dict = dict()
        rot_angles_dict = dict()
        coord_w_dict = dict()
        coord_h_dict = dict()

        crop_image_list_dict = dict()
        crop_video_mask_list_dict = dict()

        for key in frame_keys:
            aug_image_list, aug_video_mask_list, rot_angles, \
                coord_w, coord_h = \
                utils.diff_rotation_aug(
                    [coarse_image_dict[key].unsqueeze(dim=0),
                     fine_image_dict[key].unsqueeze(dim=0)],
                    [coarse_image_dict["mask"].unsqueeze(dim=0),
                     fine_image_dict["mask"].unsqueeze(dim=0)],
                    aug_rot_limit, "bilinear")
            aug_image_list_dict[key] = aug_image_list
            aug_video_mask_list_dict[key] = aug_video_mask_list
            rot_angles_dict[key] = rot_angles
            coord_w_dict[key] = coord_w
            coord_h_dict[key] = coord_h

        for key in frame_keys:
            crop_image_list, crop_video_mask_list = \
                utils.diff_rotation_aug_reverse(aug_image_list_dict[key],
                                                aug_video_mask_list_dict[key], rot_angles_dict[key],
                                                mode='bilinear')
            crop_image_list_dict[key] = crop_image_list
            crop_video_mask_list_dict[key] = crop_video_mask_list

        return aug_image_list_dict, aug_video_mask_list_dict, crop_image_list_dict, crop_video_mask_list_dict, \
            rot_angles_dict, coord_w_dict, coord_h_dict

    @staticmethod
    def generate_far_close_frame_idx(src_frame_idx, num_frames, selected_hdf5_file, frame_interval, far_frame_interval):
        frame_idx_1 = np.random.randint(low=max(0, src_frame_idx - frame_interval),
                                        high=min(num_frames, src_frame_idx + frame_interval + 1))
        frame_idx_2 = np.random.randint(low=0, high=max(
            1, src_frame_idx - far_frame_interval))
        frame_idx_3 = np.random.randint(low=min(num_frames - 1, src_frame_idx + far_frame_interval),
                                        high=num_frames)
        frame_idx_list = [frame_idx_1, frame_idx_2, frame_idx_3]

        dist_list = list()
        src_cam_pose = selected_hdf5_file["extrinsics"][src_frame_idx]
        src_cam_translation = src_cam_pose[:3, 3].reshape((3, 1))
        for idx in frame_idx_list:
            dist_list.append(np.sum(
                np.square(selected_hdf5_file["extrinsics"][idx][:3, 3].reshape((3, 1)) - src_cam_translation)))

        far_frame_idx = frame_idx_list[np.argmax(np.asarray(dist_list))]
        tgt_frame_idx = frame_idx_list[np.argmin(np.asarray(dist_list))]

        return tgt_frame_idx, far_frame_idx

    def generate_hdf5_dict(self):
        self.sequence_hdf5_dict = dict()
        self.remove_path_list = list()

        num_frames_list = list()
        for hdf5_path in self.sequence_hdf5_path_list:
            path = str(hdf5_path)
            ind = path.find("bag_")
            ind2 = path.find("/", ind)
            bag_id = int(path[ind + len("bag_"):ind2])

            if bag_id in self.patient_id_list:
                temp = h5py.File(path, 'r', libver='latest', swmr=True)
                self.sequence_hdf5_dict[path] = temp
                num_frames_list.append(temp['color'].shape[0])
            else:
                self.remove_path_list.append(hdf5_path)

        for remove_path in self.remove_path_list:
            self.sequence_hdf5_path_list.remove(remove_path)

        self.sequence_probability = np.sqrt(np.asarray(num_frames_list))
        self.sequence_probability = self.sequence_probability / \
            np.sum(self.sequence_probability)

    @torch.no_grad()
    def __getitem__(self, _):
        if self.sequence_hdf5_dict is None:
            self.generate_hdf5_dict()

        hdf5_path_idx = \
            np.random.choice(np.arange(start=0, stop=len(self.sequence_hdf5_path_list)),
                             size=1, replace=False, p=self.sequence_probability)[0]

        selected_hdf5_file = self.sequence_hdf5_dict[str(
            self.sequence_hdf5_path_list[hdf5_path_idx])]
        num_frames, ori_height, ori_width = selected_hdf5_file['color'].shape[:3]
        gt_flow_map = None

        fine_image_dict = dict()
        coarse_image_dict = dict()

        while True:
            src_frame_idx = np.random.randint(low=0, high=num_frames)
            tgt_frame_idx, far_frame_idx = \
                self.generate_far_close_frame_idx(
                    src_frame_idx, num_frames, selected_hdf5_file,
                    frame_interval=self.frame_interval, far_frame_interval=self.far_frame_interval)

            frame_idx_dict = {"source": src_frame_idx,
                              "target": tgt_frame_idx, "far": far_frame_idx}

            for key in frame_idx_dict.keys():
                fine_image_dict[key] = self.get_tensor_from_hdf5(selected_hdf5_file, 'color', frame_idx_dict[key],
                                                                 self.input_image_size[:2], 255.0, 3, erode=False, interp=cv2.INTER_LINEAR)
                coarse_image_dict[key] = self.get_tensor_from_hdf5(selected_hdf5_file, 'color', frame_idx_dict[key],
                                                                   self.output_map_size[:2], 255.0, 3, erode=False, interp=cv2.INTER_LINEAR)

            fine_image_dict["mask"] = self.get_tensor_from_hdf5(selected_hdf5_file, 'mask', 0,
                                                                self.input_image_size[:2], 1.0,
                                                                channel=1, erode=False, interp=cv2.INTER_NEAREST)
            coarse_image_dict["mask"] = self.get_tensor_from_hdf5(selected_hdf5_file, 'mask', 0,
                                                                  self.output_map_size[:2], 1.0,
                                                                  channel=1, erode=False, interp=cv2.INTER_NEAREST)

            ratio_w = self.output_map_size[1] / ori_width
            ratio_h = self.output_map_size[0] / ori_height
            camera_intrinsic_matrix = torch.from_numpy(
                selected_hdf5_file['intrinsics'][0]).float()
            camera_intrinsics = torch.stack(
                [camera_intrinsic_matrix[0, 0] * ratio_w, camera_intrinsic_matrix[1, 1] * ratio_h,
                 camera_intrinsic_matrix[0, 2] * ratio_w, camera_intrinsic_matrix[1, 2] * ratio_h],
                dim=0).reshape(4, )

            aug_image_list_dict, aug_video_mask_list_dict, crop_image_list_dict, crop_video_mask_list_dict, \
                rot_angles_dict, _, _ \
                = self.augmentation_computing(fine_image_dict=fine_image_dict, coarse_image_dict=coarse_image_dict,
                                               frame_keys=["source", "target", "far"], aug_rot_limit=self.aug_rot_limit)

            sampled_2d_locations = \
                utils.sample_2d_locations_in_mask(
                    mask=crop_video_mask_list_dict["source"][0].reshape(
                        *self.output_map_size).cpu().numpy(),
                    num_points=self.num_photo_samples).reshape(-1, 2).float()

            gt_depth_mask_dict = dict()
            gt_depth_dict = dict()
            cam_pose_dict = dict()
            scale = None

            for key in frame_idx_dict.keys():
                gt_depth_mask_dict[key] = self.get_tensor_from_hdf5(selected_hdf5_file, 'render_mask',
                                                                    frame_idx_dict[key],
                                                                    self.output_map_size[:2],
                                                                    1.0, 1, erode=False, interp=cv2.INTER_NEAREST)
                gt_depth_dict[key] = \
                    self.get_tensor_from_hdf5(selected_hdf5_file, 'render_depth', frame_idx_dict[key],
                                              self.output_map_size[:2],
                                              1.0, 1, erode=False, interp=cv2.INTER_LINEAR)

                cam_pose_dict[key] = selected_hdf5_file["extrinsics"][frame_idx_dict[key]]

                if scale is None:
                    temp = gt_depth_mask_dict[key].reshape(
                        (-1)) * gt_depth_dict[key].reshape((-1))
                    scale = 1.0 / torch.median(temp[torch.nonzero(temp)])
                gt_depth_dict[key] = scale * gt_depth_dict[key]

                cam_pose_dict[key][:3, 3] = scale * cam_pose_dict[key][:3, 3]

            gt_rotation_dict = dict()
            gt_translation_dict = dict()
            for key in ["target", "far"]:
                gt_rotation_dict[key] = torch.from_numpy(np.matmul(np.transpose(cam_pose_dict[key][:3, :3]),
                                                                   cam_pose_dict["source"][:3, :3])).float()
                gt_translation_dict[key] = torch.from_numpy(np.matmul(np.transpose(cam_pose_dict[key][:3, :3]),
                                                                      cam_pose_dict["source"][:3, 3].reshape((3, 1)) -
                                                                      cam_pose_dict[key][:3, 3].reshape(
                                                                          (3, 1)))).float()

            resample_data = False

            keypoint_candidates_dict = dict()
            no_match_candidates_dict = dict()

            far_overlap_valid = True
            for key in ["target", "far"]:
                point_within_mask_ratio, warp_area_ratio = \
                    utils.compute_scene_overlap(gt_rotation_dict[key].reshape(1, 3, 3),
                                                gt_translation_dict[key].reshape(
                        1, 3, 1),
                        gt_depth_dict["source"].unsqueeze(
                        dim=0),
                        (gt_depth_mask_dict["source"].unsqueeze(dim=0) *
                         crop_video_mask_list_dict["source"][0].
                         reshape(1, 1, *self.output_map_size)),
                        crop_video_mask_list_dict[key][0].
                        reshape(
                        1, 1, *self.output_map_size),
                        camera_intrinsics.reshape(
                                                    1, 4),
                        image_size=self.output_map_size,
                        depth_eps=self.depth_eps)

                keypoint_candidates, no_match_candidates = \
                    self.extract_keypoints(self.fast, gt_rotation_dict[key].reshape(1, 3, 3),
                                           gt_translation_dict[key].reshape(
                        1, 3, 1),
                        fine_image_dict["source"],
                        gt_depth_dict["source"].unsqueeze(
                        dim=0),
                        (gt_depth_mask_dict["source"].unsqueeze(dim=0) *
                         crop_video_mask_list_dict["source"][0].
                         reshape(1, 1, *self.output_map_size)),
                        coarse_image_dict["mask"].
                        reshape(
                        1, 1, *self.output_map_size),
                        camera_intrinsics.reshape(1, 4),
                        coarse_image_size=self.output_map_size,
                        fine_image_size=self.input_image_size,
                        depth_eps=self.depth_eps)

                keypoint_candidates_dict[key] = keypoint_candidates
                no_match_candidates_dict[key] = no_match_candidates
                if key == "target" and \
                        (point_within_mask_ratio < self.overlap_ratio_dict[key]
                         or warp_area_ratio < self.overlap_ratio_dict[key]):
                    # print("Resampling due to target ratio not satisfied")
                    resample_data = True
                elif key == "far" and \
                        (point_within_mask_ratio > self.overlap_ratio_dict[key]
                         and warp_area_ratio > self.overlap_ratio_dict[key]):
                    far_overlap_valid = False

            if resample_data:
                continue

            sampled_keypoint_2d_hw_locations_dict = dict()
            sampled_no_match_2d_hw_locations_dict = dict()
            for key in ["target"]:  # , "far"
                # Sample source keypoint 2d locations that have point match in the target image
                indexes = np.random.choice(np.arange(start=0, stop=keypoint_candidates_dict[key].shape[0]),
                                           size=self.num_reproj_samples, replace=True)
                keypoint_candidates_np = keypoint_candidates_dict[key].numpy()
                sampled_keypoint_2d_hw_locations_dict[key] = torch.from_numpy(
                    keypoint_candidates_np[indexes, :]).float() \
                    .reshape(self.num_reproj_samples, 2)

                if no_match_candidates_dict[key].shape[0] > 0:
                    num_no_match_samples = no_match_candidates_dict[key]
                    indexes = np.random.choice(np.arange(start=0, stop=num_no_match_samples.shape[0]),
                                               size=self.num_reproj_samples, replace=True)
                else:
                    indexes = []

                no_match_candidates_np = no_match_candidates_dict[key].numpy()
                sampled_no_match_2d_hw_locations_dict[key] = torch.from_numpy(
                    no_match_candidates_np[indexes, :]).float() \
                    .reshape(-1, 2)

            # generate dense flow map using gt src depth map and gt relative camera pose
            # 2 x H x W (x y order)
            gt_flow_map, gt_flow_mask = \
                utils.generate_dense_flow_map(depth_map=gt_depth_dict["source"].unsqueeze(dim=0),
                                              valid_mask=crop_video_mask_list_dict["source"][0].
                                              reshape(1, *self.output_map_size) *
                                              gt_depth_mask_dict["source"],
                                              rotation=gt_rotation_dict["target"],
                                              translation=gt_translation_dict["target"],
                                              camera_intrinsics=camera_intrinsics.reshape(
                    1, 4),
                    depth_eps=self.depth_eps)

            sampled_src_keypoint_2d_locations, match_tgt_keypoint_2d_locations = \
                utils.transform_keypoint_locations(src_depth_map=gt_depth_dict["source"].unsqueeze(dim=0),
                                                   tgt_valid_mask=crop_video_mask_list_dict["target"][0].
                                                   reshape(
                    1, 1, *self.output_map_size),
                    src_keypoint_2d_hw_locations=sampled_keypoint_2d_hw_locations_dict["target"].unsqueeze(
                    dim=0),
                    rotation=gt_rotation_dict["target"],
                    translation=gt_translation_dict["target"],
                    camera_intrinsics=camera_intrinsics.reshape(
                    1, 4),
                    depth_eps=self.depth_eps
                )
            src_flow_mask = crop_video_mask_list_dict["source"][0]. \
                reshape(1, *self.output_map_size) * gt_flow_mask.squeeze(dim=0)

            # Find one initial pose so that the initial
            # scene overlap ratio between source and target frame is enough
            failure_count = 0
            while True:
                # 3 x 3
                random_rotation = \
                    utils.generate_random_rotation(gt_rotation=gt_rotation_dict["target"],
                                                   max_rot_dir_rad=self.max_rot_dir_rad,
                                                   max_rot_angle_rad=self.max_rot_angle_rad)
                # 3 x 1
                random_translation = \
                    utils.generate_random_translation(gt_translation=gt_translation_dict["target"],
                                                      max_trans_dir_rad=self.max_trans_dir_rad,
                                                      max_trans_dist_offset=self.max_trans_dist_offset)
                random_rotation = torch.matmul(
                    random_rotation, gt_rotation_dict["target"])
                random_translation = torch.matmul(
                    random_rotation, gt_translation_dict["target"]) + random_translation

                point_within_mask_ratio, warp_area_ratio = utils.compute_scene_overlap(
                    random_rotation.reshape(1, 3, 3),
                    random_translation.reshape(1, 3, 1),
                    gt_depth_dict["source"].unsqueeze(dim=0),
                    (crop_video_mask_list_dict["source"][0].reshape(1, 1, *self.output_map_size) *
                     gt_depth_mask_dict["source"].unsqueeze(dim=0)),
                    crop_video_mask_list_dict["target"][0].reshape(
                        1, 1, *self.output_map_size),
                    camera_intrinsics.reshape(1, 4),
                    image_size=self.output_map_size,
                    depth_eps=self.depth_eps)

                if point_within_mask_ratio > self.overlap_ratio_dict["random"] \
                        and warp_area_ratio > self.overlap_ratio_dict["random"]:
                    init_overlap_ratio = torch.tensor(
                        min(point_within_mask_ratio, warp_area_ratio))
                    break
                elif failure_count <= 10:
                    failure_count += 1
                    pass
                else:
                    resample_data = True
                    break

            if resample_data:
                continue

            break

        return {"crop_coarse_src_image": crop_image_list_dict["source"][0].squeeze(dim=0),
                "crop_coarse_close_image": crop_image_list_dict["target"][0].squeeze(dim=0),
                "aug_fine_src_image": aug_image_list_dict["source"][1].squeeze(dim=0),
                "aug_fine_close_image": aug_image_list_dict["target"][1].squeeze(dim=0),
                "aug_fine_far_image": aug_image_list_dict["far"][1].squeeze(dim=0),
                "fine_src_image": fine_image_dict["source"],
                "fine_close_image": fine_image_dict["target"],
                "fine_far_image": fine_image_dict["far"],
                "crop_coarse_src_video_mask": crop_video_mask_list_dict["source"][0].
                reshape(1, *self.output_map_size),
                "crop_coarse_close_video_mask": crop_video_mask_list_dict["target"][0].
                reshape(1, *self.output_map_size),
                "crop_coarse_far_video_mask": crop_video_mask_list_dict["far"][0].
                reshape(1, *self.output_map_size),
                "aug_fine_src_video_mask": aug_video_mask_list_dict["source"][1].reshape(1, *self.input_image_size),
                "aug_fine_close_video_mask": aug_video_mask_list_dict["target"][1].reshape(1, *self.input_image_size),
                "aug_fine_far_video_mask": aug_video_mask_list_dict["far"][1].reshape(1, *self.input_image_size),
                "src_gt_depth_mask": gt_depth_mask_dict["source"].reshape(1, *self.output_map_size),
                "close_gt_depth_mask": gt_depth_mask_dict["target"].reshape(1, *self.output_map_size),
                "fine_video_mask": fine_image_dict["mask"].reshape(1, *self.input_image_size),
                "coarse_video_mask": coarse_image_dict["mask"].reshape(1, *self.output_map_size),
                "src_rot_angles": rot_angles_dict["source"].reshape(-1),
                "close_rot_angles": rot_angles_dict["target"].reshape(-1),
                "far_rot_angles": rot_angles_dict["far"].reshape(-1),
                "src_flow_mask": src_flow_mask.reshape(1, *self.output_map_size),
                "gt_flow_map": gt_flow_map.squeeze(dim=0),
                "gt_rotation": gt_rotation_dict["target"],
                "gt_translation": gt_translation_dict["target"],
                "random_rotation": random_rotation,
                "random_translation": random_translation,
                "src_gt_depth": gt_depth_dict["source"],
                "close_gt_depth": gt_depth_dict["target"],
                "camera_intrinsics": camera_intrinsics,
                "sampled_2d_locations": sampled_2d_locations,
                "reproj_src_keypoint_2d_locations": sampled_src_keypoint_2d_locations,
                "reproj_src_no_match_2d_locations": sampled_no_match_2d_hw_locations_dict["target"],
                "corr_close_2d_locations": match_tgt_keypoint_2d_locations,
                "far_overlap_valid": far_overlap_valid,
                "init_overlap_ratio": init_overlap_ratio}
