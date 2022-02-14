from pathlib import Path
import numpy as np
import torch
import torch.utils.data
import torch.backends.cudnn
import random
import tqdm
import datetime
from torch.utils.tensorboard import SummaryWriter
import logging
import sys
import argparse
import warnings
import json
import os
import signal

from utils import logger, formatter
import losses
import models
import utils
import datasets

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default=None, help='config file')

    args = parser.parse_args()
    if args.config_path is None or not Path(args.config_path).exists():
        logging.error(
            f"specified config path does not exist {args.config_path}")
        exit()

    with open(args.config_path, 'r') as f:
        args.__dict__ = json.load(f)

    date = datetime.datetime.now()
    log_root = Path(args.experiment_root) / \
        "SAGE-SLAM_{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(
        date.month,
        date.day,
        date.hour,
        date.minute,
        date.second)
    if not log_root.exists():
        log_root.mkdir(parents=True)

    if args.debug_logging_mode == "debug":
        mode = logging.DEBUG
    elif args.debug_logging_mode == "info":
        mode = logging.INFO
    else:
        mode = logging.ERROR

    logger.setLevel(mode)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(mode)
    ch.setFormatter(formatter)
    fh = logging.FileHandler(str(Path(log_root) / "training.log"))
    fh.setLevel(mode)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info("Logging at {}".format(str(log_root)))

    if args.lm_reproj_nsamples > args.lm_photo_nsamples:
        logger.warning(
            "lm_reproj_nsamples should not be larger than lm_photo_nsamples, capping it")

    args.lm_reproj_nsamples = min(
        args.lm_reproj_nsamples, args.lm_photo_nsamples)

    writer = None
    if not args.export_jit_model:
        writer = SummaryWriter(log_dir=str(log_root), flush_secs=10)
        name = "tensorboard"
        # iterating through each instance of the proess
        for line in os.popen("ps ax | grep " + name + " | grep -v grep"):
            fields = line.split()
            # extracting Process ID from the output
            pid = fields[0]
            # terminating process
            os.kill(int(pid), signal.SIGKILL)
        code = os.system(
            f"tensorboard --logdir=\"{str(log_root)}\" --port=6006 --reload_multifile=true &")
        if code != 0:
            logger.error(
                "Tensorboard visualization failed to start, error code is {0}".format(code))
            exit(-1)

    with open(os.path.join(log_root, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed(0)

    feat_model = models.FeatureNet(in_channel=3,
                                   num_pre_steps=args.net_pre_downconv_steps,
                                   filter_list=args.net_feat_net_filter_list,
                                   bottle_neck_filter=args.net_feat_net_bottleneck_filter,
                                   desc_inner_filter_list=args.net_feat_net_desc_inner_filter_list,
                                   map_inner_filter_list=args.net_feat_net_map_inner_filter_list,
                                   desc_out_activation=args.net_feat_net_desc_out_activation,
                                   map_out_activation=args.net_feat_net_map_out_activation,
                                   merge_mode='concat', gn_group_size=args.net_gn_group_size
                                   )
    depth_model = models.DepthNet(in_channel=3, num_pre_steps=args.net_pre_downconv_steps,
                                  filter_list=args.net_depth_net_filter_list,
                                  bottle_neck_filter=args.net_depth_net_bottleneck_filter,
                                  bias_inner_filter_list=args.net_depth_net_bias_inner_filter_list,
                                  basis_inner_filter_list=args.net_depth_net_basis_inner_filter_list,
                                  bias_out_activation=args.net_depth_net_bias_out_activation,
                                  basis_out_activation=args.net_depth_net_basis_out_activation,
                                  merge_mode='concat', gn_group_size=args.net_gn_group_size
                                  )
    disc_model = models.DiscNet(img_height=args.net_output_map_size[0], img_width=args.net_output_map_size[1],
                                input_nc=4, filter_base=args.net_disc_filter_base,
                                num_block=args.net_disc_num_block,
                                group_size=args.net_gn_group_size)

    feat_model = utils.init_net(
        net=feat_model, type="glorot", distribution="uniform")
    depth_model = utils.init_net(
        net=depth_model, type="glorot", distribution="uniform")
    disc_model = utils.init_net(
        net=disc_model, type="glorot", distribution="uniform")

    if args.summarize_network:
        utils.count_parameters(feat_model)
        utils.count_parameters(depth_model)
        utils.count_parameters(disc_model)

    diff_ba = \
        models.DiffBundleAdjustment(match_geom_param_factor=args.lm_match_geom_param_factor,
                                    match_geom_term_weight=args.lm_match_geom_term_weight,
                                    code_term_weight=args.lm_code_term_weight,
                                    geometry_cauchy_param_factor=args.lm_geometry_cauchy_param_factor,
                                    geometry_term_weight=args.lm_geometry_term_weight,
                                    scale_term_weight=args.lm_scale_term_weight,
                                    photo_pow_factor=args.lm_photo_pow_factor,
                                    photo_weight=args.lm_photo_weight,
                                    num_photo_level=args.lm_num_photo_level,
                                    depth_eps=args.lm_depth_eps,
                                    num_display_matches=args.lm_display_matches).cuda()

    step = 0
    epoch = 0

    diff_fm = models.DiffFeatureMatcher(input_image_size=args.net_output_map_size,
                                        response_sigma=args.net_fm_response_sigma,
                                        cycle_consis_threshold=args.cycle_consis_threshold).cuda()

    if args.net_load_weights:
        if args.net_depth_model_path is not None and Path(args.net_depth_model_path).exists():
            _, epoch, step = \
                utils.load_model(model=depth_model, trained_model_path=Path(args.net_depth_model_path),
                                 partial_load=True)

        if args.net_feat_model_path is not None and Path(args.net_feat_model_path).exists():
            _, epoch, step = \
                utils.load_model(model=feat_model, trained_model_path=Path(args.net_feat_model_path),
                                 partial_load=True)

        if args.net_ba_model_path is not None and Path(args.net_ba_model_path).exists():
            _, epoch, step = \
                utils.load_model(model=diff_ba, trained_model_path=Path(args.net_ba_model_path),
                                 partial_load=True)

        if args.net_disc_model_path is not None and Path(args.net_disc_model_path).exists():
            _, epoch, step = \
                utils.load_model(model=disc_model, trained_model_path=Path(args.net_disc_model_path),
                                 partial_load=True)

    loss_func_dict = {"depth": losses.ScaleInvariantLoss(args.depth_loss_eps),
                      "flow": losses.NormalizedMaskedL2Loss(args.flow_loss_eps),
                      "hist": losses.TripletLoss(margin=args.hist_loss_margin),
                      "decor": losses.BasisDecorrelationLoss()}

    if args.export_jit_model is True:
        depth_model.eval()
        feat_model.eval()
        scripted_depth_model = torch.jit.script(depth_model)
        scripted_feat_model = torch.jit.script(feat_model)
        scripted_depth_model.save(str(log_root / "jit_depth_model.pt"))
        scripted_feat_model.save(str(log_root / "jit_feat_model.pt"))
        logger.info("JIT ScriptModule has been generated, exit now")
        exit(0)

    optimizer = torch.optim.SGD(list(depth_model.parameters()) +
                                list(feat_model.parameters()) +
                                list(diff_ba.parameters()), lr=args.min_max_lr[1], momentum=0.9)

    optimizer_D = torch.optim.SGD(
        disc_model.parameters(), lr=args.min_max_lr[1], momentum=0.9)

    optimizer_dict = {"G": optimizer, "D": optimizer_D}

    scheduler = utils.CyclicLR(
        optimizer, base_lr=args.min_max_lr[0], max_lr=args.min_max_lr[1])

    # creating dataset
    train_dataset, eval_dataset = \
        [datasets.EndoscopyDataset(data_root=Path(args.data_root),
                                   patient_id_list=id_list,
                                   frame_interval=args.frame_interval,
                                   input_image_size=args.net_input_image_size,
                                   output_map_size=args.net_output_map_size,
                                   max_rot_dir_rad=args.max_rot_dir_rad,
                                   max_rot_angle_rad=args.max_rot_angle_rad,
                                   max_trans_dir_rad=args.max_trans_dir_rad,
                                   max_trans_dist_offset=args.max_trans_dist_offset,
                                   hdf5_pattern=args.hdf5_pattern,
                                   num_iter_per_epoch=iter_per_epoch,
                                   num_photo_samples=args.lm_photo_nsamples,
                                   num_reproj_samples=args.lm_reproj_nsamples,
                                   aug_rot_limit=args.aug_rot_limit,
                                   far_frame_interval=args.far_frame_interval,
                                   tgt_overlap_ratio=args.tgt_overlap_ratio,
                                   far_overlap_ratio=args.far_overlap_ratio,
                                   random_overlap_ratio=args.random_overlap_ratio,
                                   args=args)
         for id_list, iter_per_epoch in zip([args.train_id_list, args.eval_id_list],
                                            [args.train_num_iter_per_epoch, args.eval_num_iter_per_epoch])]

    train_loader, eval_loader = \
        [torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2 ** 32 - 1)))
            for dataset in [train_dataset, eval_dataset]]

    model_dict = {"depth": depth_model, "feat": feat_model,
                  "ba": diff_ba, "fm": diff_fm, "disc": disc_model}

    eval_step = 0
    train_step = step
    global_step = 0
    args.train_mode = "separate"

    for curr_epoch in range(epoch, args.max_epoch):
        if not args.eval_only:
            torch.manual_seed(1 + curr_epoch)
            np.random.seed(1 + curr_epoch)
            random.seed(1 + curr_epoch)

            for key, model in model_dict.items():
                model.train()

            tq = tqdm.tqdm(total=len(train_dataset))
            mean_loss_dict = None

            if curr_epoch < args.separate_train_epoch:
                args.train_mode = "separate"
            else:
                args.train_mode = "joint"

            if args.train_mode == "separate":
                mean_loss_dict = {"total": None, "depth": None,
                                  "desc": None, "hist": None, "nm": 0.0}
            elif args.train_mode == "joint":
                mean_loss_dict = {"total": None, "depth": None, "desc": None, "flow": None, "hist": None,
                                  "nm": 0.0, 'g': None, 'd': None, "decor": None}

            for curr_iter, input_dict in enumerate(train_loader):
                scheduler.batch_step(batch_iteration=train_step)
                tq.set_description(
                    'Train-epoch:{:d},lr:{:.5f}'.format(curr_epoch, float(scheduler.get_lr()[0])))

                for key in input_dict.keys():
                    input_dict[key] = input_dict[key].to(args.device)

                x = train(model_dict=model_dict, input_dict=input_dict, loss_func_dict=loss_func_dict,
                          optimizer_dict=optimizer_dict, args=args, log_root=log_root, writer=writer, step=train_step)
                if x is None:
                    tq.update(1)
                    global_step += 1
                    train_step += 1
                    continue

                loss_dict, num_lm_steps, use_match_geom = x
                for key in loss_dict.keys():
                    if mean_loss_dict[key] is not None:
                        if loss_dict[key] != 0:
                            mean_loss_dict[key] = (
                                mean_loss_dict[key] * curr_iter + loss_dict[key]) / (curr_iter + 1.0)
                    else:
                        mean_loss_dict[key] = loss_dict[key]

                global_step += 1
                train_step += 1

                tq.update(1)

                postfix = dict()
                for key in mean_loss_dict.keys():
                    if mean_loss_dict[key] is not None:
                        postfix[key] = 'a:{:.2f},c:{:.2f}'.format(mean_loss_dict[key],
                                                                  loss_dict[key] if key in loss_dict else -1.0)

                tq.set_postfix(**postfix,
                               phof='{:.2f}'.format(
                                   diff_ba.photo_pow_factor.item()),
                               phow='{:.2f}'.format(
                                   torch.abs(10.0 * diff_ba.photo_weight).item()),
                               lm='{:d}'.format(num_lm_steps),
                               po='{}'.format(not use_match_geom)
                               )

                writer.add_scalars('Train/cur_loss', loss_dict, train_step)
                writer.add_scalars(
                    'Train/avg_loss', mean_loss_dict, train_step)
                writer.add_scalars('Train/misc', {'num_lm_steps': num_lm_steps,
                                                  'photof': diff_ba.photo_pow_factor.item(),
                                                  'photow': torch.abs(10.0 * diff_ba.photo_weight).item(),
                                                  'po': not use_match_geom
                                                  }, train_step)
                writer.flush()

                torch.cuda.empty_cache()

            tq.close()
            save_model(log_root, curr_epoch, mean_loss_dict, train_step, model_dict,
                       title="Train", exclude_name_list=["fm"])

        if args.eval_only or (args.train_mode == "joint" and (curr_epoch % args.model_save_freq == 0)):
            torch.manual_seed(1)
            np.random.seed(1)
            random.seed(1)

            tq = tqdm.tqdm(total=len(eval_dataset))

            for key, model in model_dict.items():
                model.eval()

            mean_loss_dict = {"total": None, "depth": None, "desc": None, "flow": None, "hist": None,
                              "nm": 0.0}

            for curr_iter, input_dict in enumerate(eval_loader):
                tq.set_description('Eval-epoch:{:d}'.format(curr_epoch))

                for key in input_dict.keys():
                    input_dict[key] = input_dict[key].to(args.device)

                x = evaluation(model_dict=model_dict, input_dict=input_dict, loss_func_dict=loss_func_dict,
                               args=args, log_root=log_root, writer=writer, step=eval_step)
                if x is None:
                    tq.update(1)
                    eval_step += 1
                    global_step += 1
                    continue

                loss_dict, num_lm_steps, use_match_geom = x
                for key in loss_dict.keys():
                    if mean_loss_dict[key] is not None:
                        if loss_dict[key] > 0:
                            mean_loss_dict[key] = (
                                mean_loss_dict[key] * curr_iter + loss_dict[key]) / (curr_iter + 1.0)
                    else:
                        mean_loss_dict[key] = loss_dict[key]

                eval_step += 1
                global_step += 1

                tq.update(1)

                postfix = dict()
                for key in mean_loss_dict.keys():
                    if mean_loss_dict[key] is not None:
                        postfix[key] = 'a:{:.2f},c:{:.2f}'.format(mean_loss_dict[key],
                                                                  loss_dict[key] if key in loss_dict else -1.0)

                tq.set_postfix(**postfix,
                               lm='{:d}'.format(num_lm_steps),
                               po='{}'.format(not use_match_geom)
                               )

                writer.add_scalars('Eval/cur_loss', loss_dict,eval_step)
                writer.add_scalars('Eval/misc', {'num_lm_steps': num_lm_steps,
                                                 'po': not use_match_geom
                                                 },eval_step)
                writer.flush()
                torch.cuda.empty_cache()

            tq.close()

            if args.eval_only:
                logger.info("Evaluation finished, program exiting")
                exit(0)
            else:
                save_model(log_root, curr_epoch, mean_loss_dict, train_step, model_dict,
                           title="Eval", exclude_name_list=["fm"])


def save_model(log_root, curr_epoch, mean_loss_dict, step, model_dict, title, exclude_name_list):
    loss_string = f"{curr_epoch:03d}_"
    for name in mean_loss_dict.keys():
        loss_string += name + f"_{mean_loss_dict[name]:.4f}_"

    for name in model_dict.keys():
        if name in exclude_name_list:
            continue
        model_path_epoch = log_root / \
            (title + "_" + name + "_" + loss_string + ".pt")
        utils.save_checkpoint(epoch=curr_epoch, step=step,
                              model=model_dict[name], path=model_path_epoch)


def train(model_dict, input_dict, loss_func_dict, optimizer_dict, args, log_root, writer, step):
    d_adv_loss = None
    g_adv_loss = None
    no_match_response_map = None

    aug_src_feature_map, aug_src_desc_feature_map = \
        model_dict["feat"](input_dict["aug_fine_src_image"],
                           input_dict["aug_fine_src_video_mask"])
    aug_close_feature_map, aug_close_desc_feature_map = \
        model_dict["feat"](input_dict["aug_fine_close_image"],
                           input_dict["aug_fine_close_video_mask"])
    _, aug_far_desc_feature_map = \
        model_dict["feat"](input_dict["aug_fine_far_image"],
                           input_dict["aug_fine_far_video_mask"])

    crop_src_feat_list, _ = \
        utils.diff_rotation_aug_reverse([aug_src_feature_map,
                                         aug_src_desc_feature_map], [], input_dict["src_rot_angles"], 'bilinear')
    crop_close_feat_list, _ = \
        utils.diff_rotation_aug_reverse([aug_close_feature_map,
                                         aug_close_desc_feature_map], [], input_dict["close_rot_angles"], 'bilinear')
    crop_far_feat_list, _ = \
        utils.diff_rotation_aug_reverse(
            [aug_far_desc_feature_map], [], input_dict["far_rot_angles"], 'bilinear')

    crop_src_feature_map = crop_src_feat_list[0]
    crop_close_feature_map = crop_close_feat_list[0]
    crop_src_desc_feature_map = crop_src_feat_list[1]
    crop_close_desc_feature_map = crop_close_feat_list[1]
    crop_far_desc_feature_map = crop_far_feat_list[0]

    _, src_cdf_histograms = \
        utils.diff_1d_histogram_generation(crop_src_desc_feature_map,
                                           mask=input_dict["crop_coarse_src_video_mask"],
                                           num_bins=args.num_hist_bins)
    _, close_cdf_histograms = \
        utils.diff_1d_histogram_generation(crop_close_desc_feature_map,
                                           mask=input_dict["crop_coarse_close_video_mask"],
                                           num_bins=args.num_hist_bins)
    _, far_cdf_histograms = \
        utils.diff_1d_histogram_generation(crop_far_desc_feature_map,
                                           mask=input_dict["crop_coarse_far_video_mask"],
                                           num_bins=args.num_hist_bins)

    # checkpoint seeems to not work on class method!
    # Here we should use all sampled locations regardless of
    # whether or not there is a groundtruth match in the other image to mimic the reality
    x = model_dict["fm"]. \
        matching_location_estimation_cycle_consis(src_feature_map=crop_src_desc_feature_map,
                                                  tgt_feature_map=crop_close_desc_feature_map,
                                                  src_keypoint_2d_hw_locations=torch.cat([input_dict[
                                                      "reproj_src_keypoint_2d_locations"],
                                                      input_dict[
                                                      "reproj_src_no_match_2d_locations"]],
                                                      dim=1))

    if x is None:
        return None

    inlier_keypoint_2d_hw_locations, inlier_matched_2d_hw_locations = x

    if args.train_mode == "joint":
        aug_src_depth_bias, aug_src_depth_basis_list = \
            model_dict["depth"].forward_train(x=input_dict["aug_fine_src_image"],
                                              mask=input_dict["aug_fine_src_video_mask"],
                                              return_basis=True)

        aug_close_depth_bias = model_dict["depth"].forward_train(x=input_dict["aug_fine_close_image"],
                                                                 mask=input_dict["aug_fine_close_video_mask"],
                                                                 return_basis=False)

        crop_src_depth_list, _ = \
            utils.diff_rotation_aug_reverse([aug_src_depth_bias,
                                             *aug_src_depth_basis_list], [], input_dict["src_rot_angles"], 'bilinear')
        crop_src_depth_bias = crop_src_depth_list[0]
        crop_src_depth_basis_list = crop_src_depth_list[1:]

        crop_close_depth_bias, _ = \
            utils.diff_rotation_aug_reverse(
                [aug_close_depth_bias], [], input_dict["close_rot_angles"], 'bilinear')
        crop_close_depth_bias = crop_close_depth_bias[0]

        init_scale = (torch.sum(crop_close_depth_bias * input_dict["crop_coarse_close_video_mask"]) *
                      torch.sum(input_dict["crop_coarse_src_video_mask"])) / \
                     (torch.sum(crop_src_depth_bias * input_dict["crop_coarse_src_video_mask"]) *
                      torch.sum(input_dict["crop_coarse_close_video_mask"]))

        # from coarse to fine hierarchy for depth basis
        init_code_list = list()
        for depth_basis in crop_src_depth_basis_list:
            init_code_list.append(torch.zeros(depth_basis.shape[1],
                                              dtype=depth_basis.dtype, device=depth_basis.device))
    elif args.train_mode == "separate":
        src_depth_bias, src_depth_basis_list = \
            model_dict["depth"].forward_train(x=input_dict["fine_src_image"],
                                              mask=input_dict["fine_video_mask"],
                                              return_basis=True)

        close_depth_bias = model_dict["depth"].forward_train(x=input_dict["fine_close_image"],
                                                             mask=input_dict["fine_video_mask"],
                                                             return_basis=False)

        init_scale = torch.sum(close_depth_bias * input_dict["coarse_video_mask"]) / \
            torch.sum(src_depth_bias * input_dict["coarse_video_mask"])

        # from coarse to fine hierarchy for depth basis
        init_code_list = list()
        for depth_basis in src_depth_basis_list:
            init_code_list.append(torch.zeros(depth_basis.shape[1],
                                              dtype=depth_basis.dtype, device=depth_basis.device))

    if step % args.train_result_display_freq == 0:
        visualize_lm = True
    else:
        visualize_lm = False

    use_match_geom = True
    use_geom = True

    if args.train_mode == "joint":
        if args.debug_visualize_lm_result:
            if step % args.train_result_display_freq == 0:
                produce_video = True
            else:
                produce_video = False
        else:
            produce_video = False

        x = model_dict["ba"].ba_optimize(keypoint_2d_hw_locations=inlier_keypoint_2d_hw_locations,
                                         matched_2d_hw_locations=inlier_matched_2d_hw_locations,
                                         sampled_2d_hw_locations=input_dict["sampled_2d_locations"],
                                         camera_intrinsics=input_dict["camera_intrinsics"],
                                         src_feature_map=crop_src_feature_map,
                                         tgt_feature_map=crop_close_feature_map,
                                         src_valid_mask=input_dict["crop_coarse_src_video_mask"],
                                         tgt_valid_mask=input_dict["close_gt_depth_mask"] * input_dict[
                                             "crop_coarse_close_video_mask"],
                                         refer_tgt_depth_map=input_dict["close_gt_depth"] * input_dict[
                                             "crop_coarse_close_video_mask"],
                                         depth_map_bias=crop_src_depth_bias,
                                         depth_map_basis_list=crop_src_depth_basis_list,
                                         init_rotation=input_dict["random_rotation"],
                                         init_translation=input_dict["random_translation"],
                                         init_code_list=init_code_list, init_scale=init_scale,
                                         gradient_checkpoint=args.gradient_checkpoint,
                                         max_num_iters=args.lm_train_max_niters, init_damp=args.lm_init_damp,
                                         damp_min_max_cap=args.lm_damp_min_max_cap,
                                         damp_inc_dec_scale=args.lm_damp_inc_dec_factor,
                                         grad_thresh=args.lm_grad_threshold, param_thresh=args.lm_param_threshold,
                                         max_cond=args.lm_max_condition,
                                         src_input_image=input_dict["crop_coarse_src_image"],
                                         tgt_input_image=input_dict["crop_coarse_close_image"],
                                         src_desc_feature_map=crop_src_desc_feature_map,
                                         tgt_desc_feature_map=crop_close_desc_feature_map,
                                         visualize_lm=visualize_lm, gt_translation=input_dict[
                                             "gt_translation"],
                                         gt_rotation=input_dict["gt_rotation"],
                                         gt_flow_map=input_dict["gt_flow_map"],
                                         gt_flow_mask=input_dict["src_flow_mask"],
                                         src_gt_depth_map=input_dict["src_gt_depth"],
                                         tgt_gt_depth_map=input_dict["close_gt_depth"],
                                         use_match_geom=use_match_geom,
                                         use_geom=use_geom,
                                         produce_video=produce_video)

        _, _, _, _, guess_flow_map, flow_mask, \
            num_lm_steps, depth_list, result_list = x

        # L x 1 x H x W
        group_guess_src_depth_map = torch.cat(
            [crop_src_depth_bias, torch.relu(depth_list[-1])], dim=0)

        if produce_video:
            utils.write_video(result_list=result_list,
                              log_root=log_root, step=step)

        src_depth_mask = input_dict["src_gt_depth_mask"] * input_dict[
            "crop_coarse_src_video_mask"]
        tgt_depth_mask = input_dict["close_gt_depth_mask"] * input_dict[
            "crop_coarse_close_video_mask"]
        depth_loss = args.depth_loss_weight * (0.75 * loss_func_dict["depth"]([input_dict["src_gt_depth"],
                                                                               group_guess_src_depth_map,
                                                                               src_depth_mask]) +
                                               0.25 * loss_func_dict["depth"]([input_dict["close_gt_depth"],
                                                                               crop_close_depth_bias,
                                                                               tgt_depth_mask]))

        flow_loss = args.flow_loss_weight * \
            loss_func_dict["flow"]([input_dict["gt_flow_map"], guess_flow_map,
                                    input_dict["src_flow_mask"] * flow_mask])

        if input_dict["reproj_src_no_match_2d_locations"].shape[1] > 0:
            no_match_loss, no_match_response_map = \
                model_dict["fm"].calculate_no_match_loss([crop_src_desc_feature_map,
                                                          crop_close_desc_feature_map,
                                                          input_dict["reproj_src_no_match_2d_locations"]],
                                                         return_response=True)
            no_match_loss = args.no_match_loss_weight * no_match_loss
        else:
            no_match_loss = torch.tensor(
                0.0, dtype=torch.float32, device=flow_loss.device)

        desc_loss_1, close_desc_response_map = \
            model_dict["fm"].calculate_rr_loss([crop_src_desc_feature_map,
                                                crop_close_desc_feature_map,
                                                input_dict["reproj_src_keypoint_2d_locations"],
                                                input_dict["corr_close_2d_locations"]],
                                               return_response=True)
        desc_loss_2, src_desc_response_map = \
            model_dict["fm"].calculate_rr_loss([crop_close_desc_feature_map,
                                                crop_src_desc_feature_map,
                                                input_dict["corr_close_2d_locations"],
                                                input_dict["reproj_src_keypoint_2d_locations"]],
                                               return_response=True)

        desc_loss = args.desc_loss_weight * 0.5 * (desc_loss_1 + desc_loss_2)

        decor_loss = args.decor_loss_weight * loss_func_dict["decor"](crop_src_depth_basis_list[0],
                                                                      input_dict["crop_coarse_src_video_mask"])

        if input_dict["far_overlap_valid"][0]:
            hist_loss = args.hist_loss_weight * \
                loss_func_dict["hist"](
                    [src_cdf_histograms, close_cdf_histograms, far_cdf_histograms])
        else:
            hist_loss = torch.tensor(
                0.0, dtype=torch.float32, device=desc_loss.device)

        cat_pred_depth = torch.cat(depth_list, dim=0) * src_depth_mask
        steps, _, _, _ = cat_pred_depth.shape
        cat_pred_depth = cat_pred_depth / (torch.relu(
            torch.max(cat_pred_depth.reshape(steps, -1), dim=1, keepdim=True)[0].reshape(steps, 1, 1, 1)) + 1.0e-4)

        input_dict["src_gt_depth"] = input_dict["src_gt_depth"] * \
            src_depth_mask
        input_dict["src_gt_depth"] = input_dict["src_gt_depth"] / \
            (torch.relu(
                torch.max(input_dict["src_gt_depth"].reshape(1, -1), dim=1, keepdim=True)[
                    0].reshape(1, 1, 1, 1)) + 1.0e-4)

        c_fake = model_dict["disc"](
            torch.cat([(input_dict["crop_coarse_src_image"] * src_depth_mask).expand(len(depth_list), -1, -1, -1),
                       cat_pred_depth], dim=1))
        c_real = model_dict["disc"](
            torch.cat([input_dict["crop_coarse_src_image"] * src_depth_mask,
                       input_dict["src_gt_depth"]],
                      dim=1))

        g_adv_loss = args.g_adv_loss_weight * 0.5 * (
            torch.mean((c_fake - 1.0) ** 2) + torch.mean((c_real + 1.0) ** 2))

        loss = depth_loss + flow_loss + desc_loss + hist_loss + g_adv_loss + decor_loss

        optimizer_dict["G"].zero_grad()
        if torch.isnan(loss):
            logger.error("loss is NaN")
            return None

        if args.debug_gradient_tracking:
            utils.register_hooks_fn_grad(loss)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(list(model_dict["feat"].parameters()) +
                                       list(model_dict["depth"].parameters()) +
                                       list(model_dict["ba"].parameters()), 10.0)

        if args.debug_gradient_tracking:
            if utils.check_grad(model_dict["depth"], model_dict["feat"], args.debug_gradient_threshold):
                net_graph = utils.make_grad_dot(loss, fn_dict=utils.fn_dict,
                                                params={**dict(model_dict["feat"].named_parameters()),
                                                        **dict(model_dict["depth"].named_parameters())},
                                                bad_grad=args.debug_gradient_threshold)
                net_graph.render(
                    str(log_root / f'net_graph_{step}.gv'), format='svg', view=True)

        optimizer_dict["G"].step()

        c_fake = model_dict["disc"](
            torch.cat([(input_dict["crop_coarse_src_image"] * src_depth_mask).expand(len(depth_list), -1, -1, -1),
                       cat_pred_depth.detach()], dim=1))
        c_real = model_dict["disc"](
            torch.cat([input_dict["crop_coarse_src_image"] * src_depth_mask,
                       input_dict["src_gt_depth"]],
                      dim=1))

        d_adv_loss = args.d_adv_loss_weight * 0.5 * (
            torch.mean((c_real - 1.0) ** 2) + torch.mean((c_fake + 1.0) ** 2))
        d_adv_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(model_dict["disc"].parameters()), 10.0)

        optimizer_dict["D"].step()

        # display feature response map here to see how well the descriptor performs
        if step % args.train_result_display_freq == 0:
            with torch.no_grad():
                match_heatmap = utils.display_desc_heatmap(src_keypoint_2d_hw_location=input_dict["reproj_src_keypoint_2d_locations"][0, 0:1, :],
                                                           tgt_gt_2d_hw_location=input_dict[
                                                               "corr_close_2d_locations"][0, 0:1, :],
                                                           tgt_desc_response_map=close_desc_response_map,
                                                           src_desc_response_map=src_desc_response_map,
                                                           valid_mask=input_dict["crop_coarse_close_video_mask"],
                                                           src_input_image=input_dict["crop_coarse_src_image"],
                                                           tgt_input_image=input_dict["crop_coarse_close_image"],
                                                           sigma=1.0)
                utils.stack_and_display(phase="Train",
                                        title="match",
                                        step=step, writer=writer,
                                        image_list=[match_heatmap])

                if input_dict["reproj_src_no_match_2d_locations"].shape[1] > 0:
                    no_match_heatmap = utils.display_no_match_heatmap(src_no_match_2d_hw_location=input_dict["reproj_src_no_match_2d_locations"][0,
                                                                                                                                                 0:1, :],
                                                                      tgt_desc_response_map=no_match_response_map,
                                                                      valid_mask=input_dict[
                                                                          "crop_coarse_close_video_mask"],
                                                                      src_input_image=input_dict[
                                                                          "crop_coarse_src_image"],
                                                                      tgt_input_image=input_dict[
                                                                          "crop_coarse_close_image"],
                                                                      sigma=1.0)
                    utils.stack_and_display(phase="Train",
                                            title="no match",
                                            step=step, writer=writer,
                                            image_list=[no_match_heatmap])

                utils.stack_and_display(phase="Train",
                                        title="LM",
                                        step=step, writer=writer,
                                        image_list=[result_list[0], result_list[1], result_list[-1]])

                depth_optim_display = utils.display_depth_list(depth_list, height=args.net_input_image_size[0],
                                                               width=args.net_input_image_size[1])
                utils.stack_and_display(phase="Train",
                                        title="depth optim",
                                        step=step, writer=writer,
                                        image_list=[depth_optim_display])

                depth_basis_display = utils.display_basis_list(crop_src_depth_basis_list,
                                                               mask=input_dict["crop_coarse_src_video_mask"],
                                                               height=args.net_input_image_size[0],
                                                               width=args.net_input_image_size[1])
                utils.stack_and_display(phase="Train",
                                        title="depth basis",
                                        step=step, writer=writer,
                                        image_list=depth_basis_display)
                writer.flush()

        return {"total": loss.item(), "depth": depth_loss.item(), "flow": flow_loss.item(),
                "desc": desc_loss.item(), "hist": hist_loss.item(),
                "nm": no_match_loss.item(),
                "g": g_adv_loss.item(), "d": d_adv_loss.item(),
                "decor": decor_loss.item(),
                }, \
            num_lm_steps, use_match_geom

    elif args.train_mode == "separate":
        result_list = \
            model_dict["ba"].ba_optimize(keypoint_2d_hw_locations=inlier_keypoint_2d_hw_locations,
                                         matched_2d_hw_locations=inlier_matched_2d_hw_locations,
                                         sampled_2d_hw_locations=input_dict["sampled_2d_locations"],
                                         camera_intrinsics=input_dict["camera_intrinsics"],
                                         src_feature_map=crop_src_feature_map,
                                         tgt_feature_map=crop_close_feature_map,
                                         src_valid_mask=input_dict["crop_coarse_src_video_mask"],
                                         tgt_valid_mask=input_dict["crop_coarse_close_video_mask"],
                                         refer_tgt_depth_map=close_depth_bias,
                                         depth_map_bias=src_depth_bias,
                                         depth_map_basis_list=src_depth_basis_list,
                                         init_rotation=input_dict["random_rotation"],
                                         init_translation=input_dict["random_translation"],
                                         init_code_list=init_code_list, init_scale=init_scale,
                                         gradient_checkpoint=args.gradient_checkpoint,
                                         max_num_iters=0, init_damp=args.lm_init_damp,
                                         damp_min_max_cap=args.lm_damp_min_max_cap,
                                         damp_inc_dec_scale=args.lm_damp_inc_dec_factor,
                                         grad_thresh=args.lm_grad_threshold,
                                         param_thresh=args.lm_param_threshold,
                                         max_cond=args.lm_max_condition,
                                         src_input_image=input_dict["crop_coarse_src_image"],
                                         tgt_input_image=input_dict["crop_coarse_close_image"],
                                         src_desc_feature_map=crop_src_desc_feature_map,
                                         tgt_desc_feature_map=crop_close_desc_feature_map,
                                         visualize_lm=visualize_lm,
                                         gt_translation=input_dict["gt_translation"],
                                         gt_rotation=input_dict["gt_rotation"],
                                         gt_flow_map=input_dict["gt_flow_map"],
                                         gt_flow_mask=input_dict["src_flow_mask"],
                                         src_gt_depth_map=input_dict["src_gt_depth"],
                                         tgt_gt_depth_map=input_dict["close_gt_depth"],
                                         use_match_geom=use_match_geom,
                                         use_geom=use_geom,
                                         produce_video=False)

        src_depth_mask = input_dict["src_gt_depth_mask"] * \
            input_dict["coarse_video_mask"]
        tgt_depth_mask = input_dict["close_gt_depth_mask"] * \
            input_dict["coarse_video_mask"]
        depth_loss = args.depth_loss_weight * (0.75 * loss_func_dict["depth"]([input_dict["src_gt_depth"],
                                                                               src_depth_bias,
                                                                               src_depth_mask]) +
                                               0.25 * loss_func_dict["depth"]([input_dict["close_gt_depth"],
                                                                               close_depth_bias,
                                                                               tgt_depth_mask]))
        if input_dict["reproj_src_no_match_2d_locations"].shape[1] > 0:
            no_match_loss, no_match_response_map = \
                model_dict["fm"].calculate_no_match_loss([crop_src_desc_feature_map,
                                                          crop_close_desc_feature_map,
                                                          input_dict["reproj_src_no_match_2d_locations"]],
                                                         return_response=True)
            no_match_loss = args.no_match_loss_weight * no_match_loss
        else:
            no_match_loss = torch.tensor(
                0.0, dtype=torch.float32, device=depth_loss.device)

        desc_loss_1, close_desc_response_map = \
            model_dict["fm"].calculate_rr_loss([crop_src_desc_feature_map,
                                                crop_close_desc_feature_map,
                                                input_dict["reproj_src_keypoint_2d_locations"],
                                                input_dict["corr_close_2d_locations"]],
                                               return_response=True)
        desc_loss_2, src_desc_response_map = \
            model_dict["fm"].calculate_rr_loss([crop_close_desc_feature_map,
                                                crop_src_desc_feature_map,
                                                input_dict["corr_close_2d_locations"],
                                                input_dict["reproj_src_keypoint_2d_locations"]],
                                               return_response=True)

        desc_loss = args.desc_loss_weight * 0.5 * (desc_loss_1 + desc_loss_2)

        if input_dict["far_overlap_valid"][0]:
            hist_loss = args.hist_loss_weight * \
                loss_func_dict["hist"](
                    [src_cdf_histograms, close_cdf_histograms, far_cdf_histograms])
        else:
            hist_loss = torch.tensor(
                0.0, dtype=torch.float32, device=desc_loss.device)

        loss = depth_loss + desc_loss + hist_loss

        optimizer_dict["G"].zero_grad()
        if torch.isnan(loss):
            logger.error("loss is NaN")
            return None

        if args.debug_gradient_tracking:
            utils.register_hooks_fn_grad(loss)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(list(model_dict["feat"].parameters()) +
                                       list(model_dict["depth"].parameters()) +
                                       list(model_dict["ba"].parameters()), 10.0)

        if args.debug_gradient_tracking:
            if utils.check_grad(model_dict["depth"], model_dict["feat"], args.debug_gradient_threshold):
                net_graph = utils.make_grad_dot(loss, fn_dict=utils.fn_dict,
                                                params={**dict(model_dict["feat"].named_parameters()),
                                                        **dict(model_dict["depth"].named_parameters())},
                                                bad_grad=args.debug_gradient_threshold)
                net_graph.render(
                    str(log_root / f'net_graph_{step}.gv'), format='svg', view=True)

        optimizer_dict["G"].step()

        if step % args.train_result_display_freq == 0:
            with torch.no_grad():
                match_heatmap = utils.display_desc_heatmap(src_keypoint_2d_hw_location=input_dict["reproj_src_keypoint_2d_locations"][0, 0:1, :],
                                                           tgt_gt_2d_hw_location=input_dict[
                                                               "corr_close_2d_locations"][0, 0:1, :],
                                                           tgt_desc_response_map=close_desc_response_map,
                                                           src_desc_response_map=src_desc_response_map,
                                                           valid_mask=input_dict["crop_coarse_close_video_mask"],
                                                           src_input_image=input_dict["crop_coarse_src_image"],
                                                           tgt_input_image=input_dict["crop_coarse_close_image"],
                                                           sigma=1.0)
                utils.stack_and_display(phase="Train",
                                        title="match",
                                        step=step, writer=writer,
                                        image_list=[match_heatmap])

                if input_dict["reproj_src_no_match_2d_locations"].shape[1] > 0:
                    no_match_heatmap = utils.display_no_match_heatmap(src_no_match_2d_hw_location=input_dict["reproj_src_no_match_2d_locations"][0,
                                                                                                                                                 0:1, :],
                                                                      tgt_desc_response_map=no_match_response_map,
                                                                      valid_mask=input_dict[
                                                                          "crop_coarse_close_video_mask"],
                                                                      src_input_image=input_dict[
                                                                          "crop_coarse_src_image"],
                                                                      tgt_input_image=input_dict[
                                                                          "crop_coarse_close_image"],
                                                                      sigma=1.0)
                    utils.stack_and_display(phase="Train",
                                            title="no match",
                                            step=step, writer=writer,
                                            image_list=[no_match_heatmap])

                utils.stack_and_display(phase="Train",
                                        title="LM",
                                        step=step, writer=writer,
                                        image_list=result_list)

                writer.flush()

        return {"total": loss.item(), "depth": depth_loss.item(),
                "desc": desc_loss.item(), "hist": hist_loss.item(),
                "nm": no_match_loss.item()}, 0, use_match_geom

    else:
        logger.error(f"{args.train_mode} not supported")
        exit(0)


def evaluation(model_dict, input_dict, loss_func_dict, args, log_root, writer, step):
    with torch.no_grad():
        aug_src_feature_map, aug_src_desc_feature_map = \
            model_dict["feat"](
                input_dict["aug_fine_src_image"], input_dict["aug_fine_src_video_mask"])
        aug_close_feature_map, aug_close_desc_feature_map = \
            model_dict["feat"](
                input_dict["aug_fine_close_image"], input_dict["aug_fine_close_video_mask"])
        _, aug_far_desc_feature_map = \
            model_dict["feat"](
                input_dict["aug_fine_far_image"], input_dict["aug_fine_far_video_mask"])

        crop_src_feat_list, _ = \
            utils.diff_rotation_aug_reverse([aug_src_feature_map,
                                             aug_src_desc_feature_map], [], input_dict["src_rot_angles"], 'bilinear')
        crop_close_feat_list, _ = \
            utils.diff_rotation_aug_reverse([aug_close_feature_map,
                                             aug_close_desc_feature_map], [], input_dict["close_rot_angles"],
                                            'bilinear')
        crop_far_feat_list, _ = \
            utils.diff_rotation_aug_reverse(
                [aug_far_desc_feature_map], [], input_dict["far_rot_angles"], 'bilinear')

        crop_src_feature_map = crop_src_feat_list[0]
        crop_close_feature_map = crop_close_feat_list[0]
        crop_src_desc_feature_map = crop_src_feat_list[1]
        crop_close_desc_feature_map = crop_close_feat_list[1]
        crop_far_desc_feature_map = crop_far_feat_list[0]

        _, src_cdf_histograms = \
            utils.diff_1d_histogram_generation(crop_src_desc_feature_map,
                                               mask=input_dict["crop_coarse_src_video_mask"],
                                               num_bins=args.num_hist_bins)
        _, close_cdf_histograms = \
            utils.diff_1d_histogram_generation(crop_close_desc_feature_map,
                                               mask=input_dict["crop_coarse_close_video_mask"],
                                               num_bins=args.num_hist_bins)
        _, far_cdf_histograms = \
            utils.diff_1d_histogram_generation(crop_far_desc_feature_map,
                                               mask=input_dict["crop_coarse_far_video_mask"],
                                               num_bins=args.num_hist_bins)

        # checkpoint seeems to not work on class method!
        # Here we should use all sampled locations regardless of
        # whether or not there is a groundtruth match in the other image to mimic the reality
        x = model_dict["fm"]. \
            matching_location_estimation_cycle_consis(src_feature_map=crop_src_desc_feature_map,
                                                      tgt_feature_map=crop_close_desc_feature_map,
                                                      src_keypoint_2d_hw_locations=torch.cat([input_dict[
                                                          "reproj_src_keypoint_2d_locations"],
                                                          input_dict[
                                                          "reproj_src_no_match_2d_locations"]],
                                                          dim=1))

        if x is None:
            return None

        inlier_keypoint_2d_hw_locations, inlier_matched_2d_hw_locations = x

        src_depth_bias, src_depth_basis_list = model_dict["depth"].forward_train(x=input_dict["fine_src_image"],
                                                                                 mask=input_dict["fine_video_mask"],
                                                                                 return_basis=True)

        close_depth_bias = model_dict["depth"].forward_train(x=input_dict["fine_close_image"],
                                                             mask=input_dict["fine_video_mask"],
                                                             return_basis=False)

        sum_1 = torch.sum(close_depth_bias * input_dict["coarse_video_mask"]) / torch.sum(
            input_dict["coarse_video_mask"])
        sum_2 = torch.sum(src_depth_bias * input_dict["coarse_video_mask"]) / torch.sum(
            input_dict["coarse_video_mask"])
        init_scale = sum_1 / sum_2

        # from coarse to fine hierarchy for depth basis
        init_code_list = list()
        for depth_basis in src_depth_basis_list:
            init_code_list.append(torch.zeros(depth_basis.shape[1],
                                              dtype=depth_basis.dtype, device=depth_basis.device))

        if step % args.eval_result_display_freq == 0:
            visualize_lm = True
        else:
            visualize_lm = False

        use_match_geom = True
        use_geom = True

        if args.debug_visualize_lm_result:
            if step % args.eval_result_display_freq == 0:
                produce_video = True
            else:
                produce_video = False
        else:
            produce_video = False

        x = model_dict["ba"].ba_optimize(keypoint_2d_hw_locations=inlier_keypoint_2d_hw_locations,
                                         matched_2d_hw_locations=inlier_matched_2d_hw_locations,
                                         sampled_2d_hw_locations=input_dict["sampled_2d_locations"],
                                         camera_intrinsics=input_dict["camera_intrinsics"],
                                         src_feature_map=crop_src_feature_map,
                                         tgt_feature_map=crop_close_feature_map,
                                         src_valid_mask=input_dict["crop_coarse_src_video_mask"],
                                         tgt_valid_mask=input_dict["crop_coarse_close_video_mask"],
                                         refer_tgt_depth_map=close_depth_bias,
                                         depth_map_bias=src_depth_bias,
                                         depth_map_basis_list=src_depth_basis_list,
                                         init_rotation=input_dict["random_rotation"],
                                         init_translation=input_dict["random_translation"],
                                         init_code_list=init_code_list, init_scale=init_scale,
                                         gradient_checkpoint=False,
                                         max_num_iters=args.lm_eval_max_niters, init_damp=args.lm_init_damp,
                                         damp_min_max_cap=args.lm_damp_min_max_cap,
                                         damp_inc_dec_scale=args.lm_damp_inc_dec_factor,
                                         grad_thresh=args.lm_grad_threshold, param_thresh=args.lm_param_threshold,
                                         max_cond=args.lm_max_condition,
                                         src_input_image=input_dict["crop_coarse_src_image"],
                                         tgt_input_image=input_dict["crop_coarse_close_image"],
                                         src_desc_feature_map=crop_src_desc_feature_map,
                                         tgt_desc_feature_map=crop_close_desc_feature_map,
                                         visualize_lm=visualize_lm, gt_translation=input_dict[
                                             "gt_translation"],
                                         gt_rotation=input_dict["gt_rotation"],
                                         gt_flow_map=input_dict["gt_flow_map"],
                                         gt_flow_mask=input_dict["src_flow_mask"],
                                         src_gt_depth_map=input_dict["src_gt_depth"],
                                         tgt_gt_depth_map=input_dict["close_gt_depth"],
                                         use_match_geom=use_match_geom,
                                         use_geom=use_geom,
                                         produce_video=produce_video)

        _, _, _, _, guess_flow_map, flow_mask, \
            num_lm_steps, depth_list, result_list = x

        # L x 1 x H x W
        group_guess_src_depth_map = torch.cat(
            [src_depth_bias, torch.relu(depth_list[-1])], dim=0)

        if produce_video:
            utils.write_video(result_list=result_list,
                              log_root=log_root, step=step)

        src_depth_mask = input_dict["src_gt_depth_mask"] * \
            input_dict["coarse_video_mask"]
        tgt_depth_mask = input_dict["close_gt_depth_mask"] * \
            input_dict["coarse_video_mask"]
        depth_loss = args.depth_loss_weight * (0.75 * loss_func_dict["depth"]([input_dict["src_gt_depth"],
                                                                               group_guess_src_depth_map,
                                                                               src_depth_mask]) +
                                               0.25 * loss_func_dict["depth"]([input_dict["close_gt_depth"],
                                                                               close_depth_bias,
                                                                               tgt_depth_mask]))

        flow_loss = args.flow_loss_weight * \
            loss_func_dict["flow"]([input_dict["gt_flow_map"], guess_flow_map,
                                    input_dict["src_flow_mask"] * flow_mask])

        if input_dict["reproj_src_no_match_2d_locations"].shape[1] > 0:
            no_match_loss, no_match_response_map = \
                model_dict["fm"].calculate_no_match_loss([crop_src_desc_feature_map,
                                                          crop_close_desc_feature_map,
                                                          input_dict["reproj_src_no_match_2d_locations"]],
                                                         return_response=True)
            no_match_loss = args.no_match_loss_weight * no_match_loss
        else:
            no_match_loss = torch.tensor(
                0.0, dtype=torch.float32, device=flow_loss.device)

        desc_loss_1, close_desc_response_map = \
            model_dict["fm"].calculate_rr_loss([crop_src_desc_feature_map,
                                                crop_close_desc_feature_map,
                                                input_dict["reproj_src_keypoint_2d_locations"],
                                                input_dict["corr_close_2d_locations"]],
                                               return_response=True)
        desc_loss_2, src_desc_response_map = \
            model_dict["fm"].calculate_rr_loss([crop_close_desc_feature_map,
                                                crop_src_desc_feature_map,
                                                input_dict["corr_close_2d_locations"],
                                                input_dict["reproj_src_keypoint_2d_locations"]],
                                               return_response=True)

        desc_loss = args.desc_loss_weight * 0.5 * (desc_loss_1 + desc_loss_2)

        if input_dict["far_overlap_valid"][0]:
            hist_loss = args.hist_loss_weight * \
                loss_func_dict["hist"](
                    [src_cdf_histograms, close_cdf_histograms, far_cdf_histograms])
        else:
            hist_loss = torch.tensor(
                0.0, dtype=torch.float32, device=desc_loss.device)

        loss = depth_loss + flow_loss + desc_loss + hist_loss

        if torch.isnan(loss):
            logger.error("loss is NaN")
            return None

        # display feature response map here to see how well the descriptor performs
        if step % args.eval_result_display_freq == 0:
            with torch.no_grad():
                match_heatmap = utils.display_desc_heatmap(src_keypoint_2d_hw_location=input_dict["reproj_src_keypoint_2d_locations"][0, 0:1, :],
                                                           tgt_gt_2d_hw_location=input_dict[
                                                               "corr_close_2d_locations"][0, 0:1, :],
                                                           tgt_desc_response_map=close_desc_response_map,
                                                           src_desc_response_map=src_desc_response_map,
                                                           valid_mask=input_dict["crop_coarse_close_video_mask"],
                                                           src_input_image=input_dict["crop_coarse_src_image"],
                                                           tgt_input_image=input_dict["crop_coarse_close_image"],
                                                           sigma=1.0)
                utils.stack_and_display(phase="Eval",
                                        title="match",
                                        step=step, writer=writer,
                                        image_list=[match_heatmap])

                if input_dict["reproj_src_no_match_2d_locations"].shape[1] > 0:
                    no_match_heatmap = utils.display_no_match_heatmap(src_no_match_2d_hw_location=input_dict["reproj_src_no_match_2d_locations"][0,
                                                                                                                                                 0:1, :],
                                                                      tgt_desc_response_map=no_match_response_map,
                                                                      valid_mask=input_dict[
                                                                          "crop_coarse_close_video_mask"],
                                                                      src_input_image=input_dict[
                                                                          "crop_coarse_src_image"],
                                                                      tgt_input_image=input_dict[
                                                                          "crop_coarse_close_image"],
                                                                      sigma=1.0)
                    utils.stack_and_display(phase="Eval",
                                            title="no match",
                                            step=step, writer=writer,
                                            image_list=[no_match_heatmap])

                utils.stack_and_display(phase="Eval",
                                        title="LM",
                                        step=step, writer=writer,
                                        image_list=[result_list[0], result_list[1], result_list[-1]])

                depth_optim_display = utils.display_depth_list(depth_list, height=args.net_input_image_size[0],
                                                               width=args.net_input_image_size[1])
                utils.stack_and_display(phase="Eval",
                                        title="depth optim",
                                        step=step, writer=writer,
                                        image_list=[depth_optim_display])

                depth_basis_display = utils.display_basis_list(src_depth_basis_list,
                                                               mask=input_dict["coarse_video_mask"],
                                                               height=args.net_input_image_size[0],
                                                               width=args.net_input_image_size[1])
                utils.stack_and_display(phase="Eval",
                                        title="depth basis",
                                        step=step, writer=writer,
                                        image_list=depth_basis_display)
                writer.flush()

        return {"total": loss.item(), "depth": depth_loss.item(), "flow": flow_loss.item(),
                "desc": desc_loss.item(), "hist": hist_loss.item(),
                "nm": no_match_loss.item()}, num_lm_steps, use_match_geom


if __name__ == "__main__":
    main()
