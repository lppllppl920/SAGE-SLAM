import copy
import numpy as np
import pickle
import os
from pathlib import Path
import shutil
import argparse
import h5py
from colorama import init

from evo.tools import plot, file_interface, log
from evo.core import sync, metrics
import evo.core.filters
from rpg_trajectory_evaluation import transformations
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Cardo']})
rc('text', usetex=True)


init(autoreset=True)


log.configure_logging(verbose=True, debug=False, silent=False)


def make_views(ax, angles, elevation=None, width=4, height=3,
               prefix='tmprot_', **kwargs):
    """
    Makes jpeg pictures of the given 3d ax, with different angles.
    Args:
        ax (3D axis): te ax
        angles (list): the list of angles (in degree) under which to
                       take the picture.
        width,height (float): size, in inches, of the output images.
        prefix (str): prefix for the files created.

    Returns: the list of files created (for later removal)
    """

    files = []
    ax.figure.set_size_inches(width, height)

    for i, angle in enumerate(angles):
        ax.view_init(elev=elevation, azim=angle)
        fname = str(Path(prefix) / "tmprot_{:03d}.jpeg".format(i))
        ax.figure.savefig(fname)
        files.append(fname)

    return files


def make_movie(files, output, fps=10, bitrate=1800, **kwargs):
    """
    Uses mencoder, produces a .mp4/.ogv/... movie from a list of
    picture files.
    """

    output_name, output_ext = os.path.splitext(output)
    command = {'.mp4': 'mencoder "mf://%s" -mf fps=%d -o %s.mp4 -ovc lavc\
                         -lavcopts vcodec=msmpeg4v2:vbitrate=%d'
                       % (",".join(files), fps, output_name, bitrate)}

    command['.ogv'] = command['.mp4'] + \
        '; ffmpeg -i %s.mp4 -r %d %s' % (output_name, fps, output)

    print(command[output_ext])
    output_ext = os.path.splitext(output)[1]
    os.system(command[output_ext])


def make_gif(files, output, delay=100, repeat=True, **kwargs):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    """

    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s \"%s\"'
              % (delay, loop, " ".join(["\"" + fn + "\"" for fn in files]), output))


def make_strip(files, output, **kwargs):
    """
    Uses imageMagick to produce a .jpeg strip from a list of
    picture files.
    """

    os.system('montage -tile 1x -geometry +0+0 %s %s' %
              (" ".join(files), output))


def rotanimate(ax, angles, output, **kwargs):
    """
    Produces an animation (.mp4,.ogv,.gif,.jpeg,.png) from a 3D plot on
    a 3D ax

    Args:
        ax (3D axis): the ax containing the plot of interest
        angles (list): the list of angles (in degree) under which to
                       show the plot.
        output : name of the output file. The extension determines the
                 kind of animation used.
        **kwargs:
            - width : in inches
            - heigth: in inches
            - framerate : frames per second
            - delay : delay between frames in milliseconds
            - repeat : True or False (.gif only)
    """

    output_ext = os.path.splitext(output)[1]

    files = make_views(ax, angles, **kwargs)

    D = {'.mp4': make_movie,
         '.ogv': make_movie,
         '.gif': make_gif,
         '.jpeg': make_strip,
         '.png': make_strip}

    D[output_ext](files, output, **kwargs)

    for f in files:
        os.remove(f)


def orb_hdf5_to_traj_txt(hdf5_path):
    orb_slam_result = h5py.File(str(hdf5_path), 'r', libver='latest')
    pose_dataset = orb_slam_result["t_cw"]
    timestamp_dataset = orb_slam_result["timestamp"]

    # f_endoscopy_trajectory stores the poses of every frame in the largest map
    # kf_endsocopy_trajectory stores the poses of all keyframes in all maps
    # p_w saves the world coordinate of all keypoints in each frame
    # t_cw is T^{camera}_{world}
    hdf5_root = hdf5_path.parent
    fp = open(str(hdf5_root / "orb_traj_estimate.txt"), "w")
    fp.write("# timestamp tx ty tz qx qy qz qw\n")
    for i in range(pose_dataset.shape[0]):
        pose_cw = pose_dataset[i]
        pose_wc = np.eye(4)
        pose_wc[:3, :3] = np.transpose(pose_cw[:3, :3])
        pose_wc[:3, 3] = np.matmul(np.transpose(
            pose_cw[:3, :3]), -pose_cw[:3, 3])
        rotation_4x4 = transformations.convert_3x3_to_4x4(pose_wc[:3, :3])
        # qx qy qz qw
        quaternion = transformations.quaternion_from_matrix(rotation_4x4)
        # tx ty tz
        translation = pose_wc[:3, 3]
        fp.write(f"{timestamp_dataset[i]} {translation[0]} {translation[1]} {translation[2]} "
                 f"{quaternion[0]} {quaternion[1]} {quaternion[2]} {quaternion[3]}\n")


def animate(i, ax):
    ax.view_init(elev=10., azim=i)
    return ax


def plot_traj(est_traj_path_list, gt_traj_path, log_root, num_frames_to_align, display_num_rotate,
              display_delay, rpe_delta, overwrite):
    colors = ['green', 'red', 'blue', 'purple']
    display_angles = np.linspace(0, 360, display_num_rotate)[:-1]
    plot_mode = plot.PlotMode.xyz
    rpe_delta_unit = metrics.Unit.frames
    rpe_all_pairs = True

    rpe_delta_list = list()
    ours_kf_interval = None
    ours_rpe_delta = rpe_delta

    gt_traj = file_interface.read_tum_trajectory_file(str(gt_traj_path))
    sync_sim3_est_traj_list = list()
    sync_gt_traj_list = list()
    for i, traj_path in enumerate(est_traj_path_list):
        est_traj = file_interface.read_tum_trajectory_file(str(traj_path))
        sync_est_traj, sync_gt_traj = sync.associate_trajectories(
            est_traj, gt_traj, max_diff=1.0e-4)
        sync_est_traj_sim3 = copy.deepcopy(sync_est_traj)
        _, _, s_gt_est = \
            sync_est_traj_sim3.align(
                sync_gt_traj, correct_scale=True, n=num_frames_to_align)
        with open(str(traj_path.parent / (traj_path.name[:-4] + "_scale_gt_est.txt")), "w") as fp:
            fp.write(f"{s_gt_est}")

        if i == 0 and "stamped" not in str(traj_path):
            raise IOError("ours result should be in the first item")

        if "stamped" in str(traj_path):
            rpe_delta_list.append(ours_rpe_delta)
            ours_kf_interval = (sync_est_traj_sim3.timestamps[-1] - sync_est_traj_sim3.timestamps[0]) / \
                               (len(sync_est_traj_sim3.timestamps) - 1)
        else:
            kf_interval = (sync_est_traj_sim3.timestamps[-1] - sync_est_traj_sim3.timestamps[0]) / \
                          (len(sync_est_traj_sim3.timestamps) - 1)
            rpe_delta_list.append(
                round(ours_kf_interval / kf_interval * ours_rpe_delta))

        sync_gt_traj_list.append(sync_gt_traj)
        sync_sim3_est_traj_list.append(sync_est_traj_sim3)

    pdf_path = log_root / "traj_overlay_sim3.pdf"
    gif_path = log_root / "traj_overlay_sim3.gif"
    if overwrite or not pdf_path.exists() or not gif_path.exists():
        fig = plt.figure(figsize=(8, 8))
        ax = plot.prepare_axis(fig, plot_mode)
        plot.traj(ax, plot_mode, gt_traj, '--', color='black',
                  label=str(gt_traj_path.name[:-4]).replace("_", "-"))
        for i, est_traj in enumerate(sync_sim3_est_traj_list):
            plot.traj(ax, plot_mode, est_traj, '-',
                      color=colors[i % len(colors)],
                      label=str(est_traj_path_list[i].name[:-4]).replace("_", "-"))
            positions_xyz = est_traj.positions_xyz
            ax.scatter(positions_xyz[0, 0], positions_xyz[0, 1], positions_xyz[0, 2], color=colors[i % len(colors)],
                       marker='o')
            ax.scatter(positions_xyz[-1, 0], positions_xyz[-1, 1], positions_xyz[-1, 2], color=colors[i % len(colors)],
                       marker='+')
        fig.tight_layout()
        multipage(str(pdf_path))

        if overwrite or not gif_path.exists():
            # create an animated gif
            rotanimate(ax, display_angles, str(gif_path),
                       delay=display_delay, height=8, width=8, prefix=str(log_root))

    plt.close('all')

    for i in range(len(sync_sim3_est_traj_list)):
        pdf_path = log_root / \
            (str(est_traj_path_list[i].name[:-4]).replace("_", "-") + ".pdf")

        sim3_est_traj = sync_sim3_est_traj_list[i]

        data = (sync_gt_traj_list[i], sim3_est_traj)
        data_rpe = (sync_gt_traj_list[i], sim3_est_traj)

        # translation part of APE
        ape_trans_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ape_trans_metric.process_data(data)
        ape_trans_statistics = ape_trans_metric.get_all_statistics()

        # rotation part of APE
        ape_rot_metric = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
        ape_rot_metric.process_data(data)
        ape_rot_statistics = ape_rot_metric.get_all_statistics()

        ape_trans_title = "APE w.r.t. " + ape_trans_metric.pose_relation.value + "--" + str(
            est_traj_path_list[i].name[:-4]).replace("_", "-")
        ape_rot_title = "APE w.r.t. " + ape_rot_metric.pose_relation.value + "--" + str(
            est_traj_path_list[i].name[:-4]).replace("_", "-")
        with open(str(log_root / f'{ape_trans_title}.pkl'), 'wb') as fp:
            pickle.dump(ape_trans_statistics, fp,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(str(log_root / f'{ape_rot_title}.pkl'), 'wb') as fp:
            pickle.dump(ape_rot_statistics, fp,
                        protocol=pickle.HIGHEST_PROTOCOL)

        # Plot error figure for APE translation and rotation
        if overwrite or not pdf_path.exists():
            seconds_from_start = [t - sim3_est_traj.timestamps[0]
                                  for t in sim3_est_traj.timestamps]
            fig = plt.figure()
            plot.error_array(fig.gca(), ape_trans_metric.error, x_array=seconds_from_start,
                             statistics={
                                 s: v for s, v in ape_trans_statistics.items() if s != "sse"},
                             name="APE translation",
                             title=ape_trans_title, xlabel="frame")
            fig = plt.figure()
            plot.error_array(fig.gca(), ape_rot_metric.error, x_array=seconds_from_start,
                             statistics={
                                 s: v for s, v in ape_rot_statistics.items() if s != "sse"},
                             name="APE rotation ($\degree$)",
                             title=ape_rot_title, xlabel="frame")

        # Plot APE error colored trajectory
        gif_path = log_root / ("Traj " + ape_trans_title + ".gif")
        if overwrite or not gif_path.exists() or not pdf_path.exists():
            fig = plt.figure()
            ax = plot.prepare_axis(fig, plot_mode)
            plot.traj(ax, plot_mode, gt_traj, '--', "black",
                      str(gt_traj_path.name[:-4]).replace("_", "-"))
            plot.traj_colormap(ax, sim3_est_traj, ape_trans_metric.error,
                               plot_mode, min_map=ape_trans_statistics["min"], max_map=ape_trans_statistics["max"],
                               title="Traj APE w.r.t. " + ape_trans_metric.pose_relation.value + "--" + str(
                                   est_traj_path_list[i].name[:-4]).replace("_", "-"))
            if overwrite or not gif_path.exists():
                rotanimate(ax, display_angles, str(gif_path),
                           delay=display_delay, height=8, width=8, prefix=str(log_root))

        gif_path = log_root / ("Traj " + ape_rot_title + ".gif")
        if overwrite or not gif_path.exists() or not pdf_path.exists():
            fig = plt.figure()
            ax = plot.prepare_axis(fig, plot_mode)
            plot.traj(ax, plot_mode, gt_traj, '--', "black",
                      str(gt_traj_path.name[:-4]).replace("_", "-"))
            plot.traj_colormap(ax, sim3_est_traj, ape_rot_metric.error,
                               plot_mode, min_map=ape_rot_statistics["min"], max_map=ape_rot_statistics["max"],
                               title="Traj APE-w.r.t. " + ape_rot_metric.pose_relation.value + "--" + str(
                                   est_traj_path_list[i].name[:-4]).replace("_", "-"))
            if overwrite or not gif_path.exists():
                rotanimate(ax, display_angles, str(gif_path),
                           delay=display_delay, height=8, width=8, prefix=str(log_root))

        try:
            # translation part of RPE
            rpe_trans_metric = metrics.RPE(metrics.PoseRelation.translation_part,
                                           delta=rpe_delta_list[i], delta_unit=rpe_delta_unit,
                                           all_pairs=rpe_all_pairs)
            rpe_trans_metric.process_data(data_rpe)
            rpe_trans_statistics = rpe_trans_metric.get_all_statistics()
            # rotation part of RPE
            rpe_rot_metric = metrics.RPE(metrics.PoseRelation.rotation_angle_deg,
                                         delta=rpe_delta_list[i], delta_unit=rpe_delta_unit,
                                         all_pairs=rpe_all_pairs)
            rpe_rot_metric.process_data(data_rpe)
            rpe_rot_statistics = rpe_rot_metric.get_all_statistics()

            rpe_trans_title = "RPE w.r.t. " + rpe_trans_metric.pose_relation.value + "--" + str(
                est_traj_path_list[i].name[:-4]).replace("_", "-")
            rpe_rot_title = "RPE w.r.t. " + rpe_rot_metric.pose_relation.value + "--" + str(
                est_traj_path_list[i].name[:-4]).replace("_", "-")
            with open(str(log_root / f'{rpe_trans_title}.pkl'), 'wb') as fp:
                pickle.dump(rpe_trans_statistics, fp,
                            protocol=pickle.HIGHEST_PROTOCOL)
            with open(str(log_root / f'{rpe_rot_title}.pkl'), 'wb') as fp:
                pickle.dump(rpe_rot_statistics, fp,
                            protocol=pickle.HIGHEST_PROTOCOL)

            est_plot_traj = copy.deepcopy(sim3_est_traj)
            est_plot_traj.reduce_to_ids(rpe_trans_metric.delta_ids)
            seconds_from_start_rpe = [
                t - est_plot_traj.timestamps[0] for t in est_plot_traj.timestamps]

            # Plot error figure for RPE translation and rotation
            if overwrite or not pdf_path.exists():
                fig = plt.figure()
                plot.error_array(fig.gca(), rpe_trans_metric.error, x_array=seconds_from_start_rpe,
                                 statistics={
                                     s: v for s, v in rpe_trans_statistics.items() if s != "sse"},
                                 name="RPE translation",
                                 title=rpe_trans_title, xlabel="frame")
                fig = plt.figure()
                plot.error_array(fig.gca(), rpe_rot_metric.error, x_array=seconds_from_start_rpe,
                                 statistics={
                                     s: v for s, v in rpe_rot_statistics.items() if s != "sse"},
                                 name="RPE rotation ($\degree$)",
                                 title=rpe_rot_title, xlabel="frame")

            # Plot RPE error colored trajectory
            gif_path = log_root / ("Traj " + rpe_trans_title + ".gif")
            if overwrite or not gif_path.exists() or not pdf_path.exists():
                fig = plt.figure()
                ax = plot.prepare_axis(fig, plot_mode)
                plot.traj(ax, plot_mode, gt_traj, '--', "black",
                          str(gt_traj_path.name[:-4]).replace("_", "-"))
                plot.traj_colormap(ax, est_plot_traj, rpe_trans_metric.error,
                                   plot_mode, min_map=rpe_trans_statistics[
                                       "min"], max_map=rpe_trans_statistics["max"],
                                   title="Traj " + rpe_trans_title)
                if not gif_path.exists():
                    rotanimate(ax, display_angles, str(gif_path),
                               delay=display_delay, height=8, width=8, prefix=str(log_root))

            gif_path = log_root / ("Traj " + rpe_rot_title + ".gif")
            if overwrite or not gif_path.exists() or not pdf_path.exists():
                fig = plt.figure()
                ax = plot.prepare_axis(fig, plot_mode)
                plot.traj(ax, plot_mode, gt_traj, '--', "black",
                          str(gt_traj_path.name[:-4]).replace("_", "-"))
                plot.traj_colormap(ax, est_plot_traj, rpe_rot_metric.error,
                                   plot_mode, min_map=rpe_rot_statistics["min"], max_map=rpe_rot_statistics["max"],
                                   title="Traj " + rpe_rot_title)
                if overwrite or not gif_path.exists():
                    rotanimate(ax, display_angles, str(gif_path),
                               delay=display_delay, height=8, width=8, prefix=str(log_root))

            if overwrite or not pdf_path.exists():
                multipage(str(pdf_path))

            plt.close('all')
        except evo.core.filters.FilterException:
            continue


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf', dpi=dpi)
    pp.close()


def main():
    parser = argparse.ArgumentParser(
        description='Run SAGE-SLAM and visualize and evaluate the results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--slam_exe_path', type=str, required=True,
                        help='path to SAGE-SLAM executable')
    parser.add_argument('--slam_config_path', type=str, required=True,
                        help='path to SAGE-SLAM config file')
    parser.add_argument('--input_data_root', type=str, required=True,
                        help='root of input data to SAGE-SLAM')
    parser.add_argument('--python_path', type=str, required=True,
                        help='path to python binary')
    parser.add_argument('--log_root', type=str, required=True,
                        help='root to SAGE-SLAM results')
    parser.add_argument('--depth_net_path', type=str, required=True,
                        help='path to depth network JIT model')
    parser.add_argument('--feat_net_path', type=str, required=True,
                        help='path to feature network JIT model')
    parser.add_argument('--eval_id_list', nargs='+', type=int,
                        default=None, help='subject ids for evaluation')
    parser.add_argument('--display_num_rotate', type=int, default=21,
                        help='number of rotates in trajectory GIF')
    parser.add_argument('--trunc_multiplier', type=float, default=20.0,
                        help='multipler for tsdf fusion')
    parser.add_argument('--rpe_delta', type=int, default=7,
                        help='frame interval between a pair of poses for RPE evaluation')
    args = parser.parse_args()

    display_delay = 1000 / args.display_num_rotate
    input_data_root = Path(args.input_data_root)
    fly_through_script_path = Path(
        __file__).parent / "generate_reconstruction_fly_through.py"

    log_root = Path(args.log_root)
    if not log_root.exists():
        log_root.mkdir()

    overwrite_slam = False
    overwrite_plot_1 = True
    overwrite_fly_through = False

    # standard setting
    disable_gui = True
    only_loop_smg_init = False
    no_loop_smg = False
    no_tracker_reproj = False
    no_mapping_reproj = True
    no_tracker_photo = False
    no_mapping_photo = False
    no_global_loop = False
    no_local_loop = False
    no_code_optim = False

    for eval_id in args.eval_id_list:
        hdf5_path_list = list(
            input_data_root.rglob(f"bag_{eval_id}/*/*fusion*.hdf5"))

        hdf5_path_list = sorted(hdf5_path_list)
        for hdf5_path in hdf5_path_list:
            log_root_sequence = log_root / \
                hdf5_path.parents[1].name / hdf5_path.parent.name

            print(f"Processing {str(log_root_sequence)}")
            est_traj_path_list = sorted(
                list(log_root_sequence.rglob("stamped_traj_estimate.txt")))

            if overwrite_slam or len(est_traj_path_list) == 0:
                if not log_root_sequence.exists():
                    log_root_sequence.mkdir(parents=True)
                print(f"Processing {str(hdf5_path)}...")
                exe_str = (
                    f"{str(args.slam_exe_path)} --flagfile=\"{str(args.slam_config_path)}\" --source_url=\"hdf5://{str(hdf5_path)}\" "
                    f"--run_log_dir=\"{str(log_root_sequence)}\" --quit_on_finish "
                    f"--depth_network_path=\"{args.depth_net_path}\" --feature_network_path=\"{args.feat_net_path}\" "
                    + (f"--use_reprojection=false " if no_mapping_reproj else " ")
                    + (f"--loop_use_match_geom=false " if no_loop_smg else " ")
                    + (f"--use_tracker_reprojection=false " if no_tracker_reproj else " ")
                    + (f"--use_tracker_photometric=false " if no_tracker_photo else " ")
                    + (f"--use_photometric=false " if no_mapping_photo else " ")
                    + (f"--loop_use_match_geom=true --tracker_match_geom_factor_weight=0.0 "
                       if only_loop_smg_init else " ")
                    + (f"--enable_gui=false " if disable_gui else " ")
                    + (f"--use_global_loop=false " if no_global_loop else " ")
                    + (f"--use_local_loop=false " if no_local_loop else " ")
                    + (f"--code_factor_weight=10000.0 " if no_code_optim else " ")
                    + ("--v=0 ")
                )

                print(f"Executed command {exe_str}")
                os.system(exe_str)

            est_traj_path_list = sorted(
                list(log_root_sequence.rglob("stamped_traj_estimate.txt")))
            print(f"Existing est traj list: {est_traj_path_list}")
            for est_traj_path in est_traj_path_list:
                evo_log_root = est_traj_path.parent / "evo_plots" / \
                    f"all_frames_to_align"
                gt_traj_path_list = sorted(
                    list(hdf5_path.parent.rglob("stamped_groundtruth.txt")))
                if len(gt_traj_path_list) != 0 and (overwrite_plot_1 or not evo_log_root.exists()):
                    shutil.copy(src=str(gt_traj_path_list[0]), dst=str(
                        est_traj_path.parent))
                    gt_traj_path = est_traj_path.parent / "stamped_groundtruth.txt"
                    our_traj_path = est_traj_path

                    if not evo_log_root.exists():
                        evo_log_root.mkdir(parents=True)
                    # Plot trajectory
                    plot_traj([our_traj_path], gt_traj_path, evo_log_root, -1,
                              display_num_rotate=args.display_num_rotate,
                              display_delay=display_delay, rpe_delta=args.rpe_delta, overwrite=overwrite_plot_1)

                else:
                    if len(gt_traj_path_list) == 0:
                        print("gt traj not exist")
                    if evo_log_root.exists():
                        print("previous plots exist")

            # Copy gt results to the slam result folder and run traj evaluation
            est_traj_path_list = sorted(
                list(log_root_sequence.rglob("stamped_traj_estimate.txt")))
            for est_traj_path in est_traj_path_list:
                if overwrite_fly_through or len(list(est_traj_path.parent.rglob("fused_mesh.avi"))) == 0:
                    code = os.system(
                        f"{args.python_path} {fly_through_script_path} --result_root \"{str(est_traj_path.parent)}\" --trunc_margin_multiplier {args.trunc_multiplier} --overwrite_video")
                    if code != 0:
                        print(
                            "fly-through video generation failed, error code: {}".format(code))


if __name__ == "__main__":
    main()
