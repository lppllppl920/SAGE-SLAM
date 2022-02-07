from pathlib import Path
import argparse
import numpy as np
from skimage import measure
import cv2
import tqdm
import zipfile

try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule

except Exception as err:
    print('Warning: %s' % (str(err)))
    print('Failed to import PyCUDA.')
    exit()

import trimesh
from autolab_core import RigidTransform
from perception import CameraIntrinsics
# NOTE: Need to installed the in-house version of this meshrender library to render point-wise textures
from meshrender import Scene, MaterialProperties, AmbientLight, PointLight, SceneObject, VirtualCamera


# Load 3D mesh model as a PyOpenGL Scene object
def load_3d_model(model_path):
    # Start with an empty scene
    scene = Scene()
    # Add objects to the scene
    # Begin by loading meshes
    pawn_mesh = trimesh.load_mesh(model_path)
    # Set up object's pose in the world
    pawn_pose = RigidTransform(
        rotation=np.eye(3),
        translation=np.array([0.0, 0.0, 0.0]),
        from_frame='obj',
        to_frame='world'
    )
    # Set up each object's material properties
    pawn_material = MaterialProperties(
        color=np.array([1.0, 1.0, 1.0]),
        k_a=1.0,
        k_d=1.0,
        k_s=0.0,
        alpha=1.0,
        smooth=False,
        wireframe=False
    )
    # Create SceneObjects for each object
    pawn_obj = SceneObject(pawn_mesh, pawn_pose, pawn_material)
    # Add the SceneObjects to the scene
    scene.add_object('pawn', pawn_obj)
    return scene, pawn_mesh


def get_depth_image_from_3d_model(scene, camera_intrinsics, image_height, image_width, camera_pose, z_near, z_far,
                                  point_light_strength, ambient_strength):
    # Add a point light source to the scene
    pointlight = PointLight(location=camera_pose[:3, 3], color=np.array([1.0, 1.0, 1.0]),
                            strength=point_light_strength)
    scene.add_light('point_light', pointlight)

    # Add lighting to the scene
    # Create an ambient light
    ambient = AmbientLight(
        color=np.array([1.0, 1.0, 1.0]),
        strength=ambient_strength
    )
    # Add the lights to the scene
    scene.ambient_light = ambient  # only one ambient light per scene

    # Add a camera to the scene
    # Set up camera intrinsics
    ci = CameraIntrinsics(
        frame='camera',
        fx=camera_intrinsics[0, 0],
        fy=camera_intrinsics[1, 1],
        cx=camera_intrinsics[0, 2],
        cy=camera_intrinsics[1, 2],
        skew=0.0,
        height=image_height,
        width=image_width
    )
    # Set up the camera pose (z axis faces away from scene, x to right, y up)
    cp = RigidTransform(
        rotation=camera_pose[:3, :3],
        translation=camera_pose[:3, 3],
        from_frame='camera',
        to_frame='world'
    )
    # Create a VirtualCamera
    camera = VirtualCamera(ci, cp, z_near=z_near, z_far=z_far)
    # Add the camera to the scene
    scene.camera = camera
    # Render raw numpy arrays containing color and depth
    color_image_raw, depth_image_raw = scene.render(
        render_color=True, front_and_back=True)
    return color_image_raw, depth_image_raw


def surface_mesh_global_scale(surface_mesh):
    max_bound = np.max(surface_mesh.vertices, axis=0)
    min_bound = np.min(surface_mesh.vertices, axis=0)

    return np.linalg.norm(max_bound - min_bound, ord=2), np.linalg.norm(min_bound, ord=2), np.abs(
        max_bound[2] - min_bound[0])


def display_depth_map(depth_map, min_value=None, max_value=None, colormode=cv2.COLORMAP_JET, scale=None):
    if (min_value is None or max_value is None) and scale is None:
        if len(depth_map[depth_map > 0]) > 0:
            min_value = np.min(depth_map[depth_map > 0])
        else:
            min_value = 0.0
    elif scale is not None:
        min_value = 0.0
        max_value = scale
    else:
        pass

    depth_map_visualize = np.abs(
        (depth_map - min_value) / (max_value - min_value + 1.0e-8) * 255)
    depth_map_visualize[depth_map_visualize > 255] = 255
    depth_map_visualize[depth_map_visualize <= 0.0] = 0
    depth_map_visualize = cv2.applyColorMap(
        np.uint8(depth_map_visualize), colormode)
    return depth_map_visualize


class TSDFVolume(object):
    def __init__(self, vol_bnds, voxel_size, trunc_margin):
        # Define voxel volume parameters
        self._vol_bnds = vol_bnds  # rows: x,y,z columns: min,max in world coordinates in meters
        self._voxel_size = voxel_size
        self._trunc_margin = trunc_margin

        # Adjust volume bounds
        self._vol_dim = np.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size).copy(
            order='C').astype(int)  # ensure C-order contiguous
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + \
            self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(
            order='C').astype(np.float32)  # ensure C-order contiguous
        print("Voxel volume size: %d x %d x %d" %
              (self._vol_dim[0], self._vol_dim[1], self._vol_dim[2]))

        # Initialize pointers to voxel volume in CPU memory
        # Assign oversized tsdf volume
        self._tsdf_vol_cpu = np.zeros(
            self._vol_dim).astype(np.float32)  # -2.0 *
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(
            np.float32)
        self._uncertainty_vol_cpu = np.zeros(self._vol_dim).astype(
            np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        # Copy voxel volumes to GPU
        self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
        cuda.memcpy_htod(self._tsdf_vol_gpu, self._tsdf_vol_cpu)
        self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
        cuda.memcpy_htod(self._weight_vol_gpu, self._weight_vol_cpu)
        self._uncertainty_vol_gpu = cuda.mem_alloc(
            self._uncertainty_vol_cpu.nbytes)
        cuda.memcpy_htod(self._uncertainty_vol_gpu, self._uncertainty_vol_cpu)
        self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
        cuda.memcpy_htod(self._color_vol_gpu, self._color_vol_cpu)

        # Cuda kernel function (C++)
        self._cuda_src_mod_with_confidence_map = SourceModule("""
          __global__ void integrate(float * tsdf_vol,
                                    float * weight_vol,
                                    float * uncertainty_vol,
                                    float * color_vol,
                                    float * vol_dim,
                                    float * vol_origin,
                                    float * cam_intr,
                                    float * cam_pose,
                                    float * other_params,
                                    float * color_im,
                                    float * depth_im,
                                    float * std_im) {

            // Get voxel index
            int gpu_loop_idx = (int) other_params[0];
            int max_threads_per_block = blockDim.x;
            int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
            int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;

            int vol_dim_x = (int) vol_dim[0];
            int vol_dim_y = (int) vol_dim[1];
            int vol_dim_z = (int) vol_dim[2];

            if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
                return;

            // Get voxel grid coordinates (note: be careful when casting)
            float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
            float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
            float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);

            // Voxel grid coordinates to world coordinates
            float voxel_size = other_params[1];
            float pt_x = vol_origin[0]+voxel_x*voxel_size;
            float pt_y = vol_origin[1]+voxel_y*voxel_size;
            float pt_z = vol_origin[2]+voxel_z*voxel_size;

            // World coordinates to camera coordinates
            float tmp_pt_x = pt_x-cam_pose[0*4+3];
            float tmp_pt_y = pt_y-cam_pose[1*4+3];
            float tmp_pt_z = pt_z-cam_pose[2*4+3];
            float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
            float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
            float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;

            // Because of the long tube of endoscope, the minimum depth to consider is not zero
            float min_depth = other_params[6];
            if (cam_pt_z < min_depth) {
                return;
            }

            // Camera coordinates to image pixels
            int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
            int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);

            // Skip if outside view frustum
            int im_h = (int) other_params[2];
            int im_w = (int) other_params[3];
            if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h)
                return;

            // Skip invalid depth
            float depth_value = depth_im[pixel_y*im_w+pixel_x];
            float std_value = std_im[pixel_y*im_w + pixel_x];
            if (depth_value <= 0 || std_value <= 0) {
                return;
            }

            // Get std value for the current observation
            float trunc_margin = other_params[4];
            float depth_diff = depth_value - cam_pt_z;
            if (depth_diff < -trunc_margin)
                return;

            float dist = fmin(1.0f, depth_diff / std_value);

            float w_old = weight_vol[voxel_idx];
            float obs_weight = other_params[5];
            float w_new = w_old + obs_weight;
            tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx] * w_old + dist * obs_weight) / w_new;
            weight_vol[voxel_idx] = w_new;


            // Integrate color
            float new_color = color_im[pixel_y * im_w + pixel_x];
            float new_b = floorf(new_color / (256 * 256));
            float new_g = floorf((new_color - new_b * 256 * 256) / 256);
            float new_r = new_color - new_b * 256 * 256 - new_g * 256;

            float old_color = color_vol[voxel_idx];
            float old_b = floorf(old_color / (256 * 256));
            float old_g = floorf((old_color - old_b * 256 * 256) / 256);
            float old_r = old_color - old_b * 256 * 256 - old_g * 256;

            new_b = fmin(roundf((old_b * w_old + new_b * obs_weight) / w_new), 255.0f);
            new_g = fmin(roundf((old_g * w_old + new_g * obs_weight) / w_new), 255.0f);
            new_r = fmin(roundf((old_r * w_old + new_r * obs_weight) / w_new), 255.0f);

            color_vol[voxel_idx] = new_b * 256 * 256 + new_g * 256 + new_r;
          }""")

        self._cuda_integrate = self._cuda_src_mod_with_confidence_map.get_function(
            "integrate")
        # Determine block/grid size on GPU
        gpu_dev = cuda.Device(0)
        self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
        n_blocks = int(np.ceil(float(np.prod(self._vol_dim)) /
                       float(self._max_gpu_threads_per_block)))
        grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X,
                         int(np.floor(np.cbrt(n_blocks))))
        grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y, int(
            np.floor(np.sqrt(n_blocks / grid_dim_x))))
        grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z, int(
            np.ceil(float(n_blocks) / float(grid_dim_x * grid_dim_y))))
        self._max_gpu_grid_dim = np.array(
            [grid_dim_x, grid_dim_y, grid_dim_z]).astype(int)
        # _n_gpu_loops specifies how many loops for the GPU to process the entire volume
        self._n_gpu_loops = int(np.ceil(float(np.prod(self._vol_dim)) / float(
            np.prod(self._max_gpu_grid_dim) * self._max_gpu_threads_per_block)))

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, min_depth, std_im, obs_weight=1.):
        im_h = depth_im.shape[0]
        im_w = depth_im.shape[1]

        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(
            color_im[:, :, 2] * 256 * 256 + color_im[:, :, 1] * 256 + color_im[:, :, 0])

        # integrate voxel volume (calls CUDA kernel)
        for gpu_loop_idx in range(self._n_gpu_loops):
            self._cuda_integrate(self._tsdf_vol_gpu,
                                 self._weight_vol_gpu,
                                 self._uncertainty_vol_gpu,
                                 self._color_vol_gpu,
                                 cuda.InOut(self._vol_dim.astype(np.float32)),
                                 cuda.InOut(
                                     self._vol_origin.astype(np.float32)),
                                 cuda.InOut(
                                     cam_intr.reshape(-1).astype(np.float32)),
                                 cuda.InOut(
                                     cam_pose.reshape(-1).astype(np.float32)),
                                 cuda.InOut(np.asarray(
                                     [gpu_loop_idx, self._voxel_size, im_h, im_w, self._trunc_margin,
                                      obs_weight, min_depth],
                                     np.float32)),
                                 cuda.InOut(
                                     color_im.reshape(-1).astype(np.float32)),
                                 cuda.InOut(
                                     depth_im.reshape(-1).astype(np.float32)),
                                 cuda.InOut(
                                     std_im.reshape(-1).astype(np.float32)),
                                 block=(self._max_gpu_threads_per_block, 1, 1), grid=(
                                     int(self._max_gpu_grid_dim[0]), int(
                                         self._max_gpu_grid_dim[1]),
                                     int(self._max_gpu_grid_dim[2])))

    # Copy voxel volume to CPU
    def get_volume(self):
        cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
        cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
        cuda.memcpy_dtoh(self._weight_vol_cpu, self._weight_vol_gpu)
        return self._tsdf_vol_cpu, self._color_vol_cpu, self._weight_vol_cpu

    # Get mesh of voxel volume via marching cubes
    def get_mesh(self, only_visited=False):
        tsdf_vol, color_vol, weight_vol = self.get_volume()

        verts, faces, norms, vals = measure.marching_cubes_lewiner(
            tsdf_vol, level=0, gradient_direction='ascent')

        verts_ind = np.round(verts).astype(int)
        # voxel grid coordinates to world coordinates
        verts = verts * self._voxel_size + self._vol_origin

        # Get vertex colors
        std_vals = weight_vol[verts_ind[:, 0],
                              verts_ind[:, 1], verts_ind[:, 2]]
        std_vals = np.uint8(std_vals / np.max(std_vals) * 255)
        std_colors = std_vals.astype(np.uint8).reshape(-1, 1)
        std_colors = cv2.cvtColor(cv2.applyColorMap(
            std_colors, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB).reshape(-1, 3)

        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / (256 * 256))
        colors_g = np.floor((rgb_vals - colors_b * 256 * 256) / 256)
        colors_r = rgb_vals - colors_b * 256 * 256 - colors_g * 256
        colors = np.transpose(
            np.uint8(np.floor(np.asarray([colors_r, colors_g, colors_b])))).reshape(-1, 3)

        if only_visited:
            verts_indxes = verts_ind[:, 0] * weight_vol.shape[1] * weight_vol.shape[2] + \
                verts_ind[:, 1] * weight_vol.shape[2] + verts_ind[:, 2]
            weight_vol = weight_vol.reshape((-1))
            valid_vert_indexes = np.nonzero(weight_vol[verts_indxes] >= 1)[0]
            valid_vert_indexes = set(valid_vert_indexes)

            indicators = np.array([face in valid_vert_indexes for face in faces[:, 0]]) \
                & np.array([face in valid_vert_indexes for face in faces[:, 1]]) \
                & np.array([face in valid_vert_indexes for face in faces[:, 2]])

            return verts, faces[indicators], norms, colors, std_colors

        return verts, faces, norms, colors, std_colors


# Get corners of 3D camera view frustum of depth image
def get_view_frustum(depth_im, cam_intr, cam_pose):
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array([(np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2]) * np.array(
        [0, max_depth, max_depth, max_depth, max_depth]) / cam_intr[0, 0],
        (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2]) * np.array(
        [0, max_depth, max_depth, max_depth, max_depth]) / cam_intr[1, 1],
        np.array([0, max_depth, max_depth, max_depth, max_depth])])
    view_frust_pts = np.dot(cam_pose[:3, :3], view_frust_pts) + np.tile(cam_pose[:3, 3].reshape(3, 1), (
        1, view_frust_pts.shape[1]))  # from camera to world coordinates
    return view_frust_pts


# Save 3D mesh to a polygon .ply file
def meshwrite(filename, verts, faces, norms, colors):
    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
            verts[i, 0], verts[i, 1], verts[i, 2], norms[i, 0], norms[i,
                                                                      1], norms[i, 2], colors[i, 0], colors[i, 1],
            colors[i, 2]))

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" %
                       (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()


def quaternion_matrix(quaternion):
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < np.finfo(float).eps * 4.0:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def read_cam_trajectory(file_path):
    cam_pose_dict = dict()
    frame_idx_list = list()
    # trajectory.txt format: frame-idx x y z qx qy qz qw
    with open(str(file_path), "r") as fp:
        while True:
            line = fp.readline()
            if line is None:
                break
            words = line.split(sep=" ")
            if len(words) <= 1:
                break

            pose = quaternion_matrix([float(words[7]), float(
                words[4]), float(words[5]), float(words[6])])
            pose[0][3] = float(words[1])
            pose[1][3] = float(words[2])
            pose[2][3] = float(words[3])

            idx = int(float(words[0]))
            cam_pose_dict[idx] = pose
            frame_idx_list.append(idx)

    return cam_pose_dict, frame_idx_list


def read_cam_intrinsics(file_path):
    intrinsics = np.zeros((3, 3), dtype=np.float32)
    with open(str(file_path), "r") as fp:
        line = fp.readline()
        words = line.split(sep=" ")
        intrinsics[0][0] = float(words[0])
        intrinsics[1][1] = float(words[1])
        intrinsics[0][2] = float(words[2])
        intrinsics[1][2] = float(words[3])
        intrinsics[2][2] = 1.0
    return intrinsics


def main():
    parser = argparse.ArgumentParser(
        description='Read SAGE-SLAM results and generate reconstruction and fly-through visualization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--result_root', type=str, required=True,
                        help='root storing the sequence result')
    parser.add_argument('--visualize_fused_model', action='store_true',
                        help='whether or not to visualize fused 3d model')
    parser.add_argument('--max_voxel_count', type=float, default=400.0 ** 3,
                        help='maximum count of voxels for depth map fusion')
    parser.add_argument('--trunc_margin_multiplier', type=float, default=5.0,
                        help='truncate margin factor of the signed distance function')
    parser.add_argument('--overwrite_video', action='store_true',
                        help='whether or not to overwrite previous results')
    args = parser.parse_args()

    overall_result_root = Path(args.result_root)

    # Read camera trajectories
    trajectory_path_list = sorted(
        list(overall_result_root.rglob("stamped_traj_estimate.txt")))

    for trajectory_path in trajectory_path_list:
        result_root = trajectory_path.parent

        if not args.overwrite_video:
            if (result_root / "fly_through.avi").exists():
                continue

        print(f"Processing {str(result_root)}...")
        intrinsics_path = result_root / "keyframes" / "intrinsics.txt"
        cam_pose_dict, frame_idx_list = read_cam_trajectory(trajectory_path)
        cam_intr = read_cam_intrinsics(intrinsics_path)

        print("Estimating voxel volume bounds...")
        vol_bnds = np.zeros((3, 2))
        n_imgs = len(frame_idx_list)
        depth_map_list = list()

        color_img = cv2.imread(
            str(result_root / "keyframes" / "rgb_{:d}_{:d}.png".format(1, frame_idx_list[0])))
        fine_height, fine_width = color_img.shape[:2]

        depth_map_path = result_root / "keyframes" / \
            "dpt_{:d}_{:d}.pt".format(1, frame_idx_list[0])
        archive = zipfile.ZipFile(str(depth_map_path), "r")
        depth_map = np.frombuffer(archive.read("dpt_{:d}_{:d}/data/0".format(1, frame_idx_list[0])),
                                  dtype=np.float32)

        scale = np.sqrt(float(color_img.size) / float(3 * depth_map.size))

        coarse_height = int(fine_height / scale)
        coarse_width = int(fine_width / scale)

        depth_map_path = sorted(
            list((result_root / "keyframes").rglob("dpt_*.pt")))[0]
        archive = zipfile.ZipFile(str(depth_map_path), "r")

        name = depth_map_path.name[:-3]
        depth_map = np.frombuffer(archive.read(name + "/data/0"),
                                  dtype=np.float32).reshape((coarse_height, coarse_width, 1))

        mask = (depth_map > 0.0).astype(np.float32)
        color_mask = cv2.resize(mask, dsize=(
            fine_width, fine_height), interpolation=cv2.INTER_NEAREST)
        color_mask = color_mask.reshape(
            (fine_height, fine_width, 1)).astype(np.uint8)

        for i in range(n_imgs):
            depth_map_path = result_root / "keyframes" / \
                "dpt_{:d}_{:d}.pt".format(i + 1, frame_idx_list[i])
            archive = zipfile.ZipFile(str(depth_map_path), "r")
            depth_map = np.frombuffer(archive.read("dpt_{:d}_{:d}/data/0".format(i + 1, frame_idx_list[i])),
                                      dtype=np.float32).reshape((coarse_height, coarse_width, 1))

            depth_map = depth_map * \
                mask.reshape((coarse_height, coarse_width, 1))
            resized_depth_map = \
                cv2.resize(depth_map, dsize=(fine_width, fine_height),
                           interpolation=cv2.INTER_LINEAR)

            depth_map_list.append(resized_depth_map)

            # Compute camera view frustum and extend convex hull
            view_frust_pts = get_view_frustum(
                depth_map, cam_intr, cam_pose_dict[frame_idx_list[i]])
            vol_bnds[:, 0] = np.minimum(
                vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:, 1] = np.maximum(
                vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

        voxel_size = 0.1
        vol_dim = (vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size
        # Adaptively change the size of one voxel to fit into the GPU memory
        volume = vol_dim[0] * vol_dim[1] * vol_dim[2]
        factor = (volume / args.max_voxel_count) ** (1.0 / 3.0)
        voxel_size *= factor
        print("voxel size: {}".format(voxel_size))

        # Initialize voxel volume
        print("Initializing voxel volume...")
        tsdf_vol = TSDFVolume(vol_bnds, voxel_size=voxel_size,
                              trunc_margin=voxel_size * args.trunc_margin_multiplier)

        # Loop through images and fuse them together
        print("Integrating depth images...")
        tq = tqdm.tqdm(total=n_imgs)
        tq.set_description('Depth fusion')

        color_image_list = list()

        resized_cam_intr = scale * cam_intr
        resized_cam_intr[2][2] = 1.0

        for i in range(n_imgs):
            color_img_path = result_root / "keyframes" / \
                "rgb_{:d}_{:d}.png".format(i + 1, frame_idx_list[i])
            color_img = cv2.imread(str(color_img_path))
            color_image_list.append(color_img)

            resized_depth_map = depth_map_list[i]
            cam_pose = cam_pose_dict[frame_idx_list[i]]

            valid_mask = (resized_depth_map > 0.0).astype(np.float32)
            std_map = np.ones_like(resized_depth_map) * valid_mask

            # Integrate observation into voxel volume (assume color aligned with depth)
            tsdf_vol.integrate(color_img, resized_depth_map, resized_cam_intr, cam_pose,
                               min_depth=1.0e-4, std_im=std_map, obs_weight=1.)
            tq.update(1)

        tq.close()
        verts, faces, norms, colors, _ = tsdf_vol.get_mesh(only_visited=True)
        fused_model_path = str(result_root / "fused_mesh.ply")
        print("Writing mesh model...")
        meshwrite(fused_model_path, verts, faces, -norms, colors)
        print("Loading scene model...")
        scene, surface_mesh = load_3d_model(fused_model_path)

        mesh_global_scale, _, _ = surface_mesh_global_scale(
            surface_mesh)
        print("Mesh global scale: {}...".format(mesh_global_scale))

        render_color_image_list = []
        render_depth_image_list = []
        print("Fly-through rendering...")
        tq = tqdm.tqdm(total=n_imgs)
        tq.set_description('Rendering')
        for i in range(n_imgs):
            # 4x4 rigid transformation matrix T^(world)_(camera)
            cam_pose = cam_pose_dict[frame_idx_list[i]]
            render_color_image, render_depth_image = \
                get_depth_image_from_3d_model(scene, resized_cam_intr,
                                              fine_height, fine_width, cam_pose,
                                              z_near=1.0e-1,
                                              z_far=mesh_global_scale,
                                              point_light_strength=3.0 * voxel_size,
                                              ambient_strength=1.0)

            render_color_image = cv2.cvtColor(
                render_color_image, cv2.COLOR_RGBA2RGB)
            render_color_image_list.append(
                render_color_image.reshape((fine_height, fine_width, 3)).astype(np.uint8))

            render_depth_image_list.append(render_depth_image)
            tq.update(1)
        tq.close()
        max_depth = np.max(np.asarray(render_depth_image_list))
        GIF_image_list = []

        ratio = 256 / color_mask.shape[0]
        for i, render_depth_image in enumerate(render_depth_image_list):
            display_render_depth_image = display_depth_map(depth_map=render_depth_image.
                                                           reshape(
                                                               (fine_height, fine_width)),
                                                           colormode=cv2.COLORMAP_JET, scale=max_depth)
            display_render_depth_image = cv2.cvtColor(
                display_render_depth_image, cv2.COLOR_BGR2RGB)

            display_pred_depth_image = display_depth_map(depth_map=depth_map_list[i],
                                                         colormode=cv2.COLORMAP_JET, scale=max_depth)
            display_pred_depth_image = cv2.cvtColor(
                display_pred_depth_image, cv2.COLOR_BGR2RGB)

            GIF_image_list.append(cv2.resize(cv2.hconcat(
                [color_image_list[i] * color_mask, render_color_image_list[i] * color_mask,
                 display_render_depth_image * color_mask,
                 display_pred_depth_image * color_mask]), dsize=(0, 0), fx=ratio, fy=ratio))

            if args.visualize_fused_model:
                cv2.imshow("video_fly_through", GIF_image_list[i])
                cv2.waitKey(1)

        print("Writing fly-through video of fused mesh...")
        result_video_fp = cv2.VideoWriter(
            str(result_root / "fly_through.avi"),
            cv2.VideoWriter_fourcc(*'DIVX'), 5,
            (GIF_image_list[0].shape[1], GIF_image_list[0].shape[0]))
        for i in range(len(GIF_image_list)):
            result_video_fp.write(GIF_image_list[i])
        result_video_fp.release()
        if args.visualize_fused_model:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
