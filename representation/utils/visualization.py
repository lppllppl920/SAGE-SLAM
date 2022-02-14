from torch.autograd import Variable
from graphviz import Digraph
from collections import namedtuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import umap


@torch.no_grad()
def draw_flow_map(flows, max_v=None):
    _, channel, height, width = flows.shape
    flows_display = flows.reshape(2, height, width).cpu().numpy()
    flows_display = np.moveaxis(flows_display, source=[
                                0, 1, 2], destination=[2, 0, 1])
    fx, fy = flows_display[:, :, 0], flows_display[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((height, width, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    if max_v is None:
        hsv[..., 2] = np.uint8(np.minimum(v / np.max(v), 1.0) * 255)
    else:
        hsv[..., 2] = np.uint8(np.minimum(v / max_v, 1.0) * 255)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), np.max(v)


@torch.no_grad()
def write_video(result_list, log_root, step):
    result_video_fp = cv2.VideoWriter(
        str(log_root / f"result_{step}.avi"),
        cv2.VideoWriter_fourcc(*'DIVX'), 5,
        (result_list[0].shape[1], result_list[0].shape[0]))
    for i in range(len(result_list)):
        result_video_fp.write(cv2.cvtColor(result_list[i], cv2.COLOR_BGR2RGB))
    result_video_fp.release()

    return


@torch.no_grad()
def stack_and_display(phase, title, step, writer, image_list):
    writer.add_image(phase + '/Images/' + title,
                     np.hstack(image_list), step, dataformats='HWC')
    return


@torch.no_grad()
def display_color_depth(colors, gt_depths, pred_depths):
    colors_display = vutils.make_grid(colors, normalize=False)
    colors_display = np.moveaxis(colors_display.data.cpu().numpy(),
                                 source=[0, 1, 2], destination=[2, 0, 1])

    gt_depths_display = vutils.make_grid(
        gt_depths, normalize=True, scale_each=True)
    gt_depths_display = cv2.applyColorMap(np.uint8(255 * np.moveaxis(gt_depths_display.data.cpu().numpy(),
                                                                     source=[
                                                                         0, 1, 2],
                                                                     destination=[2, 0, 1])), cv2.COLORMAP_JET)
    gt_depths_display = cv2.cvtColor(gt_depths_display, cv2.COLOR_BGR2RGB)
    gt_depths_display = gt_depths_display.astype(np.float32) / 255.0

    pred_depth_list = list()
    for pred_depth in pred_depths:
        pred_depth_display = vutils.make_grid(
            pred_depth, normalize=True, scale_each=True)
        pred_depth_display = cv2.applyColorMap(np.uint8(255 * np.moveaxis(pred_depth_display.data.cpu().numpy(),
                                                                          source=[
                                                                              0, 1, 2],
                                                                          destination=[2, 0, 1])), cv2.COLORMAP_JET)
        pred_depth_display = cv2.cvtColor(
            pred_depth_display, cv2.COLOR_BGR2RGB)
        pred_depth_list.append(pred_depth_display.astype(
            np.float32) / 255.0)

    return (colors_display, gt_depths_display, *pred_depth_list)


@torch.no_grad()
def display_depth_list(depth_list, height=None, width=None, fx=3, fy=3):
    depths = torch.cat(depth_list, dim=0)
    depths = F.interpolate(depths, size=(height, width),
                           mode='bilinear', align_corners=False)
    min_depth = torch.min(depths)
    max_depth = torch.max(depths)
    depths_display = vutils.make_grid(depths, normalize=True, scale_each=False,
                                      range=(min_depth.item(), max_depth.item()))
    depths_display = cv2.applyColorMap(np.uint8(255 * np.moveaxis(depths_display.data.cpu().numpy(),
                                                                  source=[
                                                                      0, 1, 2],
                                                                  destination=[2, 0, 1])), cv2.COLORMAP_JET)
    if height is None or width is None:
        depths_display = cv2.resize(depths_display, dsize=(0, 0), fx=fx, fy=fy)

    depths_display = cv2.cvtColor(depths_display, cv2.COLOR_BGR2RGB)
    return depths_display


@torch.no_grad()
def display_basis_list(basis_list, mask, height=None, width=None, fx=3, fy=3):
    basis_display_list = list()
    for basis in basis_list:
        basis = F.interpolate(basis, size=(height, width),
                              mode='bilinear', align_corners=False)
        if mask.shape[2] != height or mask.shape[3] != width:
            mask = F.interpolate(mask, size=(height, width), mode='nearest')

        basis = (mask * basis).permute(1, 0, 2, 3)

        depths_display = vutils.make_grid(
            basis, normalize=True, scale_each=True)
        depths_display = cv2.applyColorMap(np.uint8(255 * np.moveaxis(depths_display.data.cpu().numpy(),
                                                                      source=[
                                                                          0, 1, 2],
                                                                      destination=[2, 0, 1])), cv2.COLORMAP_JET)
        if height is None or width is None:
            depths_display = cv2.resize(
                depths_display, dsize=(0, 0), fx=fx, fy=fy)

        basis_display_list.append(cv2.cvtColor(
            depths_display, cv2.COLOR_BGR2RGB))
    return basis_display_list


@torch.no_grad()
def display_desc_heatmap(src_keypoint_2d_hw_location,
                         tgt_gt_2d_hw_location, tgt_desc_response_map,
                         src_desc_response_map,
                         valid_mask, src_input_image, tgt_input_image, sigma):
    _, _, height, width = tgt_desc_response_map.shape
    tgt_desc_response_map = valid_mask * \
        tgt_desc_response_map.view(1, 1, height, width)
    max_response = torch.max(tgt_desc_response_map)
    tgt_desc_response_map = tgt_desc_response_map / max_response

    src_desc_response_map = valid_mask * \
        src_desc_response_map.view(1, 1, height, width)
    max_response = torch.max(src_desc_response_map)
    src_desc_response_map = src_desc_response_map / max_response

    src_keypoint_2d_hw_location = src_keypoint_2d_hw_location.reshape(
        1, 2).data.cpu().numpy()
    tgt_gt_2d_hw_location = tgt_gt_2d_hw_location.reshape(
        1, 2).data.cpu().numpy()

    src_keypoint_map = generate_heatmap_from_location(
        src_keypoint_2d_hw_location, height, width, sigma=sigma)
    tgt_gt_response_map = generate_heatmap_from_location(
        tgt_gt_2d_hw_location, height, width, sigma=sigma)

    src_input_image = src_input_image.reshape(
        3, height, width).permute(1, 2, 0).data.cpu().numpy()
    src_keypoint_overlay = np.clip(
        src_input_image - src_keypoint_map, a_min=0.0, a_max=1.0)
    src_keypoint_overlay = (255 * src_keypoint_overlay).astype(np.uint8)

    src_desc_response_map = src_desc_response_map.reshape(
        height, width, 1).data.cpu().numpy()
    src_response_overlay = np.clip(
        src_input_image - src_desc_response_map, a_min=0.0, a_max=1.0)
    src_response_overlay = (255 * src_response_overlay).astype(np.uint8)

    tgt_input_image = tgt_input_image.reshape(
        3, height, width).permute(1, 2, 0).data.cpu().numpy()
    tgt_desc_response_map = tgt_desc_response_map.reshape(
        height, width, 1).data.cpu().numpy()
    tgt_response_overlay = np.clip(
        tgt_input_image - tgt_desc_response_map, a_min=0.0, a_max=1.0)
    tgt_response_overlay = (255 * tgt_response_overlay).astype(np.uint8)

    tgt_gt_response_overlay = np.clip(
        tgt_input_image - tgt_gt_response_map, a_min=0.0, a_max=1.0)
    tgt_gt_response_overlay = (255 * tgt_gt_response_overlay).astype(np.uint8)

    tgt_desc_response_map = (255 * tgt_desc_response_map).astype(np.uint8)
    tgt_desc_response_map = cv2.applyColorMap(
        tgt_desc_response_map, colormap=cv2.COLORMAP_HOT)

    src_desc_response_map = (255 * src_desc_response_map).astype(np.uint8)
    src_desc_response_map = cv2.applyColorMap(
        src_desc_response_map, colormap=cv2.COLORMAP_HOT)

    row0 = cv2.hconcat([src_keypoint_overlay, tgt_gt_response_overlay])
    row1 = cv2.hconcat([tgt_response_overlay, tgt_desc_response_map])
    row2 = cv2.hconcat([src_response_overlay, src_desc_response_map])

    ratio = int(256 / row0.shape[0])
    result = cv2.cvtColor(cv2.vconcat([row0, row1, row2]), cv2.COLOR_BGR2RGB)
    result = cv2.resize(result, dsize=(0, 0), fx=ratio, fy=ratio)
    return result


@torch.no_grad()
def display_no_match_heatmap(src_no_match_2d_hw_location, tgt_desc_response_map,
                             valid_mask, src_input_image, tgt_input_image, sigma):
    _, _, height, width = tgt_desc_response_map.shape
    tgt_desc_response_map = valid_mask * \
        tgt_desc_response_map.view(1, 1, height, width)
    max_response = torch.max(tgt_desc_response_map)
    tgt_desc_response_map = tgt_desc_response_map / max_response

    src_no_match_2d_hw_location = src_no_match_2d_hw_location.reshape(
        1, 2).data.cpu().numpy()
    src_keypoint_map = generate_heatmap_from_location(
        src_no_match_2d_hw_location, height, width, sigma=sigma)

    src_input_image = src_input_image.reshape(
        3, height, width).permute(1, 2, 0).data.cpu().numpy()
    src_keypoint_overlay = np.clip(
        src_input_image - src_keypoint_map, a_min=0.0, a_max=1.0)
    src_keypoint_overlay = (255 * src_keypoint_overlay).astype(np.uint8)

    tgt_input_image = tgt_input_image.reshape(
        3, height, width).permute(1, 2, 0).data.cpu().numpy()
    tgt_desc_response_map = tgt_desc_response_map.reshape(
        height, width, 1).data.cpu().numpy()

    tgt_response_overlay = np.clip(
        tgt_input_image - tgt_desc_response_map, a_min=0.0, a_max=1.0)
    tgt_response_overlay = (255 * tgt_response_overlay).astype(np.uint8)

    tgt_desc_response_map = (255 * tgt_desc_response_map).astype(np.uint8)
    tgt_desc_response_map = cv2.applyColorMap(
        tgt_desc_response_map, colormap=cv2.COLORMAP_HOT)

    ratio = int(256 / src_keypoint_overlay.shape[0])
    result = cv2.cvtColor(cv2.hconcat([src_keypoint_overlay, tgt_response_overlay, tgt_desc_response_map]),
                          cv2.COLOR_BGR2RGB)
    result = cv2.resize(result, dsize=(0, 0), fx=ratio, fy=ratio)

    return result


@torch.no_grad()
def generate_heatmap_from_location(keypoint_2d_hw_location, height, width, sigma):
    sigma_2 = sigma ** 2
    # for i in range(sample_size):
    x = keypoint_2d_hw_location[0, 1]
    y = keypoint_2d_hw_location[0, 0]

    y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(
        width), sparse=False, indexing='ij')

    source_grid_x = x_grid - x
    source_grid_y = y_grid - y

    heatmap = np.exp(-(source_grid_x ** 2 +
                     source_grid_y ** 2) / (2.0 * sigma_2))
    heatmap = np.asarray(heatmap, dtype=np.float32).reshape((height, width, 1))

    return heatmap


def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)


@torch.no_grad()
def visualize_feature_maps(src_feature_map, mask=None, spatial_size=(128 * 4, 160 * 4)):
    fit = umap.UMAP(
        n_neighbors=20,
        min_dist=0.01,
        n_components=3,
        metric='euclidean'
    )
    _, feat_channel, height, width = src_feature_map.shape
    src_features = src_feature_map.reshape(feat_channel, height * width)
    # 2*H*W x C_feat
    feats = torch.cat([src_features], dim=1).permute(
        1, 0).detach().cpu().numpy().astype(np.float32)

    color_coded_feats = fit.fit_transform(feats)
    color_max = np.amax(color_coded_feats, axis=0, keepdims=True)
    color_min = np.amin(color_coded_feats, axis=0, keepdims=True)
    color_coded_feats = (color_coded_feats - color_min) / \
        (color_max - color_min)
    color_coded_feats = color_coded_feats.astype(np.float64)

    src_color_coded_feats = color_coded_feats[:height * width]

    # H x W x 3
    src_color_coded_feats = src_color_coded_feats.reshape(height, width, 3)

    src_color_coded_feats = cv2.resize(src_color_coded_feats, dsize=(spatial_size[1], spatial_size[0]),
                                       interpolation=cv2.INTER_NEAREST)

    if mask is not None:
        m_height, m_width = mask.shape[2:]
        mask = mask.data.cpu().numpy().reshape((m_height, m_width, 1))
        mask = cv2.resize(mask, dsize=(
            spatial_size[1], spatial_size[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("feature1", src_color_coded_feats *
                   mask.reshape(*spatial_size, 1))
    cv2.imshow("feature2", src_color_coded_feats)
    cv2.waitKey()
    return


fn_dict = {}


def register_hooks_fn_grad(var):
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = ([torch.max(x) if x is not None else None for x in grad_input],
                           [torch.min(
                               x) if x is not None else None for x in grad_input],
                           [torch.max(
                               x) if x is not None else None for x in grad_output],
                           [torch.min(x) if x is not None else None for x in grad_output])

        fn.register_hook(register_grad)

    iter_graph(var.grad_fn, hook_cb)


def register_hooks(var):
    fn_dict = {}

    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = (grad_input, grad_output)

        fn.register_hook(register_grad)

    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output, bad_grad):
        if grad_output is None:
            return False

        if isinstance(grad_output, tuple):
            if len(grad_output) == 2:
                _, grad_out = grad_output

                if grad_out is not None:
                    cond2 = grad_out.isnan().any() or (grad_out.abs() >= bad_grad).any()
                else:
                    cond2 = False
                cond = cond2  # cond1 or
            else:
                grad_output = grad_output[0]
                if grad_output is not None:
                    cond = grad_output.isnan().any() or (grad_output.abs() >= bad_grad).any()
                else:
                    cond = False
        else:
            cond = grad_output.isnan().any() or (grad_output.abs() >= bad_grad).any()
        return cond

    def make_dot():
        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='12',
                         ranksep='0.1',
                         height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '(' + (', ').join(map(str, size)) + ')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                fillcolor = 'lightblue'
                if u.grad is not None and is_bad_grad(u.grad, 1.0e3):
                    fillcolor = 'red'
                elif u.grad is not None and is_bad_grad(u.grad, 1.0e2):
                    fillcolor = 'yellow'

                dot.node(str(id(u)), node_name +
                         f"_{torch.max(u.grad)}_{torch.min(u.grad)}", fillcolor=fillcolor)
            else:
                if fn in fn_dict:
                    fillcolor = 'white'
                    try:
                        if any(is_bad_grad(gi, 1.0e3) for gi in fn_dict[fn]):
                            fillcolor = 'red'
                        elif any(is_bad_grad(gi, 1.0e2) for gi in fn_dict[fn]):
                            fillcolor = 'yellow'

                    except RuntimeError as err:
                        fillcolor = 'green'

                    name_list = list()
                    for gi in fn_dict[fn]:
                        if gi is not None and gi[-1] is not None:
                            try:
                                name_list.append(
                                    f"{torch.max(gi[-1])}_{torch.min(gi[-1])}")
                            except RuntimeError as err:
                                name_list.append("")
                        else:
                            name_list.append("")
                    name = "_" + "_".join(name_list)

                    dot.node(str(id(fn)), str(type(fn).__name__) + name,
                             fillcolor=fillcolor)

            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))

        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot


Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))


def make_grad_dot(var, fn_dict, params=None, bad_grad=1.0e2):
    """ Produces Graphviz representation of PyTorch autograd graph.

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    output_nodes = (var.grad_fn,) if not isinstance(
        var, tuple) else tuple(v.grad_fn for v in var)

    def make_name(grad_tuple):
        name_list = list()

        name_dict = {0: "li", 1: "si", 2: "lo", 3: "so"}
        for i, grad_list in enumerate(grad_tuple):
            name_list.append(name_dict[i])
            for grad in grad_list:
                if grad is not None:
                    name_list.append(f"{grad}")
                else:
                    name_list.append("None")

        name = "_" + "_".join(name_list)
        return name

    def is_grad_bad(grad_tuple):
        for grad_list in grad_tuple:
            for grad in grad_list:
                if grad is not None:
                    try:
                        if grad.abs() >= bad_grad:
                            return True
                    except RuntimeError:
                        continue
                else:
                    continue

        return False

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                # note: this used to show .saved_tensors in pytorch0.2, but stopped
                # working as it was moved to ATen and Variable-Tensor merged
                dot.node(str(id(var)), size_to_str(
                    var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                alternate_name = 'Variable\n '
                name = param_map[id(u)] if params is not None and id(
                    u) in param_map else alternate_name
                node_name = '%s\n %s' % (name, size_to_str(u.size()))

                fillcolor = 'lightblue'
                if u.grad is not None and torch.max(u.grad.abs()) > bad_grad:
                    fillcolor = "red"

                dot.node(str(id(var)), node_name, fillcolor=fillcolor)
            elif var in output_nodes:
                if var in fn_dict:
                    name = make_name(fn_dict[var])
                    fillcolor = 'darkolivegreen1'
                    if is_grad_bad(fn_dict[var]):
                        fillcolor = 'red'
                    dot.node(str(id(var)), str(type(var).__name__) +
                             name, fillcolor=fillcolor)
                else:
                    dot.node(str(id(var)), str(
                        type(var).__name__), fillcolor="green")

            else:
                if var in fn_dict:
                    name = make_name(fn_dict[var])
                    fillcolor = 'white'
                    if is_grad_bad(fn_dict[var]):
                        fillcolor = 'red'
                    dot.node(str(id(var)), str(type(var).__name__) +
                             name, fillcolor=fillcolor)
                else:
                    dot.node(str(id(var)), str(
                        type(var).__name__), fillcolor="green")

            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])

    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)

    resize_graph(dot)

    return dot


def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.

    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)


def visualize_depth_map(depth_map, max_depth=None):
    height, width = depth_map.shape[2:4]
    depth_map = depth_map.reshape(height, width, 1)
    if max_depth is None:
        max_depth = torch.max(depth_map)

    depth_map = torch.clamp(depth_map / max_depth, min=0, max=1.0)
    depth_map_display = cv2.applyColorMap(
        np.uint8(255 * depth_map.data.cpu().numpy()), cv2.COLORMAP_JET)

    return depth_map_display, max_depth
