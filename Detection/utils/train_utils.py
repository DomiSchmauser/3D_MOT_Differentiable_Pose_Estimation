import sys, os
import mathutils
from random import randint

import torch
from torch.nn import init
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
import trimesh
from scipy.interpolate import interpn
import numpy as np
import matplotlib.pyplot as plt

from detectron2.structures import BoxMode
from detectron2.utils.visualizer import GenericMask

sys.path.append('..') #Hack add ROOT DIR
from BlenderProc.utils import binvox_rw


def pose_loss(gt_rot, gt_loc, gt_scale, pred_rot, pred_loc, pred_scale, obj_pc, max_points=500, device=None):

    def get_homog_mat(rot, loc, scale):
        homog_mat = torch.zeros((4, 4)).to(device)
        homog_mat[:3, :3] = torch.diag(scale) @ rot.T
        homog_mat[:3, 3] = loc
        homog_mat[3, 3] = 1
        return homog_mat

    def makelist(count, max_int):
        return [randint(0, max_int) for _ in range(count)]

    num_points = min(max_points, obj_pc.shape[0])
    idxs = makelist(num_points, obj_pc.shape[0]-1)
    sample_points = obj_pc[idxs]

    gt_mat = get_homog_mat(gt_rot, gt_loc, gt_scale)
    pred_mat = get_homog_mat(pred_rot, pred_loc, pred_scale.repeat(3))

    gt_points = gt_mat[:3, :3] @ sample_points.T + gt_mat[:3, 3:]
    gt_points = gt_points.T

    pred_points = pred_mat[:3, :3] @ sample_points.T + pred_mat[:3, 3:]
    pred_points = pred_points.T

    dis = torch.mean(torch.norm((pred_points - gt_points), dim=-1))

    return dis

class PoseLoss(_Loss):
    '''
    7-DoF pose loss using sampled object points from the complete object voxel grid
    '''

    def __init__(self, max_points=500):
        super(PoseLoss, self).__init__(True)
        self.max_points = max_points
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, gt_rot, gt_loc, gt_scale, pred_rot, pred_loc, pred_scale, obj_pc):
        return pose_loss(gt_rot, gt_loc, gt_scale, pred_rot, pred_loc, pred_scale, obj_pc, max_points=self.max_points, device=self.device)


def balanced_BCE_loss(gt_voxels, pred_voxels):
    '''
    Balanced BCE loss giving occupied and non-occupied voxel regions the same weighting
    gt_voxels: GT voxel grid, num_instances x 32x32x32
    pred_voxels: Pred voxel grid, num_instances x 32x32x32
    '''

    num_occupied = torch.count_nonzero(gt_voxels)
    num_all = torch.numel(gt_voxels)
    pos_weight = (num_all - num_occupied) / num_occupied

    combined_loss = F.binary_cross_entropy_with_logits(pred_voxels, gt_voxels, reduction="mean", pos_weight=pos_weight)

    return combined_loss

def symmetry_smooth_l1_loss(gt_overlap, pred_overlap, gt_cls=None, debug_mode=False):
    '''
    Smooth l1 loss for 4 rotational settings (0, 90, 180, 270) around the Y-Axis in 90 degree steps clockwise
    gt_overlap: GT nocs patch, shape CxHxW
    pred_overlap: Pred nocs patch, shape CxHxW
    gt_cls: string with classname for class agnostic symmetry
    '''

    smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='mean', beta=0.1)  # Smooth L1 loss
    overlap_height, overlap_width = gt_overlap.shape[1], gt_overlap.shape[2]

    if gt_cls == None or gt_cls == 'chair' or gt_cls == 'sofa' or gt_cls == 'bed':
        num_rotation_steps = 1
    elif gt_cls == 'table':
        num_rotation_steps = 2
    else:
        num_rotation_steps = 1

    if num_rotation_steps == 4:
        # 90 degree rotation
        rotation_y = torch.tensor([[0, 0, 1.0],
                                   [0, 1.0, 0],
                                   [-1.0, 0, 0]])
    elif num_rotation_steps == 2:
        # 180 degree rotation
        rotation_y = torch.tensor([[-1.0, 0, 0],
                                   [0, 1.0, 0],
                                   [0, 0, -1.0]])

    losses = []
    for i in range(num_rotation_steps):
        if i == 0: # No rotation required
            degree_loss = smooth_l1_loss(pred_overlap, gt_overlap)
            losses.append(degree_loss)

            if debug_mode:
                plt.imshow(gt_overlap.permute(1, 2, 0).contiguous())
                plt.title('0 degree')
                plt.show()
        else:
            gt_overlap = gt_overlap.permute(1, 2, 0).contiguous()  # HxWxC
            gt_overlap_pts = gt_overlap.view(-1, 3) - 0.5  # Num pts x xyz, 0 center before rotating
            gt_overlap_pts[torch.sum(gt_overlap_pts, dim=1) != 1.5] = (
                    rotation_y @ gt_overlap_pts[torch.sum(gt_overlap_pts, dim=1) != 1.5].T).T  # exclude background
            gt_overlap_pts += 0.5 # shift back to 0-1 space

            assert torch.max(gt_overlap_pts) <= 1 and torch.min(gt_overlap_pts) >= 0

            gt_overlap = gt_overlap_pts.view(overlap_height, overlap_width, 3).permute(2, 0, 1).contiguous()  # CxHxW
            degree_loss = smooth_l1_loss(pred_overlap, gt_overlap) # input, target
            losses.append(degree_loss)

            if debug_mode:
                print(torch.max(gt_overlap_pts), torch.min(gt_overlap_pts))
                plt.imshow(gt_overlap.permute(1, 2, 0).contiguous())
                plt.title('Rotated {}'.format(str(i)))
                plt.show()

    loss_idx = torch.argmin(torch.tensor(losses))
    obj_loss = losses[loss_idx]
    return obj_loss


def symmetry_bin_loss(gt_overlap, pred_overlap, gt_cls=None, num_bins=32, debug_mode=False):
    '''
    CE Classification loss for 4 rotational settings (0, 90, 180, 270) around the Y-Axis in 90 degree steps clockwise
    gt_overlap: GT nocs patch, shape CxHxW
    pred_overlap: Pred nocs patch, shape num_binsxCxHxW
    gt_cls: string with classname for class agnostic symmetry
    '''

    ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')
    overlap_height, overlap_width = gt_overlap.shape[1], gt_overlap.shape[2]
    pred_overlap = torch.unsqueeze(pred_overlap, dim=0) # 1 x bins x 3 x H x W

    def discretize_gt_nocs(gt_nocs, num_bins):
        '''
        Discretize GT NOCS for categorical label == num bins, 0-31 per pixel value
        '''
        gt_nocs = gt_nocs * torch.tensor(num_bins, dtype=torch.float32) - 0.000001
        gt_nocs = torch.floor(gt_nocs).long()
        gt_nocs[gt_nocs == -1] = 0
        #disc_nocs = F.one_hot(gt_nocs, num_classes=num_bins)
        #disc_nocs = torch.unsqueeze(disc_nocs, dim=0).permute(0, 4, 1, 2, 3).contiguous() # 1 x bins x 3 x H x W
        disc_nocs = torch.unsqueeze(gt_nocs, dim=0) # 1 x 3 x H x W

        return disc_nocs

    if gt_cls == None or gt_cls == 'chair' or gt_cls == 'sofa' or gt_cls == 'bed':
        num_rotation_steps = 1
    elif gt_cls == 'table':
        num_rotation_steps = 2

    if num_rotation_steps == 4:
        # 90 degree rotation
        rotation_y = [None, torch.tensor([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]),
                      torch.tensor([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]]),
                      torch.tensor([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]])]

    elif num_rotation_steps == 2:
        # 180 degree rotation
        rotation_y = [None, torch.tensor([[-1.0, 0, 0],
                                   [0, 1.0, 0],
                                   [0, 0, -1.0]])]

    losses = []
    for i in range(num_rotation_steps):
        if i == 0: # No rotation required
            _gt_overlap = discretize_gt_nocs(gt_overlap, num_bins)
            degree_loss = ce_loss(pred_overlap, _gt_overlap)
            losses.append(degree_loss)

            if debug_mode:
                plt.imshow(gt_overlap.permute(1, 2, 0).contiguous())
                plt.title('0 degree')
                plt.show()
        else:
            _gt_overlap = gt_overlap.permute(1, 2, 0).contiguous()  # HxWxC
            gt_overlap_pts = _gt_overlap.view(-1, 3) - 0.5  # Num pts x xyz, 0 center before rotating
            gt_overlap_pts[torch.sum(gt_overlap_pts, dim=1) != 1.5] = (
                    rotation_y[i] @ gt_overlap_pts[torch.sum(gt_overlap_pts, dim=1) != 1.5].T).T  # exclude background
            gt_overlap_pts += 0.5 # shift back to 0-1 space

            assert torch.max(gt_overlap_pts) <= 1 and torch.min(gt_overlap_pts) >= 0

            _gt_overlap = gt_overlap_pts.view(overlap_height, overlap_width, 3).permute(2, 0, 1).contiguous()  # CxHxW
            _gt_overlap = discretize_gt_nocs(_gt_overlap, num_bins)

            degree_loss = ce_loss(pred_overlap, _gt_overlap) # input, target
            losses.append(degree_loss)

            if debug_mode:
                print(torch.max(gt_overlap_pts), torch.min(gt_overlap_pts))
                plt.imshow(gt_overlap.permute(1, 2, 0).contiguous())
                plt.title('Rotated {}'.format(str(i)))
                plt.show()

    loss_idx = torch.argmin(torch.tensor(losses))
    obj_loss = losses[loss_idx]
    return obj_loss

### --------------------------------------------------------------------------------------------------------------------

def nocs_prob_to_value(nocs_prob, channel, num_bins=32):
    '''
    nocs_prob: output of nocs model unnormalized scores, shape: num_obj x num_bins x Channel x 28 x 28
    channel: any of 0, 1, 2 for RGB channels
    '''

    prob = nocs_prob[:, :, channel, :, :]  # num obj x num bins x 28 x 28
    prob = prob.softmax(dim=1)
    prob = prob.permute(0, 2, 3, 1).contiguous()
    prob_shape = prob.shape
    prob_reshape = torch.reshape(prob, (-1, prob_shape[-1]))

    '''
    if prob_reshape.numel() == 0: # empty tensor with no object detections
        return torch.zeros(1, 1, 28, 28)
    '''

    ind = torch.argmax(prob_reshape, dim=-1)
    val = ind.to(dtype=torch.float32) / (num_bins - 1)  # index starts at 0 -> num_bins-1
    val = torch.reshape(val, (-1, prob_shape[0], 28, 28)).permute(1, 0, 2, 3).contiguous()  # num obj x 1 x 28 x 28

    return val


def crop_nocs(nocs_obj, pad_value=300):
    '''
    crop padded value on input nocs patch
    nocs_obj: H x W x C
    '''

    nocs_obj = torch.unsqueeze(nocs_obj, dim=0)  # 1 x H x W x 3
    nocs_obj = nocs_obj.permute(0, 3, 1, 2).contiguous()  # 1 x 3 x H x W

    if pad_value in nocs_obj:
        heigth_shape = torch.where(nocs_obj[0, 0, :, 0] == pad_value)[0]
        width_shape = torch.where(nocs_obj[0, 0, 0, :] == pad_value)[0]

        if len(heigth_shape) > 0 and len(width_shape) > 0:
            height = torch.min(heigth_shape)
            width = torch.min(width_shape)
            return nocs_obj[:, :, :height, :width]
        elif len(heigth_shape) > 0 and len(width_shape) == 0:
            height = torch.min(heigth_shape)
            return nocs_obj[:, :, :height, :]
        elif len(width_shape) > 0 and len(heigth_shape) == 0:
            width = torch.min(width_shape)
            return nocs_obj[:, :, :, :width]
    else:
        return nocs_obj

def get_voxel(voxel_path, scale):
    '''
    Load voxel grid and rescale with according scale parameters if scale is any other than [1, 1, 1]
    voxel_path: path to 3D-FUTURE model
    scale: array with scale parameter in xyz
    '''
    if not os.path.exists(voxel_path):
        raise ValueError('Voxelized model does not exist for this path!', voxel_path)

    with open(voxel_path, 'rb') as f:
        voxel = binvox_rw.read_as_3d_array(f).data

    unscaled_voxel = voxel.astype(int)

    if np.all(scale == 1): # No scaling required
        rescaled_voxel = unscaled_voxel
    else:
        rescaled_voxel = rescale_voxel(unscaled_voxel, scale)

    return torch.from_numpy(rescaled_voxel)

def rescale_voxel(unscaled_voxel, scale, debug_mode=False):
    '''
    Rescale 3D voxel grid by a given scale array
    '''

    centering = unscaled_voxel.shape[0] / 2
    max_value = unscaled_voxel.shape[0] - 1
    non_zeros = np.nonzero(unscaled_voxel)
    scale_mat = np.diag(scale)
    xyz = (np.stack(non_zeros, axis=0).T - centering) @ (scale_mat / scale.max())
    xyz = np.rint(xyz) + centering
    xyz[xyz>max_value] = max_value # all values rounded up to 32 are set to max -> 31
    x = xyz[:,0].astype(np.int32)
    y = xyz[:,1].astype(np.int32)
    z = xyz[:,2].astype(np.int32)
    rescale_ = np.zeros(unscaled_voxel.shape)
    rescale_[x, y, z] = 1

    if debug_mode:
        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(rescale_, edgecolor='k')
        #ax.voxels(unscaled_voxel, edgecolor='k')
        plt.show()
        ax = plt.figure().add_subplot(projection='3d')
        #ax.voxels(rescale_, edgecolor='k')
        ax.voxels(unscaled_voxel, edgecolor='k')
        plt.show()

    return rescale_

def polygon_to_binmask(polygon_mask):

    gm = GenericMask(polygon_mask, 240, 320)
    bin_mask = gm.polygons_to_mask(polygon_mask)
    binary_mask = bin_mask[:, :, None]
    return binary_mask


def crop_segmask(nocs_img, bbox, segmap, color_depth_max=65535):
    '''
    Crop nocs image and set bg to zero, normalize between 0 and 1 per patch, also works for all equal pixel values
    color_depth_max: depends on discretization of nocs map (8 bit = 255, 16bit = 65535)
    '''

    abs_bbox = torch.tensor(BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS), dtype=torch.float32)
    gm = GenericMask(segmap, 240, 320)
    bin_mask = gm.polygons_to_mask(segmap)
    binary_mask = bin_mask[:, :, None]
    crop_im = np.multiply(nocs_img, binary_mask)  # makes it black
    crop_im[crop_im == 0] = color_depth_max  # Background white
    cropped_im = np.array(crop_im[int(abs_bbox[1]):int(abs_bbox[3]), int(abs_bbox[0]):int(abs_bbox[2]), :])  # H x W x C

    if (cropped_im.max() - cropped_im.min()) != 0:
        cropped_im = (cropped_im - cropped_im.min()) / (
                    cropped_im.max() - cropped_im.min())  # Normalize patches between 0 and 1
    else:
        cropped_im /= color_depth_max  # for patches with bad bounding box cropping only equal values (e.g. all white pixels)

    return torch.from_numpy(cropped_im).to(torch.float32)

def crop_segmask_gt(nocs_img, abs_bbox, segmap, color_depth_max=65535):
    '''
    Crop nocs image and set bg to zero, normalize between 0 and 1 per patch, also works for all equal pixel values
    color_depth_max: depends on discretization of nocs map (8 bit = 255, 16bit = 65535)
    '''

    gm = GenericMask(segmap, 240, 320)
    bin_mask = gm.polygons_to_mask(segmap)
    binary_mask = bin_mask[:, :, None]
    crop_im = np.multiply(nocs_img, binary_mask)  # makes it black
    crop_im[crop_im == 0] = color_depth_max  # Background white
    cropped_im = np.array(crop_im[int(abs_bbox[1]):int(abs_bbox[3]), int(abs_bbox[0]):int(abs_bbox[2]), :])  # H x W x C

    if (cropped_im.max() - cropped_im.min()) != 0:
        cropped_im = (cropped_im - cropped_im.min()) / (
                    cropped_im.max() - cropped_im.min())  # Normalize patches between 0 and 1
    else:
        cropped_im /= color_depth_max  # for patches with bad bounding box cropping only equal values (e.g. all white pixels)

    return torch.from_numpy(cropped_im).to(torch.float32)

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)

            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)

            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)

            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('Initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def backproject_rgb(rgb, depth, intrinsics):
    '''
    Backproject depth map to camera space, with additional rgb values
    Returns: Depth PC with according RGB values in camspace, used idxs in pixel space
    '''

    intrinsics_inv = np.linalg.inv(intrinsics)
    non_zero_mask = (depth > 0)

    idxs = np.where(non_zero_mask)
    grid = np.array([idxs[1], idxs[0]])

    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid  # [3, num_pixel]
    xyz = np.transpose(xyz)  # [num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    pts[:, 1] = -pts[:, 1]
    pts[:, 2] = -pts[:, 2]

    rgb_vals = rgb[idxs[0], idxs[1]]

    rgb_pts = np.concatenate((pts, rgb_vals), axis=-1)

    return rgb_pts

def rgb2pc(rgb_img, depth, campose):
    '''
    rgb as H x W x C
    '''
    img = rgb_img.permute(1, 2, 0).contiguous().numpy()[:, :, ::-1]  # H x W x C
    cx, cy = (img.shape[1] / 2) - 0.5, (img.shape[0] / 2) - 0.5  # 0,0 is center top-left pixel -> -0,5
    fx, fy = 292.87803547399, 292.87803547399  # focal length from blenderproc
    cam_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    pc = backproject_rgb(img, depth, cam_intrinsics)
    # Cam2World transform
    trans = campose[:3, 3:]
    rot = campose[:3, :3]
    vis_pc = np.dot(rot, pc[:,:3].transpose()) + trans
    vis_pc = vis_pc.transpose()
    vis_pc = np.concatenate((vis_pc, pc[:,3:]), axis=-1)
    return vis_pc