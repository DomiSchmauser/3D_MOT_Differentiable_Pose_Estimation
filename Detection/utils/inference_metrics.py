import sys

import numpy as np
import torch
import mathutils
import math
import open3d as o3d
from eulerangles import euler2matrix


def compute_voxel_iou(generated_volume, ground_truth_volume):
    '''
    3D voxel IoU between two voxel grids
    '''

    _volume = torch.ge(generated_volume, 0.5).float()
    intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
    union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float() # if _volume+ gt_volume >= 1 is union
    voxel_iou = (intersection / union).detach().cpu().item()

    return voxel_iou

def get_rotation_diff(gt_rotation, pred_rotation):
    '''
    gt_rotation: as euler coordinates xyz in radiants
    pred_rotation: as rotation matrix
    cls_name: indicating rotation symmetry
    calculate rotations difference between gt and predicted rotation matrix, min for two poses in y rotated by 180 degree, distinc
    '''

    euler = mathutils.Euler(gt_rotation)
    gt_rotation = np.array(euler.to_matrix())

    R1 = pred_rotation / np.cbrt(np.linalg.det(pred_rotation)) # R1 = pred
    R2 = gt_rotation / np.cbrt(np.linalg.det(gt_rotation)) # R2 = GT

    y_180_RT = np.diag([-1.0, 1.0, -1.0])
    R = R1 @ R2.transpose()

    R_rot = R1 @ y_180_RT @ R2.transpose()
    theta = min(np.arccos((np.trace(R) - 1) / 2),
                np.arccos((np.trace(R_rot) - 1) / 2))

    theta_deg = theta * 180 / np.pi

    return theta_deg

def get_location_diff(gt_location, pred_location):
    '''
    Calculate location difference of predicted pose in meter
    gt_location: xyz location in world coords
    pred_location: pred xyz location
    '''

    dist = np.linalg.norm(gt_location - pred_location)

    return dist


def get_location_diff_boxcenter(gt_3dbox, pred_3dbox):
    '''
    Calculate location difference of predicted pose in meter based on bounding box centers
    gt_3dbox: 8x3 array
    pred_3dbox: 8x3 array
    '''

    gtloc_box = o3d.geometry.OrientedBoundingBox()

    if gt_3dbox.sum() == 0:
        return None
    try:
        gtloc_box = gtloc_box.create_from_points(o3d.utility.Vector3dVector(gt_3dbox))
    except:
        return None

    center_gtbox = gtloc_box.get_center()

    predloc_box = o3d.geometry.OrientedBoundingBox()
    try:
        predloc_box = predloc_box.create_from_points(o3d.utility.Vector3dVector(pred_3dbox))
    except:
        return None
    center_predbox = predloc_box.get_center()

    dist = np.linalg.norm(center_gtbox - center_predbox)
    #print('Box location center', center_gtbox, center_predbox)

    return dist

def get_mean_iou(voxel_list):

    if voxel_list:
        voxel_arr = np.array(voxel_list)
        voxel_arr = voxel_arr[~np.isnan(voxel_arr)]
        voxel_arr = voxel_arr.mean()
    else:
        voxel_arr = 'No Data'

    return voxel_arr

def get_median_iou(voxel_list):

    if voxel_list:
        voxel_arr = np.array(voxel_list)
        voxel_arr = voxel_arr[~np.isnan(voxel_arr)]
        voxel_arr = np.median(voxel_arr)
    else:
        voxel_arr = 'No Data'

    return voxel_arr