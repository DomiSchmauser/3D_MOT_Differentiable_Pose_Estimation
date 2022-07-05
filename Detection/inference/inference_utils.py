import h5py
import torch
import numpy as np
import os, sys, cv2
import open3d as o3d
import copy

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

sys.path.append('..') #Hack add ROOT DIR

from BlenderProc.utils import binvox_rw
from baseconfig import CONF
from Detection.inference.inference_metrics import get_mean_iou, get_median_iou
from PoseEst.pose_estimation import backproject, cam2world, sort_bbox


def get_scale(m):
    if type(m) == torch.Tensor:
        return m.norm(dim=0)
    return np.linalg.norm(m, axis=0)

def transform_icp_points(source, transformation):
    '''
    transforms source pc to align with target point cloud based on a learned icp transformation
    '''
    source_temp = copy.deepcopy(source)
    return source_temp.transform(transformation)

def draw_registration_result(source, target, transformation):
    '''
    Visualise ICP Matching results
    '''
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def construct_box(segpc, ax_aligned=False):
    '''
    calculates 3D bounding box around segmentation pointcloud
    '''
    if ax_aligned:
        bbox3d_obj = o3d.geometry.AxisAlignedBoundingBox()
    else:
        bbox3d_obj = o3d.geometry.OrientedBoundingBox()
    bbox_3d = bbox3d_obj.create_from_points(o3d.utility.Vector3dVector(segpc))
    center_3d = bbox_3d.get_center()

    pred_box = sort_bbox(np.array(bbox_3d.get_box_points()))

    if not ax_aligned:
        scale = bbox_3d.extent
        rotation = bbox_3d.R
        cad2world = np.diag([0, 0, 0, 1]).astype(np.float32)
        cad2world[:3, :3] = np.diag(scale) @ rotation
        cad2world[:3, 3] = center_3d

        return torch.tensor(pred_box), center_3d, cad2world

    return torch.tensor(pred_box), center_3d

def project_segmask_F2F(pred_bin_mask, abs_bbox, depth, intrinsics):
    '''
    Projection segmask to pointcloud for F2F - MaskRCNN baseline
    '''

    depth = np.array(depth, dtype=np.float32)  # HxW

    # Zero pad depth image
    depth_pad = np.zeros((240, 320))
    depth_pad[int(abs_bbox[1]):int(abs_bbox[3]), int(abs_bbox[0]):int(abs_bbox[2])] = depth[int(abs_bbox[1]):int(abs_bbox[3]),
                                                                                            int(abs_bbox[0]):int(abs_bbox[2])]
    depth = depth_pad

    depth_pts, _ = backproject(depth, intrinsics, np.array(pred_bin_mask.cpu())) # depth in camera space

    return depth_pts

def project_segmask(pred_bin_mask, abs_bbox, depth, campose):
    '''
    Projection segmask to pointcloud for F2F - MaskRCNN baseline
    '''

    depth = np.array(depth, dtype=np.float32)  # HxW

    # Zero pad depth image
    depth_pad = np.zeros((240, 320))
    depth_pad[int(abs_bbox[1]):int(abs_bbox[3]), int(abs_bbox[0]):int(abs_bbox[2])] = depth[int(abs_bbox[1]):int(abs_bbox[3]),
                                                                                            int(abs_bbox[0]):int(abs_bbox[2])]
    depth = depth_pad

    img_width = depth.shape[1]
    img_height = depth.shape[0]
    cx = (img_width / 2) - 0.5  # 0,0 is center top-left pixel -> -0,5
    cy = (img_height / 2) - 0.5  # 0,0 is center top-left pixel -> -0,5
    fx = 292.87803547399
    fy = 292.87803547399
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    depth_pts, _ = backproject(depth, intrinsics, np.array(pred_bin_mask.cpu())) # depth in camera space
    depth_world = cam2world(depth_pts, campose)

    return depth_world

def convert_voxel_to_pc(voxel_grid, rot, trans, scale):
    '''
    Converts a voxel grid to a point cloud with according pose
    voxel_grid: 32x32x32 tensor binary
    rot, trans, scale: output from run pose function
    scale already encoded in rotation
    returns pc: n x 3 array
    '''

    nonzero_inds = np.nonzero(voxel_grid)[:-1]

    points = nonzero_inds / 32 - 0.5
    points = points.detach().cpu().numpy()

    #global_scalerot = (np.identity(3) * scale.copy()) @ rot
    world_pc = rot @ points.transpose() + np.expand_dims(trans.copy(), axis=-1)
    world_pc = world_pc.transpose()

    return world_pc

def add_halfheight(location, box):
    '''
    Object location z-center is at bottom, calculate half height of the object
    and add to shift z-center to correct location
    '''
    z_coords = []
    for pt in box:
        z = pt[-1]
        z_coords.append(z)
    z_coords = np.array(z_coords)
    half_height = np.abs(z_coords.max() - z_coords.min()) / 2
    location[-1] = half_height  # Center location is at bottom object

    return location

def load_hdf5(path):
    with h5py.File(path, 'r') as data:
        for key in data.keys():
            if key == 'depth':
                depth = np.array(data[key])
            elif key == 'campose':
                campose = np.array(data[key])

    return depth, campose

def get_nocs(nocs_path):
    '''
    loads GT nocs image
    cv2.imread -1 for using all color depth values
    '''

    nocs = cv2.imread(nocs_path, -1) #BGRA
    nocs = nocs[:,:,:3]
    nocs = np.array(nocs[:, :, ::-1], dtype=np.float32) # RGB

    return nocs

def log_results(metrics):

    voxel_iou = []
    chair_iou = []
    table_iou = []
    sofa_iou = []
    bed_iou = []
    tv_stand_iou = []
    cooler_iou = []
    night_stand_iou = []
    distances = []
    thetas = []
    for seq in metrics:
        for img in seq:
            for key, value in img.items():
                if key == 'voxel_ious':
                    voxel_iou.append(value)
                elif key == 'chair_ious':
                    chair_iou.append(value)
                elif key == 'table_ious':
                    table_iou.append(value)
                elif key == 'sofa_ious':
                    sofa_iou.append(value)
                elif key == 'bed_ious':
                    bed_iou.append(value)
                elif key == 'tv_stand_ious':
                    tv_stand_iou.append(value)
                elif key == 'cooler_ious':
                    cooler_iou.append(value)
                elif key == 'night_stand_ious':
                    night_stand_iou.append(value)
                elif key == 'pose_distance':
                    for entity in value:
                        distances.append(entity)
                elif key == 'pose_rotationdiff':
                    for entity in value:
                        thetas.append(entity)


    mean_voxel_iou = get_mean_iou(voxel_iou)
    mean_chair_iou = get_mean_iou(chair_iou)
    mean_table_iou = get_mean_iou(table_iou)
    mean_sofa_iou = get_mean_iou(sofa_iou)
    mean_bed_iou = get_mean_iou(bed_iou)
    mean_tv_iou = get_mean_iou(tv_stand_iou)
    mean_cooler_iou = get_mean_iou(cooler_iou)
    mean_night_iou = get_mean_iou(night_stand_iou)

    mean_rotation_diff = get_median_iou(thetas)
    mean_distance = get_median_iou(distances)

    print('Voxel_IoU :', mean_voxel_iou, ', Voxel_Chair_IoU :', mean_chair_iou, ', Voxel_Table_IoU :', mean_table_iou,
          ', Voxel_Sofa_IoU :', mean_sofa_iou, ', Voxel_Bed_IoU :', mean_bed_iou,
          ', Voxel_TVstand_IoU :', mean_tv_iou, ', Voxel_WineCooler_IoU :', mean_cooler_iou,
          ', Voxel_NightStand_IoU :', mean_night_iou,
          ', Rotation Difference [°] :', mean_rotation_diff, ', Location Difference [m] :', mean_distance )

def calculate_F2F_metrics(outputs):

    overall_gt_objects = 0
    overall_misses = 0
    overall_fps = 0
    overall_predictions = []
    overall_targets = []

    for seq in outputs:

        overall_gt_objects += seq['total_gt_objs']
        overall_misses += seq['misses']
        overall_fps += seq['false_positives']
        overall_predictions.append(seq['prediction'])
        overall_targets.append(seq['target'])

    predictions = np.concatenate(overall_predictions)
    targets = np.concatenate(overall_targets)

    F1 = f1_score(targets, predictions, zero_division='warn')  # warn only once
    Prec = precision_score(targets, predictions, zero_division=0)
    Rec = recall_score(targets, predictions, zero_division=0)

    id_switches = np.count_nonzero(targets - predictions)
    MOTA = 1.0 - (float(overall_misses + overall_fps + id_switches) / float(overall_gt_objects))

    print('MOTA score :', MOTA, ', F1 score :', F1, ', Precision :', Prec,
          ', Recall :', Rec)

def log_F2F_results(metrics):
    '''
    F2F-MaskRCNN result logging
    metrics: list of sequences (tuple(MOTA,F1,Precision,Recall))
    '''

    overall_mota = []
    overall_F1 = []
    overall_precision = []
    overall_recall = []

    for seq in metrics:

        mota, f1, precision, recall = seq[0], seq[1], seq[2], seq[3]

        overall_mota.append(mota)
        overall_F1.append(f1)
        overall_precision.append(precision)
        overall_recall.append(recall)

    mean_mota = np.array(overall_mota).mean()
    mean_f1 = np.array(overall_F1).mean()
    mean_precision = np.array(overall_precision).mean()
    mean_recall = np.array(overall_recall).mean()

    print('MOTA score :', mean_mota, ', F1 score :', mean_f1, ', Precision :', mean_precision,
          ', Recall :', mean_recall)
