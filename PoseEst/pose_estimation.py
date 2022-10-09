import os, sys, json
import pickle
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from detectron2.utils.visualizer import GenericMask
from detectron2.structures import BoxMode

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

from PoseEst.pose_utils import estimateSimilarityTransform, umeyama_torch, estimateSimilarityTransform_torch

def backproject_torch(depth, intrinsics, bin_mask, device=None):
    '''
    Backproject depth map to camera space
    Returns: Depth PC in camspace, used idxs in pixel space
    '''

    intrinsics_inv = torch.linalg.inv(intrinsics)
    non_zero_mask = (depth > 0)
    final_instance_mask = torch.logical_and(bin_mask.to(device), non_zero_mask)

    idxs = torch.tensor(final_instance_mask).nonzero()

    grid = torch.cat([torch.unsqueeze(idxs[:,1], dim=-1), torch.unsqueeze(idxs[:,0], dim=-1)], dim=-1).T #2, len

    length = grid.shape[1]
    ones = torch.ones([1, length]).to(device)
    uv_grid = torch.cat((grid, ones), dim=0) # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid # [3, num_pixel]
    xyz = xyz.T  # [num_pixel, 3]

    z = depth[idxs[:,0], idxs[:,1]]

    pts = xyz * z[:, None] / xyz[:, -1:]
    pts[:, 1] = -pts[:, 1]
    pts[:, 2] = -pts[:, 2]

    return pts, idxs

def backproject(depth, intrinsics, bin_mask):
    '''
    Backproject depth map to camera space
    Returns: Depth PC in camspace, used idxs in pixel space
    '''

    intrinsics_inv = np.linalg.inv(intrinsics)
    non_zero_mask = (depth > 0)
    #bin_mask = np.ones(bin_mask.shape)
    final_instance_mask = np.logical_and(bin_mask, non_zero_mask)

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0) # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid # [3, num_pixel]
    xyz = np.transpose(xyz)  # [num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    pts[:, 1] = -pts[:, 1]
    pts[:, 2] = -pts[:, 2]

    return pts, idxs

def transform_pc(scale, rot, trans, pc):
    '''
    Transform PointCloud based on Umeyama results
    pc: N points x 3
    '''
    aligned_RT = np.zeros((4, 4), dtype=np.float32)
    aligned_RT[:3, :3] = np.diag(scale) @ rot.transpose()
    aligned_RT[:3, 3] = trans
    aligned_RT[3, 3] = 1
    coord_pts_rotated = aligned_RT[:3, :3] @ pc.transpose() + aligned_RT[:3, 3:]
    coord_pts_rotated = coord_pts_rotated.transpose()

    return coord_pts_rotated

def cam2world_torch(cam_pc, campose):
    '''
    Transfrom PointCloud from camera space to world space
    cam_pc: N points x XYZ(3)
    '''
    trans = campose[:3, 3:]
    rot = campose[:3, :3]

    world_pc = rot.to(torch.float32) @ cam_pc.T.to(torch.float32) + trans
    world_pc = world_pc.T

    return world_pc

def cam2world(cam_pc, campose):
    '''
    Transfrom PointCloud from camera space to world space
    cam_pc: N points x XYZ(3)
    '''
    trans = campose[:3, 3:]
    rot = campose[:3, :3]

    world_pc = np.dot(rot, cam_pc.transpose()) + trans
    world_pc = world_pc.transpose()

    return world_pc

def sort_bbox(bboxs):
    '''
    bbox: 8x3 np array
    Sort in counter clockwise order
    '''
    sort_y = np.flip(np.argsort(bboxs[:, 1]))
    y_sorted = bboxs[sort_y] # bottom points first

    sort_yx1 = np.flip(np.argsort(y_sorted[0:4, 0]))
    sort_yx2 = np.flip(np.argsort(y_sorted[4:8, 0])) + 4
    sort_yx = np.concatenate((sort_yx1, sort_yx2), axis=None)
    yx_sorted = y_sorted[sort_yx] # closer to origin first

    sort_zyx1 = np.argsort(yx_sorted[0:2, 2])
    sort_zyx1 = np.flip(sort_zyx1)
    sort_zyx2 = np.argsort(yx_sorted[2:4, 2]) + 2
    sort_zyx3 = np.argsort(yx_sorted[4:6, 2]) + 4
    sort_zyx3 = np.flip(sort_zyx3)
    sort_zyx4 = np.argsort(yx_sorted[6:8, 2]) + 6
    sort_zyx = np.concatenate((sort_zyx1, sort_zyx2, sort_zyx3, sort_zyx4), axis=None)
    zyx_sorted = yx_sorted[sort_zyx]
    return zyx_sorted

def sort_pointcloud(pc):
   '''
    pc: Nx3 np array
    Sort by distance to 0
   '''
   norm_values = np.linalg.norm(pc, axis=1) #Nx1
   sort_idxs = np.argsort(norm_values)
   sorted_pc = pc[sort_idxs]

   return sorted_pc

def clean_depth_torch(depth_pts_ol, obj_gt_3Dbbox, campose):
    '''
    Clean depth map based on GT 3D bounding box, for e.g chairs with holes in backseat depth map is very bad
    box format: 8x3
    '''

    gt_xmin = obj_gt_3Dbbox[:, 0].min()
    gt_xmax = obj_gt_3Dbbox[:, 0].max()
    gt_ymin = obj_gt_3Dbbox[:, 1].min()
    gt_ymax = obj_gt_3Dbbox[:, 1].max()
    gt_zmin = obj_gt_3Dbbox[:, 2].min()
    gt_zmax = obj_gt_3Dbbox[:, 2].max()

    copy_depth = torch.clone(depth_pts_ol)

    copy_depth_world_pts = cam2world_torch(copy_depth, campose)

    new_depth = []
    indicies_used = []
    for index, depth_cpt in enumerate(copy_depth_world_pts):
        if depth_cpt[0] > gt_xmin and depth_cpt[0] < gt_xmax and depth_cpt[1] > gt_ymin and depth_cpt[1] < gt_ymax \
                and depth_cpt[2] > gt_zmin and depth_cpt[2] < gt_zmax:
            new_depth.append(torch.unsqueeze(depth_pts_ol[index], dim=-1))
            indicies_used.append(index)
    if len(new_depth) > 60:
        depth_no_ol = torch.cat(new_depth, dim=-1)
    else:
        depth_no_ol = None

    return depth_no_ol, indicies_used


def clean_depth(depth_pts_ol, obj_gt_3Dbbox, campose):
    '''
    Clean depth map based on GT 3D bounding box, for e.g chairs with holes in backseat depth map is very bad
    box format: 8x3
    '''

    gt_xmin = obj_gt_3Dbbox[:, 0].min()
    gt_xmax = obj_gt_3Dbbox[:, 0].max()
    gt_ymin = obj_gt_3Dbbox[:, 1].min()
    gt_ymax = obj_gt_3Dbbox[:, 1].max()
    gt_zmin = obj_gt_3Dbbox[:, 2].min()
    gt_zmax = obj_gt_3Dbbox[:, 2].max()

    copy_depth = depth_pts_ol.copy()

    copy_depth_world_pts = cam2world(copy_depth, campose)

    new_depth = []
    indicies_used = []
    for index, depth_cpt in enumerate(copy_depth_world_pts):
        if depth_cpt[0] > gt_xmin and depth_cpt[0] < gt_xmax and depth_cpt[1] > gt_ymin and depth_cpt[1] < gt_ymax \
                and depth_cpt[2] > gt_zmin and depth_cpt[2] < gt_zmax:
            new_depth.append(depth_pts_ol[index])
            indicies_used.append(index)

    depth_no_ol = np.array(new_depth)

    return depth_no_ol, indicies_used

def crop_gt_bbox(depth_world_pts, gt_3Dbbox_obj):
    '''
    Crop 3D bbox based on depth, assumes specific counter-clockwise order of 8 box points
    '''

    depth_xmin = depth_world_pts[:, 0].min()
    depth_xmax = depth_world_pts[:, 0].max()
    depth_ymin = depth_world_pts[:, 1].min()
    depth_ymax = depth_world_pts[:, 1].max()
    depth_zmin = depth_world_pts[:, 2].min()
    depth_zmax = depth_world_pts[:, 2].max()

    # Crop GT 3D BBOX
    cop_bbox = gt_3Dbbox_obj.copy()
    for cidx, cpt in enumerate(cop_bbox):
        x_tmp = cpt[0]
        y_tmp = cpt[1]
        z_tmp = cpt[2]
        if cidx == 6:
            cop_bbox[cidx][0] = max(x_tmp, depth_xmin)
            cop_bbox[cidx][1] = max(y_tmp, depth_ymin)
            cop_bbox[cidx][2] = max(z_tmp, depth_zmin)
        elif cidx == 5:
            cop_bbox[cidx][0] = min(x_tmp, depth_xmax)
            cop_bbox[cidx][1] = max(y_tmp, depth_ymin)
            cop_bbox[cidx][2] = max(z_tmp, depth_zmin)
        elif cidx == 1:
            cop_bbox[cidx][0] = min(x_tmp, depth_xmax)
            cop_bbox[cidx][1] = min(y_tmp, depth_ymax)
            cop_bbox[cidx][2] = max(z_tmp, depth_zmin)
        elif cidx == 2:
            cop_bbox[cidx][0] = max(x_tmp, depth_xmin)
            cop_bbox[cidx][1] = min(y_tmp, depth_ymax)
            cop_bbox[cidx][2] = max(z_tmp, depth_zmin)
        elif cidx == 7:
            cop_bbox[cidx][0] = max(x_tmp, depth_xmin)
            cop_bbox[cidx][1] = max(y_tmp, depth_ymin)
            cop_bbox[cidx][2] = min(z_tmp, depth_zmax)
        elif cidx == 4:
            cop_bbox[cidx][0] = min(x_tmp, depth_xmax)
            cop_bbox[cidx][1] = max(y_tmp, depth_ymin)
            cop_bbox[cidx][2] = min(z_tmp, depth_zmax)
        elif cidx == 0:
            cop_bbox[cidx][0] = min(x_tmp, depth_xmax)
            cop_bbox[cidx][1] = min(y_tmp, depth_ymax)
            cop_bbox[cidx][2] = min(z_tmp, depth_zmax)
        elif cidx == 3:
            cop_bbox[cidx][0] = max(x_tmp, depth_xmin)
            cop_bbox[cidx][1] = min(y_tmp, depth_ymax)
            cop_bbox[cidx][2] = min(z_tmp, depth_zmax)

    return cop_bbox

def run_crop_3dbbox(depth, campose, gt_3Dbbox, gt_2Dbbox, gt_bin_mask): #per Object
    '''
    Crops GT 3D bounding box based on depth map per object -> enables object tight bounding boxes
    '''

    # Sort GT bbox
    gt_3Dbbox = sort_bbox(gt_3Dbbox)
    depth = np.array(depth)  # HxW zero at mask

    # Zero pad depth image
    depth_pad = np.zeros((240, 320))
    depth_pad[int(gt_2Dbbox[1]):int(gt_2Dbbox[3]), int(gt_2Dbbox[0]):int(gt_2Dbbox[2])] = depth[int(gt_2Dbbox[1]):int(gt_2Dbbox[3]), int(gt_2Dbbox[0]):int(gt_2Dbbox[2])]
    gt_depth = depth_pad

    img_width = depth.shape[1]
    img_height = depth.shape[0]

    cx = (img_width / 2) - 0.5  # 0,0 is center top-left pixel -> -0,5
    cy = (img_height / 2) - 0.5  # 0,0 is center top-left pixel -> -0,5

    fx = 292.8781  # focal length from blenderproc
    fy = 292.8781

    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    gt_depth_pts, idxs = backproject(gt_depth, intrinsics, np.array(gt_bin_mask))

    # Clean depth
    clean_depth_pts, ind = clean_depth(gt_depth_pts, gt_3Dbbox, campose)  # indicies used
    if not ind:
        return gt_3Dbbox

    # Crop GT BBox if not fully visible in Image
    depth_world_pts = cam2world(clean_depth_pts, campose)
    cropped_gt_3Dbbox = crop_gt_bbox(depth_world_pts, gt_3Dbbox)

    vis_obj = False
    if vis_obj:
        depth_pc_obj = o3d.geometry.PointCloud()
        depth_pc_obj.points = o3d.utility.Vector3dVector(depth_world_pts)
        depth_pc_obj.paint_uniform_color([0.3, 0.3, 0.3])

        nocs_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0])

        crop_box = o3d.geometry.OrientedBoundingBox()
        crop_box = crop_box.create_from_points(o3d.utility.Vector3dVector(cropped_gt_3Dbbox))

        full_box = o3d.geometry.OrientedBoundingBox()
        full_box = full_box.create_from_points(o3d.utility.Vector3dVector(gt_3Dbbox))

        o3d.visualization.draw_geometries([depth_pc_obj, full_box])
        o3d.visualization.draw_geometries([depth_pc_obj, crop_box, full_box, nocs_origin])

    return cropped_gt_3Dbbox

def run_pose_torch(nocs, depth, campose, bin_mask, abs_bbox, gt_3d_box=None, device=None, use_RANSAC=False):
    '''
    Pose Estimation with Umeyama Algorithm and RANSAC outlier removal per Object
    TORCH IMPLEMENTATION FOR BACKPROP
    nocs: predicted nocs map, shape of patch format HxWxC in RGB, normalized between 0 and 1
    depth: full depth 240 x 320 = H x W
    campose: 4x4 homogeneous camera matrix
    bin_mask: binary segmentation mask 240 x 320, predicted
    abs_bbox: bounding box in absolute coordinates XYXY, predicted
    use_depth_box: use 3D bounding box of the depth pointcloud for IoU Matching
    '''

    # Zero pad depth image
    depth_pad = torch.zeros((240, 320)).to(device)
    depth_pad[int(abs_bbox[1]):int(abs_bbox[3]), int(abs_bbox[0]):int(abs_bbox[2])] = depth[int(abs_bbox[1]):int(abs_bbox[3]), int(abs_bbox[0]):int(abs_bbox[2])]
    depth = depth_pad

    # Zero pad nocs image
    nocs_pad = torch.zeros((240, 320, 3)).to(device)
    nocs_pad[int(abs_bbox[1]):int(abs_bbox[3]), int(abs_bbox[0]):int(abs_bbox[2]), :] = nocs
    nocs = nocs_pad

    img_width = depth.shape[1]
    img_height = depth.shape[0]

    cx = (img_width / 2) - 0.5  # 0,0 is center top-left pixel -> -0,5
    cy = (img_height / 2) - 0.5  # 0,0 is center top-left pixel -> -0,5

    fx = 292.87803547399 # focal length from blenderproc # for fov=1 -> 292.8781
    fy = 292.87803547399
    intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).to(device)

    depth_pts, idxs = backproject_torch(depth, intrinsics, bin_mask, device=device)

    # Clean depth
    if gt_3d_box is not None:
        new_depth_pts, new_idxs = clean_depth_torch(depth_pts, gt_3d_box, campose) # clean depth for objects with holes
        if new_depth_pts is not None:
            depth_pts = new_depth_pts.T
            idxs_x = idxs[:, 0][new_idxs]
            idxs_y = idxs[:, 1][new_idxs]
            idxs = (idxs_x, idxs_y)

    # Clean depth
    clean_depth_pts = depth_pts
    ind = list(np.arange(clean_depth_pts.shape[0]))

    if new_depth_pts is not None:
        idxs_x = idxs[0][ind]
        idxs_y = idxs[1][ind]
    else:
        idxs_x = idxs[:, 0][ind]
        idxs_y = idxs[:, 1][ind]

    nocs_pts = nocs[idxs_x, idxs_y, :] - 0.5   # 0 centering -> back to cad space

    cleaned_nocs_pts = nocs_pts

    # RANSAC & Umeyama
    outlier_thres = 1
    if cleaned_nocs_pts.shape[0] == 0:
        return None, None, None, None, None, None

    if not use_RANSAC:
        Rotation, Scales, Translation, _ = umeyama_torch(cleaned_nocs_pts, clean_depth_pts) # CAD2CAM
    else:
        Rotation, Scales, Translation, _ = estimateSimilarityTransform_torch(cleaned_nocs_pts, clean_depth_pts,
                                                                       verbose=False, ratio_adapt=outlier_thres, device=device)  # CAD2CAM

    if Scales is None:
        return None, None, None, None, None, None

    # Chain object to camera space and cam space to world space transformation matricies
    obj_tocam = torch.eye(4).to(device) # CAD2Cam
    obj_tocam[:3,:3] = torch.diag(Scales.repeat(3)) @ Rotation.T #Rotation.T weirdly correct
    obj_tocam[:3,3] = Translation
    global_transform = campose.to(torch.float32) @ obj_tocam.to(torch.float32) # Cam2World @ CAD2Cam = CAD2World
    global_trans = global_transform[:3,3]
    global_rot = global_transform[:3,:3] # Scale already embedded into Rotation
    global_scale = Scales

    return global_rot, global_trans, global_scale, None, None, None

def run_pose(nocs, depth, campose, bin_mask, abs_bbox, vis_obj=False, gt_pc=None, gt_3d_box=None, use_depth_box=True):
    '''
    Pose Estimation with Umeyama Algorithm and RANSAC outlier removal per Object
    nocs: predicted nocs map, shape of patch format HxWxC in RGB, normalized between 0 and 1
    depth: full depth 240 x 320 = H x W
    campose: 4x4 homogeneous camera matrix
    bin_mask: binary segmentation mask 240 x 320, predicted
    abs_bbox: bounding box in absolute coordinates XYXY, predicted
    use_depth_box: use 3D bounding box of the depth pointcloud for IoU Matching
    '''

    nocs = np.array(nocs.cpu())  # HxWx3 in RGB format
    depth = np.array(depth, dtype=np.float32)  # HxW

    # Zero pad depth image
    depth_pad = np.zeros((240, 320))
    depth_pad[int(abs_bbox[1]):int(abs_bbox[3]), int(abs_bbox[0]):int(abs_bbox[2])] = depth[int(abs_bbox[1]):int(abs_bbox[3]), int(abs_bbox[0]):int(abs_bbox[2])]
    depth = depth_pad

    # Zero pad nocs image
    nocs_pad = np.zeros((240, 320, 3))
    nocs_pad[int(abs_bbox[1]):int(abs_bbox[3]), int(abs_bbox[0]):int(abs_bbox[2]), :] = nocs
    nocs = nocs_pad

    img_width = depth.shape[1]
    img_height = depth.shape[0]

    cx = (img_width / 2) - 0.5  # 0,0 is center top-left pixel -> -0,5
    cy = (img_height / 2) - 0.5  # 0,0 is center top-left pixel -> -0,5

    fx = 292.87803547399 # focal length from blenderproc # for fov=1 -> 292.8781
    fy = 292.87803547399

    '''
    Debug Depth 
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(img_width, img_height, fx, fy, cx, cy)
    depth = o3d.geometry.Image(depth)
    depth_p = o3d.geometry.PointCloud()
    depth_p = depth_p.create_from_depth_image(depth, intrinsics)
    o3d.visualization.draw_geometries([depth_p])
    '''

    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    depth_pts, idxs = backproject(depth, intrinsics, np.array(bin_mask.cpu()))

    # Clean depth
    if gt_3d_box is not None: #todo check if problem here
        new_depth_pts, new_idxs = clean_depth(depth_pts, gt_3d_box, campose) # clean depth for objects with holes
        if len(new_idxs) > 20:
            depth_pts = new_depth_pts
            idxs_x = idxs[0][new_idxs]
            idxs_y = idxs[1][new_idxs]
            idxs = (idxs_x, idxs_y)


    depth_pcd = o3d.geometry.PointCloud()
    depth_pcd.points = o3d.utility.Vector3dVector(depth_pts)

    nocs_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])
    #o3d.visualization.draw_geometries([depth_pcd, nocs_origin])


    # Clean depth
    if depth_pts.shape[0] > 100:
        cleaned_depth, ind = depth_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                                  std_ratio=2)
        clean_depth_pts = np.asarray(cleaned_depth.points)  # num points x 3
    else:
        clean_depth_pts = depth_pts
        cleaned_depth = depth_pcd
        ind = list(np.arange(clean_depth_pts.shape[0]))

    idxs_x = np.array(idxs[0])[ind]
    idxs_y = np.array(idxs[1])[ind]

    nocs_pts = nocs[idxs_x, idxs_y, :] - 0.5   # 0 centering -> back to cad space #todo wrong for y axis
    '''
    # To CAD space
    y_min = nocs_pts[:,1].min()
    nocs_pts[:,0] -= 0.5
    nocs_pts[:,2] -= 0.5
    nocs_pts[:,1] -= y_min
    '''

    nocs_pcd = o3d.geometry.PointCloud()
    nocs_pcd.points = o3d.utility.Vector3dVector(nocs_pts)

    nocs_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    #o3d.visualization.draw_geometries([nocs_pcd, nocs_origin])
    if gt_pc is not None:
        o3d.visualization.draw_geometries([nocs_pcd, nocs_origin, gt_pc])

    # Clean Nocs
    if nocs_pts.shape[0] > 100:
        cleaned_nocs, ind_nocs = nocs_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                                std_ratio=2)
        cleaned_nocs_pts = np.asarray(cleaned_nocs.points)
        # Truncate depth accordingly
        clean_depth_pts = clean_depth_pts[ind_nocs, :]
    else:
        cleaned_nocs = nocs_pcd
        cleaned_nocs_pts = nocs_pts

    '''
    # 3D Depth BBOX
    box_pts_depth = np.array(cleaned_depth.points)
    bbox3d_depth = o3d.geometry.OrientedBoundingBox()
    bbox_3d_depth = bbox3d_depth.create_from_points(o3d.utility.Vector3dVector(box_pts_depth))
    box_pts_depth_world = cam2world(np.array(bbox_3d_depth.get_box_points()), campose)
    '''

    # RANSAC & Umeyama
    outlier_thres = 1
    if cleaned_nocs_pts.shape[0] == 0:
        return None, None, None, None, None, None
    Scales, Rotation, Translation, _ = estimateSimilarityTransform(cleaned_nocs_pts, clean_depth_pts,
                                                                              verbose=False, ratio_adapt=outlier_thres) # CAD2CAM
    if Scales is None:
        return None, None, None, None, None, None
    transformed_nc_pc = transform_pc(Scales, Rotation, Translation, cleaned_nocs_pts) # Object space to camera space

    # Cam2world transform
    world_pc = cam2world(transformed_nc_pc, campose)

    # 3D BBOX
    bbox3d_obj = o3d.geometry.AxisAlignedBoundingBox()
    if use_depth_box:
        depth_world = cam2world(clean_depth_pts, campose)
        bbox_3d = bbox3d_obj.create_from_points(o3d.utility.Vector3dVector(depth_world))
        world_box = sort_bbox(np.array(bbox_3d.get_box_points()))
    else:
        bbox_3d = bbox3d_obj.create_from_points(o3d.utility.Vector3dVector(world_pc))
        world_box = sort_bbox(np.array(bbox_3d.get_box_points()))

    world_pc_obj = o3d.geometry.PointCloud()
    world_pc_obj.points = o3d.utility.Vector3dVector(world_pc)
    world_pc_obj.paint_uniform_color([0.6, 0.6, 0.6])

    if vis_obj:
        depth_world = cam2world(clean_depth_pts, campose)
        depth_pc_obj = o3d.geometry.PointCloud()
        depth_pc_obj.points = o3d.utility.Vector3dVector(depth_world)
        depth_pc_obj.paint_uniform_color([0.1, 0.1, 0.8])

        nocs_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0])

        vis_box = o3d.geometry.OrientedBoundingBox()
        vis_box = vis_box.create_from_points(o3d.utility.Vector3dVector(world_box))

        o3d.visualization.draw_geometries([depth_pc_obj, world_pc_obj, vis_box, nocs_origin])

    # Chain object to camera space and cam space to world space transformation matricies
    obj_tocam = np.identity(4) # CAD2Cam
    obj_tocam[:3,:3] = np.diag(Scales) @ Rotation.T #Rotation.T weirdly correct
    obj_tocam[:3,3] = Translation
    global_transform = campose @ obj_tocam # Cam2World @ CAD2Cam = CAD2World
    global_trans = global_transform[:3,3]
    global_rot = global_transform[:3,:3] # Scale already embedded into Rotation
    global_scale = Scales[0]


    depth_world = cam2world(clean_depth_pts, campose)

    return global_rot, global_trans, global_scale, world_box, depth_world, world_pc


def run_pose_office(nocs, depth, cam_intrinsics, bin_mask, abs_bbox, vis_obj=False, gt_pc=None, gt_3d_box=None, use_depth_box=True):
    '''
    Pose Estimation with Umeyama Algorithm and RANSAC outlier removal per Object
    nocs: predicted nocs map, shape of patch format HxWxC in RGB, normalized between 0 and 1
    depth: full depth 240 x 320 = H x W
    campose: 4x4 homogeneous camera matrix
    bin_mask: binary segmentation mask 240 x 320, predicted
    abs_bbox: bounding box in absolute coordinates XYXY
    use_depth_box: use 3D bounding box of the depth pointcloud for IoU Matching
    '''

    nocs = np.array(nocs.cpu())  # HxWx3 in RGB format
    depth = np.array(torch.squeeze(depth), dtype=np.float32)  # HxW

    # Zero pad depth image
    depth_pad = np.zeros((240, 320))
    depth_pad[int(abs_bbox[1]):int(abs_bbox[3]), int(abs_bbox[0]):int(abs_bbox[2])] = depth[int(abs_bbox[1]):int(abs_bbox[3]), int(abs_bbox[0]):int(abs_bbox[2])]
    depth = depth_pad

    # Zero pad nocs image
    nocs_pad = np.zeros((240, 320, 3))
    nocs_pad[int(abs_bbox[1]):int(abs_bbox[3]), int(abs_bbox[0]):int(abs_bbox[2]), :] = nocs
    nocs = nocs_pad

    depth_pts, idxs = backproject(depth, torch.squeeze(cam_intrinsics), np.array(bin_mask.cpu()))

    depth_pcd = o3d.geometry.PointCloud()
    depth_pcd.points = o3d.utility.Vector3dVector(depth_pts)

    # Clean depth
    if depth_pts.shape[0] > 100:
        cleaned_depth, ind = depth_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                                  std_ratio=2)
        clean_depth_pts = np.asarray(cleaned_depth.points)  # num points x 3
    else:
        clean_depth_pts = depth_pts
        cleaned_depth = depth_pcd
        ind = list(np.arange(clean_depth_pts.shape[0]))

    idxs_x = np.array(idxs[0])[ind]
    idxs_y = np.array(idxs[1])[ind]

    nocs_pts = nocs[idxs_x, idxs_y, :] - 0.5   # 0 centering -> back to cad space

    nocs_pcd = o3d.geometry.PointCloud()
    nocs_pcd.points = o3d.utility.Vector3dVector(nocs_pts)

    nocs_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    if gt_pc is not None:
        o3d.visualization.draw_geometries([nocs_pcd, nocs_origin, gt_pc])

    # Clean Nocs
    if nocs_pts.shape[0] > 100:
        cleaned_nocs, ind_nocs = nocs_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                                std_ratio=2)
        cleaned_nocs_pts = np.asarray(cleaned_nocs.points)
        # Truncate depth accordingly
        clean_depth_pts = clean_depth_pts[ind_nocs, :]
    else:
        cleaned_nocs = nocs_pcd
        cleaned_nocs_pts = nocs_pts

    # RANSAC & Umeyama
    outlier_thres = 1
    if cleaned_nocs_pts.shape[0] == 0:
        return None, None, None, None, None, None
    Scales, Rotation, Translation, _ = estimateSimilarityTransform(cleaned_nocs_pts, clean_depth_pts,
                                                                              verbose=False, ratio_adapt=outlier_thres) # CAD2CAM
    if Scales is None:
        return None, None, None, None, None, None
    transformed_nc_pc = transform_pc(Scales, Rotation, Translation, cleaned_nocs_pts) # Object space to camera space

    # 3D BBOX
    bbox3d_obj = o3d.geometry.AxisAlignedBoundingBox()
    if use_depth_box:
        depth_world = clean_depth_pts
        bbox_3d = bbox3d_obj.create_from_points(o3d.utility.Vector3dVector(depth_world))
        world_box = sort_bbox(np.array(bbox_3d.get_box_points()))
    else:
        bbox_3d = bbox3d_obj.create_from_points(o3d.utility.Vector3dVector(transformed_nc_pc))
        world_box = sort_bbox(np.array(bbox_3d.get_box_points()))

    world_pc_obj = o3d.geometry.PointCloud()
    world_pc_obj.points = o3d.utility.Vector3dVector(transformed_nc_pc)
    world_pc_obj.paint_uniform_color([0.6, 0.6, 0.6])

    # Chain object to camera space and cam space to world space transformation matricies
    obj_tocam = np.identity(4) # CAD2Cam
    obj_tocam[:3,:3] = np.diag(Scales) @ Rotation.T #Rotation.T weirdly correct
    obj_tocam[:3,3] = Translation
    global_transform = obj_tocam #
    global_trans = global_transform[:3,3]
    global_rot = global_transform[:3,:3] # Scale already embedded into Rotation
    global_scale = Scales[0]

    depth_world = clean_depth_pts

    return global_rot, global_trans, global_scale, world_box, depth_world, transformed_nc_pc


