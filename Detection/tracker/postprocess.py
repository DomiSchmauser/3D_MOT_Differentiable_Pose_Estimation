import sys, re
import numpy as np
import torch
import mathutils
import open3d as o3d

from detectron2.utils.visualizer import GenericMask
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.layers import roi_align

# required so that .register() calls are executed in module scope
import data
import roi_heads

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

from PoseEst.pose_estimation import run_pose, run_crop_3dbbox, sort_bbox, run_pose_office
from Detection.inference.inference_utils import get_scale, project_segmask_F2F


def postprocess_dets(inputs, outputs, obj_threshold=0.35, iou_threshold=0.35, mode='train', vis=False):
    '''
    Postprocessing module:
     - Crop GT bounding boxes (should be done only once at the start of the training)
     - Prune detections using objectness score and iou_overlap
     - Apply Umeyama for Pose Estimation
     - Return windowed tracking data in format: List(img_dicts) -> len(window_size)
    '''

    seq_window = []
    for img_gt, img_pred in zip(inputs, outputs):

        # GT annotations
        name = img_gt['file_name']
        seq_pattern = mode + "/(.*?)/coco_data"
        scan_pattern = "rgb_(.*?).png"
        seq_name = re.search(seq_pattern, name).group(1)
        scan_idx = int(re.search(scan_pattern, name).group(1))

        gt_2dbbox_anns = img_gt['boxes']
        gt_3dbbox_anns = img_gt['3dboxes']
        campose = img_gt['campose']
        depth = img_gt['depth_map']
        gt_segmaps = img_gt['segmap']
        gt_obj_ids = img_gt['object_id']
        gt_locations = img_gt['locations']
        gt_rotations = img_gt['rotations']
        gt_scales = img_gt['3dscales']
        gt_voxels = img_gt['instances'].get('gt_voxels') # load voxels
        gt_classes = img_gt['cat_id'] # load voxels

        # Crop GT bbox
        gt_bboxes, gt_ids, gt_locations, gt_compl_boxes = crop_gt_3dbox(gt_3dbbox_anns, gt_2dbbox_anns, gt_segmaps, gt_obj_ids, gt_locations, depth, campose)

        # Predictions by detection and pose network
        instances = img_pred['instances']
        clss = instances.get('pred_classes')  # Tensor len(num_instances)
        num_instances = len(clss)
        objectness_scores = instances.get('scores')

        bboxes = instances.get('pred_boxes')  # Boxes class 2D bbox
        bin_masks = instances.get('pred_masks')  # num_instances x img_H x img_W

        # Not useable batch if has no predicted voxel
        if not instances.has('pred_voxels'):
            print('Predicted voxels empty...')
            if not vis:
                seq_window.append({
                    'gt_object_id': torch.tensor(gt_ids),
                    'gt_3Dbbox': torch.cat(gt_bboxes, dim=0),
                    'gt_locations': torch.tensor(gt_locations),
                    'gt_classes': torch.tensor(gt_classes),
                    #'gt_rotations': torch.tensor(gt_rotations), # needed
                    #'gt_scales': torch.tensor(gt_rotations),  # needed
                    #'gt_voxels': gt_voxels, #needed
                    #'gt_compl_box': gt_compl_box, #needed
                    'image': scan_idx,
                    'scene': seq_name
                })
            else:
                seq_window.append({
                    'gt_object_id': torch.tensor(gt_ids),
                    'gt_3Dbbox': torch.cat(gt_bboxes, dim=0),
                    'gt_locations': torch.tensor(gt_locations),
                    'gt_rotations': gt_rotations,
                    'gt_scales': gt_scales,
                    'gt_voxels': gt_voxels,
                    'gt_compl_box': torch.cat(gt_compl_boxes, dim=0),
                    'gt_classes': torch.tensor(gt_classes),
                    'image': scan_idx,
                    'scene': seq_name
                })
            continue

        # Voxel preds in logits
        voxels = instances.get('pred_voxels').sigmoid()  # num_instances x 32 x 32 x 32
        voxels[voxels >= 0.5] = 1
        voxels[voxels < 0.5] = 0
        voxels = voxels.type(torch.float32)

        nocs = instances.get('pred_nocs')  # num_instances x RGB x 28 x 28

        # Store
        rotations = []
        translations = []
        scales = []
        pred_boxes = []
        classes = []
        pred_voxels = []
        for i in range(num_instances):  # number predicted instances

            obj_score = objectness_scores[i]
            if obj_score > obj_threshold:

                # Get object with maximum 2D IoU
                same_bbox = [BoxMode.convert(list(gt_2dbbox_anns[n]), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for n in
                             range(len(gt_2dbbox_anns))]  # XYWH_AB
                gt_target_boxes = Boxes(torch.tensor(same_bbox, dtype=torch.float32, device=torch.device("cuda")))
                pred_iou_box = bboxes[i].tensor.type(torch.float32)[0]
                pred_iou_box = Boxes(torch.unsqueeze(pred_iou_box, dim=0))

                ious = pairwise_iou(gt_target_boxes, pred_iou_box)
                idx_max_iou = torch.argmax(ious)
                max_iou = ious[idx_max_iou]

                if max_iou >= iou_threshold:

                    gt_bbox_loc = torch.squeeze(gt_bboxes[idx_max_iou]) # formate needs to be 8x3
                    # Pose Estimation Module ---------------------------------------------------------------------------
                    noc = nocs[i, :, :, :]  # C x 28 x 28
                    abs_bbox = bboxes[i].tensor.type(torch.int)[0]  # pred box

                    patch_heigth = torch.abs(abs_bbox[1] - abs_bbox[3])
                    patch_width = torch.abs(abs_bbox[0] - abs_bbox[2])

                    if patch_width < 3 and patch_heigth < 3:  # remove instances with patch size < 5
                        print('Predicted obj bounding box to small skipping ...')
                        continue

                    noc_bbox = [torch.unsqueeze(
                        torch.tensor([0, 0, noc.shape[1], noc.shape[2]], dtype=torch.float32,
                                     device=torch.device("cuda")),
                        dim=0)]
                    reshape_nocs = roi_align(torch.unsqueeze(noc.to(device=torch.device("cuda")), dim=0), noc_bbox,
                                             output_size=(patch_heigth, patch_width), aligned=True)
                    reshape_nocs = torch.squeeze(reshape_nocs, dim=0).permute(1, 2, 0).contiguous()  # HxWxC


                    global_rot, global_trans, global_scale, pred_3d_bbox, \
                    _, _ = run_pose(reshape_nocs.detach(), depth, campose, bin_masks[i, :, :],
                                                              abs_bbox, gt_3d_box=gt_bbox_loc, use_depth_box=True)

                    if global_rot is None or global_rot.size == 0 or global_trans.size == 0 or global_scale.size == 0:
                        #print('Skip instances due to fail in umeyama...')
                        continue

                    unscaled_rot = global_rot / get_scale(global_rot)
                    r = mathutils.Matrix(unscaled_rot)
                    euler = np.array(r.to_euler())  # X,Y,Z

                    rotations.append(torch.unsqueeze(torch.from_numpy(euler), dim=0))
                    translations.append(torch.unsqueeze(torch.from_numpy(global_trans), dim=0))
                    scales.append(global_scale)
                    pred_boxes.append(torch.unsqueeze(torch.from_numpy(pred_3d_bbox), dim=0))
                    pred_voxels.append(voxels[i:i+1,:,:,:])
                    classes.append(clss[i])

                    # End Pose Estimation Module -----------------------------------------------------------------------

        # Cleanup empty objects

        # No object predictions in that scan
        if rotations and not vis:
            img_dict = {'classes': classes,
                        'rotations': torch.cat(rotations, dim=0).to(torch.device("cuda")),
                        'translations': torch.cat(translations, dim=0).to(torch.device("cuda")),
                        'scales': torch.tensor(scales).to(torch.device("cuda")),
                        'voxels': torch.cat(pred_voxels, dim=0),
                        'pred_3Dbbox': torch.cat(pred_boxes, dim=0),
                        'gt_object_id': torch.tensor(gt_ids),
                        'gt_3Dbbox': torch.cat(gt_bboxes, dim=0),
                        'gt_locations': torch.tensor(gt_locations),
                        'gt_classes': torch.tensor(gt_classes),
                        # 'gt_rotations': torch.tensor(gt_rotations), # needed
                        # 'gt_scales': torch.tensor(gt_rotations),  # needed
                        # 'gt_voxels': gt_voxels, #needed
                        # 'gt_compl_box': gt_compl_box, #needed
                        'image': scan_idx,
                        'scene': seq_name
                        }
        elif rotations and vis:
            img_dict = {'classes': classes,
                        'rotations': torch.cat(rotations, dim=0).to(torch.device("cuda")),
                        'translations': torch.cat(translations, dim=0).to(torch.device("cuda")),
                        'scales': torch.tensor(scales).to(torch.device("cuda")),
                        'voxels': torch.cat(pred_voxels, dim=0),
                        'pred_3Dbbox': torch.cat(pred_boxes, dim=0),
                        'gt_object_id': torch.tensor(gt_ids),
                        'gt_3Dbbox': torch.cat(gt_bboxes, dim=0),
                        'gt_locations': torch.tensor(gt_locations),
                        'gt_rotations': gt_rotations,
                        'gt_scales': gt_scales,
                        'gt_voxels': gt_voxels,
                        'gt_compl_box': torch.cat(gt_compl_boxes, dim=0),
                        'gt_classes': torch.tensor(gt_classes),
                        'image': scan_idx,
                        'scene': seq_name
                        }
        elif not rotations and not vis:
            print('No object detections in this sequence {} in frame {}'.format(seq_name, scan_idx))
            img_dict = {
                'gt_object_id': torch.tensor(gt_ids),
                'gt_3Dbbox': torch.cat(gt_bboxes, dim=0),
                'gt_locations': torch.tensor(gt_locations),
                'gt_classes': torch.tensor(gt_classes),
                #'gt_rotations': torch.tensor(gt_rotations),
                #'gt_voxels': gt_voxels,
                'image': scan_idx,
                'scene': seq_name
            }
        elif not rotations and vis:
            print('No object detections in this sequence {} in frame {}'.format(seq_name, scan_idx))
            img_dict = {
                'gt_object_id': torch.tensor(gt_ids),
                'gt_3Dbbox': torch.cat(gt_bboxes, dim=0),
                'gt_locations': torch.tensor(gt_locations),
                'gt_rotations': gt_rotations,
                'gt_scales': gt_scales,
                'gt_voxels': gt_voxels,
                'gt_classes': torch.tensor(gt_classes),
                'gt_compl_box': torch.cat(gt_compl_boxes, dim=0),
                'image': scan_idx,
                'scene': seq_name
            }
        seq_window.append(img_dict)

    return seq_window

def postprocess_dets_office(inputs, outputs, obj_threshold=0.01, mode='train'):
    '''
    Postprocessing module:
     - Crop GT bounding boxes (should be done only once at the start of the training)
     - Prune detections using objectness score and iou_overlap
     - Apply Umeyama for Pose Estimation
     - Return windowed tracking data in format: List(img_dicts) -> len(window_size)
    '''

    seq_window = []
    for img_gt, img_pred in zip(inputs, outputs):

        # GT annotations
        scan_idx = None
        seq_name = None
        name = img_gt['seq_id'][0]
        depth = img_gt['depth']
        camera_intrinsics = img_gt['camera_intrinsics']

        # Predictions by detection and pose network
        instances = img_pred[0]['instances']
        clss = instances.get('pred_classes')  # Tensor len(num_instances)
        num_instances = len(clss)
        objectness_scores = instances.get('scores')

        bboxes = instances.get('pred_boxes')  # Boxes class 2D bbox
        bin_masks = instances.get('pred_masks')  # num_instances x img_H x img_W

        # Not useable batch if has no predicted voxel
        if not instances.has('pred_voxels'):
            print('Predicted voxels empty...')
            seq_window.append(None)
            continue

        # Voxel preds in logits
        voxels = instances.get('pred_voxels').sigmoid()  # num_instances x 32 x 32 x 32
        voxels[voxels >= 0.5] = 1
        voxels[voxels < 0.5] = 0
        voxels = voxels.type(torch.float32)

        nocs = instances.get('pred_nocs')  # num_instances x RGB x 28 x 28

        # Store
        rotations = []
        translations = []
        scales = []
        pred_boxes = []
        classes = []
        pred_voxels = []
        for i in range(num_instances):  # number predicted instances

            obj_score = objectness_scores[i]
            if obj_score > obj_threshold:

                # Pose Estimation Module ---------------------------------------------------------------------------
                noc = nocs[i, :, :, :]  # C x 28 x 28
                abs_bbox = bboxes[i].tensor.type(torch.int)[0]  # pred box

                patch_heigth = torch.abs(abs_bbox[1] - abs_bbox[3])
                patch_width = torch.abs(abs_bbox[0] - abs_bbox[2])

                if patch_width < 3 and patch_heigth < 3:  # remove instances with patch size < 5
                    print('Predicted obj bounding box to small skipping ...')
                    continue

                noc_bbox = [torch.unsqueeze(
                    torch.tensor([0, 0, noc.shape[1], noc.shape[2]], dtype=torch.float32,
                                 device=torch.device("cuda")),
                    dim=0)]
                reshape_nocs = roi_align(torch.unsqueeze(noc.to(device=torch.device("cuda")), dim=0), noc_bbox,
                                         output_size=(patch_heigth, patch_width), aligned=True)
                reshape_nocs = torch.squeeze(reshape_nocs, dim=0).permute(1, 2, 0).contiguous()  # HxWxC


                global_rot, global_trans, global_scale, pred_3d_bbox, \
                _, _ = run_pose_office(reshape_nocs.detach(), depth, camera_intrinsics, bin_masks[i, :, :],
                                                          abs_bbox, gt_3d_box=None, use_depth_box=True)

                if global_rot is None or global_rot.size == 0 or global_trans.size == 0 or global_scale.size == 0:
                    continue

                unscaled_rot = global_rot / get_scale(global_rot)
                r = mathutils.Matrix(unscaled_rot)
                euler = np.array(r.to_euler())  # X,Y,Z

                rotations.append(torch.unsqueeze(torch.from_numpy(euler), dim=0))
                translations.append(torch.unsqueeze(torch.from_numpy(global_trans), dim=0))
                scales.append(global_scale)
                pred_boxes.append(torch.unsqueeze(torch.from_numpy(pred_3d_bbox), dim=0))
                pred_voxels.append(voxels[i:i+1,:,:,:])
                classes.append(clss[i])

                # End Pose Estimation Module -----------------------------------------------------------------------

        # Cleanup empty objects

        # No object predictions in that scan
        if rotations:
            img_dict = {'classes': classes,
                        'rotations': torch.cat(rotations, dim=0).to(torch.device("cuda")),
                        'translations': torch.cat(translations, dim=0).to(torch.device("cuda")),
                        'scales': torch.tensor(scales).to(torch.device("cuda")),
                        'voxels': torch.cat(pred_voxels, dim=0),
                        'pred_3Dbbox': torch.cat(pred_boxes, dim=0),
                        'image': scan_idx,
                        'scene': seq_name
                        }
        else:
            print('No object detections in this sequence {} in frame {}'.format(seq_name, scan_idx))
            img_dict = None
        seq_window.append(img_dict)

    return seq_window

def postprocess_dets_office_F2F(img_gt, img_pred, obj_threshold=0.01, mode='train'):
    '''
    Postprocessing module:
     - Crop GT bounding boxes (should be done only once at the start of the training)
     - Prune detections using objectness score and iou_overlap
     - Apply Umeyama for Pose Estimation
     - Return windowed tracking data in format: List(img_dicts) -> len(window_size)
    '''

    # GT annotations
    name = img_gt['seq_id'][0]
    depth = torch.squeeze(img_gt['depth'])
    camera_intrinsics = torch.squeeze(img_gt['camera_intrinsics'])
    frame_objs = []

    # Predictions by detection and pose network
    instances = img_pred[0]['instances']
    clss = instances.get('pred_classes')  # Tensor len(num_instances)
    num_instances = len(clss)
    objectness_scores = instances.get('scores')

    bboxes = instances.get('pred_boxes')  # Boxes class 2D bbox
    bin_masks = instances.get('pred_masks')  # num_instances x img_H x img_W

    # Not useable batch if has no predicted voxel
    if not instances.has('pred_voxels'):
        print('Predicted voxels empty...')
        return None

    # Store
    classes = []

    for i in range(num_instances):  # number predicted instances

        obj_score = objectness_scores[i]
        if obj_score > obj_threshold:

            abs_bbox = bboxes[i].tensor.type(torch.int)[0]  # pred box
            pred_bin_mask = bin_masks[i]
            classes.append(clss[i])

            world_segpc = project_segmask_F2F(pred_bin_mask, abs_bbox, depth, camera_intrinsics)
            frame_objs.append(world_segpc)

    '''
    vis_objs = []
    for frame_obj in frame_objs:
        pc_obj = o3d.geometry.PointCloud()
        pc_obj.points = o3d.utility.Vector3dVector(frame_obj)
        vis_objs.append(pc_obj)
    vis_objs += [o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])]
    o3d.visualization.draw_geometries(vis_objs)
    '''

    if frame_objs:
        return {'segpc':frame_objs, 'pred_classes': classes}
    else:
        return {'segpc':None,  'pred_classes': classes}


def crop_gt_3dbox(gt_3dbbox_anns, gt_2dbbox_anns, gt_segmaps, gt_obj_ids, gt_locations_in, depth, campose):
    '''
    Crops 3D box based on depth map per object -> 3D boxes which are not fully visible in the image
    Returns according cropped box and object id
    '''

    gt_bboxes = []
    gt_ids = []
    gt_locations = []
    gt_compl_boxes = []

    for m, gt_3dbbox_ann in enumerate(gt_3dbbox_anns): # number of gt_instances

        segmap = gt_segmaps[m]
        gt_id = int(gt_obj_ids[m])
        gm = GenericMask(segmap, 240, 320)
        gt_bin_mask = gm.polygons_to_mask(segmap)
        gt_location = gt_locations_in[m]

        # skip empty seg mask
        if gt_bin_mask.max() == 0:
            gt_bboxes.append(torch.unsqueeze(torch.from_numpy(sort_bbox(gt_3dbbox_ann)), dim=0)) # sort in counter clockwise order
            gt_compl_boxes.append(torch.unsqueeze(torch.from_numpy(sort_bbox(gt_3dbbox_ann)), dim=0)) # sort in counter clockwise order
            gt_ids.append(gt_id)
            gt_locations.append(gt_location)
            continue

        gt_2dbbox_ann = np.array(BoxMode.convert(gt_2dbbox_anns[m], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS))
        crop_gt_3d_bbox = run_crop_3dbbox(depth, campose, gt_3dbbox_ann, gt_2dbbox_ann, gt_bin_mask)
        gt_compl_boxes.append(torch.unsqueeze(torch.from_numpy(sort_bbox(gt_3dbbox_ann)), dim=0))
        gt_bboxes.append(torch.unsqueeze(torch.from_numpy(crop_gt_3d_bbox), dim=0))
        gt_ids.append(gt_id)
        gt_locations.append(gt_location)

    return gt_bboxes, gt_ids, gt_locations, gt_compl_boxes