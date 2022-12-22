import cv2, os, sys, json
import h5py
import numpy as np
import torch
import mathutils
import matplotlib.pyplot as plt
import traceback
import logging

from Detection.utils.cfg_setup import get_inference_cfg
from Detection.data.analyse_dataset import get_dataset_info
from Detection.data.register_dataset import RegisterDataset

from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import GenericMask
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.layers import roi_align
import open3d as o3d

# required so that .register() calls are executed in module scope

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

from Detection.pose.pose_estimation import run_pose, run_crop_3dbbox, sort_bbox
from Detection.utils.inference_metrics import compute_voxel_iou, get_rotation_diff, get_location_diff
from Detection.utils.inference_utils import load_hdf5, log_results, convert_voxel_to_pc, add_halfheight, get_nocs, \
    get_scale, visualize_segmentation_results, get_annotations
from Detection.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def inference_on_sequence(instances, depth, campose, gt_annotations , hdf5_dir, idx, img,
                    write_files=True, visualize_pose=False, visualize_voxel=False):
    '''
    Main Inference function: generates voxel and pose predictions and stores them together with meta data in a
    per image hdf5 file
    '''


    output = {}

    # GT Annotations
    gt_2dbbox_anns = gt_annotations['2Dbbox']
    gt_nocsmap = gt_annotations['nocs']
    gt_bboxes = []
    gt_ids = []
    gt_voxels = []
    gt_rotations = []
    gt_locations = []
    gt_compl_boxes = []
    gt_scales = []
    gt_classes = []

    # Crops GT 3D bounding box based on depth map per object
    for m, gt_3dbbox_ann in enumerate(
            gt_annotations['3Dbbox']):  # number of gt_instances

        segmap = gt_annotations['segmentation'][m]
        gt_id = int(gt_annotations['obj_id'][m])
        gm = GenericMask(segmap, 240, 320)
        gt_bin_mask = gm.polygons_to_mask(segmap)
        gt_voxel = gt_annotations['voxel'][m]
        gt_rotation = gt_annotations['3Drot'][m]
        gt_location = gt_annotations['3Dloc'][m]
        gt_scale = gt_annotations['3Dscale'][m]
        gt_compl_boxes.append(sort_bbox(gt_3dbbox_ann))
        gt_scales.append(gt_scale)
        gt_classes.append(gt_annotations['cls'][m])

        if gt_bin_mask.max() == 0:  # skip empty seg mask
            # gt_bboxes.append(np.zeros((8,3)))
            gt_bboxes.append(sort_bbox(gt_3dbbox_ann))  # sort in counter clockwise order
            gt_ids.append(gt_id)
            gt_voxels.append(torch.unsqueeze(gt_voxel, dim=0))
            gt_rotations.append(gt_rotation)
            gt_locations.append(gt_location)
            # print('Sequence {} has empty box, use for debugging'.format(gt_annotations['seq']))
            continue

        gt_2dbbox_ann = np.array(BoxMode.convert(gt_2dbbox_anns[m], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS))
        crop_gt_3d_bbox = run_crop_3dbbox(depth, campose, gt_3dbbox_ann, gt_2dbbox_ann,
                                          gt_bin_mask)  # for 3D box which is not fully visible in img
        # Store GT data
        gt_bboxes.append(crop_gt_3d_bbox)
        gt_ids.append(gt_id)
        gt_voxels.append(torch.unsqueeze(gt_voxel, dim=0))
        gt_rotations.append(gt_rotation)
        gt_locations.append(gt_location)

    # Predictions by detection and pose network
    clss = instances.get('pred_classes') #Tensor len(num_instances)
    num_instances = len(clss)
    objectness_scores = instances.get('scores')

    bboxes = instances.get('pred_boxes') # Boxes class 2D bbox
    bin_masks = instances.get('pred_masks') # num_instances x img_H x img_W

    # ROI Align pred nocs
    rotations = []
    translations = []
    scales = []
    pred_bboxes = []

    # Metrics data
    voxel_metrics = {'voxel_ious': [], 'chair_ious': [], 'table_ious': [], 'sofa_ious': [], 'bed_ious': [],
                     'tv_stand_ious': [], 'cooler_ious': [], 'night_stand_ious': []}
    distances = []
    rotation_diff = []
    rm_indicies = []  # Broken objects which will be removed from the predictions

    if instances.has('pred_voxels'):
        voxels = instances.get('pred_voxels').sigmoid() # num_instances x 32 x 32 x 32
        voxels[voxels >= 0.5] = 1
        voxels[voxels < 0.5] = 0
        voxels = voxels.type(torch.float32)

        nocs = instances.get('pred_nocs') # num_instances x RGB x 28 x 28

        # Visualization data
        gt_visobjects = []
        pred_visobjects = []
        pred_pcobjects = []

        for i in range(num_instances): # number predicted instances

            obj_score = objectness_scores[i]
            obj_threshold = 0.35
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

                if max_iou >= 0.35:

                    # Get GT according to max IoU
                    gt_voxel = gt_annotations['voxel'][idx_max_iou]
                    gt_clsobj = gt_annotations['cls'][idx_max_iou]
                    gt_bbox_loc = gt_bboxes[idx_max_iou]
                    gt_rotation = gt_annotations['3Drot'][idx_max_iou] #Cad2World
                    gt_location = gt_annotations['3Dloc'][idx_max_iou] #Cad2World

                    # pose Estimation Module --------------------------------------------------------------------------------
                    noc = nocs[i, :, :, :]  # C x 28 x 28
                    abs_bbox = bboxes[i].tensor.type(torch.int)[0]  # pred box

                    patch_heigth = torch.abs(abs_bbox[1] - abs_bbox[3])
                    patch_width = torch.abs(abs_bbox[0] - abs_bbox[2])

                    if patch_width < 5 and patch_heigth < 5:  # remove instances with patch size < 5
                        rm_indicies.append(i)
                        continue

                    noc_bbox = [torch.unsqueeze(
                        torch.tensor([0, 0, noc.shape[1], noc.shape[2]], dtype=torch.float32, device=torch.device("cuda")),
                        dim=0)]
                    reshape_nocs = roi_align(torch.unsqueeze(noc.to(device=torch.device("cuda")), dim=0), noc_bbox,
                                             output_size=(patch_heigth, patch_width), aligned=True)
                    reshape_nocs = torch.squeeze(reshape_nocs, dim=0).permute(1, 2, 0).contiguous()  # HxWxC

                    '''
                    cv2.imshow('Nocs Pred Patch', reshape_nocs.cpu().detach().numpy())
                    cv2.waitKey(0)
                    '''

                    # Noc2World Transforms, Rotation includes scale!!
                    global_rot, global_trans, global_scale, pred_3d_bbox, \
                    gt_pointcloud, pred_pointcloud = run_pose(reshape_nocs, depth, campose, bin_masks[i, :, :], abs_bbox, gt_3d_box=gt_bbox_loc, use_depth_box=True)

                    if global_rot is None or global_rot.size == 0 or global_trans.size == 0 or global_scale.size == 0:
                        rm_indicies.append(i)
                        continue

                    unscaled_rot = global_rot / get_scale(global_rot)
                    r = mathutils.Matrix(unscaled_rot)
                    euler = np.array(r.to_euler())  # X,Y,Z
                    # End pose Estimation Module ----------------------------------------------------------------------------

                    '''
                    # For debugging vis GT nocs
                    gt_nocs = crop_segmask_gt(gt_nocsmap, abs_bbox, gt_annotations['segmentation'][idx_max_iou]) # H x W x C
                    idxs = np.where(gt_nocs.detach().cpu().numpy() != 1)[:-1]
                    nc_pts = gt_nocs[idxs[0], idxs[1], :] - 0.5
                    nocs_pcd = o3d.geometry.PointCloud()
                    nocs_pcd.points = o3d.utility.Vector3dVector(nc_pts.detach().cpu().numpy())
                    nocs_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, -0.07, 0])
                    o3d.visualization.draw_geometries([nocs_pcd, nocs_origin])
                    global_rot, global_trans, global_scale, pred_3d_bbox, \
                    gt_pointcloud, pred_pointcloud = run_pose(gt_nocs, depth, campose, bin_masks[i, :, :], abs_bbox, gt_pc=nocs_pcd, gt_3d_box=gt_bbox_loc)
                    '''

                    # Visualize pose Estimation
                    if visualize_pose and gt_pointcloud is not None:
                        # Projected depth as GT
                        gt_pc_obj = o3d.geometry.PointCloud()
                        gt_pc_obj.points = o3d.utility.Vector3dVector(gt_pointcloud)
                        gt_pc_obj.paint_uniform_color([0.1, 0.1, 0.6])  # GT Blue
                        gt_visobjects.append(gt_pc_obj)

                        # Predicted Pointcloud from Nocs
                        pred_pc_obj = o3d.geometry.PointCloud()
                        pred_pc_obj.points = o3d.utility.Vector3dVector(pred_pointcloud)
                        pred_pc_obj.paint_uniform_color([0.1, 0.6, 0.1])  # PC Green
                        pred_pcobjects.append(pred_pc_obj)

                        #voxel_grid = voxels[idx_max_iou, :, :, :] predicted voxel instead of GT
                        world_pc = convert_voxel_to_pc(gt_voxel, global_rot, global_trans, global_scale)
                        world_pc_obj = o3d.geometry.PointCloud()
                        world_pc_obj.points = o3d.utility.Vector3dVector(world_pc)
                        world_pc_obj.paint_uniform_color([0.6, 0.3, 0.05]) #Pred brown
                        pred_visobjects.append(world_pc_obj)

                    if visualize_voxel:
                        if idx == 0:
                            im = img[int(abs_bbox[1]):int(abs_bbox[3]),int(abs_bbox[0]):int(abs_bbox[2]),:]
                            plt.imshow(im)
                            # and plot everything
                            ax = plt.figure().add_subplot(projection='3d')
                            ax.voxels(pred_voxel.permute(2, 0, 1).contiguous(), facecolors='whitesmoke', edgecolor='k')
                            ax.view_init(elev=31, azim=26)
                            plt.show()

                            ax = plt.figure().add_subplot(projection='3d')
                            ax.voxels(gt_voxel.permute(2, 0, 1).contiguous(), facecolors='dodgerblue', edgecolor='k')
                            ax.view_init(elev=31, azim=26)
                            plt.show()

                    pred_voxel = voxels[i, :, :, :]
                    #dvis(pred_voxel, c=2, vs=1)
                    voxel_iou = compute_voxel_iou(pred_voxel.to(torch.device("cuda")), gt_voxel.to(torch.device("cuda")))
                    voxel_metrics['voxel_ious'].append(voxel_iou)
                    if gt_clsobj == 1:
                        voxel_metrics['chair_ious'].append(voxel_iou)
                    elif gt_clsobj == 2:
                        voxel_metrics['table_ious'].append(voxel_iou)
                    elif gt_clsobj == 3:
                        voxel_metrics['sofa_ious'].append(voxel_iou)
                    elif gt_clsobj == 4:
                        voxel_metrics['bed_ious'].append(voxel_iou)
                    elif gt_clsobj == 5:
                        voxel_metrics['tv_stand_ious'].append(voxel_iou)
                    elif gt_clsobj == 6:
                        voxel_metrics['cooler_ious'].append(voxel_iou)
                    elif gt_clsobj == 7:
                        voxel_metrics['night_stand_ious'].append(voxel_iou)
                    else:
                        raise Exception('GT class id is not known!')

                    # Translation
                    dist = get_location_diff(np.array(gt_location), global_trans)
                    distances.append(dist)

                    # Rotation
                    theta = get_rotation_diff(gt_rotation, unscaled_rot) #cad2world
                    rotation_diff.append(theta)

                    rotations.append(euler)
                    translations.append(global_trans)
                    scales.append(global_scale)
                    pred_bboxes.append(pred_3d_bbox)

                else:
                    rm_indicies.append(i)
            else:
                rm_indicies.append(i)

    if visualize_pose and (idx == 0 or idx == 12):
        nocs_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0])
        combined = gt_visobjects + pred_visobjects + pred_pcobjects
        combined.append(nocs_origin)
        o3d.visualization.draw_geometries(combined)

    for vox_key, vox_val in voxel_metrics.items():
        if vox_val:
            mean_iou = np.array(vox_val).mean()
            output.update({vox_key:mean_iou})

    if rotation_diff:
        mean_rotation_diff = np.array(rotation_diff)
        output.update({'pose_rotationdiff': mean_rotation_diff})
    if distances:
        mean_distance = np.array(distances)
        output.update({'pose_distance': mean_distance})

    # delete empty objects
    if rm_indicies: # move to cpu
        voxels = voxels.detach().cpu().numpy()
        clss = clss.detach().cpu().numpy()
        objectness_scores = objectness_scores.detach().cpu().numpy()

    rm_indicies.sort(reverse=True) # delete in reverse order
    for rm_idx in rm_indicies:
        clss = np.delete(clss, rm_idx, axis=0)
        objectness_scores = np.delete(objectness_scores, rm_idx, axis=0)
        voxels = np.delete(voxels, rm_idx, axis=0)

    if rm_indicies: # move back to cuda
        voxels = torch.tensor(voxels).to(dtype=torch.float32, device=torch.device("cuda"))
        objectness_scores = torch.tensor(objectness_scores).to(dtype=torch.float32, device=torch.device("cuda"))
        clss = torch.tensor(clss).to(dtype=torch.int, device=torch.device("cuda"))

    # Deprecated else but keep for safety
    if gt_voxels:
        gt_voxels = torch.cat(gt_voxels, dim=0)
    else:
        gt_voxels = torch.tensor([])

    assert len(rotations) == len(translations) == len(scales)

    if len(rotations) == 0:
        print('Empty predictions store None for predicted...')

    # Write Hdf5 file for tracking
    if write_files and not len(rotations) == 0:
        fname = os.path.join(hdf5_dir, str(idx) + '.h5')
        hf = h5py.File(fname, 'w')
        hf.create_dataset('classes', data=clss.cpu())
        hf.create_dataset('objectness_scores', data=objectness_scores.cpu())
        hf.create_dataset('voxels', data=voxels.cpu())
        hf.create_dataset('rotations', data=rotations)
        hf.create_dataset('translations', data=translations)
        hf.create_dataset('scales', data=scales)
        hf.create_dataset('pred_3Dbbox', data=pred_bboxes)
        hf.create_dataset('gt_3Dbbox', data=gt_bboxes) # all gt 3dbboxes
        hf.create_dataset('gt_objid', data=gt_ids) # all gt ids according to boxes
        hf.create_dataset('gt_voxels', data=gt_voxels) # gt voxel retrieval
        hf.create_dataset('gt_rotations', data=gt_rotations) # gt rotation for vis tracking
        hf.create_dataset('gt_locations', data=gt_locations) # gt location for vis tracking
        hf.create_dataset('gt_scales', data=gt_scales) # gt location for vis tracking
        hf.create_dataset('gt_compl_box', data=gt_compl_boxes) # gt location for vis tracking
        hf.create_dataset('gt_cls', data=gt_classes) # gt location for vis tracking
        hf.close()
    elif write_files and len(rotations) == 0:
        filler = np.array([])
        fname = os.path.join(hdf5_dir, str(idx) + '.h5')
        hf = h5py.File(fname, 'w')
        hf.create_dataset('classes', data=filler)
        hf.create_dataset('objectness_scores', data=filler)
        hf.create_dataset('voxels', data=filler)
        hf.create_dataset('rotations', data=filler)
        hf.create_dataset('translations', data=filler)
        hf.create_dataset('scales', data=filler)
        hf.create_dataset('pred_3Dbbox', data=filler)
        hf.create_dataset('gt_3Dbbox', data=gt_bboxes)  # all gt 3dbboxes
        hf.create_dataset('gt_objid', data=gt_ids)  # all gt ids according to boxes
        hf.create_dataset('gt_voxels', data=gt_voxels)  # gt voxel retrieval
        hf.create_dataset('gt_rotations', data=gt_rotations)  # gt rotation for vis tracking
        hf.create_dataset('gt_locations', data=gt_locations)  # gt location for vis tracking
        hf.create_dataset('gt_scales', data=gt_scales)  # gt location for vis tracking
        hf.create_dataset('gt_compl_box', data=gt_compl_boxes)  # gt location for vis tracking
        hf.create_dataset('gt_cls', data=gt_classes)  # gt location for vis tracking
        hf.close()
    return output

def run_inference(split, write_files=True, overwrite=False):

    inference_results = []
    data_dir = os.path.join(CONF.PATH.DETECTDATA, split)
    datafolders = [f for f in os.listdir(os.path.abspath(data_dir))]

    for seq_idx, seq in enumerate(datafolders):
        seq_pred = []

        if (seq_idx + 1) % 50 == 0 or seq_idx == len(datafolders) - 1:
            logger.info(f"Inference for sequence {seq_idx+1} of {len(datafolders)} sequences.")
            log_results(inference_results)

        png_path = os.path.join(data_dir, seq, 'coco_data')
        hdf5_dir = os.path.join(CONF.PATH.TRACKDATA, split, seq)

        if not os.path.exists(os.path.join(CONF.PATH.TRACKDATA, split)):
            os.mkdir(os.path.join(CONF.PATH.TRACKDATA, split))

        if not overwrite and os.path.exists(hdf5_dir):
            continue

        if not os.path.isdir(hdf5_dir):
            os.mkdir(hdf5_dir)

        imgs = [f for f in os.listdir(os.path.abspath(png_path)) if not 'json' in f and not 'nocs' in f]
        imgs.sort()

        json_file = os.path.join(data_dir, seq, "coco_data/coco_annotations.json")
        with open(json_file) as f:
            imgs_anns = json.load(f)

        for i, img in enumerate(imgs):

            gt_annotations = get_annotations(imgs_anns, png_path, seq, i)
            depth, campose = load_hdf5(os.path.join(data_dir, seq, str(i) + '.hdf5'))
            img = np.array(cv2.imread(os.path.join(png_path, img)))  # BGR
            instances = predictor(img)['instances']
            try:
                output = inference_on_sequence(instances, depth, campose, gt_annotations, hdf5_dir, i, img,
                                         write_files=write_files, visualize_pose=False, visualize_voxel=False)
                seq_pred.append(output)
            except:
                logger.error(f"Inference on sequence {seq} failed.")
                traceback.print_exc()
                if os.path.exists('broken_scenes.txt'):
                    append_write = 'a'
                else:
                    append_write = 'w'
                with open('broken_scenes.txt', append_write) as f:
                    f.write('{}\n'.format(seq))
                break

        inference_results.append(seq_pred)
    return inference_results

def setup(dataset_split="test", num_classes=7):
    '''
    Inference of the Detection, Reconstruction and Pose Estimation Pipeline
     - Loads a pretrained network (best_model.pth) from the model folder
     - Stores inference results in the predicted_data folder in a hdf5 format

    num_classes: Number of object classes in the dataset the model was trained on
    '''

    # Dataset setup
    TRAIN_IMG_DIR = CONF.PATH.DETECTTRAIN

    mapping_list, name_list = get_dataset_info(TRAIN_IMG_DIR)
    mapping_list, name_list = zip(*sorted(zip(mapping_list, name_list)))

    if isinstance(name_list, tuple) and isinstance(mapping_list, tuple):
        name_list = list(name_list)
        mapping_list = list(mapping_list)

    register_dataset = RegisterDataset(mapping_list, name_list)
    register_dataset.register_to_catalog()

    # Create predictor
    my_cfg = get_inference_cfg(num_classes)
    predictor = DefaultPredictor(my_cfg)
    dataset_dicts = RegisterDataset.get_front_dicts(os.path.join(TRAIN_IMG_DIR[:-6], dataset_split))
    return predictor, dataset_dicts

if __name__=="__main__":
    SPLIT = "test"
    VISUALIZE_SEGMENTATIONS = False
    setup_logging()
    predictor, dataset_dicts = setup(dataset_split=SPLIT, num_classes=7)

    # Visualize Segmentation predictions
    if VISUALIZE_SEGMENTATIONS:
        visualize_segmentation_results(predictor, dataset_dicts, SPLIT)

    # Make predictions
    inference_results = run_inference(SPLIT, write_files=True, overwrite=True)

    logger.info(f"Finished inference on {SPLIT} set, results: ")
    log_results(inference_results)
