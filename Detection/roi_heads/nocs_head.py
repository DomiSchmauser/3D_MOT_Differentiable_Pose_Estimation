# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
import numpy as np
import mathutils
from detectron2.layers import ShapeSpec, cat, roi_align
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.structures import BitMasks
from torch import nn
from typing import Dict
import sys
sys.path.append('..') #Hack add ROOT DIR
from Detection.utils.train_utils import init_weights, symmetry_smooth_l1_loss, symmetry_bin_loss, crop_nocs, nocs_prob_to_value
from Detection.utils.train_utils import PoseLoss
from Detection.inference.inference_utils import vox2pc
from PoseEst.pose_estimation import run_pose, run_pose_torch

import matplotlib.pyplot as plt

ROI_NOCS_HEAD_REGISTRY = Registry("ROI_NOCS_HEAD")


def nocs_loss(pred_nocsmap, instances, pred_boxes, l1_loss_weight=3, pose_loss_weight=1.5,
              iou_thres=0.5, cls_mapping=None, use_bin_loss=False, num_bins=32, start_iter_pose=None, current_iter=None):
    '''
    Calculate loss between predicted and gt nocs map if same category id and max IoU box > threshold
    per batch
    iou_thres: IoU threshold used for positive samples for loss calculation
    cls_mapping: class id to name mapping used for symmetry in loss
    use_bin_loss: if true use classification loss else use smooth l1 loss
    '''

    l1_loss = 0
    gen_pose_loss = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #batch_size = len(instances)
    start_instance = 0
    num_instances_overlap = 0
    pose_loss_criterion = PoseLoss(max_points=500)


    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue

        end_instance = start_instance + len(instances_per_image)

        gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
        gt_boxes_per_image = instances_per_image.gt_boxes
        gt_nocs_per_image = instances_per_image.gt_nocs
        gt_binmasks = BitMasks.from_polygon_masks(instances_per_image.gt_masks, 240, 320).tensor
        gt_depth_objs = instances_per_image.gt_depth
        gt_campose = instances_per_image.gt_campose
        gt_3dboxs = instances_per_image.gt_3dboxes
        gt_rots = instances_per_image.gt_rot
        gt_locs = instances_per_image.gt_loc
        gt_scales = instances_per_image.gt_scale
        gt_voxel_logits = instances_per_image.gt_voxels.to(dtype=torch.float)

        for i in range(start_instance, end_instance):

            abs_pred_box = pred_boxes[i,:].to(dtype=torch.int64)
            pred_box = Boxes(torch.unsqueeze(abs_pred_box, dim=0)) # XYXY
            patch_heigth = int(abs_pred_box[3] - abs_pred_box[1])  # Y
            patch_width = int(abs_pred_box[2] - abs_pred_box[0])  # X

            pred_nocs = pred_nocsmap[i] #  (32) x C x 28 x 28 (bin)

            ious = pairwise_iou(gt_boxes_per_image, pred_box)
            idx_max_iou = int(torch.argmax(ious))
            max_iou = ious[idx_max_iou]

            if max_iou >= iou_thres:

                num_instances_overlap += 1

                gt_box = gt_boxes_per_image.tensor[idx_max_iou,:].to(dtype=torch.int64)

                gt_nocs = gt_nocs_per_image[idx_max_iou, :, :, :] # H x W x C
                gt_nocs = torch.squeeze(crop_nocs(gt_nocs), dim=0).to(device=device)  # C x H x W

                gt_cls = cls_mapping[gt_classes_per_image[idx_max_iou]]

                # Get overlapping pixels for loss computation -> Positive ROIs
                x_min = int(torch.max(torch.tensor([gt_box[0], abs_pred_box[0]])))
                x_max = int(torch.min(torch.tensor([gt_box[2], abs_pred_box[2]])))
                y_min = int(torch.max(torch.tensor([gt_box[1], abs_pred_box[1]])))
                y_max = int(torch.min(torch.tensor([gt_box[3], abs_pred_box[3]])))

                # Symmetry Loss: Rotate gt_overlap 90,180,270 degree around y_axis and take min
                if use_bin_loss:
                    # Roi Align pred nocs to pred box shape
                    tmp_box = [torch.unsqueeze(
                        torch.tensor([0, 0, pred_nocs.shape[3], pred_nocs.shape[2]], dtype=torch.float32,
                                     device=device), dim=0)] * num_bins
                    pred_patch = roi_align(pred_nocs.to(device=device), tmp_box,
                                           output_size=(patch_heigth, patch_width), aligned=True) # num_bins x 3 x H x W

                    # Full image patches
                    full_patch = torch.zeros(num_bins, 3, 240, 320)
                    full_patch[:, :, abs_pred_box[1]:abs_pred_box[3], abs_pred_box[0]:abs_pred_box[2]] = pred_patch

                    gt_patch = torch.zeros(3, 240, 320)
                    gt_patch[:, gt_box[1]:gt_box[3], gt_box[0]:gt_box[2]] = gt_nocs

                    # Loss only on overlap ROI with GT
                    pred_overlap = full_patch[:, :, y_min:y_max, x_min:x_max]  # binsxCxHxW
                    gt_overlap = gt_patch[:, y_min:y_max, x_min:x_max]  # CxHxW
                    # print(pred_overlap.shape, max_iou, pred_patch.shape)

                    obj_loss = symmetry_bin_loss(gt_overlap, pred_overlap, gt_cls=gt_cls, num_bins=num_bins)

                else:
                    # Roi Align pred nocs to pred box shape
                    tmp_box = [torch.unsqueeze(
                        torch.tensor([0, 0, pred_nocs.shape[2], pred_nocs.shape[1]], dtype=torch.float32,
                                     device=device), dim=0)]
                    pred_patch = roi_align(torch.unsqueeze(pred_nocs.to(device=device), dim=0), tmp_box,
                                           output_size=(patch_heigth, patch_width), aligned=True)
                    pred_patch = torch.squeeze(pred_patch, dim=0)  # C x H x W of predicted box
                    reshaped_patch = torch.clone(pred_patch.permute(1, 2, 0).contiguous())  # HxWxC

                    # Pose Predictions
                    gt_depth = gt_depth_objs[idx_max_iou, :, :]  # H x W
                    campose = gt_campose[idx_max_iou, :, :]  # 4 x 4
                    gt_bbox_loc = gt_3dboxs[idx_max_iou, :, :] # 8 x 3
                    gt_rot = gt_rots[idx_max_iou, :] # 3

                    euler = mathutils.Euler(gt_rot.detach().cpu().numpy())
                    gt_rot = torch.from_numpy(np.array(euler.to_matrix())).to(device)

                    gt_loc = gt_locs[idx_max_iou, :] # 3
                    gt_scale = gt_scales[idx_max_iou, :] # 3
                    gt_binmask = gt_binmasks[idx_max_iou, :, :] # H x W
                    gt_voxel = gt_voxel_logits[idx_max_iou, :, : ,:]
                    obj_pc = vox2pc(gt_voxel)
                    pred_rot = None

                    if current_iter > int(start_iter_pose):

                        pred_rot, pred_trans, pred_scale, _, _, _ = \
                            run_pose_torch(reshaped_patch, gt_depth, campose,
                                    gt_binmask, abs_pred_box, gt_3d_box=gt_bbox_loc, use_RANSAC=False, device=device)


                    if pred_rot is not None:
                        #pred_rot = torch.from_numpy(global_rot).to(device)
                        #pred_trans = torch.from_numpy(global_trans).to(device)
                        #pred_scale = torch.tensor(global_scale).to(device)
                        #pred_rot.requires_grad, pred_trans.requires_grad, pred_scale.requires_grad = True, True, True

                        # 7 DoF Object Pose Loss using sampled points from complete object geometry
                        obj_pose_loss = pose_loss_criterion(gt_rot, gt_loc, gt_scale, pred_rot, pred_trans,
                                                            pred_scale, obj_pc)
                    else:
                        obj_pose_loss = 0

                    # Full image patches
                    full_patch = torch.zeros(3, 240, 320)
                    full_patch[:, abs_pred_box[1]:abs_pred_box[3], abs_pred_box[0]:abs_pred_box[2]] = pred_patch

                    gt_patch = torch.zeros(3, 240, 320)
                    gt_patch[:, gt_box[1]:gt_box[3], gt_box[0]:gt_box[2]] = gt_nocs

                    # Loss only on overlap ROI with GT
                    pred_overlap = full_patch[:, y_min:y_max, x_min:x_max]  # CxHxW
                    gt_overlap = gt_patch[:, y_min:y_max, x_min:x_max]  # CxHxW
                    # print(pred_overlap.shape, max_iou, pred_patch.shape)

                    obj_loss = symmetry_smooth_l1_loss(gt_overlap, pred_overlap, gt_cls=gt_cls)

                l1_loss += obj_loss
                gen_pose_loss += obj_pose_loss

        start_instance = end_instance

    l1_loss = l1_loss * l1_loss_weight / num_instances_overlap
    gen_pose_loss = gen_pose_loss * pose_loss_weight / num_instances_overlap
    total_loss = l1_loss + gen_pose_loss
    #print('L1 :', l1_loss, ', POSE :', gen_pose_loss)

    return total_loss, None

def nocs_inference(pred_nocsmap, pred_instances, use_bin_loss=False, num_bins=32): # shape num obj 3x  28 x 28  (RGB), Num img x num obj ...

    num_boxes_per_image = [len(i) for i in pred_instances]
    nocs_pred = pred_nocsmap.split(num_boxes_per_image, dim=0)

    if np.array(num_boxes_per_image).sum() == 0:
        return

    # instances and predictions always same len just empty
    for prob, instances in zip(nocs_pred, pred_instances):

        if len(instances) == 0:
            print('No predicted instances found ...')
            continue

        num_pred_instances = prob.shape[0]
        num_dims = len(prob.shape)

        if use_bin_loss and num_pred_instances != 0 and num_dims == 5:

            x_prob = nocs_prob_to_value(prob, channel=0, num_bins=num_bins)
            y_prob = nocs_prob_to_value(prob, channel=1, num_bins=num_bins)
            z_prob = nocs_prob_to_value(prob, channel=2, num_bins=num_bins)
            prob = torch.cat((x_prob, y_prob, z_prob), dim=1)

        instances.pred_nocs = prob


class NocsModel(torch.nn.Module):
    """
    Decoder Module NOCS
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(NocsModel, self).__init__()

        self.input_shape = input_shape
        self.use_bin_loss = cfg.MODEL.ROI_NOCS_HEAD.USE_BIN_LOSS
        self.num_bins = cfg.MODEL.ROI_NOCS_HEAD.NUM_BINS

        # Layer Definition
        if self.use_bin_loss:
            self.layer1_R = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, bias=True, padding=1),  # 14
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(128)
            )
            self.layer1_G = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, bias=True, padding=1),  # 14
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(128)
            )
            self.layer1_B = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, bias=True, padding=1),  # 14
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(128)
            )
            # Layer 2
            self.layer2_R = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, bias=True, padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(64)
            )
            self.layer2_G = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, bias=True, padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(64)
            )
            self.layer2_B = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, bias=True, padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(64)
            )
            # Layer 3
            self.layer3_R = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(64, self.num_bins, kernel_size=3, stride=1, bias=True, padding=1),  # 28 x num_bins R/x - head
                #torch.nn.LogSoftmax(dim=1)
            )
            self.layer3_G = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(64, self.num_bins, kernel_size=3, stride=1, bias=True, padding=1),  # 28 x num_bins G/y - head
                #torch.nn.LogSoftmax(dim=1)
            )
            self.layer3_B = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(64, self.num_bins, kernel_size=3, stride=1, bias=True, padding=1),  # 28 x num_bins B/z - head
                #torch.nn.LogSoftmax(dim=1)
            )
        else:
            self.layer0 = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, bias=True, padding=1),  # 14
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(256)
            )
            self.layer1 = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, bias=True, padding=1),  # 14
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(128)
            )
            self.layer2 = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, bias=True, padding=1),# 28 # use kernel size divisible by stride
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(64)
            )
            self.layer3 = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, bias=True, padding=1),  # 28 x RGB
                torch.nn.Sigmoid()
            )

    def forward(self, features):
        """
        input features from ROI Pool, dim num instances, 256 x 14 x 14
        """

        if self.use_bin_loss:
            R_features = self.layer1_R(features)
            R_features = self.layer2_R(R_features)
            R_features = torch.unsqueeze(self.layer3_R(R_features), dim=1)  # num obj x 1 x num_bins x 28 x 28

            G_features = self.layer1_G(features)
            G_features = self.layer2_G(G_features)
            G_features = torch.unsqueeze(self.layer3_G(G_features), dim=1)  # num obj x 1 x num_bins x 28 x 28

            B_features = self.layer1_B(features)
            B_features = self.layer2_B(B_features)
            B_features = torch.unsqueeze(self.layer3_B(B_features), dim=1)  # num obj x 1 x num_bins x 28 x 28

            features = torch.cat((R_features, G_features, B_features), dim=1).permute(0, 2, 1, 3, 4).contiguous() # num obj x num_bins x 3 x 28 x 28

        else:
            features = self.layer0(features)
            features = self.layer1(features)
            features = self.layer2(features)
            features = self.layer3(features) # num obj x 3 x 28 x 28

        return features


@ROI_NOCS_HEAD_REGISTRY.register()
class NocsDecoder(nn.Module):
    """
    A Nocs head with upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape):
        super(NocsDecoder, self).__init__()

        ### Model
        self.nocs_layers = NocsModel(cfg, input_shape)
        init_weights(self.nocs_layers, init_type='kaiming', init_gain=0.02)


    def forward(self, x):

        x = self.nocs_layers(x) #BS x C x H x W

        return x


def build_nocs_head(cfg, input_shape):
    name = cfg.MODEL.ROI_NOCS_HEAD.NAME
    return ROI_NOCS_HEAD_REGISTRY.get(name)(cfg, input_shape)
