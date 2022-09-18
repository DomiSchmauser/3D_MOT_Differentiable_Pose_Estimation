# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import fvcore.nn.weight_init as weight_init
import sys
import torch
import numpy as np
import torchvision.transforms as T
#import matplotlib.pyplot as plt

from detectron2.layers import ShapeSpec, cat
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from torch import nn
from torch.nn import functional as F
from typing import Dict

sys.path.append('..') #Hack add ROOT DIR
from Detection.inference.inference_metrics import compute_voxel_iou
from Detection.utils.train_utils import init_weights, balanced_BCE_loss



ROI_VOXEL_HEAD_REGISTRY = Registry("ROI_VOXEL_HEAD")


def voxel_loss(pred_voxel_logits, instances, pred_boxes, loss_weight=1, iou_thres=0.5, use_refiner=True,
               refiner=None):
    '''
    Calculate BCE loss between predicted 32³ voxel grid and GT voxel grid if IoU larger threshold
    '''

    start_instance = 0
    pred_voxel_logits = torch.squeeze(pred_voxel_logits, dim=1)  # Num obj x 32x32x32
    mean_voxel_iou = []
    loss_gt_voxels = []
    loss_pred_voxels = []
    resize_transform = T.Resize((64, 64))


    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue

        end_instance = start_instance + len(instances_per_image)

        gt_voxel_logits = instances_per_image.gt_voxels.to(dtype=torch.float)
        gt_depth_objs = instances_per_image.gt_depth
        gt_boxes_per_image = instances_per_image.gt_boxes

        for i in range(start_instance, end_instance):

            abs_pred_box = pred_boxes[i, :].to(dtype=torch.int64)
            pred_box = Boxes(torch.unsqueeze(abs_pred_box, dim=0))  # XYXY

            pred_voxel = pred_voxel_logits[i,:,:,:]

            if torch.sum(pred_voxel) == 0: # empty detections
                continue

            ious = pairwise_iou(gt_boxes_per_image, pred_box)
            idx_max_iou = int(torch.argmax(ious))
            max_iou = ious[idx_max_iou]

            if max_iou >= iou_thres:

                gt_voxel = gt_voxel_logits[idx_max_iou,:,:,:]
                gt_depth = gt_depth_objs[idx_max_iou, :, :] # H x W
                gt_box = torch.squeeze(gt_boxes_per_image[idx_max_iou].tensor, dim=0)

                if use_refiner and refiner is not None:
                    depth_crop = gt_depth[int(gt_box[1]):int(gt_box[3]), int(gt_box[0]):int(gt_box[2])]  # H x W
                    norm_depth_crop = resize_transform(torch.unsqueeze(depth_crop, dim=0))
                    pred_voxel = refiner(torch.unsqueeze(pred_voxel, dim=0), norm_depth_crop)
                    pred_voxel = torch.squeeze(pred_voxel)

                voxel_iou = compute_voxel_iou(pred_voxel, gt_voxel)
                mean_voxel_iou.append(voxel_iou)
                loss_gt_voxels.append(torch.unsqueeze(gt_voxel, dim=0))
                loss_pred_voxels.append(torch.unsqueeze(pred_voxel, dim=0))

        start_instance = end_instance

    if mean_voxel_iou:
        get_event_storage().put_scalar("training/voxel_iou", np.array(mean_voxel_iou).mean())

    gt_voxels = cat(loss_gt_voxels, dim=0)
    pred_voxels = cat(loss_pred_voxels, dim=0)

    assert pred_voxels.shape == gt_voxels.shape

    voxel_loss = balanced_BCE_loss(gt_voxels, pred_voxels)
    voxel_loss = voxel_loss * loss_weight

    return voxel_loss, gt_voxels


def voxel_inference(pred_voxel_logits, pred_instances,
                    use_refiner=True, refiner=None, depth=None): # shape Num obj x 1 x D x H x W, Num img x Instance class

    voxel_probs_pred = pred_voxel_logits
    num_boxes_per_image = [len(i) for i in pred_instances]
    resize_transform = T.Resize((64, 64))

    if np.array(num_boxes_per_image).sum() == 0:
        print('No predicted instances found for batch...')
        return

    depth = depth[0]
    voxel_probs_pred = voxel_probs_pred.split(num_boxes_per_image, dim=0)

    # Assign predicted voxels   # instances and predictions different len -> moving idx
    for inst, prob, d in zip(pred_instances, voxel_probs_pred, depth):

        if len(inst) == 0:
            print('No predicted instances found ...')
            continue

        if use_refiner:
            prob_preds = []
            img_bboxs = inst.get('pred_boxes').tensor
            for i in range(img_bboxs.shape[0]): # per object
                bbox = img_bboxs[i]
                depth_crop = d[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]  # H x W
                norm_depth_crop = resize_transform(torch.unsqueeze(depth_crop, dim=0))
                pred_voxel = refiner(torch.squeeze(prob[i], dim=1), norm_depth_crop)
                prob_preds.append(pred_voxel)
            prob = torch.cat(prob_preds, dim=0)



        if prob.sum() == 0: # sigmoid of 0 = 0.5 -< (prob.numel() * 0.5)
            inst.pred_voxels = torch.tensor([]).cuda()
        else:
            inst.pred_voxels = torch.squeeze(prob, dim=1)  # (Num inst in 1 img, D, H, W)

@ROI_VOXEL_HEAD_REGISTRY.register()
class VoxRefiner(torch.nn.Module):
    """
    Voxel grid refinement network
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(VoxRefiner, self).__init__()

        self.input_shape = input_shape # has to be bs x 2 x 32 x 32 x 32 -> output has to be bs x 1 x 32 x 32 x 32

        # Layer Definition
        self.depth_upscale = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(1, 1, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.ReLU()
        )
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(2, 16, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(16),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 32, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, bias=False, padding=1),
        )

    def forward(self, coarse_voxel, depth):
        '''
        coarse_voxel: dim = bs x 32 x 32 x 32
        depth: dim = bs x 64 x 64
        '''
        depth = torch.unsqueeze(depth, dim=1) # bs x 1 x 64 x 64
        coarse_voxel = torch.unsqueeze(coarse_voxel, dim=1) # bs x 1 x 32 x 32 x 32
        num_obj = depth.shape[0] #num of pred objects

        # Unet like architecture
        if coarse_voxel.sum() != 0:
            depth_volume = depth.view(num_obj, -1, 16, 16, 16).to(torch.float)
            depth_st1 = self.depth_upscale(depth_volume) # bs x 1 x 32 x 32 x 32
            comb_volume = torch.cat((depth_st1, coarse_voxel), dim=1)  # bs x 2 x 32 x 32 x 32
            c_st1 = self.layer1(comb_volume) # bs x 16 x 16 x 16 x 16
            c_st2 = self.layer2(c_st1) # bs x 32 x 8 x 8 x 8
            c_st3 = self.layer3(c_st2) + c_st1  # bs x 16 x 16 x 16 x 16
            gen_volume = self.layer4(c_st3) + coarse_voxel  # bs x 1 x 32 x 32 x 32
        else:
            gen_volume = coarse_voxel

        return gen_volume


class Decoder(torch.nn.Module):
    """
    Decoder Module from Pix2Vox++ Implementation
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(Decoder, self).__init__()

        self.input_shape = input_shape

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(784, 512, kernel_size=3, stride=1, bias=False, padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=False),
        )

    def forward(self, features):
        """
        """
        num_obj = features.shape[0] #num of pred objects
        if num_obj != 0:
            gen_volume = features.view(num_obj, -1, 4, 4, 4)
            #print(gen_volume.size())   # torch.Size([num_obj, 784, 4, 4, 4])
            gen_volume = self.layer1(gen_volume)
            #print(gen_volume.size())   # torch.Size([num_obj, 512, 4, 4, 4])
            gen_volume = self.layer2(gen_volume)
            #print(gen_volume.size())   # torch.Size([num_obj, 128, 8, 8, 8])
            gen_volume = self.layer3(gen_volume)
            #print(gen_volume.size())   # torch.Size([num_obj, 32, 16, 16, 16])
            gen_volume = self.layer4(gen_volume)
            #print(gen_volume.size())   # torch.Size([num_obj, 8, 32, 32, 32])
            gen_volume = self.layer5(gen_volume)
            #print(gen_volume.size())   # torch.Size([num_obj, 1, 32, 32, 32])
        else:
            gen_volume = torch.zeros([1, 1, 32, 32, 32])

        return gen_volume


@ROI_VOXEL_HEAD_REGISTRY.register()
class Pix2VoxDecoder(nn.Module):
    """
    A voxel head with several conv layers, plus an upsample layer.
    """

    def __init__(self, cfg, input_shape):
        super(Pix2VoxDecoder, self).__init__()

        # Model
        self.decoder = Decoder(cfg, input_shape)
        #init_weights(self.decoder, init_type='kaiming', init_gain=0.02)


    def forward(self, x):

        x = self.decoder(x) #Batchsize x channels x H x W

        return x


def build_voxel_head(cfg, input_shape):
    name = cfg.MODEL.ROI_VOXEL_HEAD.NAME
    return ROI_VOXEL_HEAD_REGISTRY.get(name)(cfg, input_shape)

def build_voxel_refiner(cfg, input_shape):
    name = cfg.MODEL.ROI_VOXEL_HEAD.REFINER_NAME
    return ROI_VOXEL_HEAD_REGISTRY.get(name)(cfg, input_shape)