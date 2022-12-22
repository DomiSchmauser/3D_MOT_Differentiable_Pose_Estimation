import logging
import sys
import torch
import numpy as np
import torchvision.transforms as T

from detectron2.layers import ShapeSpec, cat
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.structures import Boxes, pairwise_iou
from torch import nn
from typing import Dict

sys.path.append('..')
from Detection.utils.inference_metrics import compute_voxel_iou
from Detection.utils.train_utils import balanced_BCE_loss

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


ROI_VOXEL_HEAD_REGISTRY = Registry("ROI_VOXEL_HEAD")


def voxel_loss(
        pred_voxel_logits, instances, pred_boxes, loss_weight=1, iou_threshold=0.5, refiner=None
):
    '''
    Calculates the balanced BCE loss between a predicted 32³ voxel grid and a GT voxel grid.
    Only calcluate the loss if the 2D bounding box IoU is larger than a threshold.
    '''

    start_instance = 0
    pred_voxel_logits = torch.squeeze(pred_voxel_logits, dim=1)  # Num obj x 32x32x32
    mean_voxel_iou = []
    loss_gt_voxels = []
    loss_pred_voxels = []
    resize_transform = T.Resize((64, 64))


    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            logger.warning('No instances in the given image.')
            continue

        end_instance = start_instance + len(instances_per_image)

        gt_voxel_logits = instances_per_image.gt_voxels.to(dtype=torch.float)
        gt_depth_map = instances_per_image.gt_depth
        gt_boxes_per_image = instances_per_image.gt_boxes

        for i in range(start_instance, end_instance):

            abs_pred_box = pred_boxes[i, :].to(dtype=torch.int64)
            pred_box = Boxes(torch.unsqueeze(abs_pred_box, dim=0))  # XYXY

            pred_voxel = pred_voxel_logits[i, :, :, :]

            if torch.sum(pred_voxel) == 0:
                logger.warning("Empty instance given to voxel head, skipping. ")
                continue

            ious = pairwise_iou(gt_boxes_per_image, pred_box)
            idx_max_iou = int(torch.argmax(ious))
            max_iou = ious[idx_max_iou]

            if max_iou >= iou_threshold:

                gt_voxel = gt_voxel_logits[idx_max_iou,:,:,:]
                gt_depth = gt_depth_map[idx_max_iou, :, :] # H x W
                gt_box = torch.squeeze(gt_boxes_per_image[idx_max_iou].tensor, dim=0) # XYXY

                if refiner is not None:
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


def voxel_inference(pred_voxel_logits, pred_instances, refiner=None, depth=None):
    """
    Args:
        pred_voxel_logits shape: number_objects x 1 x dims x H x W
        pred_instances shape: number_images x instance class
    """

    num_boxes_per_image = [len(i) for i in pred_instances]
    resize_transform = T.Resize((64, 64))

    if np.array(num_boxes_per_image).sum() == 0:
        logger.warning('No predicted instances found for batch.')
        return

    if depth is None:
        logger.warning('No depth annotation given.')
        return

    voxel_pred_split = pred_voxel_logits.split(num_boxes_per_image, dim=0)

    for pred_instances_per_img, pred_voxels_per_img in zip(pred_instances, voxel_pred_split):

        if len(pred_instances_per_img) == 0:
            logger.warning('No predicted instances found.')
            continue

        voxel_preds = []
        if refiner is not None:
            img_bboxes = pred_instances_per_img.get('pred_boxes').tensor
            num_objs = img_bboxes.shape[0]
            for instance_idx in range(num_objs):
                bbox = img_bboxes[instance_idx]  # xyxy absolute
                depth_crop = depth[0][0][int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]  # H x W
                if (depth_crop.shape[0] < 2) or (depth_crop.shape[1] < 2):
                    pred_voxel = pred_voxels_per_img[instance_idx]
                    logger.warning('Bounding box prediction width or height < 2 pixel, skipping refinement step.')
                else:
                    norm_depth_crop = resize_transform(torch.unsqueeze(depth_crop, dim=0))
                    pred_voxel = refiner(torch.squeeze(pred_voxels_per_img[instance_idx], dim=1), norm_depth_crop)
                voxel_preds.append(pred_voxel)
            voxel_preds = torch.cat(voxel_preds, dim=0)
        else:
            raise ValueError("Voxel Refiner is None.")

        if voxel_preds.sum() == 0:
            pred_instances_per_img.pred_voxels = torch.tensor([]).cuda()
        else:
            pred_instances_per_img.pred_voxels = torch.squeeze(voxel_preds, dim=1)  # inst_per_img x 32 x 32 x 32

@ROI_VOXEL_HEAD_REGISTRY.register()
class VoxelRefiner(torch.nn.Module):
    """
    Voxel grid refinement network using GT depth information.
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(VoxelRefiner, self).__init__()

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
        coarse_voxel shape: BS x 32 x 32 x 32
        depth_shape: BS x 64 x 64
        '''
        depth = torch.unsqueeze(depth, dim=1)
        coarse_voxel = torch.unsqueeze(coarse_voxel, dim=1)
        num_obj = depth.shape[0]

        # U-net like architecture
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
    Voxel decoder module similar to the Pix2Vox++ implementation.
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
        num_obj = features.shape[0]
        if num_obj != 0:
            gen_volume = features.view(num_obj, -1, 4, 4, 4)  # torch.Size([num_obj, 784, 4, 4, 4])
            gen_volume = self.layer1(gen_volume)  # torch.Size([num_obj, 512, 4, 4, 4])
            gen_volume = self.layer2(gen_volume)   # torch.Size([num_obj, 128, 8, 8, 8])
            gen_volume = self.layer3(gen_volume)  # torch.Size([num_obj, 32, 16, 16, 16])
            gen_volume = self.layer4(gen_volume)  # torch.Size([num_obj, 8, 32, 32, 32])
            gen_volume = self.layer5(gen_volume)  # torch.Size([num_obj, 1, 32, 32, 32])
        else:
            gen_volume = torch.zeros([1, 1, 32, 32, 32])

        return gen_volume


@ROI_VOXEL_HEAD_REGISTRY.register()
class VoxelDecoder(nn.Module):
    """
    A voxel head with several conv layers, plus an upsample layer.
    """

    def __init__(self, cfg, input_shape):
        super(VoxelDecoder, self).__init__()

        # Model
        self.decoder = Decoder(cfg, input_shape)
    def forward(self, x):
        x = self.decoder(x)  # BS x C x H x W
        return x


def build_voxel_head(cfg, input_shape):
    name = cfg.MODEL.ROI_VOXEL_HEAD.NAME
    return ROI_VOXEL_HEAD_REGISTRY.get(name)(cfg, input_shape)

def build_voxel_refiner(cfg, input_shape):
    name = cfg.MODEL.ROI_VOXEL_HEAD.REFINER_NAME
    return ROI_VOXEL_HEAD_REGISTRY.get(name)(cfg, input_shape)