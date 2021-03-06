import os, sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch
import cv2
import h5py

from detectron2.config import configurable
from detectron2.structures import BoxMode
from detectron2.structures import polygons_to_bitmask
from detectron2.utils.visualizer import GenericMask
from detectron2.data import DatasetMapper

from BlenderProc.utils import binvox_rw

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T


sys.path.append('..') #Hack add ROOT DIR
from Detection.utils.train_utils import crop_segmask, get_voxel

__all__ = ["VoxNocsMapper", "VoxMapper"]

class VoxNocsMapper:
    '''
    Dataset mapper class to handle MOTFront data with a Detectron2 network training pipeline with Voxel and NOCs head
    '''

    def __init__(self, cfg, use_instance_mask: bool = False, instance_mask_format: str = "polygon",
            recompute_boxes: bool = False, is_train=True, dataset_names=None,):

        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        self.is_train = is_train
        self.augmentations = None  # list with augmentations NOT IMPLEMENTED YET
        self.cfg = cfg
        self.image_format = cfg.INPUT.FORMAT
        self.use_instance_mask = use_instance_mask
        self.instance_mask_format = instance_mask_format
        self.recompute_boxes = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {self.augmentations}")

        self.dataset_names = dataset_names
        self.voxel_on = cfg.MODEL.VOXEL_ON
        self.nocs_on = cfg.MODEL.NOCS_ON

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=None
            )
            for obj in dataset_dict['annotations']
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        image = utils.read_image(dataset_dict["file_name"], format=self.image_format) # H x W x C
        utils.check_image_size(dataset_dict, image)

        sem_seg_gt = None
        self.augmentations = T.AugmentationList(utils.build_augmentation(self.cfg, self.is_train))
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))) # C x H x W

        if not self.is_train:
            pass
            #dataset_dict.pop("annotations", None)
            #return dataset_dict

        nocs_map = self.get_nocs(dataset_dict["nocs_map"])
        depth_map = self.load_hdf5(dataset_dict["depth_map"])

        dataset_dict["depth_map"] = depth_map
        dataset_dict["nocs_map"] = nocs_map

        for anno in dataset_dict['annotations']:
            voxel = get_voxel(anno["voxel"], anno["scale"])
            nocs_obj = crop_segmask(nocs_map, anno['bbox'], anno['segmentation'])
            depth_obj = self.crop_depth(depth_map, anno['bbox'], anno['segmentation'])

            anno["voxel"] = voxel
            anno["nocs"] = nocs_obj
            anno["depth"] = depth_obj

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

            if self.voxel_on:
                count = 0
                for anno in dataset_dict['annotations']:
                    if count == 0:
                        gt_voxels = torch.unsqueeze(anno['voxel'], 0)
                    else:
                        gt_voxel = torch.unsqueeze(anno['voxel'], 0)
                        gt_voxels = torch.cat((gt_voxels, gt_voxel), 0)
                    count += 1

                dataset_dict['instances'].set('gt_voxels', gt_voxels)

            if self.nocs_on:
                max_height, max_width = self.get_max_dims(dataset_dict['annotations'])
                count = 0
                for anno in dataset_dict['annotations']:
                    width = anno['nocs'].shape[1]
                    height = anno['nocs'].shape[0]
                    p2d = (0, 0, 0, max_width - width, 0, max_height - height)  # pad image to right
                    if count == 0:
                        gt_nocs = torch.unsqueeze(anno['nocs'], 0)  # 1 x H x W x 3
                        gt_nocs = F.pad(gt_nocs, p2d, "constant", 300)  # 300 not a pixel value # 1 x maxH x maxW x 3
                    else:
                        gt_noc = torch.unsqueeze(anno['nocs'], 0)
                        gt_noc = F.pad(gt_noc, p2d, "constant", 300)  # 300 not a pixel value
                        gt_nocs = torch.cat((gt_nocs, gt_noc), 0)
                    count += 1

                dataset_dict['instances'].set('gt_nocs', gt_nocs)

        return dataset_dict


    @staticmethod
    def get_max_dims(dset):
        '''
        padding image crops
        '''

        max_height = 0
        max_width = 0

        for anno in dset:
            height, width = anno['nocs'].shape[0], anno['nocs'].shape[1]

            if height >= max_height:
                max_height = height

            if width >= max_width:
                max_width = width

        return max_height, max_width

    @staticmethod
    def get_nocs(nocs_path):

        nocs = cv2.imread(nocs_path, -1) #BGRA
        nocs = nocs[:,:,:3]
        nocs = np.array(nocs[:, :, ::-1], dtype=np.float32) # RGB

        return nocs

    @staticmethod
    def crop_depth(depth_img, bbox, segmap):

        abs_bbox = torch.tensor(BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS), dtype=torch.float32)

        gm = GenericMask(segmap, 240, 320)
        bin_mask = gm.polygons_to_mask(segmap)
        binary_mask = bin_mask[:, :]
        crop_im = np.multiply(depth_img, binary_mask)
        cropped_im = np.array(crop_im[int(abs_bbox[1]):int(abs_bbox[3]),int(abs_bbox[0]):int(abs_bbox[2])])

        return torch.from_numpy(cropped_im).to(torch.float32)

    @staticmethod
    def load_hdf5(path):
        with h5py.File(path, 'r') as data:
            for key in data.keys():
                if key == 'depth':
                    depth = np.array(data[key])

        return depth


class VoxMapper:
    '''
        Dataset mapper class to handle MOTFront data with a Detectron2 network training pipeline with Voxel head
    '''

    def __init__(
            self,
            cfg,
            use_instance_mask: bool = False,
            instance_mask_format: str = "polygon",
            recompute_boxes: bool = False,
            is_train=True,
            dataset_names=None,
    ):
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train = is_train
        self.augmentations = None  # list with augmentations NOT IMPLEMENTED YET
        self.cfg = cfg
        self.image_format = cfg.INPUT.FORMAT
        self.use_instance_mask = use_instance_mask
        self.instance_mask_format = instance_mask_format
        self.recompute_boxes = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {self.augmentations}")

        self.dataset_names = dataset_names
        self.voxel_on = cfg.MODEL.VOXEL_ON
        self.nocs_on = cfg.MODEL.NOCS_ON


    def _transform_annotations(self, dataset_dict, transforms, image_shape):


        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=None
            )
            for obj in dataset_dict['annotations']
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        image = utils.read_image(dataset_dict["file_name"], format=self.image_format) # H x W x C
        utils.check_image_size(dataset_dict, image)

        sem_seg_gt = None
        self.augmentations = T.AugmentationList(utils.build_augmentation(self.cfg, self.is_train))
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))) # C x H x W

        if not self.is_train:
            pass
            #dataset_dict.pop("annotations", None)
            #return dataset_dict

        for anno in dataset_dict['annotations']:
            voxel = anno["voxel"]
            anno["voxel"] = voxel


        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

            if self.voxel_on:
                count = 0
                for anno in dataset_dict['annotations']:
                    if count == 0:
                        gt_voxels = torch.unsqueeze(anno['voxel'], 0)
                    else:
                        gt_voxel = torch.unsqueeze(anno['voxel'], 0)
                        gt_voxels = torch.cat((gt_voxels, gt_voxel), 0)
                    count += 1

                dataset_dict['instances'].set('gt_voxels', gt_voxels)

        return dataset_dict


    @staticmethod
    def get_max_dims(dset):
        '''
        padding image crops
        '''

        max_height = 0
        max_width = 0

        for anno in dset:
            height, width = anno['nocs'].shape[0], anno['nocs'].shape[1]

            if height >= max_height:
                max_height = height

            if width >= max_width:
                max_width = width

        return max_height, max_width