from typing import Dict
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals, StandardROIHeads
from detectron2.data import MetadataCatalog
from detectron2.utils.registry import Registry
from roi_heads.voxel_head import (
    build_voxel_head,
    build_voxel_refiner,
    voxel_inference,
    voxel_loss,
)
from roi_heads.nocs_head import (
    build_nocs_head,
    nocs_inference,
    nocs_loss,
)

@ROI_HEADS_REGISTRY.register()
class VoxelNocsHeads(StandardROIHeads):
    """
    The ROI specific heads for Voxel and Nocs branch.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)
        self._init_voxel_head(cfg, input_shape)
        self._init_nocs_head(cfg, input_shape)
        self._misc = {}

        self.train_dataset_names = cfg.DATASETS.TRAIN[0]
        self.test_dataset_names = cfg.DATASETS.TEST[0]
        self.metadata = MetadataCatalog.get(self.train_dataset_names)
        if 'thing_classes' in self.metadata.as_dict():
            self.class_mapping = self.metadata.thing_classes
        self.iteration = 0


    def _init_voxel_head(self, cfg, input_shape):

        self.voxel_on = cfg.MODEL.VOXEL_ON
        self.voxel_loss_weight = cfg.MODEL.ROI_VOXEL_HEAD.LOSS_WEIGHT

        if not self.voxel_on:
            return

        voxel_pooler_resolution = cfg.MODEL.ROI_VOXEL_HEAD.POOLER_RESOLUTION
        voxel_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        voxel_sampling_ratio = cfg.MODEL.ROI_VOXEL_HEAD.POOLER_SAMPLING_RATIO
        voxel_pooler_type = cfg.MODEL.ROI_VOXEL_HEAD.POOLER_TYPE

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.voxel_pooler = ROIPooler(
            output_size=voxel_pooler_resolution,
            scales=voxel_pooler_scales,
            sampling_ratio=voxel_sampling_ratio,
            pooler_type=voxel_pooler_type,
        )
        shape = ShapeSpec(
            channels=in_channels, width=voxel_pooler_resolution, height=voxel_pooler_resolution
        )
        self.voxel_head = build_voxel_head(cfg, shape)
        self.voxel_refiner = build_voxel_refiner(cfg, shape)

    def _init_nocs_head(self, cfg, input_shape):

        self.nocs_on = cfg.MODEL.NOCS_ON
        self.nocs_loss_weight = cfg.MODEL.ROI_NOCS_HEAD.LOSS_WEIGHT
        self.start_iteration_pose = cfg.MODEL.ROI_NOCS_HEAD.START_ITERATION_POSE
        self.pose_loss_weight = cfg.MODEL.ROI_NOCS_HEAD.POSE_LOSS_WEIGHT
        self.iou_threshold = cfg.MODEL.ROI_NOCS_HEAD.IOU_THRES
        self.use_bin_loss = cfg.MODEL.ROI_NOCS_HEAD.USE_BIN_LOSS
        self.num_bins = cfg.MODEL.ROI_NOCS_HEAD.NUM_BINS

        if not self.nocs_on:
            return
        nocs_pooler_resolution = cfg.MODEL.ROI_NOCS_HEAD.POOLER_RESOLUTION
        nocs_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        nocs_sampling_ratio = cfg.MODEL.ROI_NOCS_HEAD.POOLER_SAMPLING_RATIO
        nocs_pooler_type = cfg.MODEL.ROI_NOCS_HEAD.POOLER_TYPE


        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.nocs_pooler = ROIPooler(
            output_size=nocs_pooler_resolution,
            scales=nocs_pooler_scales,
            sampling_ratio=nocs_sampling_ratio,
            pooler_type=nocs_pooler_type,
        )
        shape = ShapeSpec(
            channels=in_channels, width=nocs_pooler_resolution, height=nocs_pooler_resolution
        )
        self.nocs_head = build_nocs_head(cfg, shape)

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        images: ImageList
        features: Dict with backbone features from the FPN
        proposals: Anchor proposals
        targets: GT instances
        """
        instances, losses = super().forward(images, features, proposals, targets)
        del images
        if self.training:
            del targets
            losses.update(self._forward_voxel(features, instances))
            losses.update(self._forward_nocs(features, instances))
            self.iteration += 1
            return [], losses

        else:
            pred_instances = self.forward_with_given_boxes_voxnocs(features, instances, depth=targets)
            return pred_instances, {}


    def forward_with_given_boxes_voxnocs(self, features, instances, depth=None):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.
        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.
        Returns:
            instances (Instances): the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_voxels` or 'pred_nocs.
        """
        assert not self.training
        #instances = super().forward_with_given_boxes(features, instances)

        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes") and instances[0].has("pred_masks")

        instances = self._forward_voxel(features, instances, depth=depth)
        instances = self._forward_nocs(features, instances)

        return instances

    def _forward_voxel(self, features, instances, depth=None):
        """
        Forward logic for the voxel branch.
        Args:
            features (list[Tensor]): #level input features for voxel prediction
            instances (list[Instances]): the per-image instances to train/predict meshes.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_voxels" and return it.
        """
        if not self.voxel_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            losses = {}
            # dim: Number of boxes aggregated over all images in a batch  x 256 x 14 x 14
            voxel_features = self.voxel_pooler(features, proposal_boxes)
            voxel_logits = self.voxel_head(voxel_features) #Num objs x 1 x 32 x 32 x 32, zeros for empty detection
            src_boxes = cat([p.tensor for p in proposal_boxes])
            loss_voxel, _ = voxel_loss(
                voxel_logits, proposals, src_boxes, loss_weight=self.voxel_loss_weight,
                iou_threshold=self.iou_threshold, refiner=self.voxel_refiner
            )
            losses.update({"loss_voxel": loss_voxel})

            return losses
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            voxel_features = self.voxel_pooler(features, pred_boxes)  # BS x 256 x 14 x 14
            voxel_logits = self.voxel_head(voxel_features)
            voxel_inference(
                voxel_logits, instances, refiner=self.voxel_refiner, depth=depth
            )
            return instances

    def _forward_nocs(self, features, instances):
        """
        Forward logic for the voxel branch.
        Args:
            features (list[Tensor]): #level input features for nocs prediction
            instances (list[Instances]): the per-image instances to train/predict nocs.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_nocs" and return it.
        """
        if not self.nocs_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            losses = {}
            get_pose_loss = self.iteration > self.start_iteration_pose
            nocs_features = self.nocs_pooler(features, proposal_boxes)
            nocs_map_rgb = self.nocs_head(nocs_features)  # num_obj x 3 x 28 x 28
            src_boxes = cat([p.tensor for p in proposal_boxes])
            loss_nocs = nocs_loss(
                nocs_map_rgb, proposals, src_boxes, l1_loss_weight=self.nocs_loss_weight,
                pose_loss_weight=self.pose_loss_weight, iou_threshold=self.iou_threshold,
                cls_mapping=self.class_mapping, use_bin_loss=self.use_bin_loss, num_bins=self.num_bins,
                get_pose_loss=get_pose_loss
            )
            losses.update({"loss_nocs": loss_nocs})

            return losses
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            nocs_features = self.nocs_pooler(features, pred_boxes) # BS x 256 x 14 x 14
            nocs_map_rgb = self.nocs_head(nocs_features) # BS x 3 x 28 x 28 (RGB)
            nocs_inference(nocs_map_rgb, instances, use_bin_loss=self.use_bin_loss, num_bins=self.num_bins)

            return instances
