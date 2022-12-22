import contextlib
import copy
import io
import itertools
import logging
import pickle
import matplotlib.pyplot as plt

import numpy as np
import os, sys
from collections import OrderedDict
import detectron2.utils.comm as comm
import open3d as o3d
import torch

from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table
from detectron2.layers import roi_align
from detectron2.utils.events import get_event_storage

from Detection.evaluator.coco import COCO

sys.path.append('..') #Hack add ROOT DIR

from Detection.utils.inference_metrics import compute_voxel_iou, get_rotation_diff, get_location_diff
from Detection.utils.inference_utils import convert_voxel_to_pc
from Detection.pose.pose_estimation import run_pose
from Detection.utils.train_utils import crop_segmask, get_voxel, symmetry_smooth_l1_loss, balanced_BCE_loss


class FrontEvaluator(DatasetEvaluator):
    '''
    Evaluator class for a Detectron2 network pipeline evaluated on the MOTFront dataset
    '''

    def __init__(
            self,
            dataset_name,
            tasks=None,
            distributed=True,
            output_dir=None,
            *,
            use_fast_impl=True,
            use_bin_loss=True,
    ):
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl
        self.use_bin_loss = use_bin_loss

        # GT data
        self.gt_data = {'cat_ids':[], 'gt_voxels':[], 'gt_boxes':[], 'gt_nocs':[],
                        'depth':[], 'campose':[], 'gt_rotations':[], 'gt_locations':[], 'gt_3dboxes':[]}

        self.iou_thres = 0.5
        self._tasks = tasks
        self._cpu_device = torch.device("cpu")
        self._device = torch.device("cuda")

        self._metadata = MetadataCatalog.get(dataset_name)
        self.class_mapping = self._metadata.thing_classes

        if not hasattr(self._metadata, "json_file"):
            self._logger.info(
                f"'{dataset_name}' is not registered by `register_coco_instances`."
                " Therefore trying to convert it to COCO format ..."
            )

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)


        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset


    def reset(self):

        self._predictions = []
        self.gt_data = {'cat_ids': [], 'gt_voxels': [], 'gt_boxes': [], 'gt_nocs': [],
                        'depth': [], 'campose': [], 'gt_rotations': [], 'gt_locations': [], 'gt_3dboxes': []}

    def process(self, inputs, outputs):
        """
        prepare model inputs and outputs for evaluation
        """
        for inp, output in zip(inputs, outputs): # Inputs and Predictions image-wise, Inputs len = BS = 1
            # Load GT data
            num_objs = len(inp['cat_id'])
            nocs = inp['nocs_map']
            nocs = torch.from_numpy(nocs).to(self._cpu_device)
            all_nocs = [] # Image-wise nocs objects
            all_voxel = [] # Image-wise voxel objects
            for m in range(num_objs):
                obj_nocs = crop_segmask(nocs, inp['boxes'][m], inp['segmap'][m]).to(self._cpu_device)
                obj_vox = get_voxel(inp['vox'][m], inp['3dscales'][m]).to(self._cpu_device)
                all_nocs.append(obj_nocs)
                all_voxel.append(obj_vox)

            prediction = {"image_id": inp["image_id"]}
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, inp["image_id"])
                self.gt_data['cat_ids'].append(torch.tensor(inp['cat_id']).to(self._cpu_device))
                self.gt_data['gt_voxels'].append(all_voxel)
                self.gt_data['gt_boxes'].append(torch.tensor(inp['boxes']).to(self._cpu_device))
                self.gt_data['gt_nocs'].append(all_nocs)
                self.gt_data['depth'].append(inp['depth_map'])
                self.gt_data['campose'].append(inp['campose'])
                self.gt_data['gt_3dboxes'].append(torch.tensor(np.stack(inp['3dboxes'], axis=0)).to(self._cpu_device))
                self.gt_data['gt_rotations'].append(torch.tensor(np.stack(inp['rotations'], axis=0)).to(self._cpu_device))
                self.gt_data['gt_locations'].append(torch.tensor(np.stack(inp['locations'], axis=0)).to(self._cpu_device))
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)


    def evaluate(self, batch_idx=None, save_img_pred=False):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
            batch_idx: for saving images to tensorboard
        """

        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(predictions, batch_idx, save_img_pred)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)


    def _tasks_from_predictions(self, predictions):
        """
        Get COCO API "tasks" (i.e. iou_type) from COCO-format predictions.
        """
        tasks = {}
        for pred in predictions:

            if "voxel" in pred:
                tasks.add("vox")
            if "nocs" in pred:
                tasks.add("nocs")

        return sorted(tasks)


    def _eval_predictions(self, predictions, batch_idx, save_img_pred):
        tasks = self._tasks

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints", "vox", "nocs"}, f"Got unknown task: {task}!"

            if task == "vox":
                res = _evaluate_voxel(predictions, self.gt_data, class_mapping=self.class_mapping,
                                      thres=self.iou_thres, vis_vox=save_img_pred, device=self._device, batch_idx=batch_idx)

            elif task == "nocs":
                res = _evaluate_nocs(predictions, self.gt_data, class_mapping=self.class_mapping,
                                      thres=self.iou_thres, vis_nocs=save_img_pred, device=self._device,
                                     batch_idx=batch_idx, use_bin_loss=self.use_bin_loss)

            self._results[task] = res



    def _eval_box_proposals(self, predictions):
        """
        Evaluate the box proposals in predictions.
        Fill self._results with the metrics for "box_proposals" task.
        """
        if self._output_dir:
            # Saving generated box proposals to file.
            # Predicted box_proposals are in XYXY_ABS mode.
            bbox_mode = BoxMode.XYXY_ABS.value
            ids, boxes, objectness_logits = [], [], []
            for prediction in predictions:
                ids.append(prediction["image_id"])
                boxes.append(prediction["proposals"].proposal_boxes.tensor.numpy())
                objectness_logits.append(prediction["proposals"].objectness_logits.numpy())

            proposal_data = {
                "boxes": boxes,
                "objectness_logits": objectness_logits,
                "ids": ids,
                "bbox_mode": bbox_mode,
            }
            with PathManager.open(os.path.join(self._output_dir, "box_proposals.pkl"), "wb") as f:
                pickle.dump(proposal_data, f)

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating bbox proposals ...")
        res = {}
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = _evaluate_box_proposals(predictions, self._coco_api, area=area, limit=limit)
                key = "AR{}@{:d}".format(suffix, limit)
                res[key] = float(stats["ar"].item() * 100)
        self._logger.info("Proposal metrics: \n" + create_small_table(res))
        self._results["box_proposals"] = res

#------------------------- Static functions ----------------------------------------------------------------------------

def _evaluate_voxel(predictions, gt_data, class_mapping=None, thres=0.4, vis_vox=False, device=None, batch_idx=0):

    if device is None:
        device = torch.device("cpu")

    gt_voxels = gt_data['gt_voxels']
    cat_ids = gt_data['cat_ids']
    gt_bboxes = gt_data['gt_boxes']

    mean_vox_acc = []
    mean_vox_loss = []
    id_storage = []
    for img_pred, gt_voxel, cat_id, gt_bbox in zip(predictions,gt_voxels,cat_ids, gt_bboxes):

        for inst in img_pred['instances']:

            pred_bbox = torch.tensor(BoxMode.convert(inst['bbox'], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS),
                                     dtype=torch.float32, device=device)
            pred_bbox = Boxes(torch.unsqueeze(pred_bbox, dim=0))
            pred_catid = int(inst['category_id'])
            pred_vox = torch.tensor(inst['voxel']) # can be a empty tensor

            if pred_vox.numel() == 0: # skip empty predicted voxels
                continue

            same_catid = np.where((np.array(cat_id) - pred_catid) == 0)[0] #same category check
            same_bbox = [BoxMode.convert(list(gt_bbox[val]), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for val in
                         same_catid]  # XYWH_AB
            if not same_bbox:
                continue
            gt_target_boxes = Boxes(torch.tensor(same_bbox, dtype=torch.float32, device=device))

            # compare bbox
            ious = pairwise_iou(gt_target_boxes, pred_bbox)
            idx_max_iou = torch.argmax(ious)
            max_iou = ious[idx_max_iou]

            if max_iou >= thres:

                idx_gt_voxel = same_catid[idx_max_iou]
                gt_same_voxel = gt_voxel[idx_gt_voxel]
                gt_cat_id = cat_id[idx_gt_voxel]

                voxel_iou = compute_voxel_iou(pred_vox.sigmoid(), gt_same_voxel) # add sigmoid
                mean_vox_acc.append(voxel_iou)

                voxel_loss = balanced_BCE_loss(gt_same_voxel.type('torch.FloatTensor'), pred_vox)
                mean_vox_loss.append(voxel_loss.detach().cpu().numpy())

                if vis_vox:

                    if gt_cat_id not in id_storage:

                        pred_vox = pred_vox > 0.5

                        fig = plt.figure()
                        ax = fig.gca(projection='3d')
                        ax.voxels(gt_same_voxel,
                                  edgecolor='k',
                                  linewidth=0.5)
                        ax.set(xlabel='X', ylabel='Y', zlabel='Z')

                        fig.canvas.draw()

                        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        img = np.transpose(img,(2, 0, 1))

                        gt_docstring = class_mapping[int(gt_cat_id.detach().cpu().item())] + '_voxel_' + str(batch_idx) + "/gt"
                        get_event_storage().put_image(gt_docstring, img)

                        fig = plt.figure()
                        ax = fig.gca(projection='3d')
                        ax.voxels(pred_vox,
                                  edgecolor='k',
                                  linewidth=0.5)
                        ax.set(xlabel='X', ylabel='Y', zlabel='Z')

                        fig.canvas.draw()

                        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        img = np.transpose(img, (2, 0, 1))

                        pred_docstring = class_mapping[int(gt_cat_id.detach().cpu().item())] + '_voxel_' + str(batch_idx) + "/pred"
                        get_event_storage().put_image(pred_docstring, img)

                        # Only one image per category
                        id_storage.append(gt_cat_id)
                        plt.close("all")

    if not mean_vox_acc:
        voxel_mean = None
    else:
        voxel_mean = np.array(mean_vox_acc).mean()

    if not mean_vox_loss:
        voxel_loss_mean = None
    else:
        voxel_loss_mean = np.array(mean_vox_loss).mean()

    return {'voxel valacc': voxel_mean, 'voxel valloss': voxel_loss_mean}

def _evaluate_nocs(predictions, gt_data, class_mapping=None, thres=0.4, vis_nocs=False,
                   device=None, batch_idx=0, use_bin_loss=True):

    if device is None:
        device = torch.device("cpu")

    # Unpack GT data
    gt_nocs = gt_data['gt_nocs']
    cat_ids = gt_data['cat_ids']
    gt_bboxes = gt_data['gt_boxes']
    camposes = gt_data['campose']
    depths = gt_data['depth']
    rotations = gt_data['gt_rotations']
    locations = gt_data['gt_locations']
    gt_3dboxes = gt_data['gt_3dboxes']
    gt_voxels = gt_data['gt_voxels']

    # Data storage
    mean_nocs_loss = []
    mean_rotation_error = []
    mean_location_error = []
    cls_rotation_error = {'chair': [], 'table': [], 'sofa': [], 'bed': [], 'tv stand': [], 'wine cooler': [],
                          'nightstand': []}
    cls_location_error = {'chair': [], 'table': [], 'sofa': [], 'bed': [], 'tv stand': [], 'wine cooler': [],
                          'nightstand': []}
    id_storage = []
    gt_objs = []
    pred_objs = []

    for img_pred, gt_nocs_obj, cat_id, gt_bbox, depth, campose, gt_rot, gt_loc, gt_3dbox, gt_voxel in \
            zip(predictions,gt_nocs,cat_ids,gt_bboxes,depths,camposes,rotations,locations,gt_3dboxes,gt_voxels):

        for inst in img_pred['instances']:

            # Predicted data
            abs_bbox = torch.tensor(BoxMode.convert(inst['bbox'], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS), dtype=torch.int64, device=device)
            pred_bbox = Boxes(torch.unsqueeze(abs_bbox, dim=0))
            pred_catid = int(inst['category_id'])
            pred_nocs = torch.tensor(inst['nocs'])
            pred_mask = inst['segmentation']

            same_catid = np.where((np.array(cat_id) - pred_catid) == 0)[0] #same category check
            same_bbox = [BoxMode.convert(list(gt_bbox[val]), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for val in same_catid] # XYWH_AB
            if not same_bbox:
                continue
            gt_target_boxes = Boxes(torch.tensor(same_bbox, dtype=torch.float32, device=device))

            #compare bbox
            ious = pairwise_iou(gt_target_boxes, pred_bbox)
            idx_max_iou = torch.argmax(ious)
            max_iou = ious[idx_max_iou]

            if max_iou >= thres:
                # Get correct GT association
                idx_gt_nocs = same_catid[idx_max_iou]
                gt_same_nocs = gt_nocs_obj[idx_gt_nocs].permute(2, 0, 1).contiguous() # C x H x W

                '''
                FOR VISUALISING GT NOCS MAPS
                idxs = np.where(gt_same_nocs.detach().cpu().numpy() != 1)[1:]
                gt_nc = gt_same_nocs.permute(1,2,0).contiguous()
                nc_pts = gt_nc[idxs[0],idxs[1],:]

                nocs_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
                nocs_pcd = o3d.geometry.PointCloud()
                nocs_pcd.points = o3d.utility.Vector3dVector(nc_pts.detach().cpu().numpy())
                o3d.visualization.draw_geometries([nocs_pcd, nocs_origin])
                '''

                gt_cat_id = cat_id[idx_gt_nocs]
                gt_same_rot = gt_rot[idx_gt_nocs]
                gt_same_voxel = gt_voxel[idx_gt_nocs]
                gt_same_3dbox = gt_3dbox[idx_gt_nocs]
                gt_same_loc = gt_loc[idx_gt_nocs]
                # gt_same_box = gt_target_boxes.tensor[idx_gt_nocs]
                cls_name = class_mapping[int(gt_cat_id.detach().cpu().item())]

                heigth = pred_nocs.shape[1] # 28
                width = pred_nocs.shape[2] # 28
                gt_heigth = gt_same_nocs.shape[1]
                gt_width = gt_same_nocs.shape[2]
                pred_heigth = int(abs_bbox[3] - abs_bbox[1])
                pred_width = int(abs_bbox[2] - abs_bbox[0])

                # ROI Align to GT patch for loss computation
                bbox = [torch.unsqueeze(torch.tensor([0, 0, width, heigth], dtype=torch.float32, device=torch.device("cuda")), dim=0)]
                pred_patch = roi_align(torch.unsqueeze(pred_nocs.to(device=device), dim=0), bbox,
                                       output_size=(gt_heigth, gt_width), aligned=True)
                pred_patch = torch.squeeze(pred_patch, dim=0)  # C x H x W


                nocs_loss = symmetry_smooth_l1_loss(gt_same_nocs, pred_patch.cpu(), gt_cls=cls_name)  # Predicted Patch reshaped to GT nocs shape for comparison

                mean_nocs_loss.append(nocs_loss.cpu().detach().numpy())

                if vis_nocs:
                    if gt_cat_id not in id_storage:
                        gt_same_nocs = gt_same_nocs.cpu().detach().numpy()
                        pred_patch = pred_patch.cpu().detach().numpy()
                        gt_docstring = class_mapping[int(gt_cat_id.detach().cpu().item())] + '_nocs_' + str(batch_idx) + "/gt"
                        pred_docstring = class_mapping[int(gt_cat_id.detach().cpu().item())] + '_nocs_' + str(batch_idx) + "/pred"
                        get_event_storage().put_image(gt_docstring, gt_same_nocs)
                        get_event_storage().put_image(pred_docstring, pred_patch)

                        # Only one image per category
                        id_storage.append(gt_cat_id)

                # pose Estimation ------------------------------------------------------------
                # ROI align to predicted box size
                pose_patch = roi_align(torch.unsqueeze(pred_nocs.to(device=device), dim=0), bbox,
                                       output_size=(pred_heigth, pred_width), aligned=True)
                pose_patch = torch.squeeze(pose_patch, dim=0).permute(1,2,0).contiguous() # H x W x C
                pred_rotation, pred_trans, pred_scale, pred_3d_bbox, gt_pointcloud, pred_nocs_pc = run_pose(pose_patch, depth, campose, pred_mask,
                                                                               abs_bbox, vis_obj=False, gt_3d_box=gt_same_3dbox)
                if pred_rotation is None or pred_rotation.size == 0:
                    continue

                rotation_error = get_rotation_diff(gt_same_rot.detach().numpy(), pred_rotation)
                location_error = get_location_diff(gt_same_loc.detach().numpy(), pred_trans)
                #location_error = get_location_diff_boxcenter(gt_same_3dbox.detach().numpy(), pred_3d_bbox)

                if location_error is None or np.isnan(location_error):
                    print('Some issue with location skip this instance for loss calculation')
                    continue

                if rotation_error is None or np.isnan(rotation_error) or np.isinf(rotation_error):
                    print('Some issue with rotation skip this instance for loss calculation')
                    continue

                mean_rotation_error.append(rotation_error)
                mean_location_error.append(location_error)

                # Class agnostic errors
                if cls_name in ['chair', 'table', 'sofa', 'bed', 'tv stand', 'wine cooler', 'nightstand']:
                    cls_rotation_error[cls_name].append(rotation_error)
                    cls_location_error[cls_name].append(location_error)
                else:
                    print('Class not found ...')
                    sys.exit()

                if vis_nocs and gt_pointcloud is not None: # Visualise an image
                    # Projected depth as GT
                    gt_pc_obj = o3d.geometry.PointCloud()
                    gt_pc_obj.points = o3d.utility.Vector3dVector(gt_pointcloud)
                    gt_pc_obj.paint_uniform_color([0.1, 0.1, 0.6]) # GT Blue

                    # Use GT voxel grid and place with predicted pose
                    pred_pointcloud = convert_voxel_to_pc(gt_same_voxel, pred_rotation, pred_trans, pred_scale)
                    pred_pc_obj = o3d.geometry.PointCloud()
                    pred_pc_obj.points = o3d.utility.Vector3dVector(pred_pointcloud)
                    pred_pc_obj.paint_uniform_color([0.6, 0.3, 0.05])
                    gt_objs.append(gt_pc_obj)
                    pred_objs.append(pred_pc_obj)

        # Visualise pose of all objects in a scene
        if False:
            nocs_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
            combined = gt_objs + pred_objs
            combined.append(nocs_origin)

            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=True)
            for obj in combined:
                vis.add_geometry(obj)
                vis.update_geometry(obj)
            ctr = vis.get_view_control()
            ctr.rotate(-45.0, -210.0)
            ctr.scale(2)
            vis.poll_events()
            vis.update_renderer()
            scene_img = vis.capture_screen_float_buffer(do_render=True)
            vis.destroy_window()
            scene_img = np.transpose(np.asarray(scene_img), (2, 0, 1)) # 3 x H x W
            get_event_storage().put_image('scene_' + str(batch_idx), scene_img)

    if not mean_nocs_loss:
        nocs_mean = None
    else:
        nocs_mean = np.array(mean_nocs_loss).mean()

    if not mean_rotation_error:
        rotation_mean = None
    else:
        rotation_mean = np.array(mean_rotation_error).mean()

    if not mean_location_error:
        location_mean = None
    else:
        location_mean = np.array(mean_location_error).mean()

    for class_id, val in cls_rotation_error.items():
        if val:
            cls_rotation_error[class_id] = np.array(val).mean()
        else:
            cls_rotation_error[class_id] = None

    for class_id, val in cls_location_error.items():
        if val:
            cls_location_error[class_id] = np.array(val).mean()
        else:
            cls_location_error[class_id] = None

    return {'nocs valloss': nocs_mean, 'rotation error':rotation_mean, 'location error': location_mean,
            'cls rotation error': cls_rotation_error, 'cls location error': cls_location_error}



def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.
    Args:
        instances (Instances):
        img_id (int): the image id
    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()


    has_voxels = instances.has("pred_voxels")

    if has_voxels:
        voxels = instances.pred_voxels.tolist()

    has_nocs = instances.has("pred_nocs")

    if has_nocs:
        nocs = instances.pred_nocs.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        bin_masks = instances.pred_masks

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k]}

        if has_voxels:
            result["voxel"] = voxels[k]

        if has_nocs:
            result["nocs"] = nocs[k]

        if has_mask:
            result["segmentation"] = bin_masks[k,:,:]

        results.append(result)
    return results

    # inspired from Detectron:
    # https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L255 # noqa


def _evaluate_box_proposals(dataset_predictions, coco_api, thresholds=None, area="all", limit=None):
    """
    Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for prediction_dict in dataset_predictions:
        predictions = prediction_dict["proposals"]

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = predictions.objectness_logits.sort(descending=True)[1]
        predictions = predictions[inds]

        ann_ids = coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
        anno = coco_api.loadAnns(ann_ids)
        gt_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for obj in anno
            if obj["iscrowd"] == 0
        ]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = Boxes(gt_boxes)
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0 or len(predictions) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if limit is not None and len(predictions) > limit:
            predictions = predictions[:limit]

        overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(predictions), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = (
        torch.cat(gt_overlaps, dim=0) if len(gt_overlaps) else torch.zeros(0, dtype=torch.float32)
    )
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }