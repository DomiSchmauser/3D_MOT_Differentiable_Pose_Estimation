import datetime
import logging
import time, sys, os, math, json
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn
import numpy as np

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils.events import get_event_storage

from detectron2.checkpoint import DetectionCheckpointer


def inference_on_dataset_voxnocs(model, data_loader, evaluator, logger, cfg, save_img_pred):

    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR)

    num_devices = get_world_size()
    logger.info("Start inference on {} images".format(len(data_loader)))


    total = len(data_loader)  # inference data loader must have a fixed length
    all_res = []
    losses = []
    all_box_loss = []
    all_mask_loss = []
    all_cls_loss = []

    evaluator.reset()
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        for idx, inputs in enumerate(data_loader):
            start_single = time.perf_counter()

            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs) # Process inputs, outputs for images in batch, here batch_size = 1
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start

            # Visualise and save 4 image predictions to tensorboard
            if (idx == 0 or idx == math.ceil((total-1)/2) or idx == total-1) and save_img_pred:
                save_ = True
            else:
                save_ = False

            results = evaluator.evaluate(batch_idx=idx, save_img_pred=save_) # Evaluate
            all_res.append(results)
            evaluator.reset()

            if (idx + 1) % 100 == 0 or idx == len(data_loader) - 1:
                print('Evaluation Image: ' + str(idx+1) + ' from ' + str(total) + ' Images')

            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

            loss_batch, box_loss, cls_loss, mask_loss = _get_loss(model, inputs)
            losses.append(loss_batch)
            all_box_loss.append(box_loss)
            all_cls_loss.append(cls_loss)
            all_mask_loss.append(mask_loss)

    mean_loss = np.mean(losses)

    # Save Model with lowest validation loss
    _save_valmodel(mean_loss, checkpointer, cfg.OUTPUT_DIR)

    mean_box_loss = np.mean(all_box_loss)
    mean_cls_loss = np.mean(all_cls_loss)
    mean_mask_loss = np.mean(all_mask_loss)
    logger.info(f"Validation Loss: {mean_loss}")
    get_event_storage().put_scalar("validation/validation_loss", mean_loss)
    get_event_storage().put_scalar("validation/box_loss", mean_box_loss)
    get_event_storage().put_scalar("validation/mask_loss", mean_mask_loss)
    get_event_storage().put_scalar("validation/cls_loss", mean_cls_loss)

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if all_res is None:
        all_res = {}

    final_results = compute_voxnocsmeans(all_res)
    evaluator.reset()

    return final_results


def inference_on_dataset_coco(model, data_loader, evaluator, logger):

    num_devices = get_world_size()
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    all_res = []
    losses = []
    all_voxel_loss = []
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            if 'annotations' in inputs[0]:
                inputs[0].pop("annotations")

            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start

            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                if (idx + 1) % 100 == 0 or idx == len(data_loader) - 1:
                    total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                    eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                    print(f"Inference done {idx + 1}/{total}. "
                            f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                            f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                            f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                            f"Total: {total_seconds_per_iter:.4f} s/iter. "
                            f"ETA={eta}")
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}

    save_ap(results)

    evaluator.reset()

    return results

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

def _get_loss(model, data):
    # How loss is calculated on train_loop
    model.train()
    metrics_dict = model(data)
    model.eval()
    metrics_dict = {
        k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
        for k, v in metrics_dict.items()
    }

    total_losses_reduced = sum(loss for loss in metrics_dict.values())
    box_loss = metrics_dict['loss_box_reg']
    cls_loss = metrics_dict['loss_cls']
    mask_loss = metrics_dict['loss_mask']

    return total_losses_reduced, box_loss, cls_loss, mask_loss

def compute_voxnocsmeans(all_res):

    voxel_val = []
    voxel_lossval = []
    nocs_val = []
    rotation_error = []
    location_error = []
    cls_rotation_error = {'chair': [], 'table': [], 'sofa': [], 'bed': [], 'tv stand': [], 'wine cooler': [], 'nightstand': []}
    cls_location_error = {'chair': [], 'table': [], 'sofa': [], 'bed': [], 'tv stand': [], 'wine cooler': [], 'nightstand': []}


    for res in all_res:
        for key, value in res.items():
            if key == 'vox':
                if value['voxel valacc'] is not None:
                    voxel_val.append(value['voxel valacc'])
                if value['voxel valloss'] is not None:
                    voxel_lossval.append(value['voxel valloss'])
            elif key == 'nocs':
                if value['nocs valloss'] is not None:
                    nocs_val.append(value['nocs valloss'])
                if value['rotation error'] is not None:
                    rotation_error.append(value['rotation error'])
                if value['location error'] is not None:
                    location_error.append(value['location error'])
                for class_id, val in value['cls rotation error'].items():
                    if val is not None and not np.isnan(val):
                        cls_rotation_error[class_id].append(val)
                for class_id, val in value['cls location error'].items():
                    if val is not None and not np.isnan(val):
                        cls_location_error[class_id].append(val)
    if not voxel_val:
        voxel_validation = 0
    else:
        voxel_validation = np.array(voxel_val).mean()
    if not voxel_lossval:
        voxel_lossvalidation = 0
    else:
        voxel_lossvalidation = np.array(voxel_lossval).mean()
    if not nocs_val:
        nocs_validation = 0
    else:
        nocs_validation = np.array(nocs_val).mean()
    if not rotation_error:
        rotation_validation = 0
    else:
        rotation_validation = np.median(np.array(rotation_error))
    if not location_error:
        location_validation = 0
    else:
        location_validation = np.median(np.array(location_error))

    for class_id, val in cls_rotation_error.items():
        if not val:
            cls_rotation_error[class_id] = 0
        else:
            cls_rotation_error[class_id] = np.median(np.array(val))

        get_event_storage().put_scalar("validation/rot_error_" + class_id, cls_rotation_error[class_id])

    for class_id, val in cls_location_error.items():
        if not val:
            cls_location_error[class_id] = 0
        else:
            cls_location_error[class_id] = np.median(np.array(val))

        get_event_storage().put_scalar("validation/loc_error_" + class_id, cls_location_error[class_id])

    final_results = OrderedDict()
    final_results['voxel'] = {'voxel-iou':voxel_validation, 'voxel-loss':voxel_lossvalidation}
    final_results['nocs'] = {'nocs-loss':nocs_validation, 'rotation': rotation_validation,
                             'location': location_validation}

    # Tensorboard logging
    get_event_storage().put_scalar("validation/voxel_iou", voxel_validation)
    get_event_storage().put_scalar("validation/nocs_loss", nocs_validation)
    get_event_storage().put_scalar("validation/voxel_loss", voxel_lossvalidation)
    get_event_storage().put_scalar("validation/rotation_error", rotation_validation)
    get_event_storage().put_scalar("validation/location_error", location_validation)

    return final_results

def _save_valmodel(mean_loss, checkpointer, output_dir):

    json_path = os.path.join(output_dir, 'val_metrics.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
            min_loss = min(data.values())
    else:
        data = {}
        min_loss = math.inf

    now = datetime.datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    data.update({dt_string: mean_loss})
    now = None

    with open(json_path, 'w+') as f:
        json.dump(data, f)

    # save best model
    if mean_loss < min_loss:
        checkpointer.save("best_model_val")


def save_ap(coco_res):
    box_apsofa = None
    box_aptable = None
    box_apchair = None
    box_apbed = None
    seg_apsofa = None
    seg_aptable = None
    seg_apchair = None
    seg_apbed = None

    for key, value in coco_res.items():
        if key == 'bbox':
            box_ap = value['AP']
            box_ap50 = value['AP50']
            if 'AP-sofa' in value.keys():
                box_apsofa = value['AP-sofa']
            if 'AP-table' in value.keys():
                box_aptable = value['AP-table']
            if 'AP-chair' in value.keys():
                box_apchair = value['AP-chair']
            if 'AP-bed' in value.keys():
                box_apbed = value['AP-bed']

        elif key == 'segm':
            seg_ap = value['AP']
            seg_ap50 = value['AP50']
            if 'AP-sofa' in value.keys():
                seg_apsofa = value['AP-sofa']
            if 'AP-table' in value.keys():
                seg_aptable = value['AP-table']
            if 'AP-chair' in value.keys():
                seg_apchair = value['AP-chair']
            if 'AP-bed' in value.keys():
                seg_apbed = value['AP-bed']

    # Tensorboard logging
    get_event_storage().put_scalar("Box_Validation/box_ap", box_ap)
    get_event_storage().put_scalar("Box_Validation/box_ap50", box_ap50)
    if box_apsofa is not None:
        get_event_storage().put_scalar("Box_Validation/box_apsofa", box_apsofa)
    if box_aptable is not None:
        get_event_storage().put_scalar("Box_Validation/box_aptable", box_aptable)
    if box_apchair is not None:
        get_event_storage().put_scalar("Box_Validation/box_apchair", box_apchair)
    if box_apbed is not None:
        get_event_storage().put_scalar("Box_Validation/box_apbed", box_apbed)

    get_event_storage().put_scalar("Mask_Validation/seg_ap", seg_ap)
    get_event_storage().put_scalar("Mask_Validation/seg_ap50", seg_ap50)
    if seg_apsofa is not None:
        get_event_storage().put_scalar("Mask_Validation/seg_apsofa", seg_apsofa)
    if seg_aptable is not None:
        get_event_storage().put_scalar("Mask_Validation/seg_aptable", seg_aptable)
    if seg_apchair is not None:
        get_event_storage().put_scalar("Mask_Validation/seg_apchair", seg_apchair)
    if seg_apbed is not None:
        get_event_storage().put_scalar("Mask_Validation/seg_apbed", seg_apbed)

