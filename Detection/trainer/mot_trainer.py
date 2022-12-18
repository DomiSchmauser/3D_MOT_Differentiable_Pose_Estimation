import logging
import os
import torch
import traceback
import roi_heads  # Required for call register()
from collections import OrderedDict
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer

from detectron2.engine import default_argument_parser, default_writers, launch
from detectron2.evaluation import print_csv_format

from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, build_detection_train_loader

from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

from data.mapper_heads import VoxNocsMapper
from evaluator.FrontEvaluator import FrontEvaluator
from evaluator.CocoEvaluator import COCOEvaluator
from evaluator.EvaluatorUtils import inference_on_dataset_voxnocs, inference_on_dataset_coco

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class MOTTrainer(DefaultTrainer):
    '''
    Base 3D MOT trainer class. Runs the 2D object detection, 3D reconstruction and 7-DoF pose estimation
    pipeline.
    '''
    @classmethod
    def build_evaluator_coco(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, ('bbox', 'segm'), True, output_folder)

    @classmethod
    def build_evaluator_voxel_nocs_pipeline(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return FrontEvaluator(dataset_name, ('vox', 'nocs'), True, output_folder)

    @classmethod
    def build_test_loader(cls, cfg):
        dataset_names = cfg.DATASETS.TEST[0]
        return build_detection_test_loader(
            cfg, dataset_names, mapper=VoxNocsMapper(cfg, is_train=False, dataset_names=dataset_names)
        )

    @classmethod
    def build_train_loader(cls, cfg):
        dataset_names = cfg.DATASETS.TRAIN[0]
        return build_detection_train_loader(
            cfg, mapper=VoxNocsMapper(cfg, is_train=True, dataset_names=dataset_names)
        )

    @classmethod
    def do_test(cls, cfg, model, save_img_pred=False):
        logger.info('Start evaluation.')
        results = OrderedDict()

        for dataset_name in cfg.DATASETS.TEST:

            data_loader = cls.build_test_loader(cfg)
            evaluator_voxnocs = cls.build_evaluator_voxel_nocs_pipeline(cfg, dataset_name)
            results_voxnocs = inference_on_dataset_voxnocs(
                model, data_loader, evaluator_voxnocs, logger, cfg, save_img_pred
            )

            evaluator_coco = cls.build_evaluator_coco(cfg, dataset_name)
            results_coco = inference_on_dataset_coco(model, data_loader, evaluator_coco, logger)

            results_coco['voxel'] = results_voxnocs['voxel']
            results_coco['nocs'] = results_voxnocs['nocs']

            results[dataset_name] = results_coco
            if comm.is_main_process():
                assert isinstance(results_coco, dict),\
                    "Evaluator must return a dict on the main process. Got {} instead.".format(results_coco)
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_coco)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def do_train(cls, cfg, model, resume=False):
        logger.info('Start training.')
        model.train()
        optimizer = build_optimizer(cfg, model)
        scheduler = build_lr_scheduler(cfg, optimizer)

        checkpointer = DetectionCheckpointer(
            model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
        )
        start_iter = (
                checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
        )
        max_iter = cfg.SOLVER.MAX_ITER

        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
        )

        writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

        data_loader = cls.build_train_loader(cfg)
        logger.info(f"Starting training from iteration {start_iter} of {max_iter} iterations.")
        with EventStorage(start_iter) as storage:
            for data, iteration in zip(data_loader, range(start_iter, max_iter)):
                storage.iter = iteration

                loss_dict = model(data)

                losses = sum(loss_dict.values())

                if (iteration + 1) % 50 == 0:
                    print(
                        f"Iteration {iteration+1} of {max_iter}, Train loss: {losses.detach().cpu().item()}."
                    )

                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()

                assert cfg.TEST.EVAL_PERIOD > 0, "Evalutation period must best set > 0"
                try:
                    if iteration != max_iter - 1 and (iteration+1) >= cfg.TEST.START_EVAL:
                        if (iteration + 1) % (cfg.TEST.IMG_SAVE_FREQ * cfg.TEST.EVAL_PERIOD) == 0:
                            cls.do_test(cfg, model, save_img_pred=True)
                            comm.synchronize()
                        elif (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0:
                            cls.do_test(cfg, model, save_img_pred=False)
                            comm.synchronize()
                except Exception:
                    traceback.print_exc()

                if (iteration + 1) % 20 == 0 or iteration == max_iter - 1:
                    for writer in writers:
                        writer.write()
                periodic_checkpointer.step(iteration)