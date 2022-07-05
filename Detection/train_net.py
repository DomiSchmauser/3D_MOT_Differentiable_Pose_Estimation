import logging
import os, sys, shutil
import torch
import roi_heads #Required for call register()
from collections import OrderedDict
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer

from detectron2.engine import default_argument_parser, default_writers, launch
from detectron2.evaluation import print_csv_format

from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, build_detection_train_loader

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

from register_dataset import RegisterDataset
from data.mapper_heads import VoxNocsMapper
from evaluator.FrontEvaluator import FrontEvaluator
from evaluator.CocoEvaluator import COCOEvaluator
from evaluator.EvaluatorUtils import inference_on_dataset_voxnocs, inference_on_dataset_coco
from Utility.analyse_datset import get_dataset_info
from cfg_setup import init_cfg

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF


logger = logging.getLogger("front_logger")


class FrontTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator_coco(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, ('bbox', 'segm'), True, output_folder)

    @classmethod
    def build_evaluator_voxnocs(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return FrontEvaluator(dataset_name, ('vox', 'nocs'), True, output_folder)

    @classmethod
    def build_fronttest_loader(cls, cfg):
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
        print('Evaluation starts...')
        results = OrderedDict()

        for dataset_name in cfg.DATASETS.TEST:

            data_loader = cls.build_fronttest_loader(cfg)
            evaluator_voxnocs = cls.build_evaluator_voxnocs(cfg, dataset_name)
            results_voxnocs = inference_on_dataset_voxnocs(model, data_loader, evaluator_voxnocs, logger, cfg, save_img_pred)

            evaluator_coco = cls.build_evaluator_coco(cfg, dataset_name)
            results_coco = inference_on_dataset_coco(model, data_loader, evaluator_coco, logger)

            results_coco['voxel'] = results_voxnocs['voxel']
            results_coco['nocs'] = results_voxnocs['nocs']

            results[dataset_name] = results_coco
            if comm.is_main_process():
                assert isinstance(results_coco, dict), "Evaluator must return a dict on the main process. Got {} instead.".format(results_coco)
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_coco)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def do_train(cls, cfg, model, resume=False):
        print('Training starts...')
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
        logger.info("Starting training from iteration {}".format(start_iter))
        with EventStorage(start_iter) as storage:
            for data, iteration in zip(data_loader, range(start_iter, max_iter)):
                storage.iter = iteration

                loss_dict = model(data)

                losses = sum(loss_dict.values())

                if (iteration + 1) % 100 == 0:
                    print('Iteration ', iteration+1,' of ', max_iter, ' , Training Loss: ', losses.detach().cpu().item())

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

                if (cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % (cfg.TEST.IMG_SAVE_FREQ * cfg.TEST.EVAL_PERIOD) == 0 and iteration != max_iter - 1 and (iteration+1) >= cfg.TEST.START_EVAL):
                    cls.do_test(cfg, model, save_img_pred=True)
                    comm.synchronize()
                elif (cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0 and iteration != max_iter - 1 and (iteration+1) >= cfg.TEST.START_EVAL):
                    cls.do_test(cfg, model, save_img_pred=False)
                    comm.synchronize()

                if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                    for writer in writers:
                        writer.write()
                periodic_checkpointer.step(iteration)

## ------------------------------ Class methods end ------------------
def setup():
    TRAIN_IMG_DIR = CONF.PATH.DETECTTRAIN
    mapping_list, name_list = get_dataset_info(TRAIN_IMG_DIR)
    mapping_list, name_list = zip(*sorted(zip(mapping_list, name_list)))

    num_classes = len(mapping_list)
    cfg = init_cfg(num_classes)
    return cfg, mapping_list, name_list


def main(args):
    cfg, mapping_list, name_list = setup()
    print('Existing Classes :', name_list)

    register_cls = RegisterDataset(mapping_list, name_list)
    register_cls.reg_dset()

    # Visualise annotations for debugging
    # register_cls.eval_annotation()

    # Remove old files
    if os.path.exists(CONF.PATH.DETECTOUTPUT):
        print('Removing old outputs ...')
        shutil.rmtree(CONF.PATH.DETECTOUTPUT)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        print('ONLY EVALUATION')
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return FrontTrainer.do_test(cfg, model, False)

    FrontTrainer.do_train(cfg, model, resume=args.resume)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
