import logging
import os, sys, shutil, re, traceback, json
import torch
import datetime
import numpy as np
import pandas as pd
#from torch.utils.data.sampler import SequentialSampler, BatchSampler
from detectron2.data.samplers import TrainingSampler

import warnings
warnings.filterwarnings('ignore')

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
from Detection.utils.postprocess import postprocess_dets

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

# Load tracking setup
from Tracking.options import Options
from Tracking.utils.vis_utils import fuse_pose
from Tracking.tracker.tracking_front import Tracker
from Tracking.visualise.visualise import visualise_gt_sequence, visualise_pred_sequence
from Detection.utils.train_utils import rgb2pc

options = Options()
opts = options.parse()
if opts.use_graph:
    from Tracking.mpn_trainer import Trainer
else:
    from Tracking.trainer import Trainer

from Tracking.utils.eval_utils import get_mota_df


logger = logging.getLogger("front_logger")


class FrontTrainer(DefaultTrainer):
    '''
    Main Trainer class for End-to-End Detection, Pose Estimation and Tracking
    '''

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
            cfg, dataset_names, mapper=VoxNocsMapper(cfg, is_train=False, dataset_names=dataset_names),
        )

    @classmethod
    def build_train_loader(cls, cfg, img_count):
        dataset_names = cfg.DATASETS.TRAIN[0]
        sampler = TrainingSampler(size=img_count, shuffle=False)
        return build_detection_train_loader(
            cfg, mapper=VoxNocsMapper(cfg, is_train=True, dataset_names=dataset_names),
            sampler=sampler
        )

    @classmethod
    def check_save_models(cls, mota_score , model, trainer, cfg):
        '''
        save models based on max mota score
        detection mode is saved without adam optimizer parameters
        '''

        checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR)
        json_path = os.path.join(cfg.OUTPUT_DIR, 'mota_metrics.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
                max_mota = max(data.values())
        else:
            data = {}
            max_mota = -10

        now = datetime.datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        data.update({dt_string: mota_score})
        now = None

        with open(json_path, 'w+') as f:
            json.dump(data, f)

        # save best model Detection and Tracking
        if mota_score > max_mota:
            print('Current model is better than existing, saving ...')
            checkpointer.save("best_model_val")
            trainer.save_end2end_model(path=os.path.join(cfg.OUTPUT_DIR, 'tracking'))


    @classmethod
    def do_test(cls, cfg, model, trainer, evaluate_voxnocs=False, evaluate_coco=False, save_img_pred=False, storage=None, mode='val',
                vis=False, seq_len=25, classwise=True, nograph=False):

        '''
        vis: for dvis visualisation of reconstruction and tracking results
        seq_len: set to length of the input sequence
        classwise: set to True for a per object class MOTA evaluation
        mode: for testing set to 'test'
        '''


        print('Evaluation starts...')
        results = OrderedDict()

        for dataset_name in cfg.DATASETS.TEST:

            data_loader = cls.build_fronttest_loader(cfg)

            # Evaluation Detection
            if evaluate_voxnocs:
                print('Evaluating Voxel and Nocs performance:')
                evaluator_voxnocs = cls.build_evaluator_voxnocs(cfg, dataset_name)
                results_voxnocs = inference_on_dataset_voxnocs(model, data_loader, evaluator_voxnocs, logger, cfg, save_img_pred)

            if evaluate_coco:
                print('Evaluating Detection performance:')
                evaluator_coco = cls.build_evaluator_coco(cfg, dataset_name)
                results_coco = inference_on_dataset_coco(model, data_loader, evaluator_coco, logger)
            else:
                results_coco = dict()

            # Evaluation Tracking
            MOTracker = Tracker(seq_len=seq_len)
            model.eval()
            trainer.set_eval()

            avg_tracking_loss = []

            seq_pattern = mode + "/(.*?)/coco_data"
            seq_outputs = []
            seq_inputs = []

            mota_df = pd.DataFrame()
            classes_df = {
                0: pd.DataFrame(), 1: pd.DataFrame(), 2: pd.DataFrame(),
                3: pd.DataFrame(), 4: pd.DataFrame(),
                5: pd.DataFrame(), 6: pd.DataFrame()
            }

            for idx, data in enumerate(data_loader):

                if int(idx + 1) % 100 == 0:
                    print('Sequence {} of {} Sequences'.format(int((idx + 1) / 25),
                                                               int(len(data_loader) / 25)))

                with torch.no_grad():

                    # For Visualisation use RGB Pointcloud as BG
                    if vis and idx == 0:
                        vis_pc = rgb2pc(data[0]['image'], data[0]['depth_map'], data[0]['campose'])

                    #if int(idx + 1) % 2500 == 0:
                    #    writer = pd.ExcelWriter('mota_comb.xlsx', engine='xlsxwriter')
                    #    mota_df.to_excel(writer, sheet_name='mota')
                    #    writer.save()

                    # Intermediate Logging all 30 sequences
                    if int(idx + 1) % 750 == 0:
                        # Logging
                        mota_score = mota_df.loc[:, 'mota'].mean(axis=0)
                        Prec = mota_df.loc[:, 'precision'].mean(axis=0)  # How many of found are correct
                        Rec = mota_df.loc[:, 'recall'].mean(axis=0)  # How many predictions found
                        num_misses = mota_df.loc[:, 'num_misses'].sum(axis=0)
                        num_false_positives = mota_df.loc[:, 'num_false_positives'].sum(axis=0)
                        id_switches = mota_df.loc[:, 'num_switches'].sum(axis=0)
                        num_objects_gt = mota_df.loc[:, 'num_objects'].sum(axis=0)
                        mota_accumulated = get_mota_df(num_objects_gt, num_misses, num_false_positives, id_switches)
                        print('Accumulated MOTA:', mota_accumulated, ' Averaged MOTA:', mota_score,
                              ' Precision:', Prec,
                              ' Recall:', Rec,
                              'ID switches:', id_switches,
                              ' Current sum Misses:', num_misses,
                              ' Current sum False Positives:', num_false_positives)

                    if int(idx + 1) % 750 == 0 and classwise:
                        cls_mapping = {
                            0: 'chair', 1: 'table', 2: 'sofa',
                            3: 'bed', 4: 'tv_stand',
                            5: 'cooler', 6: 'night_stand'
                        }
                        if classwise:
                            for clss, cls_df in classes_df.items():
                                if cls_df.empty:
                                    continue
                                cls_mota_accumulated = get_mota_df(cls_df.loc[:, 'num_objects'].sum(axis=0),
                                                                   cls_df.loc[:, 'num_misses'].sum(axis=0),
                                                                   cls_df.loc[:, 'num_false_positives'].sum(axis=0),
                                                                   cls_df.loc[:, 'num_switches'].sum(axis=0))
                                print('Class MOTA :', cls_mapping[clss], 'score:', cls_mota_accumulated)

                    # Process Tracking evaluation
                    outputs = model(data)

                    name = data[0]['file_name']
                    seq_name = re.search(seq_pattern, name).group(1)

                    if idx == 0:
                        previous_seq = seq_name

                    # If same sequence append prediction, if end sequence evaluate all 25 frames at a time
                    if (idx+1) == len(data_loader):
                        previous_seq = seq_name
                        print('Final sequence, evaluating ...')
                        seq_outputs.append(outputs[0])
                        seq_inputs.append(data[0])

                        # Postprocess and run Tracking network
                        window_seq_data = postprocess_dets(seq_inputs, seq_outputs, obj_threshold=0.35, iou_threshold=0.35,
                                                           mode=mode, vis=vis)

                        window_seq_data = [window_seq_data]
                        tracking_outputs, tracking_losses = trainer.process_batch_combined(window_seq_data, mode=mode, vis_pose=vis)
                        if not tracking_outputs: #todo might wanna remove this
                            print('Empty predictions skipping ...')
                            continue

                        if vis:
                            pred_trajectories, gt_trajectories = MOTracker.analyse_trajectories_vis(window_seq_data[0], tracking_outputs[0], vis_pc=vis_pc)
                            visualise_gt_sequence(gt_trajectories, seq_name=window_seq_data[0][0]['scene'], seq_len=seq_len)
                            pred_trajectories = fuse_pose(pred_trajectories, seq_len=seq_len)
                            visualise_pred_sequence(pred_trajectories, gt_trajectories, seq_name=window_seq_data[0][0]['scene'])
                            continue

                        tracking_losses = tracking_losses['BCE_loss']
                        avg_tracking_loss.append(tracking_losses.detach().cpu().item())

                        # Tracking
                        try:
                            if nograph:
                                pred_trajectories, gt_trajectories = MOTracker.analyse_trajectories_heur(window_seq_data[0],
                                                                                                    tracking_outputs[0])
                            else:
                                pred_trajectories, gt_trajectories = MOTracker.analyse_trajectories(window_seq_data[0],
                                                                                                tracking_outputs[0])
                            gt_traj_tables = MOTracker.get_traj_tables(gt_trajectories, 'gt')
                            pred_traj_tables = MOTracker.get_traj_tables(pred_trajectories, 'pred')
                            if classwise:
                                seq_mota_summary, mot_events = MOTracker.eval_mota_classwise(pred_traj_tables,
                                                                                                gt_traj_tables)
                            else:
                                seq_mota_summary = MOTracker.eval_mota(pred_traj_tables, gt_traj_tables)
                            mota_df = pd.concat([mota_df, seq_mota_summary], axis=0, ignore_index=True)

                            if classwise:
                                for clss, cls_df in classes_df.items():
                                    gt_cls_traj_tables = gt_traj_tables[gt_traj_tables['obj_cls'] == clss]
                                    if gt_cls_traj_tables.empty:
                                        continue
                                    matches = mot_events[mot_events['Type'] == 'MATCH']
                                    class_ids = gt_cls_traj_tables['obj_idx'].unique()
                                    filtered_matched = matches[matches['HId'].isin(class_ids)]  # all mate
                                    frame_idxs = filtered_matched.index.droplevel(1)
                                    obj_idxs = filtered_matched['HId']
                                    fp_cls_traj_tables = pred_traj_tables.loc[
                                        pred_traj_tables['scan_idx'].isin(frame_idxs) & pred_traj_tables[
                                            'obj_idx'].isin(obj_idxs)]
                                    pred_cls_traj_tables = pred_traj_tables[pred_traj_tables['obj_cls'] == clss]
                                    pred_merge_table = pd.concat(
                                        [fp_cls_traj_tables, pred_cls_traj_tables]).drop_duplicates()
                                    # print('IS EQUAL :', pred_merge_table.equals(pred_cls_traj_tables) )
                                    class_mota_summary, _ = MOTracker.eval_mota_classwise(pred_merge_table, gt_cls_traj_tables)
                                    classes_df[clss] = pd.concat([cls_df, class_mota_summary], axis=0, ignore_index=True)

                        except:
                            traceback.print_exc()
                            continue

                    elif seq_name == previous_seq:
                        previous_seq = seq_name
                        seq_outputs.append(outputs[0])
                        seq_inputs.append(data[0])

                    else:
                        previous_seq = seq_name
                        # Postprocess and run Tracking network
                        window_seq_data = postprocess_dets(seq_inputs, seq_outputs, obj_threshold=0.35, iou_threshold=0.35,
                                                           mode=mode, vis=vis)

                        # Initialize new sequence
                        seq_outputs = [outputs[0]]
                        seq_inputs = [data[0]]

                        window_seq_data = [window_seq_data]
                        tracking_outputs, tracking_losses = trainer.process_batch_combined(window_seq_data, mode=mode)
                        if not tracking_outputs: #todo might wanna remove this
                            print('Empty predictions skipping ...')
                            continue

                        tracking_losses = tracking_losses['BCE_loss']
                        avg_tracking_loss.append(tracking_losses.detach().cpu().item())

                        # Tracking
                        try:
                            if nograph:
                                pred_trajectories, gt_trajectories = MOTracker.analyse_trajectories_heur(
                                    window_seq_data[0],
                                    tracking_outputs[0])
                            else:
                                pred_trajectories, gt_trajectories = MOTracker.analyse_trajectories(window_seq_data[0],
                                                                                                    tracking_outputs[0])
                            gt_traj_tables = MOTracker.get_traj_tables(gt_trajectories, 'gt')
                            pred_traj_tables = MOTracker.get_traj_tables(pred_trajectories, 'pred')
                            if classwise:
                                seq_mota_summary, mot_events = MOTracker.eval_mota_classwise(pred_traj_tables,
                                                                                             gt_traj_tables)
                            else:
                                seq_mota_summary = MOTracker.eval_mota(pred_traj_tables, gt_traj_tables)
                            mota_df = pd.concat([mota_df, seq_mota_summary], axis=0, ignore_index=True)

                            if classwise:
                                for clss, cls_df in classes_df.items():
                                    gt_cls_traj_tables = gt_traj_tables[gt_traj_tables['obj_cls'] == clss]
                                    if gt_cls_traj_tables.empty:
                                        continue
                                    matches = mot_events[mot_events['Type'] == 'MATCH']
                                    class_ids = gt_cls_traj_tables['obj_idx'].unique()
                                    filtered_matched = matches[matches['HId'].isin(class_ids)]  # all mate
                                    frame_idxs = filtered_matched.index.droplevel(1)
                                    obj_idxs = filtered_matched['HId']
                                    fp_cls_traj_tables = pred_traj_tables.loc[
                                        pred_traj_tables['scan_idx'].isin(frame_idxs) & pred_traj_tables[
                                            'obj_idx'].isin(obj_idxs)]
                                    pred_cls_traj_tables = pred_traj_tables[pred_traj_tables['obj_cls'] == clss]
                                    pred_merge_table = pd.concat(
                                        [fp_cls_traj_tables, pred_cls_traj_tables]).drop_duplicates()
                                    # print('IS EQUAL :', pred_merge_table.equals(pred_cls_traj_tables) )
                                    class_mota_summary, _ = MOTracker.eval_mota_classwise(pred_merge_table, gt_cls_traj_tables)
                                    classes_df[clss] = pd.concat([cls_df, class_mota_summary], axis=0, ignore_index=True)
                        except:
                            traceback.print_exc()
                            continue

            # Logging
            print('Final tracking scores :')
            mota_score = mota_df.loc[:, 'mota'].mean(axis=0)
            Prec = mota_df.loc[:, 'precision'].mean(axis=0)  # How many of found are correct
            Rec = mota_df.loc[:, 'recall'].mean(axis=0)  # How many predictions found
            num_misses = mota_df.loc[:, 'num_misses'].sum(axis=0)
            num_false_positives = mota_df.loc[:, 'num_false_positives'].sum(axis=0)
            id_switches = mota_df.loc[:, 'num_switches'].sum(axis=0)
            num_objects_gt = mota_df.loc[:, 'num_objects'].sum(axis=0)
            mota_accumulated = get_mota_df(num_objects_gt, num_misses, num_false_positives, id_switches)
            print('Accumulated MOTA:', mota_accumulated, ' Averaged MOTA:', mota_score,
                  ' Precision:', Prec,
                  ' Recall:', Rec,
                  'ID switches:', id_switches,
                  ' Current sum Misses:', num_misses,
                  ' Current sum False Positives:', num_false_positives)

            cls_mapping = {
                0: 'chair', 1: 'table', 2: 'sofa',
                3: 'bed', 4: 'tv_stand',
                5: 'cooler', 6: 'night_stand'
            }
            if classwise:
                for clss, cls_df in classes_df.items():
                    if cls_df.empty:
                        continue
                    cls_mota_accumulated = get_mota_df(cls_df.loc[:, 'num_objects'].sum(axis=0),
                                                       cls_df.loc[:, 'num_misses'].sum(axis=0),
                                                       cls_df.loc[:, 'num_false_positives'].sum(axis=0),
                                                       cls_df.loc[:, 'num_switches'].sum(axis=0))
                    print('Class MOTA :', cls_mapping[clss], 'score:', cls_mota_accumulated)

            if evaluate_voxnocs:
                results_coco['voxel'] = results_voxnocs['voxel']
                results_coco['nocs'] = results_voxnocs['nocs']

            results_coco['precision'] = Prec
            results_coco['recall'] = Rec
            results_coco['f1'] = 2 * (Rec * Prec) / (Rec + Prec)
            results_coco['mota'] = mota_accumulated

            results[dataset_name] = results_coco
            if comm.is_main_process():
                assert isinstance(results_coco, dict), "Evaluator must return a dict on the main process. Got {} instead.".format(results_coco)
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_coco)

        # Tensorboard Logging
        if storage is not None:
            storage.put_scalar("validation/tracking_loss", np.array(avg_tracking_loss).mean())
            storage.put_scalar("validation/F1", 2 * (Rec * Prec) / (Rec + Prec))
            storage.put_scalar("validation/MOTA", mota_accumulated)

        # Model saving
        cls.check_save_models(mota_accumulated, model, trainer, cfg)

        if len(results) == 1:
            results = list(results.values())[0]

        # Convert back to training mode
        model.train()
        trainer.set_train()

        return results

    @classmethod
    def do_train(cls, cfg, model, resume=False, img_count=None, eval_first=True, eval_only=False):

        # Detection Classes
        print('Training starts...')
        model.train()
        optimizer = build_optimizer(cfg, model)
        scheduler = build_lr_scheduler(cfg, optimizer)

        checkpointer = DetectionCheckpointer(
            model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
        )
        start_iter = (
                checkpointer.resume_or_load(os.path.join(CONF.PATH.DETECTMODEL, 'best_model.pth'), resume=resume).get("iteration", -1) + 1
        )
        max_iter = start_iter + cfg.SOLVER.MAX_ITER

        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
        )

        writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

        data_loader = cls.build_train_loader(cfg, img_count)
        logger.info("Starting training from iteration {}".format(start_iter))

        # Tracking classes
        trainer = Trainer(opts, combined=True)
        tracking_optimizer = trainer.build_optimizer()
        trainer.set_train()

        # Load pretrained tracking network
        if resume:
            trainer.load_model()

        with EventStorage(start_iter) as storage:

            # Initial Eval step to get comparison values
            if eval_first and eval_only:
                cls.do_test(cfg, model, trainer, save_img_pred=False, storage=storage, mode='test')
                print('Evalutation done, exiting ...')
                quit()
            elif eval_first:
                cls.do_test(cfg, model, trainer, save_img_pred=False, storage=storage, mode='val')
            # For debugging
            #torch.autograd.set_detect_anomaly(False)
            for data, iteration in zip(data_loader, range(start_iter, max_iter)):

                # Assert data from same sequence otherwise skip batch for single sequence overlaps time  #todo very ugly atm, not proper for single seq overfit
                seq_names = []
                seq_pattern = 'train' + "/(.*?)/coco_data"
                for d in data:
                    name = d['file_name']
                    seq_name = re.search(seq_pattern, name).group(1)
                    seq_names.append(seq_name)

                # Skip batches with not consecutive sequences
                if not seq_names.count(seq_names[0]) == len(seq_names):
                    continue

                # Start pipeline Detection
                storage.iter = iteration

                loss_dict = model(data)

                # Model in eval mode and get outputs model.eval()
                model.eval()
                outputs = model(data)
                model.train()

                # Postprocess Pose Estimation and run Tracking network
                try: # todo precompute gt box crops
                    window_seq_data = postprocess_dets(data, outputs, obj_threshold=0.35, iou_threshold=0.35, mode='train')
                except:
                    traceback.print_exc()
                    continue

                # Penalize Detection
                process_tracking = True

                # Either skip using or keep scenes
                for win_d in window_seq_data:
                    if 'classes' not in win_d:
                        print('Batch {} contains image with no predicted objects skipping ...'.format(seq_name))
                        process_tracking = False

                if process_tracking:
                    window_seq_data = [window_seq_data]
                    _, tracking_losses = trainer.process_batch_combined(window_seq_data)
                    tracking_losses = tracking_losses['BCE_loss']
                    # Check for empty predictions and skip those instances, CE loss has singularity at 0 and 1
                    if tracking_losses == float('-inf') or tracking_losses == 0.0:
                        process_tracking = False

                # Losses Detection
                losses = sum(loss_dict.values())

                if (iteration + 1) % 5 == 0 and process_tracking:
                    print('Iteration ', iteration+1,' of ', max_iter, ' , Training Loss Detection: ', losses.detach().cpu().item(),
                          ' , Training Loss Tracking: ', tracking_losses.detach().cpu().item())

                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
                    if process_tracking:
                        storage.put_scalar("tracking loss", tracking_losses.detach().cpu().item())

                # Gradient update
                optimizer.zero_grad()
                losses.backward() #inputs only detection parameters, lr reduce
                optimizer.step()

                if process_tracking:
                    tracking_optimizer.zero_grad()
                    tracking_losses.backward(inputs=trainer.get_parameters())
                    tracking_optimizer.step()

                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()

                if (cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % (cfg.TEST.IMG_SAVE_FREQ * cfg.TEST.EVAL_PERIOD) == 0 and iteration != max_iter - 1 and (iteration+1) >= cfg.TEST.START_EVAL):
                    cls.do_test(cfg, model, trainer, save_img_pred=True, evaluate_coco=False, evaluate_voxnocs=False, storage=storage)
                    comm.synchronize()
                elif (cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0 and iteration != max_iter - 1 and (iteration+1) >= cfg.TEST.START_EVAL):
                    cls.do_test(cfg, model, trainer, save_img_pred=False, storage=storage)
                    comm.synchronize()

                if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                    for writer in writers:
                        writer.write()

                periodic_checkpointer.step(iteration)

## ------------------------------ Class methods end ------------------
def setup():
    TRAIN_IMG_DIR = CONF.PATH.DETECTTRAIN
    mapping_list, name_list, img_count = get_dataset_info(TRAIN_IMG_DIR, combined=True)
    mapping_list, name_list = zip(*sorted(zip(mapping_list, name_list)))

    if isinstance(name_list, tuple):
        name_list = list(name_list)

    if isinstance(mapping_list, tuple):
        mapping_list = list(mapping_list)

    #num_classes = len(mapping_list)
    num_classes = 7
    cfg = init_cfg(num_classes, combined=True)
    return cfg, mapping_list, name_list, img_count


def main(args, use_pretrained=True):
    cfg, mapping_list, name_list, img_count = setup()
    print('Existing Classes :', name_list)
    print('Num Images :', img_count)

    register_cls = RegisterDataset(mapping_list, name_list)
    register_cls.reg_dset()
    #register_cls.eval_annotation()

    # Remove old files
    if os.path.exists(CONF.PATH.DETECTOUTPUT):
        print('Removing old outputs ...')
        shutil.rmtree(CONF.PATH.DETECTOUTPUT)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    model_path = os.path.join(CONF.PATH.DETECTMODEL, 'best_model.pth')

    if use_pretrained:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            model_path, resume=True
        )


    '''
    Main Network training and evaluation function:
    
    Set training and evaluation parameters here:
    eval_first: executes an evaluation run before training the end-to-end pipeline
    eval_only: only executes evaluation for a validation or test run
    resume: uses a pretrained Detection and Tracking network 
    '''
    FrontTrainer.do_train(cfg, model, resume=use_pretrained, img_count=img_count, eval_first=True, eval_only=False)

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
