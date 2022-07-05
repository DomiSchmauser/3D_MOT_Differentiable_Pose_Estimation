from __future__ import absolute_import, division, print_function

import sys, os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import math
import time, datetime
import numpy as np
import pandas as pd
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch_geometric.utils import to_networkx
from torch.nn import functional as F


import json

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

from Tracking.graph_cfg import init_graph_cfg
from Tracking import networks
from Tracking import datasets
from Tracking.utils.train_utils import check_pair, sec_to_hm_str, init_weights, get_quaternion_from_euler
from Tracking.utils.eval_utils import get_precision, get_recall, get_f1, get_MOTA, get_mota_df
from Tracking.utils.vis_utils import visualize_graph
from Tracking.tracker.tracking_front import Tracker

class Trainer:

    def __init__(self, options, combined=False):
        self.opt = options
        self.combined = combined
        if not self.combined:
            self.log_path = CONF.PATH.TRACKOUTPUT

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.box_iou_thres = 0.01 # Min IoU threshold GT and predicted 3D box

        # Model setup --------------------------------------------------------------------------------------------------

        self.voxel_encoding_size = 16
        self.models["voxel_encoder"] = networks.VoxelEncoder(input_channel=1, output_channel=self.voxel_encoding_size)
        self.models["voxel_encoder"].to(self.device)

        self.graph_cfg = init_graph_cfg(node_in_size=self.voxel_encoding_size)
        if self.opt.rel_app:
            self.graph_cfg['encoder_feats_dict']['edge_in_dim'] += 1 # add 1 appearance feature

        if self.opt.as_quaternion:
            self.graph_cfg['encoder_feats_dict']['edge_in_dim'] += 1 # add 1 rotation feature

        self.time_aware_mp = self.graph_cfg['use_time_aware_mp']
        self.models['graph_net'] = networks.MPGraph(model_params=self.graph_cfg, time_aware_mp=self.time_aware_mp, use_leaky_relu=self.graph_cfg['use_leaky_relu'])
        self.models['graph_net'].to(self.device)

        classifier_input_dim = self.graph_cfg['edge_model_feats_dict']['fc_dims'][-1]
        self.is_undirected = self.graph_cfg['undirected_graph']
        self.max_frame_dist = self.graph_cfg['max_frame_dist']

        self.models["edge_classifier"] = networks.EdgeClassifier(input_dim=classifier_input_dim, intermed_dim=8)
        self.models["edge_classifier"].to(self.device)

        init_weights(self.models["edge_classifier"], init_type='kaiming', init_gain=0.02)
        init_weights(self.models["voxel_encoder"], init_type='kaiming', init_gain=0.02)
        init_weights(self.models["graph_net"], init_type='kaiming', init_gain=0.02)

        self.parameters_to_train += list(self.models["voxel_encoder"].parameters())
        self.parameters_to_train += list(self.models["edge_classifier"].parameters())
        self.parameters_to_train += list(self.models["graph_net"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate, weight_decay=self.opt.weight_decay)

        #self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, 15, 0.5)

        self.dataset = datasets.Front_dataset
        self.graph_dataset = {}

        if self.opt.use_augmentation:
            print('Using data augmentation ...')
            self.transform = transforms.Compose([transforms.RandomRotation(degrees=(-20, 20)),
                                                transforms.RandomHorizontalFlip(p=0.8)])
        else:
            self.transform = None

        # Loss Function ---------------------------------------------------------------------------------------------
        if self.opt.use_triplet:
            print('Using Triplet Loss for gradients ...')
            self.criterion = nn.TripletMarginLoss(margin=1.0, p=2.0, reduction='mean') # p=2 is euclidian dist, m=1 margin between anchor and negative sample
            self.loss_key = 'Triplet_loss'
        elif self.opt.use_l1:
            print('Using L1 Loss for gradients ...')
            self.criterion = nn.L1Loss(reduction='mean')
            self.loss_key = 'L1_loss'
        else:
            print('Using BCE Loss for gradients ...')
            self.loss_key = 'BCE_loss'

        # Tracking ---------------------------------------------------------------------------------------------------
        self.Tracker = Tracker()
        # Dataset ----------------------------------------------------------------------------------------------------
        if not self.combined:
            DATA_DIR = CONF.PATH.TRACKDATA

            train_dataset = self.dataset(
                base_dir=DATA_DIR,
                split='train',
                transform=self.transform,
                with_scene_pc=False)

            self.train_loader = DataLoader(
                train_dataset,
                self.opt.batch_size,
                shuffle=True,
                num_workers=self.opt.num_workers,
                collate_fn=lambda x:x,
                pin_memory=True,
                drop_last=True)

            val_dataset = self.dataset(
                base_dir=DATA_DIR,
                split='val',
                transform=self.transform,
                with_scene_pc=False)

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.opt.batch_size,
                shuffle=False,
                num_workers=self.opt.num_workers,
                collate_fn=lambda x:x,
                pin_memory=True,
                drop_last=False)

            test_dataset = self.dataset(
                base_dir=DATA_DIR,
                split='test',
                transform=self.transform,
                with_scene_pc=False)

            self.test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.opt.num_workers,
                collate_fn=lambda x: x,
                pin_memory=True,
                drop_last=False)

            if not os.path.exists(self.opt.log_dir):
                os.makedirs(self.opt.log_dir)

            self.writers = {}
            for mode in ["train", "val"]:
                logging_path = os.path.join(self.opt.log_dir, mode)
                self.writers[mode] = SummaryWriter(logging_path)

            num_train_samples = len(train_dataset)
            num_eval_samples = len(val_dataset)
            print("There are {} training sequences and {} validation sequences in total...".format(num_train_samples,
                                                                                           num_eval_samples))
            self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
            print('Processing with batch size :', self.opt.batch_size)

    def build_optimizer(self):
        '''
        Used in end-to-end training for optimizer export
        '''
        return self.model_optimizer

    def get_parameters(self):
        '''
        Used in end-to-end training for backprop
        '''
        return self.parameters_to_train

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """
        Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        if self.opt.resume:
            print('Loading pretrained models and resume training...')
            self.load_model()

        for self.epoch in range(self.opt.num_epochs):
            print('Start training ...')
            self.run_epoch()
            if (self.epoch+1) % self.opt.save_frequency == 0 \
                    and self.opt.save_model and (self.epoch+1) >= self.opt.start_saving:
                self.save_model(is_val=False)

    def inference(self, vis_pose=False, classwise=True):
        """
        Run the entire inference pipeline
        """
        print("Starting inference and loading models ...")
        self.start_time = time.time()
        self.load_model()
        self.set_eval()

        mota_df = pd.DataFrame()
        classes_df = {
            0: pd.DataFrame(), 1: pd.DataFrame(), 2: pd.DataFrame(),
            3: pd.DataFrame(), 4: pd.DataFrame(),
            5: pd.DataFrame(), 6: pd.DataFrame()
        }
        classes_old_df = {
            0: pd.DataFrame(), 1: pd.DataFrame(), 2: pd.DataFrame(),
            3: pd.DataFrame(), 4: pd.DataFrame(),
            5: pd.DataFrame(), 6: pd.DataFrame()
        }

        for batch_idx, inputs in enumerate(self.test_loader):
            if int(batch_idx + 1) % 20 == 0:
                print('Sequence {} of {} Sequences'.format(int((batch_idx+1)), int(len(self.test_loader))))

            with torch.no_grad():
                outputs, _ = self.process_batch(inputs, mode='test')

            # Tracking
            pred_trajectories, gt_trajectories = self.Tracker.analyse_trajectories(inputs[0], outputs[0])
            gt_traj_tables = self.Tracker.get_traj_tables(gt_trajectories, 'gt')
            pred_traj_tables = self.Tracker.get_traj_tables(pred_trajectories, 'pred')
            if classwise:
                seq_mota_summary, mot_events = self.Tracker.eval_mota_classwise(pred_traj_tables, gt_traj_tables)
            else:
                seq_mota_summary = self.Tracker.eval_mota(pred_traj_tables, gt_traj_tables)
            mota_df = pd.concat([mota_df, seq_mota_summary], axis=0, ignore_index=True)

            if classwise:

                # Just run eval MOTA on all classes dfs and aggregate seperately
                for cls, cls_df in classes_df.items():
                    gt_cls_traj_tables = gt_traj_tables[gt_traj_tables['obj_cls'] == cls]
                    if gt_cls_traj_tables.empty:
                        continue
                    # Get assignments
                    matches = mot_events[mot_events['Type'] == 'MATCH']
                    class_ids = gt_cls_traj_tables['obj_idx'].unique()
                    filtered_matched = matches[matches['HId'].isin(class_ids)] # all mate
                    frame_idxs = filtered_matched.index.droplevel(1)
                    obj_idxs = filtered_matched['HId']
                    fp_cls_traj_tables = pred_traj_tables.loc[pred_traj_tables['scan_idx'].isin(frame_idxs) & pred_traj_tables['obj_idx'].isin(obj_idxs)]
                    pred_cls_traj_tables = pred_traj_tables[pred_traj_tables['obj_cls'] == cls]
                    pred_merge_table = pd.concat([fp_cls_traj_tables, pred_cls_traj_tables]).drop_duplicates()
                    #print('IS EQUAL :', pred_merge_table.equals(pred_cls_traj_tables) )
                    class_mota_summary = self.Tracker.eval_mota(pred_merge_table, gt_cls_traj_tables)
                    classes_df[cls] = pd.concat([cls_df, class_mota_summary], axis=0, ignore_index=True)

                    #class_mota_summary_old = self.Tracker.eval_mota(pred_cls_traj_tables, gt_cls_traj_tables)
                    #classes_old_df[cls] = pd.concat([classes_old_df[cls], class_mota_summary_old], axis=0, ignore_index=True)

            if vis_pose:
                pred_trajectories, gt_trajectories = self.Tracker.analyse_trajectories_vis(inputs[0], outputs[0])

            if int(batch_idx + 1) % 10 == 0:
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

            if int(batch_idx + 1) % 40 == 0 and classwise:
                cls_mapping = {
                    0: 'chair', 1: 'table', 2: 'sofa',
                    3: 'bed', 4: 'tv_stand',
                    5: 'cooler', 6: 'night_stand'
                }
                if classwise:
                    for cls, cls_df in classes_df.items():
                        if cls_df.empty:
                            continue
                        cls_mota_accumulated = get_mota_df(cls_df.loc[:, 'num_objects'].sum(axis=0),
                                                           cls_df.loc[:, 'num_misses'].sum(axis=0),
                                                           cls_df.loc[:, 'num_false_positives'].sum(axis=0),
                                                           cls_df.loc[:, 'num_switches'].sum(axis=0))
                        print('Class MOTA :', cls_mapping[cls], 'score:', cls_mota_accumulated)
                        '''
                        cls_mota_accumulated = get_mota_df(classes_old_df[cls].loc[:, 'num_objects'].sum(axis=0),
                                                           classes_old_df[cls].loc[:, 'num_misses'].sum(axis=0),
                                                           classes_old_df[cls].loc[:, 'num_false_positives'].sum(axis=0),
                                                           classes_old_df[cls].loc[:, 'num_switches'].sum(axis=0))
                        print('Class MOTA old:', cls_mapping[cls], 'score:', cls_mota_accumulated)
                        '''



        # Final Logging
        print('Final tracking scores :')
        mota_score = mota_df.loc[:, 'mota'].mean(axis=0)
        Prec = mota_df.loc[:, 'precision'].mean(axis=0)
        Rec = mota_df.loc[:, 'recall'].mean(axis=0)
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
            for cls, cls_df in classes_df.items():
                if cls_df.empty:
                    continue
                cls_mota_accumulated = get_mota_df(cls_df.loc[:, 'num_objects'].sum(axis=0), cls_df.loc[:, 'num_misses'].sum(axis=0),
                                               cls_df.loc[:, 'num_false_positives'].sum(axis=0), cls_df.loc[:, 'num_switches'].sum(axis=0))
                print('Class MOTA :', cls_mapping[cls], 'score:', cls_mota_accumulated )


    def run_epoch(self):
        """
        Run a single epoch of training and validation
        """
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            _, losses = self.process_batch(inputs, mode='train')

            loss = losses[self.loss_key]

            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            self.opt.log_frequency = 1
            if int(batch_idx + 1) % self.opt.log_frequency == 0:
                self.log_time(batch_idx, duration, loss.cpu().data)
                self.log("train", losses)

            self.step += 1
        #self.model_lr_scheduler.step()
        self.val()

    def process_batch(self, inputs, mode='train', debug_mode=False):

        '''
        1. Siamese Network encodes voxel grids in feature space, MLP Encodes relative pose for latent edge features
        2. Get active/non-active edges (TARGETS) by computing IoU pred and GT and compare GT object ids
        3. Neural Message Passing
        4. Edge classification into active non-active
        '''

        batch_loss = 0
        batch_size = len(inputs)
        batch_output = []

        for batch_idx, input in enumerate(inputs):

            num_imgs = len(input)
            # MOTA calculation
            total_gt_objs = 0
            total_pred_objs = 0
            misses = 0

            # Graph features
            instances_count = []
            voxel_features = []
            rotations = []
            translations = []
            scales = []

            for i in range(num_imgs):

                num_instances = int(input[i]['voxels'].shape[0])
                instances_count.append(num_instances)  # count predicted instances per image

                if num_instances != 0:
                    # One voxel batch consists of all instances in one image
                    voxels = torch.unsqueeze(input[i]['voxels'], dim=1) # num_instances x 1 x 32 x 32 x 32
                    voxel_feature = self.models["voxel_encoder"](voxels.to(self.device)) # num_instances x feature_dim
                    voxel_features.append(voxel_feature)

                    rot = input[i]['rotations']  #  num_instances x 3
                    if self.opt.as_quaternion:
                        rot = get_quaternion_from_euler(rot[:, 0], rot[:, 1], rot[:, 2]) #num instances x 4
                    rotations.append(rot)
                    trans = input[i]['translations']#  num_instances x 3
                    translations.append(trans)
                    scale = torch.unsqueeze(input[i]['scales'], -1) # num_instances x 1
                    scales.append(scale)

                # For MOTA score calculation
                per_img_gt_objs = int(input[i]['gt_object_id'].shape[-1])
                total_gt_objs += per_img_gt_objs # Number of ground truth objects in one frame
                total_pred_objs += num_instances # Number of predicted objects in one frame

                # Missing detections/ False Negatives per image for a GT box no matching pred box found
                if num_instances < per_img_gt_objs:
                    misses += per_img_gt_objs - num_instances

            #voxel_features = torch.cat(voxel_features, dim=0)
            if voxel_features:
                voxel_features = torch.cat(voxel_features, dim=0)
            else:
                # Skipping empty predictions for occuring buggy scenes
                losses = dict()
                losses[self.loss_key] = torch.tensor(float(0), device=self.device, requires_grad=True)
                return batch_output, losses

            rotations = torch.cat(rotations, dim=0).to(self.device)
            translations = torch.cat(translations, dim=0).to(self.device)
            scales = torch.cat(scales, dim=0).to(self.device)

            init_node_emb = voxel_features

            assert init_node_emb.shape[0] == rotations.shape[0] == np.array(instances_count).sum()

            # Graph construction, edge attr, edge idx ------------------------------------------------------------------

            scene_id = input[0]['scene'] + '_' + mode

            if self.opt.rel_app:
                appearance = voxel_features
            else:
                appearance = None

            if scene_id not in self.graph_dataset:
                scene_dataset = datasets.GraphDataset(rotations, translations, scales, input, instances_count, num_images=num_imgs, appearance=appearance)
                graph_data = scene_dataset.construct_batch_graph(is_undirected=self.is_undirected, max_frame_dist=self.max_frame_dist, mode=mode)
                self.graph_dataset[scene_id] = graph_data

            self.graph_dataset[scene_id].x = init_node_emb # all detections from 25 frames

            false_positives = self.graph_dataset[scene_id].false_positives
            if 'vis_idxs' in self.graph_dataset[scene_id]:
                vis_idxs = self.graph_dataset[scene_id].vis_idxs
            else:
                vis_idxs = None
            consecutive_mask = self.graph_dataset[scene_id].consecutive_mask
            targets = self.graph_dataset[scene_id].y
            if 'unique_dets' in self.graph_dataset[scene_id]:
                unique_dets = self.graph_dataset[scene_id].unique_dets
            else:
                unique_dets = None

            if debug_mode:
                print(self.graph_dataset[scene_id].num_edges, self.graph_dataset[scene_id].num_nodes, self.graph_dataset[scene_id].y.shape)
                print(self.graph_dataset[scene_id].has_isolated_nodes(), self.graph_dataset[scene_id].is_undirected())
                print(self.graph_dataset[scene_id].x, self.graph_dataset[scene_id].edge_attr)
                G = to_networkx(self.graph_dataset[scene_id], to_undirected=True)
                visualize_graph(G, color=None)

            # Graph Network
            try:
                graph_outputs = self.models['graph_net'](self.graph_dataset[scene_id])
            except:
                print('Some Cuda issue', scene_id)
                print(targets)
                traceback.print_exc()
                batch_loss = 0
                losses = {}
                continue

            # Binary Classifier, batch is all combinations of instances in one sequence
            graph_loss = 0
            for edge_feature in graph_outputs:
                similarity_pred = torch.squeeze(self.models['edge_classifier'](edge_feature.type(torch.float32)), dim=-1)
                losses = self.compute_losses(similarity_pred, targets)
                graph_loss += (losses[self.loss_key]/ len(graph_outputs))

            # Accumulate loss/ outputs over a batch
            similarity_pred = torch.sigmoid(similarity_pred) # last layer similarity pred
            batch_loss += (graph_loss / batch_size)
            outputs = {'total_gt_objs': total_gt_objs, 'false_positives': false_positives, 'misses': misses,
                       'vis_idxs': vis_idxs, 'dets': unique_dets, 'prediction': similarity_pred.cpu().detach().numpy(),
                       'target': targets.cpu().detach().numpy(), 'consecutive_mask': consecutive_mask.cpu().detach().numpy()} # output per scene

            batch_output.append(outputs)

        losses[self.loss_key] = batch_loss

        return batch_output, losses

    def process_batch_combined(self, inputs, mode='train', debug_mode=False, vis_pose=False):

        '''
        Process batch for combined network -> batch size always 1
        Needs graph reconstruction after each iteration
        '''

        batch_loss = 0
        batch_output = []

        for batch_idx, input in enumerate(inputs):

            num_imgs = len(input)

            # Graph features
            instances_count = []
            voxel_features = []
            rotations = []
            translations = []
            scales = []

            for i in range(num_imgs):

                if 'voxels' not in input[i]:
                    instances_count.append(0)
                    continue

                num_instances = int(input[i]['voxels'].shape[0])
                instances_count.append(num_instances)  # count predicted instances per image

                if num_instances != 0:
                    # One voxel batch consists of all instances in one image
                    voxels = torch.unsqueeze(input[i]['voxels'], dim=1) # num_instances x 1 x 32 x 32 x 32
                    voxel_feature = self.models["voxel_encoder"](voxels.to(self.device)) # num_instances x feature_dim
                    voxel_features.append(voxel_feature)

                    rot = input[i]['rotations']  #  num_instances x 3
                    if self.opt.as_quaternion:
                        rot = get_quaternion_from_euler(rot[:, 0], rot[:, 1], rot[:, 2]) #num instances x 4
                    rotations.append(rot)
                    trans = input[i]['translations']#  num_instances x 3
                    translations.append(trans)
                    scale = torch.unsqueeze(input[i]['scales'], -1) # num_instances x 1
                    scales.append(scale)

            if voxel_features:
                voxel_features = torch.cat(voxel_features, dim=0)
            else:
                # Skipping empty predictions for occuring buggy scenes
                losses = dict()
                losses[self.loss_key] = torch.tensor(float('-inf'), device=self.device, requires_grad=True)
                return batch_output, losses

            rotations = torch.cat(rotations, dim=0).to(self.device)
            translations = torch.cat(translations, dim=0).to(self.device)
            scales = torch.cat(scales, dim=0).to(self.device)

            init_node_emb = voxel_features

            assert init_node_emb.shape[0] == rotations.shape[0] == np.array(instances_count).sum()

            # Graph construction, edge attr, edge idx ------------------------------------------------------------------

            scene_dataset = datasets.GraphDataset(rotations, translations, scales, input, instances_count, num_images=num_imgs)
            graph_data = scene_dataset.construct_batch_graph(is_undirected=self.is_undirected, max_frame_dist=self.max_frame_dist, vis_pose=vis_pose, mode=mode)
            graph_data.x = init_node_emb

            # Case of empty predictions due to non overlapping boxes
            if torch.numel(graph_data.edge_index) == 0:
                losses = dict()
                losses[self.loss_key] = torch.tensor(float('-inf'), device=self.device, requires_grad=True)
                return batch_output, losses

            if 'vis_idxs' in graph_data:
                vis_idxs = graph_data.vis_idxs
            else:
                vis_idxs = None
            consecutive_mask = graph_data.consecutive_mask
            targets = graph_data.y
            if 'unique_dets' in graph_data:
                unique_dets = graph_data.unique_dets
            else:
                unique_dets = None

            # Graph Network
            try:
                graph_outputs = self.models['graph_net'](graph_data)
            except:
                traceback.print_exc()
                batch_loss = torch.tensor(float('-inf'), device=self.device, requires_grad=True)
                losses = {}
                continue

            # Binary Classifier, batch is all combinations of instances in one sequence
            graph_loss = 0
            for edge_feature in graph_outputs:
                similarity_pred = torch.squeeze(self.models['edge_classifier'](edge_feature.type(torch.float32)), dim=-1)
                losses = self.compute_losses(similarity_pred, targets)
                graph_loss += (losses[self.loss_key]/ len(graph_outputs))

            # Accumulate loss/ outputs over a batch
            similarity_pred = torch.sigmoid(similarity_pred) # last layer similarity pred
            batch_loss += graph_loss / num_imgs # Normalize by windows size since BS is always 1

            if mode != 'train':
                outputs = {'vis_idxs': vis_idxs, 'prediction': similarity_pred.cpu().detach().numpy(), 'dets': unique_dets,
                           'target': targets.cpu().detach().numpy(), 'consecutive_mask': consecutive_mask.cpu().detach().numpy()} # output per scene

                batch_output.append(outputs)

        losses[self.loss_key] = batch_loss

        return batch_output, losses

    def process_batch_office(self, inputs, mode='train', vis_pose=False):
        '''
        Process Batch for real world office dataset
        Save memory by only getting output in non train mode
        '''

        batch_output = []

        for batch_idx, input in enumerate(inputs):

            num_imgs = len(input)

            # Graph features
            instances_count = []
            voxel_features = []
            rotations = []
            translations = []
            scales = []

            for i in range(num_imgs):

                if input[i] is None or 'voxels' not in input[i]:
                    instances_count.append(0)
                    continue

                num_instances = int(input[i]['voxels'].shape[0])
                instances_count.append(num_instances)  # count predicted instances per image

                if num_instances != 0:
                    # One voxel batch consists of all instances in one image
                    voxels = torch.unsqueeze(input[i]['voxels'], dim=1)  # num_instances x 1 x 32 x 32 x 32
                    voxel_feature = self.models["voxel_encoder"](voxels.to(self.device))  # num_instances x feature_dim
                    voxel_features.append(voxel_feature)

                    rot = input[i]['rotations']  # num_instances x 3
                    if self.opt.as_quaternion:
                        rot = get_quaternion_from_euler(rot[:, 0], rot[:, 1], rot[:, 2])  # num instances x 4
                    rotations.append(rot)
                    trans = input[i]['translations']  # num_instances x 3
                    translations.append(trans)
                    scale = torch.unsqueeze(input[i]['scales'], -1)  # num_instances x 1
                    scales.append(scale)

            voxel_features = torch.cat(voxel_features, dim=0)

            rotations = torch.cat(rotations, dim=0).to(self.device)
            translations = torch.cat(translations, dim=0).to(self.device)
            scales = torch.cat(scales, dim=0).to(self.device)

            init_node_emb = voxel_features

            assert init_node_emb.shape[0] == rotations.shape[0] == np.array(instances_count).sum()

            # Graph construction, edge attr, edge idx ------------------------------------------------------------------

            scene_dataset = datasets.GraphDataset(rotations, translations, scales, input, instances_count,
                                                  num_images=num_imgs)
            graph_data = scene_dataset.construct_batch_graph_office(is_undirected=self.is_undirected, max_frame_dist=self.max_frame_dist)
            graph_data.x = init_node_emb

            if 'vis_idxs' in graph_data:
                vis_idxs = graph_data.vis_idxs
            else:
                vis_idxs = None
            consecutive_mask = graph_data.consecutive_mask
            if 'unique_dets' in graph_data:
                unique_dets = graph_data.unique_dets
            else:
                unique_dets = None

            # Graph Network
            try:
                graph_outputs = self.models['graph_net'](graph_data)
            except:
                traceback.print_exc()
                continue

            # Binary Classifier, batch is all combinations of instances in one sequence

            for edge_feature in graph_outputs:
                similarity_pred = torch.squeeze(self.models['edge_classifier'](edge_feature.type(torch.float32)), dim=-1)

            # Accumulate loss/ outputs over a batch
            similarity_pred = torch.sigmoid(similarity_pred)  # last layer similarity pred

            outputs = {'vis_idxs': vis_idxs, 'prediction': similarity_pred.cpu().detach().numpy(),
                       'dets': unique_dets,
                       'consecutive_mask': consecutive_mask.cpu().detach().numpy()}  # output per scene

            batch_output.append(outputs)


        return batch_output

    def val(self):
        """
        Validate the model on the validation set
        """
        self.set_eval()

        print("Starting evaluation ...")
        val_loss = []
        mota_df = pd.DataFrame()

        for batch_idx, inputs in enumerate(self.val_loader):

            if int(batch_idx + 1) % 10 == 0:
                print("[Validation] Batch Idx: ", batch_idx + 1)

            with torch.no_grad():
                outputs, losses = self.process_batch(inputs, mode='val')

            if isinstance(losses[self.loss_key], float) or isinstance(losses[self.loss_key], int):
                val_loss.append(losses[self.loss_key])
            else:
                val_loss.append(losses[self.loss_key].detach().cpu().item())

            if len(outputs) == 0:
                continue

            # Eval Metrics
            for seq_idx, output in enumerate(outputs):

                # Tracking
                pred_trajectories, gt_trajectories = self.Tracker.analyse_trajectories(inputs[seq_idx], output)
                if pred_trajectories:
                    gt_traj_tables = self.Tracker.get_traj_tables(gt_trajectories, 'gt')
                    pred_traj_tables = self.Tracker.get_traj_tables(pred_trajectories, 'pred')
                    seq_mota_summary = self.Tracker.eval_mota(pred_traj_tables, gt_traj_tables)
                    mota_df = pd.concat([mota_df, seq_mota_summary], axis=0, ignore_index=True)
                else:
                    print('No predictions skipping scene {}'.format(inputs[seq_idx][0]['scene']))


            if int(batch_idx + 1) % 50 == 0:
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
                      ' Current sum Misses:', num_misses,
                      ' Current sum False Positives:', num_false_positives)

        # Final Logging
        print('Final tracking scores :')
        mota_score = mota_df.loc[:, 'mota'].mean(axis=0)
        Prec = mota_df.loc[:, 'precision'].mean(axis=0)
        Rec = mota_df.loc[:, 'recall'].mean(axis=0)
        F1 = 2 * (Prec * Rec) / (Prec + Rec)
        num_misses = mota_df.loc[:, 'num_misses'].sum(axis=0)
        num_false_positives = mota_df.loc[:, 'num_false_positives'].sum(axis=0)
        id_switches = mota_df.loc[:, 'num_switches'].sum(axis=0)
        num_objects_gt = mota_df.loc[:, 'num_objects'].sum(axis=0)
        mota_accumulated = get_mota_df(num_objects_gt, num_misses, num_false_positives, id_switches)
        print('Accumulated MOTA:', mota_accumulated, ' Averaged MOTA:', mota_score,
              ' Precision:', Prec,
              ' Recall:', Rec,
              ' Current sum Misses:', num_misses,
              ' Current sum False Positives:', num_false_positives)

        val_loss_mean = {self.loss_key: np.array(val_loss).mean(), 'Precision': Prec,
                         'Recall': Rec, 'F1_score': F1, 'MOTA': mota_accumulated}

        self.log("val", val_loss_mean)

        self._save_valmodel(val_loss_mean)
        del inputs, outputs, losses

        self.set_train()

    def compute_losses(self, inputs, targets):
        '''
        Balanced loss giving active and non-active edges same magnitude
        inputs: predictions for edge connectivity
        '''

        losses = {}

        if self.opt.use_l1:
            l1_loss = self.criterion(torch.sigmoid(inputs), targets)
            losses[self.loss_key] = l1_loss
        else:
            num_active = torch.count_nonzero(targets)
            num_all = torch.numel(targets)
            pos_weight = (num_all - num_active) / num_active
            balanced_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean", pos_weight=pos_weight)

            losses[self.loss_key] = balanced_loss

        return losses

    def compute_triplet_loss(self, anchor, positive, negative):

        losses = {}

        triplet = self.criterion(anchor, positive, negative)
        losses[self.loss_key] = triplet

        return losses


    def log(self, mode, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, is_val=True):
        """Save model weights """
        if is_val:
            best_folder = os.path.join(self.log_path, "best_model")
            if not os.path.exists(best_folder):
                os.makedirs(best_folder)

            for model_name, model in self.models.items():
                save_path = os.path.join(best_folder, "{}.pth".format(model_name + '_best'))
                to_save = model.state_dict()
                torch.save(to_save, save_path)

            if self.epoch >= self.opt.start_saving_optimizer:
                save_path = os.path.join(best_folder, "{}.pth".format("adam_best"))
                torch.save(self.model_optimizer.state_dict(), save_path)

        else:
            save_folder = os.path.join(self.log_path, "models", "epoch_{}".format(self.epoch+1))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            for model_name, model in self.models.items():
                save_path = os.path.join(save_folder, "{}.pth".format(model_name))
                to_save = model.state_dict()
                torch.save(to_save, save_path)

            if self.epoch >= self.opt.start_saving_optimizer:
                save_path = os.path.join(save_folder, "{}.pth".format("adam"))
                torch.save(self.model_optimizer.state_dict(), save_path)

    def save_end2end_model(self, path=None):
        """Save model weights """

        best_folder = path
        if not os.path.exists(best_folder):
            os.makedirs(best_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(best_folder, "{}.pth".format(model_name + '_best'))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

            save_path = os.path.join(best_folder, "{}.pth".format("adam_best"))
            torch.save(self.model_optimizer.state_dict(), save_path)

    def _save_valmodel(self, losses):

        mean_loss = losses[self.loss_key]

        json_path = os.path.join(self.opt.log_dir, 'val_metrics.json')
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
        if mean_loss < min_loss and self.epoch >= self.opt.start_saving:
            print('Current Model Loss is lower than previous model, saving ...')
            self.save_model(is_val=True)


    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("Loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            if n == 'edge_encoder':
                continue
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx + 1, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
