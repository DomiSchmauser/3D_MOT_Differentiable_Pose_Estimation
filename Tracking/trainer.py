from __future__ import absolute_import, division, print_function

import sys, os
import math
import time, datetime
import numpy as np
import pandas as pd
import traceback
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.nn import functional as F

import json

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

from Tracking import networks
from Tracking import datasets
from Tracking.utils.train_utils import check_pair, sec_to_hm_str, init_weights
from Tracking.utils.eval_utils import get_mota_df
from Tracking.datasets.siamese_dataset import construct_siamese_dataset, recompute_edge_features, compute_edge_emb, construct_siamese_dataset_vis, construct_siamese_dataset_office, compute_edge_emb_nogeo
from Tracking.tracker.tracking_front import Tracker
from Tracking.visualise.visualise import visualise_gt_sequence, visualise_pred_sequence


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
        self.voxel_out_dim = 12
        self.edge_out_dim = 8
        if not self.opt.no_geo:
            self.models["voxel_encoder"] = networks.VoxelEncoder(input_channel=1, output_channel=self.voxel_out_dim)
            self.models["voxel_encoder"].to(self.device)
            init_weights(self.models["voxel_encoder"], init_type='kaiming', init_gain=0.02)
            self.parameters_to_train += list(self.models["voxel_encoder"].parameters())

        if not self.opt.no_pose:
            classifier_in_dim = 2 * self.voxel_out_dim + self.edge_out_dim
            self.models["edge_encoder"] = networks.MLP(7, [8, self.edge_out_dim], dropout_p=None, use_batchnorm=False)
            self.models["edge_encoder"].to(self.device)
            init_weights(self.models["edge_encoder"], init_type='kaiming', init_gain=0.02)
            self.parameters_to_train += list(self.models["edge_encoder"].parameters())
        else:
            classifier_in_dim = 2 * self.voxel_out_dim

        if self.opt.no_geo:
            classifier_in_dim = self.edge_out_dim

        self.models["edge_classifier"] = networks.EdgeClassifier(input_dim=classifier_in_dim)
        self.models["edge_classifier"].to(self.device)

        init_weights(self.models["edge_classifier"], init_type='kaiming', init_gain=0.02)



        self.parameters_to_train += list(self.models["edge_classifier"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate, weight_decay=self.opt.weight_decay)

        #self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, 15, 0.5)

        self.classifier_dataset = {}

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
        self.Tracker = Tracker(seq_len=self.opt.seq_len)
        # Dataset ----------------------------------------------------------------------------------------------------
        if not self.combined:
            self.dataset = datasets.Front_dataset

            DATA_DIR = CONF.PATH.TRACKDATA
            self.precomp_dir = CONF.PATH.TRACKDATA

            train_dataset = self.dataset(
                base_dir=DATA_DIR,
                split='train',
                transform=self.transform)

            if self.opt.precompute_feats:
                shuffle = False
            else:
                shuffle = True
            self.train_loader = DataLoader(
                train_dataset,
                self.opt.batch_size,
                shuffle=shuffle,
                num_workers=self.opt.num_workers,
                collate_fn=lambda x:x,
                pin_memory=True,
                drop_last=True)

            val_dataset = self.dataset(
                base_dir=DATA_DIR,
                split='val',
                transform=self.transform)

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
                transform=self.transform)

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

    def precompute(self):
        """
        Precompute feature space only required if memory issues arise...
        """
        print("Starting feature precomputation ...")
        self.start_time = time.time()

        print('Precomputing Training features ')
        for batch_idx, inputs in enumerate(self.train_loader):
            if int(batch_idx + 1) % 20 == 0:
                print('Train Batch {} of {} Train Batches'.format(int((batch_idx+1)), int(len(self.train_loader))))

            with torch.no_grad():
                self.precompute_features(inputs, mode='train')

        print('Precomputing Validation features ')
        for batch_idx, inputs in enumerate(self.val_loader):
            if int(batch_idx + 1) % 20 == 0:
                print('Train Batch {} of {} Train Batches'.format(int((batch_idx + 1)), int(len(self.val_loader))))

            with torch.no_grad():
                self.precompute_features(inputs, mode='val')

    def inference(self, vis_pose=False, no_graph=False, classwise=True):
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

        for batch_idx, inputs in enumerate(self.test_loader):
            if int(batch_idx + 1) % 20 == 0:
                print('Sequence {} of {} Sequences'.format(int((batch_idx+1)), int(len(self.test_loader))))

            with torch.no_grad():
                outputs, _ = self.process_batch(inputs, mode='test', vis_pose=vis_pose)

            if vis_pose:
                pred_trajectories, gt_trajectories = self.Tracker.analyse_trajectories_vis(inputs[0], outputs[0])
                visualise_gt_sequence(gt_trajectories, seq_name=inputs[0][0]['scene'], seq_len=self.opt.seq_len)
                visualise_pred_sequence(pred_trajectories, gt_trajectories, seq_name=inputs[0][0]['scene'])
                continue

            # Tracking
            if no_graph:
                pred_trajectories, gt_trajectories = self.Tracker.analyse_trajectories_heur(inputs[0], outputs[0])
                #pred_trajectories, gt_trajectories = self.Tracker.analyse_trajectories_nograph(inputs[0], outputs[0])
            else:
                pred_trajectories, gt_trajectories = self.Tracker.analyse_trajectories(inputs[0], outputs[0], no_cls=False)
            gt_traj_tables = self.Tracker.get_traj_tables(gt_trajectories, 'gt')
            pred_traj_tables = self.Tracker.get_traj_tables(pred_trajectories, 'pred')
            if classwise:
                seq_mota_summary, mot_events = self.Tracker.eval_mota_classwise(pred_traj_tables, gt_traj_tables)
            else:
                seq_mota_summary = self.Tracker.eval_mota(pred_traj_tables, gt_traj_tables)
            mota_df = pd.concat([mota_df, seq_mota_summary], axis=0, ignore_index=True)

            if classwise:
                for cls, cls_df in classes_df.items():
                    gt_cls_traj_tables = gt_traj_tables[gt_traj_tables['obj_cls'] == cls]
                    if gt_cls_traj_tables.empty:
                        continue
                    # Get assignments
                    matches = mot_events[mot_events['Type'] == 'MATCH']
                    class_ids = gt_cls_traj_tables['obj_idx'].unique()
                    filtered_matched = matches[matches['HId'].isin(class_ids)]  # all mate
                    frame_idxs = filtered_matched.index.droplevel(1)
                    obj_idxs = filtered_matched['HId']
                    fp_cls_traj_tables = pred_traj_tables.loc[
                        pred_traj_tables['scan_idx'].isin(frame_idxs) & pred_traj_tables['obj_idx'].isin(obj_idxs)]
                    pred_cls_traj_tables = pred_traj_tables[pred_traj_tables['obj_cls'] == cls]
                    pred_merge_table = pd.concat([fp_cls_traj_tables, pred_cls_traj_tables]).drop_duplicates()
                    # print('IS EQUAL :', pred_merge_table.equals(pred_cls_traj_tables) )
                    class_mota_summary = self.Tracker.eval_mota(pred_merge_table, gt_cls_traj_tables)
                    classes_df[cls] = pd.concat([cls_df, class_mota_summary], axis=0, ignore_index=True)

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
                        cls_mota_accumulated = get_mota_df(cls_df.loc[:, 'num_objects'].sum(axis=0),
                                                           cls_df.loc[:, 'num_misses'].sum(axis=0),
                                                           cls_df.loc[:, 'num_false_positives'].sum(axis=0),
                                                           cls_df.loc[:, 'num_switches'].sum(axis=0))
                        print('Class MOTA :', cls_mapping[cls], 'score:', cls_mota_accumulated)

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
                cls_mota_accumulated = get_mota_df(cls_df.loc[:, 'num_objects'].sum(axis=0),
                                                   cls_df.loc[:, 'num_misses'].sum(axis=0),
                                                   cls_df.loc[:, 'num_false_positives'].sum(axis=0),
                                                   cls_df.loc[:, 'num_switches'].sum(axis=0))
                print('Class MOTA :', cls_mapping[cls], 'score:', cls_mota_accumulated)

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

            #self.opt.log_frequency = 1
            if int(batch_idx + 1) % self.opt.log_frequency == 0:
                self.log_time(batch_idx, duration, loss.cpu().data)
                self.log("train", losses)
            '''
            if int(batch_idx + 1) % int(round(len(self.train_loader)/1)) == 0: # validation n time per epoch
                self.val()
            '''
            self.step += 1
        #self.model_lr_scheduler.step()
        self.val()

    def precompute_features(self, inputs, mode='train', overwrite=False):

        for batch_idx, input in enumerate(inputs):

            graph_in_features = []
            num_imgs = len(input)
            total_gt_objs = 0
            total_pred_objs = 0
            misses = 0

            for i in range(num_imgs):

                # One voxel batch consists of all instances in one image
                voxels = torch.unsqueeze(input[i]['voxels'], dim=1) # num_instances x 1 x 32 x 32 x 32
                num_instances = int(voxels.shape[0])
                if num_instances != 0:
                    voxel_feature = torch.unsqueeze(self.models["voxel_encoder"](voxels.to(self.device)), dim=0) # 1 x num_instances x feature_dim

                    if not self.opt.no_pose: # with pose
                        rot = torch.unsqueeze(input[i]['rotations'], dim=0)  # 1 x num_instances x 3
                        trans = torch.unsqueeze(input[i]['translations'], dim=0) # 1 x num_instances x 3
                        scale = torch.unsqueeze(torch.unsqueeze(input[i]['scales'], -1), dim=0) # 1 x num_instances x 1
                        pose = torch.cat((rot, trans, scale), dim=-1) # 1 x num_instance x 7

                        img_feat = torch.cat((voxel_feature.detach().cpu(), pose), dim=-1) # 1 x num instances x 16
                    else:
                        img_feat = voxel_feature
                else:
                    # Empty predictions for this image
                    img_feat = None

                graph_in_features.append(img_feat)

                per_img_gt_objs = int(input[i]['gt_object_id'].shape[-1])
                total_gt_objs += per_img_gt_objs # Number of ground truth objects in one frame
                total_pred_objs += num_instances # Number of predicted objects in one frame

                if num_instances < per_img_gt_objs: # Missing detections/ False Negatives per image or for a GT box no matching pred box found
                    misses += per_img_gt_objs - num_instances

            # Object Association

            classifier_data = construct_siamese_dataset(input, graph_in_features, thres=self.box_iou_thres, mode=mode, device=self.device)
            edge_features = torch.cat(classifier_data['edge_features'], dim=0)

            # HDF5
            fname = os.path.join(self.precomp_dir, mode, input[0]['scene'], 'features.h5')
            if not overwrite and os.path.exists(fname):
                continue
            hf = h5py.File(fname, 'w')
            hf.create_dataset('edge_features', data=edge_features)
            hf.create_dataset('targets', data=classifier_data['targets'].detach().cpu())
            hf.create_dataset('obj_ids', data=classifier_data['obj_ids'])
            hf.create_dataset('unique_dets', data=classifier_data['unique_dets'])


    def process_batch(self, inputs, mode='train', vis_pose=False):
        '''
        Process batch:
        1. Encodes voxel grids in feature space and concat with pose for object feature
        2. Get active/non-active edges (TARGETS) by computing IoU pred and GT and compare GT object ids
        -> obj1 img1 with all obj img2, obj2 img1 with all obj img2
        3. Concat object features between consecutive frames -> feature vector of 2 x 16 object features
        5. Edge classification into active non-active
        '''

        batch_loss = 0
        batch_size = len(inputs)
        batch_output = []
        outputs = {} #for train use empty outputs

        for batch_idx, input in enumerate(inputs):

            graph_in_features = []
            num_imgs = len(input)
            total_gt_objs = 0
            total_pred_objs = 0
            misses = 0

            for i in range(num_imgs):

                # One voxel batch consists of all instances in one image
                voxels = torch.unsqueeze(input[i]['voxels'], dim=1) # num_instances x 1 x 32 x 32 x 32
                num_instances = int(voxels.shape[0])
                if num_instances != 0:
                    if not self.opt.no_geo:
                        voxel_feature = torch.unsqueeze(self.models["voxel_encoder"](voxels.to(self.device)), dim=0) # 1 x num_instances x feature_dim

                    if not self.opt.no_pose and not self.opt.no_geo: # with pose
                        rot = torch.unsqueeze(input[i]['rotations'], dim=0)  # 1 x num_instances x 3
                        trans = torch.unsqueeze(input[i]['translations'], dim=0) # 1 x num_instances x 3
                        scale = torch.unsqueeze(torch.unsqueeze(input[i]['scales'], -1), dim=0) # 1 x num_instances x 1
                        pose = torch.cat((rot, trans, scale), dim=-1) # 1 x num_instance x 7
                        #print('pose feature', pose.shape)

                        img_feat = torch.cat((voxel_feature, pose.to(self.device)), dim=-1) # 1 x num instances x 16
                    elif self.opt.no_pose:
                        img_feat = voxel_feature
                    elif self.opt.no_geo:
                        rot = torch.unsqueeze(input[i]['rotations'], dim=0)  # 1 x num_instances x 3
                        trans = torch.unsqueeze(input[i]['translations'], dim=0)  # 1 x num_instances x 3
                        scale = torch.unsqueeze(torch.unsqueeze(input[i]['scales'], -1), dim=0)  # 1 x num_instances x 1
                        pose = torch.cat((rot, trans, scale), dim=-1)  # 1 x num_instance x 7
                        img_feat = pose
                else:
                    # Empty predictions for this image
                    img_feat = None

                graph_in_features.append(img_feat)

                per_img_gt_objs = int(input[i]['gt_object_id'].shape[-1])
                total_gt_objs += per_img_gt_objs # Number of ground truth objects in one frame
                total_pred_objs += num_instances # Number of predicted objects in one frame

                if num_instances < per_img_gt_objs: # Missing detections/ False Negatives per image or for a GT box no matching pred box found
                    misses += per_img_gt_objs - num_instances

            # Object Association
            scene_id = input[0]['scene'] + '_' + mode

            if scene_id not in self.classifier_dataset:
                if not vis_pose:
                    classifier_data = construct_siamese_dataset(input, graph_in_features, thres=self.box_iou_thres, mode=mode, device=self.device)
                else:
                    classifier_data = construct_siamese_dataset_vis(input, graph_in_features, thres=self.box_iou_thres, device=self.device)
                self.classifier_dataset[scene_id] = classifier_data
                edge_features = self.classifier_dataset[scene_id]['edge_features']
            else:
                try:
                    edge_features = recompute_edge_features(graph_in_features, self.classifier_dataset[scene_id]['obj_ids'])
                except:
                    print('ID issue :', scene_id)
                    traceback.print_exc()

            targets = self.classifier_dataset[scene_id]['targets']
            false_positives = self.classifier_dataset[scene_id]['false_positives']
            vis_idxs = self.classifier_dataset[scene_id]['vis_idxs']
            unique_dets = self.classifier_dataset[scene_id]['unique_dets']
            non_empty = False

            if self.opt.use_triplet:
                # Only used for triplet loss
                anchors = self.classifier_dataset[scene_id]['anchors']
                positive_samples = self.classifier_dataset[scene_id]['positive_samples']
                negative_samples = self.classifier_dataset[scene_id]['negative_samples']

                if anchors:
                    non_empty = True
                    anchors = torch.cat(anchors, dim=0)
                    positive_samples = torch.cat(positive_samples, dim=0)
                    negative_samples = torch.cat(negative_samples, dim=0)

            if edge_features:
                edge_feature = torch.cat(edge_features, dim=0)# num instance combinations in one sequence x 32
                if not self.opt.no_pose and not self.opt.no_geo:
                    edge_feature = compute_edge_emb(edge_feature, self.models['edge_encoder'], voxel_dim=self.voxel_out_dim)
                elif self.opt.no_geo:
                    edge_feature = compute_edge_emb_nogeo(edge_feature, self.models['edge_encoder'],
                                                    device=self.device)

            else:
                print('Empty tensor', ', Bad scene:', input[0]['scene'])
                batch_loss = 1
                continue

            # Binary Classifier, batch is all combinations of instances in one sequence
            similarity_pred = torch.squeeze(self.models['edge_classifier'](edge_feature.type(torch.float32)), dim=-1) # n*m
            #targets = torch.tensor(targets, dtype=torch.float32, device=self.device) # already moved

            if self.opt.use_triplet and non_empty:
                losses = self.compute_triplet_loss(anchors, positive_samples, negative_samples) # shape n samples x 16
            elif self.opt.use_triplet:
                print('No triplet pairs found for sequence {}'.format(input[0]['scene']))
                losses = {}
                losses[self.loss_key] = 1
            else: # for BCE and L1 loss compute distance in edge prediction space
                losses = self.compute_losses(similarity_pred, targets)

            # Accumulate loss/ outputs over a batch
            similarity_pred = torch.sigmoid(similarity_pred)
            batch_loss += losses[self.loss_key] / batch_size

            if mode != 'train':
                outputs = {'total_gt_objs': total_gt_objs, 'false_positives': false_positives, 'misses': misses, 'vis_idxs': vis_idxs, 'dets': unique_dets,
                           'prediction': similarity_pred.cpu().detach().numpy(), 'target': targets.cpu().detach().numpy()} # output per scene

            batch_output.append(outputs)

        losses[self.loss_key] = batch_loss
        return batch_output, losses

    def process_batch_combined(self, inputs, mode='train', vis_pose=False):
        '''
        Process Batch for combined network
        Save memory by only getting output in non train mode
        '''

        batch_loss = 0
        losses = dict()
        batch_output = []

        for batch_idx, input in enumerate(inputs):

            graph_in_features = []
            num_imgs = len(input)
            scan_id_str = '_'

            for i in range(num_imgs):

                if 'voxels' not in input[i]: # per image
                    # Empty Predictions
                    graph_in_features.append(None)
                    continue

                # One voxel batch consists of all instances in one image
                scan_id_str += str(input[i]['image'])
                voxels = torch.unsqueeze(input[i]['voxels'], dim=1) # num_instances x 1 x 32 x 32 x 32
                num_instances = int(voxels.shape[0])
                if num_instances != 0:
                    voxel_feature = torch.unsqueeze(self.models["voxel_encoder"](voxels.to(self.device)), dim=0) # 1 x num_instances x feature_dim

                    if not self.opt.no_pose: # with pose
                        rot = torch.unsqueeze(input[i]['rotations'], dim=0)  # 1 x num_instances x 3
                        trans = torch.unsqueeze(input[i]['translations'], dim=0) # 1 x num_instances x 3
                        scale = torch.unsqueeze(torch.unsqueeze(input[i]['scales'], -1), dim=0) # 1 x num_instances x 1
                        pose = torch.cat((rot, trans, scale), dim=-1) # 1 x num_instance x 7

                        img_feat = torch.cat((voxel_feature, pose.to(self.device)), dim=-1) # 1 x num instances x 16
                    else:
                        img_feat = voxel_feature
                else:
                    # Empty predictions for this image
                    img_feat = None

                graph_in_features.append(img_feat)

            # Object Association
            if not vis_pose:
                classifier_data = construct_siamese_dataset(input, graph_in_features, thres=self.box_iou_thres, mode=mode, device=self.device)
            else:
                classifier_data = construct_siamese_dataset_vis(input, graph_in_features, thres=self.box_iou_thres, device=self.device)
            edge_features = classifier_data['edge_features']
            targets = classifier_data['targets']
            vis_idxs = classifier_data['vis_idxs']
            unique_dets = classifier_data['unique_dets']

            if edge_features:
                edge_feature = torch.cat(edge_features, dim=0) # num instance combinations in one sequence x 32
                edge_feature = compute_edge_emb(edge_feature, self.models['edge_encoder'], voxel_dim=self.voxel_out_dim)
            else: # Empty if no overlapping bounding boxes in scan
                #print('Empty tensor, no box overlaps found, bad scene:', input[0]['scene'])
                batch_loss = torch.tensor(float('-inf'), device=self.device, requires_grad=True)
                continue

            # Binary Classifier, batch is all combinations of instances in one sequence
            similarity_pred = torch.squeeze(self.models['edge_classifier'](edge_feature.type(torch.float32)), dim=-1) # n*m
            #targets = torch.tensor(targets, dtype=torch.float32, device=self.device)

            #for BCE and L1 loss compute distance in edge prediction space
            losses = self.compute_losses(similarity_pred, targets)

            # Accumulate loss/ outputs over a batch
            similarity_pred = torch.sigmoid(similarity_pred)
            batch_loss += losses[self.loss_key] / num_imgs #Batch always size 1, image window size can vary

            if mode != 'train':
                outputs = {'vis_idxs': vis_idxs, 'dets': unique_dets,
                           'prediction': similarity_pred.cpu().detach().numpy(), 'target': targets.cpu().detach().numpy()} # output per scene

                batch_output.append(outputs)

        losses[self.loss_key] = batch_loss
        return batch_output, losses

    def process_batch_office(self, inputs, mode='train'):
        '''
        Process Batch for real world office dataset
        Save memory by only getting output in non train mode
        '''

        batch_output = []

        for batch_idx, input in enumerate(inputs):

            graph_in_features = []
            num_imgs = len(input)

            for i in range(num_imgs):

                # One voxel batch consists of all instances in one image
                voxels = torch.unsqueeze(input[i]['voxels'], dim=1) # num_instances x 1 x 32 x 32 x 32
                voxel_feature = torch.unsqueeze(self.models["voxel_encoder"](voxels.to(self.device)), dim=0) # 1 x num_instances x feature_dim

                if not self.opt.no_pose: # with pose
                    rot = torch.unsqueeze(input[i]['rotations'], dim=0)  # 1 x num_instances x 3
                    trans = torch.unsqueeze(input[i]['translations'], dim=0) # 1 x num_instances x 3
                    scale = torch.unsqueeze(torch.unsqueeze(input[i]['scales'], -1), dim=0) # 1 x num_instances x 1
                    pose = torch.cat((rot, trans, scale), dim=-1) # 1 x num_instance x 7

                    img_feat = torch.cat((voxel_feature, pose.to(self.device)), dim=-1) # 1 x num instances x 16
                else:
                    img_feat = voxel_feature

                graph_in_features.append(img_feat)

            # Object Association
            classifier_data = construct_siamese_dataset_office(input, graph_in_features)
            edge_features = classifier_data['edge_features']
            vis_idxs = classifier_data['vis_idxs']

            if edge_features:
                edge_feature = torch.cat(edge_features, dim=0) # num instance combinations in one sequence x 32
                edge_feature = compute_edge_emb(edge_feature, self.models['edge_encoder'], voxel_dim=self.voxel_out_dim)
            else:
                continue

            # Binary Classifier, batch is all combinations of instances in one sequence
            similarity_pred = torch.squeeze(self.models['edge_classifier'](edge_feature.type(torch.float32)), dim=-1) # n*m

            # Accumulate loss/ outputs over a batch
            similarity_pred = torch.sigmoid(similarity_pred)

            if mode != 'train':
                outputs = {'vis_idxs': vis_idxs, 'prediction': similarity_pred.cpu().detach().numpy()} # output per scene

                batch_output.append(outputs)

        return batch_output

    def val(self):
        """
        Validate the model on the validation set
        Batch size 1
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

            if isinstance(losses[self.loss_key], float):
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

    def compute_losses(self, inputs, targets, clip_preds=True):
        '''
        Balanced loss giving active and non-active edges same magnitude
        inputs: predictions
        '''

        losses = {}

        if self.opt.use_l1:
            l1_loss = self.criterion(torch.sigmoid(inputs), targets)
            losses[self.loss_key] = l1_loss
        else:
            num_active = torch.count_nonzero(targets)
            num_all = torch.numel(targets)
            pos_weight = (num_all - num_active) / num_active
            max_weigth = torch.tensor(10.0, dtype=torch.float32, device=self.device)
            pos_weight = min(pos_weight, max_weigth) # restrict positive weight
            # Add clamping of too large input values
            if clip_preds:
                inputs = inputs.clamp(min=-100, max=100)
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
            if n == 'graph_net':
                continue
            if n == 'edge_encoder' and self.opt.no_pose:
                continue
            if n == 'voxel_encoder' and self.opt.no_geo:
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
