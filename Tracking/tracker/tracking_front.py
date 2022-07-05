import numpy as np
import torch
import motmetrics as mm
import pandas as pd
import open3d as o3d
from dvis import dvis
import mathutils

class Tracker:

    def __init__(self, seq_len=25):
        self.seq_len = seq_len
        self.quantization_size = 0.04
        self.similar_value = 0.1
        self.iou_thres = 0.3
        self.l2_thres = 0.4
        self.dist_thres = 100
        self.cls_gt_objs = {
            'chair': [], 'table': [], 'sofa': [],
            'bed': [], 'tv_stand': [],
            'cooler': [], 'night_stand': []
        }
        #
    def pred_trajectory(self, trajectories, pred, scan_idx, dist_thres=0.25):
        '''
        Heuristic to match non graph inference using a l2 distance of predictions
        '''

        if not trajectories:
            pred_obj = {'cad2world_loc': pred['loc_id_1'], 'obj_idx': pred['obj_id_1'], 'obj_cls': pred['cls_id_1']}
            trajectories.append([{'obj': pred_obj, 'scan_idx': scan_idx}])
            return trajectories

        l2_distances = []
        for traj in trajectories:
            dist = np.linalg.norm(traj[-1]['obj']['cad2world_loc'] - pred['loc_id_1'])
            l2_distances.append(dist)

        idx_min = np.argmin(np.array(l2_distances))
        l2_min = l2_distances[idx_min]
        if l2_min < dist_thres:
            has_similar = False
            for traj in trajectories:
                if traj[-1]['scan_idx'] == scan_idx:
                    if np.linalg.norm(traj[-1]['obj']['cad2world_loc'] - pred['loc_id_1']) < 0.2:
                        has_similar = True
            if not has_similar:
                pred_obj = {'cad2world_loc': pred['loc_id_1'], 'obj_idx': pred['obj_id_1'], 'obj_cls': pred['cls_id_1']}
                trajectories[idx_min].append({'obj': pred_obj, 'scan_idx': scan_idx})
        else:
            has_similar = False
            for traj in trajectories:
                if traj[-1]['scan_idx'] == scan_idx:
                    if np.linalg.norm(traj[-1]['obj']['cad2world_loc'] - pred['loc_id_1']) < 0.2:
                        has_similar = True

            if not has_similar:
                # Start a new trajectory
                pred_obj = {'cad2world_loc': pred['loc_id_1'], 'obj_idx': pred['obj_id_1'], 'obj_cls': pred['cls_id_1']}
                trajectories.append([{'obj': pred_obj, 'scan_idx': scan_idx}])

        return trajectories

    def pred_trajectory_office(self, trajectories, pred, scan_idx, dist_thres=10):
        '''
        Heuristic to match non graph inference using a l2 distance of predictions
        '''

        if not trajectories:
            pred_obj = {'cad2world': pred['cad2world_1'], 'voxel': pred['vox_1'], 'obj_idx': 0,
                        'compl_box': pred['box_1']}
            trajectories.append([{'obj': pred_obj, 'scan_idx': scan_idx}])
            return trajectories

        l2_distances = []
        for traj in trajectories:
            dist = np.linalg.norm(traj[-1]['obj']['cad2world'][:3,3] - pred['cad2world_1'][:3,3])
            l2_distances.append(dist)

        idx_min = np.argmin(np.array(l2_distances))
        l2_min = l2_distances[idx_min]
        if l2_min < dist_thres:
            has_similar = False
            for traj in trajectories:
                if traj[-1]['scan_idx'] == scan_idx:
                    if np.linalg.norm(traj[-1]['obj']['cad2world'][:3,3] - pred['cad2world_1'][:3,3]) < 10:
                        has_similar = True
            if not has_similar:
                pred_obj = {'cad2world': pred['cad2world_1'], 'voxel': pred['vox_1'], 'obj_idx': 0,
                            'compl_box': pred['box_1']}
                trajectories[idx_min].append({'obj': pred_obj, 'scan_idx': scan_idx})

        return trajectories


    def analyse_trajectories_nograph(self, gt_seq_list, pred_seq):
        '''
        Create trajectories based on match criterion
        '''

        seq_data = dict()
        predictions = pred_seq['vis_idxs']

        pred_trajectories = []
        gt_trajectories = []

        for scan_idx in range(self.seq_len):

            gt_scan_dct = gt_seq_list[scan_idx]

            # Initialize trajectory
            if scan_idx == 0:
                # Pred
                for pred in predictions:
                    if pred['image'] == scan_idx:
                        has_similar = False
                        pred_obj = {'cad2world_loc': pred['loc_id_1'], 'obj_idx': pred['obj_id_1'], 'obj_cls': pred['cls_id_1']}
                        for pred_traj in pred_trajectories:
                            if np.linalg.norm(pred_traj[0]['obj']['cad2world_loc'] - pred['loc_id_1']) < 0.2:
                                has_similar = True
                        if not has_similar:
                            pred_trajectories.append([{'obj': pred_obj, 'scan_idx': scan_idx}])
                # GT
                num_gt_objs = len(gt_scan_dct['gt_object_id'])
                for i in range(num_gt_objs):
                    gt_obj = {'cad2world_loc': gt_scan_dct['gt_locations'][i], '3D_box': gt_scan_dct['gt_3Dbbox'][i],
                              'obj_idx': gt_scan_dct['gt_object_id'][i], 'obj_cls': gt_scan_dct['gt_classes'][i]}
                    gt_trajectories.append([{'obj': gt_obj, 'scan_idx': scan_idx}])
            else:
                # Pred Match trajectories to initial trajectory
                for pred in predictions:
                    if pred['image'] == scan_idx:
                        pred_trajectories = self.pred_trajectory(pred_trajectories, pred, scan_idx)

                # GT Matching
                num_gt_objs = len(gt_scan_dct['gt_object_id'])
                for i in range(num_gt_objs):
                    gt_obj = {'cad2world_loc': gt_scan_dct['gt_locations'][i], '3D_box': gt_scan_dct['gt_3Dbbox'][i],
                              'obj_idx': gt_scan_dct['gt_object_id'][i], 'obj_cls': gt_scan_dct['gt_classes'][i]}
                    matched = False
                    for gt_traj in gt_trajectories:
                        if gt_obj['obj_idx'] == gt_traj[0]['obj']['obj_idx']:
                            gt_traj.append({'obj': gt_obj, 'scan_idx': scan_idx})  # build list per trajectory
                            matched = True
                            break
                    if not matched:
                        gt_trajectories.append([{'obj': gt_obj, 'scan_idx': scan_idx}]) # start new trajectory

        return pred_trajectories, gt_trajectories

    def analyse_trajectories_heur(self, gt_seq_list, pred_seq):
        '''
        Create trajectories based on match criterion
        '''

        seq_data = dict()

        pred_trajectories = []
        gt_trajectories = []

        # Create predicted trajectory
        predictions = pred_seq['prediction']
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0

        # for Graph setting
        if 'consecutive_mask' in pred_seq.keys():
            consecutive_mask = pred_seq['consecutive_mask']
            forward_connections = len(consecutive_mask)
            predictions = predictions[:forward_connections][consecutive_mask == 1]

        # todo stand alone predictions updated
        pos_mask = predictions == 1
        connections = np.array(pred_seq['vis_idxs'])[pos_mask]
        dets = np.array(pred_seq['dets'])

        connections = np.concatenate((connections, dets), axis=0)

        # Rearange
        scan_connections = [None] * self.seq_len
        for conn in connections:
            idx = conn['image']
            pred_obj = {'cad2world_loc': conn['loc_id_1'], 'obj_idx': conn['obj_id_1'], 'obj_cls': conn['cls_id_1']}

            if scan_connections[idx] == None:
                scan_connections[idx] = [pred_obj] #todo consider incoming and outgoing conns hence obj_id_2 for consec frame
            else:
                scan_connections[idx].append(pred_obj)

            if idx == self.seq_len-2:  # second last frame use also t+1
                and_idx = self.seq_len-1
                # todo added for empty dets
                if conn['obj_id_2'] is None:
                    continue
                pred_obj = {'cad2world_loc': conn['loc_id_2'], 'obj_idx': conn['obj_id_2'], 'obj_cls': conn['cls_id_2']}
                if scan_connections[and_idx] == None:
                    scan_connections[and_idx] = [pred_obj]
                else:
                    scan_connections[and_idx].append(pred_obj)

        for scan_idx in range(self.seq_len):

            gt_scan_dct = gt_seq_list[scan_idx]

            # Initialize trajectory
            if scan_idx == 0:
                unique_ids = []
                # Pred
                if scan_connections[scan_idx] == None:
                    pass
                else:
                    for pred_obj in scan_connections[scan_idx]:
                        if pred_obj['obj_idx'] not in unique_ids and pred_obj['obj_idx'] is not None: #todo improved by checking closer
                            pred_trajectories.append([{'obj': pred_obj, 'scan_idx': scan_idx}])
                            unique_ids.append(pred_obj['obj_idx'])
                # GT
                num_gt_objs = len(gt_scan_dct['gt_object_id'])
                for i in range(num_gt_objs):
                    gt_obj = {'cad2world_loc': gt_scan_dct['gt_locations'][i], '3D_box': gt_scan_dct['gt_3Dbbox'][i],
                              'obj_idx': gt_scan_dct['gt_object_id'][i], 'obj_cls': gt_scan_dct['gt_classes'][i],
                              }
                    gt_trajectories.append([{'obj': gt_obj, 'scan_idx': scan_idx}])
            else:
                unique_ids = []
                # Pred Match trajectories to initial trajectory
                if scan_connections[scan_idx] == None:
                    pass
                else:
                    for pred_obj in scan_connections[scan_idx]:
                        matched = False
                        for pred_traj in pred_trajectories:
                            if pred_traj[0]['obj']['obj_idx'] == pred_obj['obj_idx'] and pred_obj['obj_idx'] not in unique_ids and pred_obj['obj_idx'] is not None:
                                pred_traj.append({'obj': pred_obj, 'scan_idx': scan_idx})
                                unique_ids.append(pred_obj['obj_idx'])
                                matched = True
                                break
                        if not matched and pred_obj['obj_idx'] not in unique_ids and pred_obj['obj_idx'] is not None:
                            pred_trajectories.append([{'obj': pred_obj, 'scan_idx': scan_idx}])
                            unique_ids.append(pred_obj['obj_idx']) # todo added newly

                # GT Matching
                num_gt_objs = len(gt_scan_dct['gt_object_id'])
                for i in range(num_gt_objs):
                    gt_obj = {'cad2world_loc': gt_scan_dct['gt_locations'][i], '3D_box': gt_scan_dct['gt_3Dbbox'][i],
                              'obj_idx': gt_scan_dct['gt_object_id'][i], 'obj_cls': gt_scan_dct['gt_classes'][i],}
                    matched = False
                    for gt_traj in gt_trajectories:
                        if gt_obj['obj_idx'] == gt_traj[0]['obj']['obj_idx']:
                            gt_traj.append({'obj': gt_obj, 'scan_idx': scan_idx})  # build list per trajectory
                            matched = True
                            break
                    if not matched:
                        gt_trajectories.append([{'obj': gt_obj, 'scan_idx': scan_idx}]) # start new trajectory

        return pred_trajectories, gt_trajectories

    def analyse_trajectories(self, gt_seq_list, pred_seq, no_cls=False):
        '''
        Create trajectories based on match criterion
        '''

        seq_data = dict()

        pred_trajectories = []
        gt_trajectories = []

        # Create predicted trajectory
        predictions = pred_seq['prediction']
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0

        # for Graph setting
        if 'consecutive_mask' in pred_seq.keys():
            consecutive_mask = pred_seq['consecutive_mask']
            forward_connections = len(consecutive_mask)
            predictions = predictions[:forward_connections][consecutive_mask == 1]

        # todo stand alone predictions updated
        pos_mask = predictions == 1
        connections = np.array(pred_seq['vis_idxs'])[pos_mask]
        dets = np.array(pred_seq['dets'])

        connections = np.concatenate((connections, dets), axis=0)

        # Rearange
        scan_connections = [None] * self.seq_len
        for conn in connections:
            idx = conn['image']
            if not no_cls:
                pred_obj = {'cad2world_loc': conn['loc_id_1'], 'obj_idx': conn['obj_id_1'], 'obj_cls': conn['cls_id_1']}
                pred_obj_2 = {'cad2world_loc': conn['loc_id_2'], 'obj_idx': conn['obj_id_2'], 'obj_cls': conn['cls_id_2']}
            else:
                pred_obj = {'cad2world_loc': conn['loc_id_1'], 'obj_idx': conn['obj_id_1']}
                pred_obj_2 = {'cad2world_loc': conn['loc_id_2'], 'obj_idx': conn['obj_id_2']}
            if scan_connections[idx] == None:
                scan_connections[idx] = [pred_obj] #todo consider incoming and outgoing conns hence obj_id_2 for consec frame
            else:
                scan_connections[idx].append(pred_obj)

            if scan_connections[idx+1] == None:
                scan_connections[idx+1] = [pred_obj_2]
            else:
                scan_connections[idx+1].append(pred_obj_2)

            if idx == self.seq_len-2:  # second last frame use also t+1
                and_idx = self.seq_len-1
                # todo added for empty dets
                if conn['obj_id_2'] is None:
                    continue
                if not no_cls:
                    pred_obj = {'cad2world_loc': conn['loc_id_2'], 'obj_idx': conn['obj_id_2'], 'obj_cls': conn['cls_id_2']}
                else:
                    pred_obj = {'cad2world_loc': conn['loc_id_2'], 'obj_idx': conn['obj_id_2']}
                if scan_connections[and_idx] == None:
                    scan_connections[and_idx] = [pred_obj]
                else:
                    scan_connections[and_idx].append(pred_obj)

        for scan_idx in range(self.seq_len):

            gt_scan_dct = gt_seq_list[scan_idx]

            # Initialize trajectory
            if scan_idx == 0:
                unique_ids = []
                # Pred
                if scan_connections[scan_idx] == None:
                    pass
                else:
                    for pred_obj in scan_connections[scan_idx]:
                        if pred_obj['obj_idx'] not in unique_ids and pred_obj['obj_idx'] is not None: #todo improved by checking closer
                            pred_trajectories.append([{'obj': pred_obj, 'scan_idx': scan_idx}])
                            unique_ids.append(pred_obj['obj_idx'])
                # GT
                num_gt_objs = len(gt_scan_dct['gt_object_id'])
                for i in range(num_gt_objs):
                    if not no_cls:
                        gt_obj = {'cad2world_loc': gt_scan_dct['gt_locations'][i], '3D_box': gt_scan_dct['gt_3Dbbox'][i],
                                  'obj_idx': gt_scan_dct['gt_object_id'][i], 'obj_cls': gt_scan_dct['gt_classes'][i],
                                  }
                    else:
                        gt_obj = {'cad2world_loc': gt_scan_dct['gt_locations'][i],
                                  '3D_box': gt_scan_dct['gt_3Dbbox'][i],
                                  'obj_idx': gt_scan_dct['gt_object_id'][i]}
                    gt_trajectories.append([{'obj': gt_obj, 'scan_idx': scan_idx}])
            else:
                unique_ids = []
                # Pred Match trajectories to initial trajectory
                if scan_connections[scan_idx] == None:
                    pass
                else:
                    for pred_obj in scan_connections[scan_idx]:
                        matched = False
                        for pred_traj in pred_trajectories:
                            if pred_traj[0]['obj']['obj_idx'] == pred_obj['obj_idx'] and pred_obj['obj_idx'] not in unique_ids and pred_obj['obj_idx'] is not None:
                                pred_traj.append({'obj': pred_obj, 'scan_idx': scan_idx})
                                unique_ids.append(pred_obj['obj_idx'])
                                matched = True
                                break
                        if not matched and pred_obj['obj_idx'] not in unique_ids and pred_obj['obj_idx'] is not None:
                            pred_trajectories.append([{'obj': pred_obj, 'scan_idx': scan_idx}])
                            unique_ids.append(pred_obj['obj_idx']) # todo added newly

                # GT Matching
                num_gt_objs = len(gt_scan_dct['gt_object_id'])
                for i in range(num_gt_objs):
                    if not no_cls:
                        gt_obj = {'cad2world_loc': gt_scan_dct['gt_locations'][i], '3D_box': gt_scan_dct['gt_3Dbbox'][i],
                                  'obj_idx': gt_scan_dct['gt_object_id'][i], 'obj_cls': gt_scan_dct['gt_classes'][i],}
                    else:
                        gt_obj = {'cad2world_loc': gt_scan_dct['gt_locations'][i],
                                  '3D_box': gt_scan_dct['gt_3Dbbox'][i],
                                  'obj_idx': gt_scan_dct['gt_object_id'][i]}
                    matched = False
                    for gt_traj in gt_trajectories:
                        if gt_obj['obj_idx'] == gt_traj[0]['obj']['obj_idx']:
                            gt_traj.append({'obj': gt_obj, 'scan_idx': scan_idx})  # build list per trajectory
                            matched = True
                            break
                    if not matched:
                        gt_trajectories.append([{'obj': gt_obj, 'scan_idx': scan_idx}]) # start new trajectory

        return pred_trajectories, gt_trajectories

    def analyse_trajectories_vis(self, gt_seq_list, pred_seq, vis_pc=None):
        '''
        Create trajectories based on match criterion for vis purposes
        '''

        pred_trajectories = []
        gt_trajectories = []

        # Create predicted trajectory
        predictions = pred_seq['prediction']
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0

        # for Graph setting
        if 'consecutive_mask' in pred_seq.keys():
            consecutive_mask = pred_seq['consecutive_mask']
            forward_connections = len(consecutive_mask)
            predictions = predictions[:forward_connections][consecutive_mask == 1]

        pos_mask = predictions == 1
        connections = np.array(pred_seq['vis_idxs'])[pos_mask]
        dets = np.array(pred_seq['dets'])
        connections = np.concatenate((connections, dets), axis=0)

        # Rearange
        scan_connections = [None] * self.seq_len
        for conn in connections:
            idx = conn['image']
            pred_obj = {'cad2world': conn['cad2world_1'], 'voxel': conn['vox_1'], 'obj_idx': conn['obj_id_1'], 'compl_box': conn['box_1']}
            if scan_connections[idx] == None:
                scan_connections[idx] = [pred_obj] #todo consider incoming and outgoing conns hence obj_id_2 for consec frame
            else:
                scan_connections[idx].append(pred_obj)

            if idx == self.seq_len-2:  # second last frame use also t+1
                and_idx = self.seq_len-1
                pred_obj = {'cad2world': conn['cad2world_2'], 'voxel': conn['vox_2'], 'obj_idx': conn['obj_id_2'], 'compl_box': conn['box_2']}
                if scan_connections[and_idx] == None:
                    scan_connections[and_idx] = [pred_obj]
                else:
                    scan_connections[and_idx].append(pred_obj)

        for scan_idx in range(self.seq_len):

            gt_scan_dct = gt_seq_list[scan_idx]

            # Initialize trajectory
            if scan_idx == 0:
                unique_ids = []
                # Pred
                if scan_connections[scan_idx] == None:
                    pass
                else:
                    for pred_obj in scan_connections[scan_idx]:
                        if pred_obj['obj_idx'] not in unique_ids: #todo consider adding constraint only obj_idx once in tracklet but improved by checking closer
                            pred_trajectories.append([{'obj': pred_obj, 'scan_idx': scan_idx}])
                            unique_ids.append(pred_obj['obj_idx'])
                # GT
                num_gt_objs = len(gt_scan_dct['gt_object_id'])
                for i in range(num_gt_objs):
                    cad2world = self.cad2world_mat(gt_scan_dct, i)
                    gt_obj = {'cad2world': cad2world, 'voxel': gt_scan_dct['gt_voxels'][i],
                              '3D_box': gt_scan_dct['gt_3Dbbox'][i], 'obj_idx': gt_scan_dct['gt_object_id'][i],
                              'compl_box': self.box2minmax(gt_scan_dct['gt_compl_box'][i]), 'seq_id': gt_scan_dct['scene'],
                              'img_id': gt_scan_dct['image']} #.replace('.h5', '')
                    if i == 0:
                        if vis_pc is None:
                            gt_obj['world_pc'] = gt_scan_dct['world_pc']
                        else:
                            gt_obj['world_pc'] = vis_pc
                    gt_trajectories.append([{'obj': gt_obj, 'scan_idx': scan_idx}])
            else:
                unique_ids = []
                # Pred Match trajectories to initial trajectory
                if scan_connections[scan_idx] == None:
                    pass
                else:
                    for pred_obj in scan_connections[scan_idx]:
                        matched = False
                        for pred_traj in pred_trajectories:
                            if pred_traj[0]['obj']['obj_idx'] == pred_obj['obj_idx'] and pred_obj['obj_idx'] not in unique_ids:
                                pred_traj.append({'obj': pred_obj, 'scan_idx': scan_idx})
                                unique_ids.append(pred_obj['obj_idx'])
                                matched = True
                                break
                        if not matched and pred_obj['obj_idx'] not in unique_ids:
                            pred_trajectories.append([{'obj': pred_obj, 'scan_idx': scan_idx}])
                            unique_ids.append(pred_obj['obj_idx']) #todo added newly check

                # GT Matching
                num_gt_objs = len(gt_scan_dct['gt_object_id'])
                for i in range(num_gt_objs):
                    cad2world = self.cad2world_mat(gt_scan_dct, i)
                    gt_obj = {'cad2world': cad2world, 'voxel': gt_scan_dct['gt_voxels'][i],
                              '3D_box': gt_scan_dct['gt_3Dbbox'][i], 'obj_idx': gt_scan_dct['gt_object_id'][i],
                              'compl_box': self.box2minmax(gt_scan_dct['gt_compl_box'][i]), 'seq_id': gt_scan_dct['scene'],
                              'img_id': gt_scan_dct['image']}#.replace('.h5', '')}
                    if i == 0:
                        if vis_pc is None:
                            gt_obj['world_pc'] = gt_scan_dct['world_pc']
                        else:
                            gt_obj['world_pc'] = vis_pc
                    matched = False
                    for gt_traj in gt_trajectories:
                        if gt_obj['obj_idx'] == gt_traj[0]['obj']['obj_idx']:
                            gt_traj.append({'obj': gt_obj, 'scan_idx': scan_idx})  # build list per trajectory
                            matched = True
                            break
                    if not matched:
                        gt_trajectories.append([{'obj': gt_obj, 'scan_idx': scan_idx}]) # start new trajectory

        return pred_trajectories, gt_trajectories

    def analyse_trajectories_office_new(self, pred_seq, seq_len=None, dist_thres=0.8):
        '''
        Create trajectories based on match criterion for vis purposes
        For Real-World Dataset Office TUM
        '''

        pred_trajectories = []

        # Create predicted trajectory
        predictions = pred_seq['vis_idxs']
        for scan_idx in range(seq_len):

            # Initialize trajectory
            if scan_idx == 0:
                # Pred
                for pred in predictions:
                    if pred['image'] == scan_idx:
                        has_similar = False
                        pred_obj = {'cad2world': pred['cad2world_1'], 'voxel': pred['vox_1'], 'obj_idx': 0,
                                    'compl_box': pred['box_1']}
                        for pred_traj in pred_trajectories:
                            #print('values', np.linalg.norm(pred_traj[0]['obj']['cad2world'][:3, 3] - pred['cad2world_1'][:3, 3]))
                            if np.linalg.norm(pred_traj[0]['obj']['cad2world'][:3, 3] - pred['cad2world_1'][:3, 3]) < 5:
                                has_similar = True
                        if not has_similar:
                            pred_trajectories.append([{'obj': pred_obj, 'scan_idx': scan_idx}])
            else:

                # Pred Match trajectories to initial trajectory
                for pred in predictions:
                    if pred['image'] == scan_idx:
                        pred_trajectories = self.pred_trajectory_office(pred_trajectories, pred, scan_idx)

        return pred_trajectories

    def analyse_trajectories_office(self, pred_seq, seq_len=None, dist_thres=0.8):
        '''
        Create trajectories based on match criterion for vis purposes
        For Real-World Dataset Office TUM
        '''

        pred_trajectories = []

        # Create predicted trajectory
        predictions = pred_seq['prediction']
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0

        # for Graph setting
        if 'consecutive_mask' in pred_seq.keys():
            consecutive_mask = pred_seq['consecutive_mask']
            forward_connections = len(consecutive_mask)
            predictions = predictions[:forward_connections][consecutive_mask == 1]

        pos_mask = predictions == 1
        connections = np.array(pred_seq['vis_idxs'])[pos_mask]
        #dets = np.array(pred_seq['dets'])

        #connections = np.concatenate((connections, dets), axis=0)

        # Rearange
        scan_connections = [None] * seq_len
        mappings = [dict()] * seq_len
        #id_map = dict()
        for c_idx, conn in enumerate(connections):
            idx = conn['image']

            # New heuristic to match
            loc_1 = conn['cad2world_1'][:3, 3]
            if idx != 0:
                dists = []
                ids = []
                for m_id, loc in mappings[idx-1].items():
                    dist = np.linalg.norm(loc - loc_1)
                    dists.append(dist)
                    ids.append(m_id)
                min_dist_idx = np.argmin(np.array(dists))
                min_dist = dists[min_dist_idx]

                if min_dist < dist_thres:
                    map_id = ids[min_dist_idx]
                else:
                    map_id = conn['obj_1']

                mappings[idx][map_id] = loc_1

            else:
                map_id = conn['obj_1']
                mappings[idx][map_id] = loc_1

            pred_obj = {'cad2world': conn['cad2world_1'], 'voxel': conn['vox_1'], 'obj_idx': map_id, 'compl_box': conn['box_1']}

            if scan_connections[idx] == None:
                scan_connections[idx] = [pred_obj]
            else:
                scan_connections[idx].append(pred_obj)

            if idx == seq_len-2:  # second last frame use also t+1
                and_idx = seq_len-1

                # New heuristic to match
                map_id = conn['obj_2']

                pred_obj = {'cad2world': conn['cad2world_2'], 'voxel': conn['vox_2'], 'obj_idx': map_id, 'compl_box': conn['box_2']}
                if scan_connections[and_idx] == None:
                    scan_connections[and_idx] = [pred_obj]
                else:
                    scan_connections[and_idx].append(pred_obj)

        for scan_idx in range(seq_len):

            # Initialize trajectory
            if scan_idx == 0:
                unique_ids = []
                # Pred
                if scan_connections[scan_idx] == None:
                    pass
                else:
                    for pred_obj in scan_connections[scan_idx]:
                        if pred_obj['obj_idx'] not in unique_ids:
                            pred_trajectories.append([{'obj': pred_obj, 'scan_idx': scan_idx}])
                            unique_ids.append(pred_obj['obj_idx'])
            else:
                unique_ids = []
                # Pred Match trajectories to initial trajectory
                if scan_connections[scan_idx] == None:
                    pass
                else:
                    for pred_obj in scan_connections[scan_idx]:
                        matched = False
                        for pred_traj in pred_trajectories:
                            if pred_traj[0]['obj']['obj_idx'] == pred_obj['obj_idx'] and pred_obj['obj_idx'] not in unique_ids:
                                pred_traj.append({'obj': pred_obj, 'scan_idx': scan_idx})
                                unique_ids.append(pred_obj['obj_idx'])
                                matched = True
                                break
                        if not matched and pred_obj['obj_idx'] not in unique_ids:
                            pred_trajectories.append([{'obj': pred_obj, 'scan_idx': scan_idx}])
                            unique_ids.append(pred_obj['obj_idx']) #todo added newly

        return pred_trajectories

    def analyse_trajectories_F2F(self, gt_seq_list, pred_seq, vis=True):
        '''
        Create trajectories based on match criterion
        '''

        seq_data = dict()

        pred_trajectories = []
        gt_trajectories = []

        # Create predicted trajectory
        predictions = pred_seq['prediction']
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0
        pos_mask = predictions == 1
        connections = np.array(pred_seq['vis_idxs'])[pos_mask]
        # Rearange
        scan_connections = [None] * self.seq_len
        for conn in connections:
            idx = conn['image']
            if not vis:
                pred_obj = {'cad2world_loc': conn['loc_id_1'], 'obj_idx': conn['obj_id_1'], 'obj_cls': conn['cls_id_1']}
            else:
                pred_obj = {'cad2world_loc': conn['loc_id_1'], 'obj_idx': conn['obj_id_1'], 'obj_cls': conn['cls_id_1'],
                            'obj_pc': conn['pc_1'], 'obj_box': conn['box_1'], 'cad2world': conn['cad2world_1']}
            if scan_connections[idx] == None:
                scan_connections[idx] = [pred_obj]
            else:
                scan_connections[idx].append(pred_obj)

            if idx == 23:  # second last frame use also t+1
                and_idx = 24
                if not vis:
                    pred_obj = {'cad2world_loc': conn['loc_id_2'], 'obj_idx': conn['obj_id_2'], 'obj_cls': conn['cls_id_2']}
                else:
                    pred_obj = {'cad2world_loc': conn['loc_id_2'], 'obj_idx': conn['obj_id_2'], 'obj_cls': conn['cls_id_2'],
                                'obj_pc': conn['pc_2'], 'obj_box': conn['box_2'], 'cad2world': conn['cad2world_2']}
                if scan_connections[and_idx] == None:
                    scan_connections[and_idx] = [pred_obj]
                else:
                    scan_connections[and_idx].append(pred_obj)

        for scan_idx in range(self.seq_len):

            gt_scan_dct = gt_seq_list[scan_idx]

            # Initialize trajectory
            if scan_idx == 0:
                unique_ids = []
                # Pred
                if scan_connections[scan_idx] == None:
                    pass
                else:
                    for pred_obj in scan_connections[scan_idx]:
                        if pred_obj['obj_idx'] not in unique_ids: #todo consider adding constraint only obj_idx once in tracklet but improved by checking closer
                            pred_trajectories.append([{'obj': pred_obj, 'scan_idx': scan_idx}])
                            unique_ids.append(pred_obj['obj_idx'])
                # GT
                num_gt_objs = len(gt_scan_dct['obj_ids'])
                for i in range(num_gt_objs):
                    tmp_box = gt_scan_dct['gt_3dboxes'][i]
                    bbox3d_obj = o3d.geometry.AxisAlignedBoundingBox()
                    bbox_3d = bbox3d_obj.create_from_points(o3d.utility.Vector3dVector(tmp_box))
                    center_3d = bbox_3d.get_center()
                    gt_obj = {'cad2world_loc': center_3d, '3D_box': gt_scan_dct['gt_3dboxes'][i],
                              'obj_idx': gt_scan_dct['obj_ids'][i], 'obj_cls': gt_scan_dct['gt_classes'][i]} #todo change
                    gt_trajectories.append([{'obj': gt_obj, 'scan_idx': scan_idx}])
            else:
                unique_ids = []
                # Pred Match trajectories to initial trajectory
                if scan_connections[scan_idx] == None:
                    pass
                else:
                    for pred_obj in scan_connections[scan_idx]:
                        matched = False
                        for pred_traj in pred_trajectories:
                            if pred_traj[0]['obj']['obj_idx'] == pred_obj['obj_idx'] and pred_obj['obj_idx'] not in unique_ids:
                                pred_traj.append({'obj': pred_obj, 'scan_idx': scan_idx})
                                unique_ids.append(pred_obj['obj_idx'])
                                matched = True
                                break
                        if not matched and pred_obj['obj_idx'] not in unique_ids:
                            pred_trajectories.append([{'obj': pred_obj, 'scan_idx': scan_idx}])

                # GT Matching
                num_gt_objs = len(gt_scan_dct['obj_ids'])
                for i in range(num_gt_objs):
                    tmp_box = gt_scan_dct['gt_3dboxes'][i]
                    bbox3d_obj = o3d.geometry.AxisAlignedBoundingBox()
                    bbox_3d = bbox3d_obj.create_from_points(o3d.utility.Vector3dVector(tmp_box))
                    center_3d = bbox_3d.get_center()
                    gt_obj = {'cad2world_loc': center_3d, '3D_box': gt_scan_dct['gt_3dboxes'][i],
                              'obj_idx': gt_scan_dct['obj_ids'][i], 'obj_cls': gt_scan_dct['gt_classes'][i]}
                    matched = False
                    for gt_traj in gt_trajectories:
                        if gt_obj['obj_idx'] == gt_traj[0]['obj']['obj_idx']:
                            gt_traj.append({'obj': gt_obj, 'scan_idx': scan_idx})  # build list per trajectory
                            matched = True
                            break
                    if not matched:
                        gt_trajectories.append([{'obj': gt_obj, 'scan_idx': scan_idx}]) # start new trajectory

        return pred_trajectories, gt_trajectories

    def analyse_trajectories_F2F_office(self, pred_seq, seq_len=None, dist_thres=0.8):
        '''
        Create trajectories based on match criterion for vis purposes
        For Real-World Dataset Office TUM
        '''

        pred_trajectories = []

        # Create predicted trajectory
        predictions = pred_seq['prediction']
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0

        pos_mask = predictions == 1
        connections = np.array(pred_seq['vis_idxs'])[pos_mask]

        # Rearange
        scan_connections = [None] * seq_len
        mappings = [dict()] * seq_len
        #id_map = dict()
        for c_idx, conn in enumerate(connections):
            idx = conn['image']

            # New heuristic to match
            loc_1 = conn['cad2world_1'][:3, 3]
            if idx != 0:
                dists = []
                ids = []
                for m_id, loc in mappings[idx-1].items():
                    dist = np.linalg.norm(loc - loc_1)
                    dists.append(dist)
                    ids.append(m_id)
                min_dist_idx = np.argmin(np.array(dists))
                min_dist = dists[min_dist_idx]

                if min_dist < dist_thres:
                    map_id = ids[min_dist_idx]
                else:
                    map_id = conn['obj_1']

                mappings[idx][map_id] = loc_1

            else:
                map_id = conn['obj_1']
                mappings[idx][map_id] = loc_1

            pred_obj = {'cad2world_loc': conn['loc_id_1'], 'obj_idx': map_id, 'obj_cls': conn['cls_id_1'],
                        'obj_pc': conn['pc_1'], 'obj_box': conn['box_1'], 'cad2world': conn['cad2world_1']}

            if scan_connections[idx] == None:
                scan_connections[idx] = [pred_obj]
            else:
                scan_connections[idx].append(pred_obj)

            if idx == seq_len-2:  # second last frame use also t+1
                and_idx = seq_len-1

                # New heuristic to match
                map_id = conn['obj_2']

                pred_obj = {'cad2world_loc': conn['loc_id_2'], 'obj_idx': map_id, 'obj_cls': conn['cls_id_2'],
                            'obj_pc': conn['pc_2'], 'obj_box': conn['box_2'], 'cad2world': conn['cad2world_2']}
                if scan_connections[and_idx] == None:
                    scan_connections[and_idx] = [pred_obj]
                else:
                    scan_connections[and_idx].append(pred_obj)

        for scan_idx in range(seq_len):

            # Initialize trajectory
            if scan_idx == 0:
                unique_ids = []
                # Pred
                if scan_connections[scan_idx] == None:
                    pass
                else:
                    for pred_obj in scan_connections[scan_idx]:
                        if pred_obj['obj_idx'] not in unique_ids:
                            pred_trajectories.append([{'obj': pred_obj, 'scan_idx': scan_idx}])
                            unique_ids.append(pred_obj['obj_idx'])
            else:
                unique_ids = []
                # Pred Match trajectories to initial trajectory
                if scan_connections[scan_idx] == None:
                    pass
                else:
                    for pred_obj in scan_connections[scan_idx]:
                        matched = False
                        for pred_traj in pred_trajectories:
                            if pred_traj[0]['obj']['obj_idx'] == pred_obj['obj_idx'] and pred_obj['obj_idx'] not in unique_ids:
                                pred_traj.append({'obj': pred_obj, 'scan_idx': scan_idx})
                                unique_ids.append(pred_obj['obj_idx'])
                                matched = True
                                break
                        if not matched and pred_obj['obj_idx'] not in unique_ids:
                            pred_trajectories.append([{'obj': pred_obj, 'scan_idx': scan_idx}])
                            unique_ids.append(pred_obj['obj_idx']) #todo added newly

        return pred_trajectories


    def get_traj_table(self, traj, traj_id):
        traj_df = pd.DataFrame()

        for k in range(len(traj)):
            scan_idx = traj[k]['scan_idx']
            world_t = traj[k]['obj']['cad2world_loc']
            if 'gt' in traj_id:
                single_df = pd.DataFrame(dict(scan_idx=scan_idx,
                                              world_x = world_t[0].item(),
                                              world_y = world_t[1].item(),
                                              world_z = world_t[2].item(),
                                              obj_idx = traj[k]['obj']['obj_idx'].item() if 'obj_idx' in traj[k]['obj'] else None,
                                              obj_cls = traj[k]['obj']['obj_cls'].item() if 'obj_cls' in traj[k]['obj'] else None,
                                              ), index=[scan_idx]
                                         )
            else:
                single_df = pd.DataFrame(dict(scan_idx=scan_idx,
                                              world_x=world_t[0].item(),
                                              world_y=world_t[1].item(),
                                              world_z=world_t[2].item(),
                                              obj_idx=traj[k]['obj']['obj_idx'] if 'obj_idx' in traj[k]['obj'] else None,
                                              obj_cls=traj[k]['obj']['obj_cls'].item() if 'obj_cls' in traj[k]['obj'] else None,
                                              ), index=[scan_idx]
                                         )

            traj_df = pd.concat([traj_df, single_df], axis=0)
        return traj_df

    def get_traj_tables(self, trajectories, prefix):
        traj_tables = pd.DataFrame()
        for t in range(len(trajectories)):
            traj_table = self.get_traj_table(trajectories[t], prefix)
            traj_tables = pd.concat([traj_tables, traj_table], axis=0)
        return traj_tables

    def get_traj_table_F2F(self, traj, traj_id):
        traj_df = pd.DataFrame()

        for k in range(len(traj)):
            scan_idx = traj[k]['scan_idx']
            world_t = traj[k]['obj']['cad2world_loc']

            single_df = pd.DataFrame(dict(scan_idx=scan_idx,
                                          world_x=world_t[0],
                                          world_y=world_t[1],
                                          world_z=world_t[2],
                                          obj_idx = traj[k]['obj']['obj_idx'] if 'obj_idx' in traj[k]['obj'] else None,
                                          obj_cls = traj[k]['obj']['obj_cls'].item() if 'obj_cls' in traj[k]['obj'] else None,
                                          ), index=[scan_idx]
                                     )


            traj_df = pd.concat([traj_df, single_df], axis=0)
        return traj_df

    def get_traj_tables_F2F(self, trajectories, prefix):
        traj_tables = pd.DataFrame()
        for t in range(len(trajectories)):
            traj_table = self.get_traj_table_F2F(trajectories[t], prefix)
            traj_tables = pd.concat([traj_tables, traj_table], axis=0)
        return traj_tables

    def eval_mota_F2F(self, pred_table, mov_obj_traj_table):

        '''
        Update to class-wise MOTA
        '''
        l2_th = 0.25

        mh = mm.metrics.create()

        acc = mm.MOTAccumulator(auto_id=True)
        for scan_idx in range(self.seq_len):
            gt_cams = np.array(mov_obj_traj_table[mov_obj_traj_table['scan_idx'] == scan_idx][['world_x', 'world_y', 'world_z']]) # CAD2WORLD TRANSLATION
            # get gt position in camera frame
            gt_objects = mov_obj_traj_table[mov_obj_traj_table['scan_idx'] == scan_idx]['obj_idx'].tolist()

            hypo_table = pred_table[(pred_table['scan_idx'] == scan_idx)]
            pred_objects = []
            dist_matrix = np.nan * np.ones((len(gt_objects), len(hypo_table)))
            for j, hypo in enumerate(hypo_table.iterrows()):
                hypo_cam = np.array(hypo[1][['world_x', 'world_y', 'world_z']]) #CAD2WORLD TRANSLATION
                # get hypo position in camera frame
                hypo_id = int(hypo[1]['obj_idx'])
                pred_objects.append(hypo_id)
                for i, gt_obj in enumerate(gt_objects):
                    gt_cam = gt_cams[i,:]
                    dist_matrix[i][j] = mm.distances.norm2squared_matrix(gt_cam, hypo_cam, max_d2=l2_th) # l2 distance between gt object and hypothesis, capped to l2_th


            acc.update(
                gt_objects,  # Ground truth objects in this frame
                pred_objects,  # Detector hypotheses in this frame
                dist_matrix
            )

        all_traj_summary = mh.compute(acc, metrics=['num_frames', 'mota', 'precision', 'recall', 'num_objects', 'num_matches', 'num_misses',
                                           'num_false_positives', 'num_switches'], name='acc')

        return all_traj_summary, acc.mot_events

    def eval_mota(self, pred_table, mov_obj_traj_table):
        l2_th = self.l2_thres

        mh = mm.metrics.create()

        acc = mm.MOTAccumulator(auto_id=True)
        for scan_idx in range(self.seq_len):
            gt_cams = np.array(mov_obj_traj_table[mov_obj_traj_table['scan_idx'] == scan_idx][['world_x', 'world_y', 'world_z']]) # CAD2WORLD TRANSLATION
            # get gt position in camera frame
            gt_objects = mov_obj_traj_table[mov_obj_traj_table['scan_idx'] == scan_idx]['obj_idx'].tolist()

            hypo_table = pred_table[(pred_table['scan_idx'] == scan_idx)]
            pred_objects = []
            dist_matrix = np.nan * np.ones((len(gt_objects), len(hypo_table)))
            for j, hypo in enumerate(hypo_table.iterrows()):
                hypo_cam = np.array(hypo[1][['world_x', 'world_y', 'world_z']]) #CAD2WORLD TRANSLATION
                # get hypo position in camera frame
                hypo_id = int(hypo[1]['obj_idx'])
                pred_objects.append(hypo_id)
                for i, gt_obj in enumerate(gt_objects):
                    gt_cam = gt_cams[i,:]
                    dist_matrix[i][j] = mm.distances.norm2squared_matrix(gt_cam, hypo_cam, max_d2=l2_th) # l2 distance between gt object and hypothesis, capped to l2_th


            acc.update(
                gt_objects,  # Ground truth objects in this frame
                pred_objects,  # Detector hypotheses in this frame
                dist_matrix
            )

        all_traj_summary = mh.compute(acc, metrics=['num_frames', 'mota', 'precision', 'recall', 'num_objects', 'num_matches', 'num_misses',
                                           'num_false_positives', 'num_switches'], name='acc')

        return all_traj_summary

    def eval_mota_classwise(self, pred_table, mov_obj_traj_table):
        l2_th = self.l2_thres

        mh = mm.metrics.create()

        acc = mm.MOTAccumulator(auto_id=True)
        for scan_idx in range(self.seq_len):
            gt_cams = np.array(mov_obj_traj_table[mov_obj_traj_table['scan_idx'] == scan_idx][['world_x', 'world_y', 'world_z']]) # CAD2WORLD TRANSLATION
            # get gt position in camera frame
            gt_objects = mov_obj_traj_table[mov_obj_traj_table['scan_idx'] == scan_idx]['obj_idx'].tolist()

            hypo_table = pred_table[(pred_table['scan_idx'] == scan_idx)]
            pred_objects = []
            dist_matrix = np.nan * np.ones((len(gt_objects), len(hypo_table)))
            for j, hypo in enumerate(hypo_table.iterrows()):
                hypo_cam = np.array(hypo[1][['world_x', 'world_y', 'world_z']]) #CAD2WORLD TRANSLATION
                # get hypo position in camera frame
                hypo_id = int(hypo[1]['obj_idx'])
                pred_objects.append(hypo_id)
                for i, gt_obj in enumerate(gt_objects):
                    gt_cam = gt_cams[i,:]
                    dist_matrix[i][j] = mm.distances.norm2squared_matrix(gt_cam, hypo_cam, max_d2=l2_th) # l2 distance between gt object and hypothesis, capped to l2_th


            acc.update(
                gt_objects,  # Ground truth objects in this frame
                pred_objects,  # Detector hypotheses in this frame
                dist_matrix
            )

        all_traj_summary = mh.compute(acc, metrics=['num_frames', 'mota', 'precision', 'recall', 'num_objects', 'num_matches', 'num_misses',
                                           'num_false_positives', 'num_switches'], name='acc')

        return all_traj_summary, acc.mot_events

    def euler_to_rot(self, euler_rot, fmt='torch'):
        '''
        Euler to 3x3 Rotation Matrix transform
        '''

        euler = mathutils.Euler(euler_rot)
        rot = np.array(euler.to_matrix())

        if fmt == 'torch':
            return torch.from_numpy(rot)
        else:
            return rot

    def cad2world_mat(self, gt_scan_dct, i):
        '''
        Return cad2world matrix from annotations
        '''
        cad2world = torch.eye(4)
        if type(gt_scan_dct['gt_scales'][i]) == np.ndarray:
            scale_mat = torch.diag(torch.from_numpy(gt_scan_dct['gt_scales'][i]))
        else:
            scale_mat = torch.diag(gt_scan_dct['gt_scales'][i])
        cad2world[:3, :3] = scale_mat @ self.euler_to_rot(gt_scan_dct['gt_rotations'][i], fmt='torch')
        cad2world[:3, 3] = gt_scan_dct['gt_locations'][i]
        return cad2world

    def box2minmax(self, corner_pt_box):
        '''
        Box from 8x3 to minmax format
        '''
        xyz_min = torch.min(corner_pt_box, dim=0).values
        xyz_max = torch.max(corner_pt_box, dim=0).values
        box = np.concatenate((xyz_min.numpy(), xyz_max.numpy()))
        return box