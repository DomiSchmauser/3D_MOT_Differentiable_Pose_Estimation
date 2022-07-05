import sys
import torch
import numpy as np
import mathutils
from torch_geometric.data import Data

sys.path.append('..') #Hack add ROOT DIR
from Tracking.utils.train_utils import check_pair


class GraphDataset():
    '''
    Graph dataset class enables data handling for pytorch geometric graphs
    init_node_emb: voxel features, shape: num nodes x feature dim
    rotations, translations, scales, -> edge features, shape: num nodes x (3 or 1)
    instances_count: per image instances
    '''

    def __init__(self, rotations, translations, scales, input, instances_count, num_images=25, appearance=None):

        self.rotations = rotations
        self.translations = translations
        self.scales = scales
        self.input = input
        self.instances_count = instances_count
        self.num_images = num_images
        self.box_iou_thres = 0.01  # Min IoU threshold GT and predicted 3D box
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.appearance = appearance

    def get_edge_data(self, is_undirected=True, max_frame_dist=1, max_seq_len=125, mode=None, vis_pose=False):
        '''
        Get edge attributes and according edge indicies
        is_undirected: Graph nodes connected both ways 0-1, 1-0, duplicate idxs, edge features and targets
        max_frame_dist: maximum frame in positive timesteps to construct graph with
        max_seq_len: maximum length of any input sequence
        '''

        relative_scales = []
        relative_rotations = []
        relative_positions = []
        relative_times = []
        relative_appearances = []

        edge_idxs = []

        # Validation data
        vis_idxs = []
        unique_dets = []
        false_positives = 0
        targets = []
        consecutive_mask = []

        for t in range(self.num_images - 1):

            gt_bbox_1 = self.input[t]['gt_3Dbbox']  # num inst x 8 pts x xyz
            gt_id_1 = self.input[t]['gt_object_id']  # num inst

            # Window frames
            window_frames = torch.arange(t, t+1+max_frame_dist)
            window_mask = torch.logical_and(window_frames >= 0, window_frames != t)
            max_len = min(max_seq_len, self.num_images)
            window_mask = torch.logical_and(window_mask, window_frames < max_len)
            window_frames = window_frames[window_mask].tolist()

            start_inst_count = int(torch.sum(torch.tensor(self.instances_count[:t])))  # start count instances frame t
            end_inst_count = start_inst_count + self.instances_count[t]  # end count instances frame t

            for j, frame in enumerate(window_frames):

                gt_bbox_2 = self.input[frame]['gt_3Dbbox']
                gt_id_2 = self.input[frame]['gt_object_id']

                prior_num_inst = int(torch.sum(torch.tensor(self.instances_count[:frame])))
                consec_num_inst = prior_num_inst + self.instances_count[frame]

                for n in range(start_inst_count, end_inst_count):
                    pred_bbox_1 = self.input[t]['pred_3Dbbox']  # num inst x 8pts x xyz
                    pred_loc_1 = self.input[t]['translations']
                    pred_cls_1 = self.input[t]['classes']
                    if vis_pose:
                        pred_rot_1 = self.input[t]['rotations']
                        pred_scales_1 = self.input[t]['scales']
                        pred_vox_1 = self.input[t]['voxels']
                        # Cad2world mat
                        cad2world_1 = self.cad2world_mat(pred_rot_1[n-start_inst_count], pred_loc_1[n-start_inst_count], pred_scales_1[n-start_inst_count])

                    # Object Matching frame t
                    try:
                        obj_id_1 = check_pair(pred_bbox_1[n-start_inst_count, :, :], gt_bbox_1, gt_id_1,
                                              thres=self.box_iou_thres)
                    except:
                        obj_id_1 = None
                        print('Issue with convex hull ...')

                    if obj_id_1 is None and j == 0:
                        false_positives += 1  # No overlapping GT bounding box found and append only once
                        continue
                    elif obj_id_1 is None:
                        continue  # SKIP THIS INSTANCE FOR GRAPH CONSTRUCTION

                    if (prior_num_inst - consec_num_inst) == 0 and frame == t+1:
                        if vis_pose:
                            unique_dets.append({'image': t, 'obj_1': n - start_inst_count, 'obj_2': None,
                                                'obj_id_1': int(obj_id_1), 'obj_id_2': None,
                                                'cad2world_1': cad2world_1, 'cad2world_2': None,
                                                'vox_1': pred_vox_1[n-start_inst_count], 'vox_2': None,
                                                'box_1': self.box2minmax(pred_bbox_1[n-start_inst_count]), 'box_2': None})
                        else:
                            unique_dets.append({'image': t, 'obj_1': n-start_inst_count, 'obj_2': None,
                                                 'obj_id_1': int(obj_id_1), 'obj_id_2': None,
                                                 'loc_id_1': pred_loc_1[n-start_inst_count], 'loc_id_2': None,
                                                 'cls_id_1': pred_cls_1[n-start_inst_count], 'cls_id_2': None })

                    for m in range(prior_num_inst, consec_num_inst):  # n0-m0 n0-m1 n1-m0 n1-m1 ....
                        pred_bbox_2 = self.input[frame]['pred_3Dbbox']
                        pred_loc_2 = self.input[frame]['translations']
                        pred_cls_2 = self.input[frame]['classes']
                        if vis_pose:
                            pred_rot_2 = self.input[frame]['rotations']
                            pred_scales_2 = self.input[frame]['scales']
                            pred_vox_2 = self.input[frame]['voxels']
                            # Cad2world mat
                            cad2world_2 = self.cad2world_mat(pred_rot_2[m-prior_num_inst],
                                                             pred_loc_2[m-prior_num_inst],
                                                             pred_scales_2[m-prior_num_inst])
                        # Object Matching frame window
                        try:
                            obj_id_2 = check_pair(pred_bbox_2[m-prior_num_inst, :, :], gt_bbox_2, gt_id_2, thres=self.box_iou_thres)
                        except:
                            obj_id_2 = None
                            print('Issue with convex hull ...')

                        # ONLY FOR LAST FRAME WHICH ISNT COVERED IN OUTER LOOP ADD FP
                        if t == self.num_images - 2 and n == end_inst_count - 1:
                            if obj_id_2 is None:
                                false_positives += 1

                        # GT targets: active (1) and non-active (0) connections
                        if obj_id_1 == obj_id_2 and obj_id_1 is not None and obj_id_2 is not None:  # both objects exist and same id
                            target = 1
                        elif obj_id_1 != obj_id_2 and obj_id_1 is not None and obj_id_2 is not None:
                            target = 0
                        elif obj_id_2 is None:  # false prediction for any object in consecutive frame -> exclude
                            continue

                        # Consecutive mask for identity switches in MOTA calculation
                        if frame == t+1:
                            consecutive_mask.append(1)
                            if vis_pose:
                                vis_idxs.append({'image': t, 'obj_1': n-start_inst_count, 'obj_2': m-prior_num_inst,
                                                'obj_id_1': int(obj_id_1), 'obj_id_2': int(obj_id_2),
                                                'cad2world_1': cad2world_1, 'cad2world_2': cad2world_2,
                                                'vox_1': pred_vox_1[n-start_inst_count], 'vox_2': pred_vox_2[m-prior_num_inst],
                                                'box_1': self.box2minmax(pred_bbox_1[n-start_inst_count]), 'box_2': self.box2minmax(pred_bbox_2[m-prior_num_inst])})
                            else:
                                vis_idxs.append({'image': t, 'obj_1': n-start_inst_count, 'obj_2': m-prior_num_inst,
                                                 'obj_id_1': int(obj_id_1), 'obj_id_2': int(obj_id_2),
                                                 'loc_id_1': pred_loc_1[n-start_inst_count], 'loc_id_2': pred_loc_2[m-prior_num_inst],
                                                 'cls_id_1': pred_cls_1[n-start_inst_count], 'cls_id_2': pred_cls_2[m-prior_num_inst]}) # n and m do not start with 0 and obj 1 should be instance number in specific img
                        else:
                            consecutive_mask.append(0)

                        targets.append(target)

                        # Edge feature construction
                        edge_idxs.append([n, m]) # 0 - 1

                        relative_scale = torch.unsqueeze(torch.log(self.scales[m, :] / self.scales[n, :]),
                                                         dim=0)  # feat t+1 / feat t
                        relative_scales.append(relative_scale)
                        relative_position = torch.unsqueeze(self.translations[m, :] - self.translations[n, :], dim=0)
                        relative_positions.append(relative_position)
                        relative_rot = torch.unsqueeze(self.rotations[m, :] - self.rotations[n, :], dim=0)
                        relative_rotations.append(relative_rot)
                        relative_time = torch.unsqueeze(torch.tensor([frame - t], dtype=torch.int64),
                                                        dim=0)
                        relative_times.append(relative_time)
                        if self.appearance is not None:
                            relative_appearance = torch.unsqueeze(torch.linalg.vector_norm(self.appearance[m, :] - self.appearance[n, :]), dim=0)
                            relative_appearances.append(torch.unsqueeze(relative_appearance, dim=0))

        # Skip diverge case of empty edges due to non overlapping boxes
        if not relative_scales:
            return torch.tensor([], device=self.device), torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device), torch.tensor([], device=self.device), None, 0, None
            #edge_index, edge_attr, targets, consecutive_mask, vis_idxs, false_positives, unique_dets

        relative_scales = torch.cat(relative_scales, dim=0)  # num_edges x 1
        relative_positions = torch.cat(relative_positions, dim=0)  # num_edges x 3
        relative_rotations = torch.cat(relative_rotations, dim=0)  # num_edges x 3
        relative_times = torch.cat(relative_times, dim=0).to(self.device)  # num_edges x 1
        if relative_appearances:
            relative_appearances = torch.cat(relative_appearances, dim=0)  # num_edges x 1
            edge_attr = torch.cat((relative_positions, relative_rotations, relative_scales, relative_times, relative_appearances),
                                  dim=-1).to(dtype=torch.float32, device=self.device)  # Num edges x feat_dim
        else:
            edge_attr = torch.cat((relative_positions, relative_rotations, relative_scales, relative_times),
                                  dim=-1).to(dtype=torch.float32, device=self.device)  # Num edges x feat_dim
        edge_index = torch.tensor(edge_idxs, dtype=torch.long).t().contiguous().to(self.device)
        targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        consecutive_mask = torch.tensor(consecutive_mask, dtype=torch.int8, device=self.device)

        if is_undirected:
            edge_index = torch.cat((edge_index, torch.stack((edge_index[1], edge_index[0]))), dim=1)
            edge_attr = torch.cat((edge_attr, edge_attr), dim=0)
            targets = targets.repeat(2)

        # Save CPU memory
        #if mode == 'train':
        #    vis_idxs = None
        #    unique_dets = None

        return edge_index, edge_attr, targets, consecutive_mask, vis_idxs, false_positives, unique_dets

    def construct_batch_graph(self, is_undirected=True, max_frame_dist=5, mode=None, vis_pose=False):
        '''
        Returns batch graph data:   x: Node Embeddings, shape: Num nodes x feature dim(16)
                                    edge_idx: Edge indicies, shape: 2 x Num edges
                                    edge_attr: Edge features, shape: Num edges x feature dim(8)
                                    y: targets, shape: Num edges
        is_undirected: undirected Graph
        max_frame_dist: distance Graph nodes to connect with
        '''
        edge_idx, edge_attr, targets, consecutive_mask, vis_idxs, false_positives, unique_dets = self.get_edge_data(is_undirected=is_undirected, max_frame_dist=max_frame_dist, mode=mode, vis_pose=vis_pose)
        batch_graph = Data(x=None, edge_index=edge_idx, edge_attr=edge_attr, y=targets, consecutive_mask=consecutive_mask,
                           false_positives=false_positives, vis_idxs=vis_idxs, unique_dets=unique_dets)

        return batch_graph

    def get_edge_data_office(self, is_undirected=True, max_frame_dist=1, max_seq_len=500):

        relative_scales = []
        relative_rotations = []
        relative_positions = []
        relative_times = []
        relative_appearances = []

        edge_idxs = []

        # Validation data
        vis_idxs = []
        unique_dets = []
        consecutive_mask = []

        for t in range(self.num_images - 1):

            # Window frames
            window_frames = torch.arange(t, t+1+max_frame_dist)
            window_mask = torch.logical_and(window_frames >= 0, window_frames != t)
            max_len = min(max_seq_len, self.num_images)
            window_mask = torch.logical_and(window_mask, window_frames < max_len)
            window_frames = window_frames[window_mask].tolist()

            start_inst_count = int(torch.sum(torch.tensor(self.instances_count[:t])))  # start count instances frame t
            end_inst_count = start_inst_count + self.instances_count[t]  # end count instances frame t

            for j, frame in enumerate(window_frames):

                prior_num_inst = int(torch.sum(torch.tensor(self.instances_count[:frame])))
                consec_num_inst = prior_num_inst + self.instances_count[frame]

                for n in range(start_inst_count, end_inst_count):
                    pred_bbox_1 = self.input[t]['pred_3Dbbox']  # num inst x 8pts x xyz
                    pred_loc_1 = self.input[t]['translations']
                    pred_rot_1 = self.input[t]['rotations']
                    pred_scales_1 = self.input[t]['scales']
                    pred_vox_1 = self.input[t]['voxels']
                    # Cad2world mat
                    cad2world_1 = self.cad2world_mat(pred_rot_1[n-start_inst_count], pred_loc_1[n-start_inst_count], pred_scales_1[n-start_inst_count], constraint=False)

                    if (prior_num_inst - consec_num_inst) == 0 and frame == t+1:
                        unique_dets.append({'image': t, 'obj_1': n - start_inst_count, 'obj_2': None,
                                            'obj_id_1': None, 'obj_id_2': None,
                                            'cad2world_1': cad2world_1, 'cad2world_2': None,
                                            'vox_1': pred_vox_1[n-start_inst_count], 'vox_2': None,
                                            'box_1': self.box2minmax(pred_bbox_1[n-start_inst_count]), 'box_2': None})


                    for m in range(prior_num_inst, consec_num_inst):  # n0-m0 n0-m1 n1-m0 n1-m1 ....
                        pred_bbox_2 = self.input[frame]['pred_3Dbbox']
                        pred_loc_2 = self.input[frame]['translations']
                        pred_rot_2 = self.input[frame]['rotations']
                        pred_scales_2 = self.input[frame]['scales']
                        pred_vox_2 = self.input[frame]['voxels']
                        # Cad2world mat
                        cad2world_2 = self.cad2world_mat(pred_rot_2[m-prior_num_inst],
                                                         pred_loc_2[m-prior_num_inst],
                                                         pred_scales_2[m-prior_num_inst], constraint=False)

                        # Consecutive mask for identity switches in MOTA calculation
                        if frame == t+1:
                            consecutive_mask.append(1)
                            vis_idxs.append({'image': t, 'obj_1': n-start_inst_count, 'obj_2': m-prior_num_inst,
                                            'obj_id_1': None, 'obj_id_2': None,
                                            'cad2world_1': cad2world_1, 'cad2world_2': cad2world_2,
                                            'vox_1': pred_vox_1[n-start_inst_count], 'vox_2': pred_vox_2[m-prior_num_inst],
                                            'box_1': self.box2minmax(pred_bbox_1[n-start_inst_count]), 'box_2': self.box2minmax(pred_bbox_2[m-prior_num_inst])})
                        else:
                            consecutive_mask.append(0)

                        # Edge feature construction
                        edge_idxs.append([n, m]) # 0 - 1

                        relative_scale = torch.unsqueeze(torch.log(self.scales[m, :] / self.scales[n, :]),
                                                         dim=0)  # feat t+1 / feat t
                        relative_scales.append(relative_scale)
                        relative_position = torch.unsqueeze(self.translations[m, :] - self.translations[n, :], dim=0)
                        relative_positions.append(relative_position)
                        relative_rot = torch.unsqueeze(self.rotations[m, :] - self.rotations[n, :], dim=0)
                        relative_rotations.append(relative_rot)
                        relative_time = torch.unsqueeze(torch.tensor([frame - t], dtype=torch.int64),
                                                        dim=0)
                        relative_times.append(relative_time)
                        if self.appearance is not None:
                            relative_appearance = torch.unsqueeze(torch.linalg.vector_norm(self.appearance[m, :] - self.appearance[n, :]), dim=0)
                            relative_appearances.append(torch.unsqueeze(relative_appearance, dim=0))

        # Skip diverge case of empty edges due to non overlapping boxes
        if not relative_scales:
            return torch.tensor([], device=self.device), torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device), torch.tensor([], device=self.device), None, 0, None
            #edge_index, edge_attr, targets, consecutive_mask, vis_idxs, false_positives, unique_dets

        relative_scales = torch.cat(relative_scales, dim=0)  # num_edges x 1
        relative_positions = torch.cat(relative_positions, dim=0)  # num_edges x 3
        relative_rotations = torch.cat(relative_rotations, dim=0)  # num_edges x 3
        relative_times = torch.cat(relative_times, dim=0).to(self.device)  # num_edges x 1
        if self.appearance is not None:
            relative_appearances = torch.cat(relative_appearances, dim=0)  # num_edges x 1
            edge_attr = torch.cat((relative_positions, relative_rotations, relative_scales, relative_times, relative_appearances),
                                  dim=-1).to(dtype=torch.float32, device=self.device)  # Num edges x feat_dim
        else:
            edge_attr = torch.cat((relative_positions, relative_rotations, relative_scales, relative_times),
                                  dim=-1).to(dtype=torch.float32, device=self.device)  # Num edges x feat_dim
        edge_index = torch.tensor(edge_idxs, dtype=torch.long).t().contiguous().to(self.device)
        consecutive_mask = torch.tensor(consecutive_mask, dtype=torch.int8, device=self.device)

        if is_undirected:
            edge_index = torch.cat((edge_index, torch.stack((edge_index[1], edge_index[0]))), dim=1)
            edge_attr = torch.cat((edge_attr, edge_attr), dim=0)

        return edge_index, edge_attr, consecutive_mask, vis_idxs, unique_dets

    def construct_batch_graph_office(self, is_undirected=True, max_frame_dist=5):
        '''
        Returns batch graph data:   x: Node Embeddings, shape: Num nodes x feature dim(16)
                                    edge_idx: Edge indicies, shape: 2 x Num edges
                                    edge_attr: Edge features, shape: Num edges x feature dim(8)
                                    y: targets, shape: Num edges
        is_undirected: undirected Graph
        max_frame_dist: distance Graph nodes to connect with
        '''
        edge_idx, edge_attr, consecutive_mask, vis_idxs, unique_dets = self.get_edge_data_office(is_undirected=is_undirected, max_frame_dist=max_frame_dist)
        batch_graph = Data(x=None, edge_index=edge_idx, edge_attr=edge_attr, consecutive_mask=consecutive_mask,
                           vis_idxs=vis_idxs, unique_dets=unique_dets)

        return batch_graph



# ------ HELPER FUNCTIONS -----------------------------------------------------------------------------------------------
    def box2minmax(self, corner_pt_box):
        '''
        Box from 8x3 to minmax format
        Only works properly for axis aligned boxes
        '''
        xyz_min = torch.min(corner_pt_box, dim=0).values
        xyz_max = torch.max(corner_pt_box, dim=0).values
        box = np.concatenate((xyz_min.numpy(), xyz_max.numpy()))
        return box

    def cad2world_mat(self, rot, loc, scale, with_scale=True, constraint=False):
        '''
        Return cad2world matrix from annotations
        '''

        def euler_to_rot(euler_rot, fmt='torch', constraint=constraint):
            '''
            Euler to 3x3 Rotation Matrix transform
            '''
            if constraint:
                euler_rot = torch.tensor([0, euler_rot[1], 0])
            euler = mathutils.Euler(euler_rot)
            rot = np.array(euler.to_matrix())

            if fmt == 'torch':
                return torch.from_numpy(rot)
            else:
                return rot

        cad2world = torch.eye(4)
        scale_mat = torch.diag(torch.tensor([scale, scale, scale]))
        if with_scale:
            cad2world[:3, :3] = scale_mat @ euler_to_rot(rot, fmt='torch')
        else:
            cad2world[:3, :3] = euler_to_rot(rot, fmt='torch')

        if constraint:
            loc[1] = 0
        cad2world[:3, 3] = loc
        return cad2world