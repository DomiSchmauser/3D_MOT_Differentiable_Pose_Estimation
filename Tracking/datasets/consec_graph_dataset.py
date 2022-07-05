import sys
import torch
from torch_geometric.data import Data
from torch_geometric.utils.undirected import to_undirected

sys.path.append('..') #Hack add ROOT DIR
from Tracking.utils.train_utils import check_pair


class GraphDataset():
    '''
    Graph dataset class enables data handling for pytorch geometric graphs
    init_node_emb: voxel features, shape: num nodes x feature dim
    rotations, translations, scales, -> edge features, shape: num nodes x (3 or 1)
    instances_count: per image instances
    '''

    def __init__(self, init_node_emb, rotations, translations, scales, input, instances_count, num_images=25):

        self.init_node_emb = init_node_emb
        self.rotations = rotations
        self.translations = translations
        self.scales = scales
        self.input = input
        self.instances_count = instances_count
        self.num_images = num_images
        self.box_iou_thres = 0.01  # Min IoU threshold GT and predicted 3D box
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def get_edge_data(self, is_undirected=True):
        '''
        Get edge attributes and according edge indicies
        Currently directed graph and only consecutive frames connected
        is_undirected: Graph nodes connected both ways 0-1, 1-0, duplicate idxs, edge features and targets
        '''

        relative_scales = []
        relative_rotations = []
        relative_positions = []
        relative_times = []

        img_inst_count = 0
        edge_idxs = []

        # Validation data
        vis_idxs = []
        false_positives = 0
        targets = []
        node_color = []

        for t in range(self.num_images - 1):

            gt_bbox_1 = self.input[t]['gt_3Dbbox']  # num inst x 8 pts x xyz
            gt_bbox_2 = self.input[t+1]['gt_3Dbbox']

            gt_id_1 = self.input[t]['gt_object_id']  # num inst
            gt_id_2 = self.input[t+1]['gt_object_id']

            pred_bbox_1 = self.input[t]['pred_3Dbbox']  # num inst x 8pts x xyz
            pred_bbox_2 = self.input[t+1]['pred_3Dbbox']

            start_inst_count = img_inst_count # start count instances frame t
            img_inst_count += self.instances_count[t] # start count instances frame t+1
            consecutive_inst_count = img_inst_count + self.instances_count[t+1]

            for n in range(start_inst_count, img_inst_count):

                # Object Matching frame t
                try:
                    obj_id_1 = check_pair(pred_bbox_1[n-start_inst_count, :, :], gt_bbox_1, gt_id_1,
                                          thres=self.box_iou_thres)
                except:
                    obj_id_1 = None
                    print('Issue with convex hull', ', Bad scene:', input[0]['scene'])

                if obj_id_1 is None:
                    false_positives += 1  # No overlapping GT bounding box found
                    node_color.append(1)
                    continue  # SKIP THIS INSTANCE FOR GRAPH CONSTRUCTION
                else:
                    node_color.append(0)

                for m in range(img_inst_count, consecutive_inst_count):  # n0-m0 n0-m1 n1-m0 n1-m1 ....

                    # Object Matching frame t+1
                    try:
                        obj_id_2 = check_pair(pred_bbox_2[m-img_inst_count, :, :], gt_bbox_2, gt_id_2, thres=self.box_iou_thres)
                    except:
                        obj_id_2 = None
                        print('Issue with convex hull', ', Bad scene:', input[0]['scene'])

                    # ONLY FOR LAST FRAME WHICH ISNT COVERED IN OUTER LOOP ADD FP
                    if t == self.num_images - 2 and n == img_inst_count - 1:
                        if obj_id_2 is None:
                            false_positives += 1
                            node_color.append(1)
                        else:
                            node_color.append(0)

                    # GT targets: active (1) and non-active (0) connections
                    if obj_id_1 == obj_id_2 and obj_id_1 is not None and obj_id_2 is not None:  # both objects exist and same id
                        target = 1
                    elif obj_id_1 != obj_id_2 and obj_id_1 is not None and obj_id_2 is not None:
                        target = 0
                    elif obj_id_2 is None:  # false prediction for any object in consecutive frame -> exclude
                        continue

                    vis_idxs.append({'image': t, 'obj_1': n, 'obj_2': m, 'obj_id_1': int(obj_id_1),
                                     'obj_id_2': int(obj_id_2)})

                    if is_undirected:
                        targets.append(target)  # obj1 img1 with all obj img2, obj2 img1 with all obj img2 ... per sequence
                        targets.append(target)  # twice for undirected 0-1 and 1-0
                    else:
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
                    relative_time = torch.unsqueeze(torch.tensor([t + 1 - t], dtype=torch.int64),
                                                    dim=0)  # always 1 for consecutive frames
                    relative_times.append(relative_time)
                    # relative_appearance -> could be also an edge feature but is already encoded in the node

                    if is_undirected:
                        edge_idxs.append([m, n]) # 1 - 0

                        relative_scale = torch.unsqueeze(torch.log(self.scales[m, :] / self.scales[n, :]),
                                                         dim=0)  # feat t / feat t+1
                        relative_scales.append(relative_scale)
                        relative_position = torch.unsqueeze(self.translations[m, :] - self.translations[n, :], dim=0)
                        relative_positions.append(relative_position)
                        relative_rot = torch.unsqueeze(self.rotations[m, :] - self.rotations[n, :], dim=0)
                        relative_rotations.append(relative_rot)
                        relative_time = torch.unsqueeze(torch.tensor([t + 1 - t], dtype=torch.int64),
                                                        dim=0)  # always 1 for consecutive frames
                        relative_times.append(relative_time)
                        # relative_appearance -> could be also an edge feature but is already encoded in the node


        relative_scales = torch.cat(relative_scales, dim=0)  # num_edges x 1
        relative_positions = torch.cat(relative_positions, dim=0)  # num_edges x 3
        relative_rotations = torch.cat(relative_rotations, dim=0)  # num_edges x 3
        relative_times = torch.cat(relative_times, dim=0)  # num_edges x 1

        edge_attr = torch.cat((relative_positions, relative_rotations, relative_scales, relative_times),
                              dim=-1).to(dtype=torch.float32, device=self.device)  # Num edges x feat_dim
        edge_index = torch.tensor(edge_idxs, dtype=torch.long).t().contiguous().to(self.device)

        return edge_index, edge_attr, torch.tensor(targets, dtype=torch.float32, device=self.device), vis_idxs, false_positives, torch.tensor(node_color)

    def construct_batch_graph(self, is_undirected=True):
        '''
        Returns batch graph data:   x: Node Embeddings, shape: Num nodes x feature dim(9)
                                    edge_idx: Edge indicies, shape: 2 x Num edges
                                    edge_attr: Edge features, shape: Num edges x feature dim(8)
                                    y: targets, shape: Num edges
        '''

        edge_idx, edge_attr, targets, vis_idxs, false_positives, node_color = self.get_edge_data(is_undirected=is_undirected)
        batch_graph = Data(x=self.init_node_emb, edge_index=edge_idx, edge_attr=edge_attr, y=targets)

        return batch_graph, vis_idxs, false_positives, node_color