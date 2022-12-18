import torch
import mathutils
import numpy as np
from Tracking.utils.train_utils import check_pair
from Tracking.utils.vis_utils import cad2world_mat, box2minmax, box2minmax_axaligned
import traceback

def compute_edge_emb(edge_feature, edge_encoder, voxel_dim=12):
    '''
    Computes edge embedding for a concatenated edge feature tensor
    edge_feature: tensor concatenated two consecutive objects
    edge_encoder: MLP to transform edge embeddings
    '''
    obj_dim = int(edge_feature.shape[-1] / 2) # 19, 12 vox 7 pose
    obj_1_feature = edge_feature[:, :obj_dim]
    obj_2_feature = edge_feature[:, obj_dim:]

    # voxels feature
    voxel_1 = obj_1_feature[:, :voxel_dim]
    voxel_2 = obj_2_feature[:, :voxel_dim]

    # pose features
    rot_1  = obj_1_feature[:, voxel_dim:voxel_dim+3]
    loc_1 = obj_1_feature[:, voxel_dim+3:voxel_dim+6]
    scale_1 = torch.unsqueeze(obj_1_feature[:, -1], dim=-1)

    rot_2 = obj_2_feature[:, voxel_dim:voxel_dim+3]
    loc_2 = obj_2_feature[:, voxel_dim+3:voxel_dim+6]
    scale_2 = torch.unsqueeze(obj_2_feature[:, -1], dim=-1)

    relative_scale = torch.log(scale_2 / scale_1)
    relative_position = loc_2 - loc_1
    relative_rot = rot_2 - rot_1

    edge_emb = torch.cat((relative_scale, relative_position, relative_rot), dim=-1).type(torch.float32)

    edge_emb = edge_encoder(edge_emb)

    edge_embedding = torch.cat((voxel_1, voxel_2, edge_emb), dim=-1)

    return edge_embedding

def compute_edge_emb_nogeo(edge_feature, edge_encoder, device=None):
    '''
    Computes edge embedding for a concatenated edge feature tensor
    edge_feature: tensor concatenated two consecutive objects
    edge_encoder: MLP to transform edge embeddings
    '''
    obj_dim = int(edge_feature.shape[-1] / 2) # 19, 12 vox 7 pose
    obj_1_feature = edge_feature[:, :obj_dim]
    obj_2_feature = edge_feature[:, obj_dim:]


    # pose features
    rot_1  = obj_1_feature[:, :3]
    loc_1 = obj_1_feature[:, 3:6]
    scale_1 = torch.unsqueeze(obj_1_feature[:, -1], dim=-1)

    rot_2 = obj_2_feature[:, :3]
    loc_2 = obj_2_feature[:, 3:6]
    scale_2 = torch.unsqueeze(obj_2_feature[:, -1], dim=-1)

    relative_scale = torch.log(scale_2 / scale_1)
    relative_position = loc_2 - loc_1
    relative_rot = rot_2 - rot_1

    edge_emb = torch.cat((relative_scale, relative_position, relative_rot), dim=-1).type(torch.float32)

    edge_emb = edge_encoder(edge_emb.to(device))

    edge_embedding = edge_emb

    return edge_embedding


def recompute_edge_features(graph_in_features, obj_ids):
    '''
    Recompute edge feature matrix
    Skip instance pairs with obj ids = None
    obj_ids = list in list for 25 frames with n objs each
    '''

    num_combined_frames = len(graph_in_features) - 1
    edge_features = []

    for t in range(num_combined_frames):  # always check two neighboring frames

        img_1 = graph_in_features[t]
        img_2 = graph_in_features[t + 1]  # 1 x num instances x 16

        if img_1 is None or img_2 is None:
            continue

        obj_ids_1 = obj_ids[t]
        obj_ids_2 = obj_ids[t+1]

        # CHECK BOUNDING BOX OVERLAP GT AND PREDICTED IF IOU > THRES ASSIGN GT OBJ ID TO OBJECT
        for n in range(img_1.shape[1]):  # n = num instances in img 1
            obj_id_1 = obj_ids_1[n]

            if obj_id_1 is None:
                continue  # SKIP THIS INSTANCE FOR LOSS COMPUTATION

            for m in range(img_2.shape[1]):  # m = num instances in img 2
                obj_id_2 = obj_ids_2[m]

                if obj_id_2 is None:
                    continue
                # Concat object embeddings for edge features
                edge_feat = torch.cat((img_1[:, n, :], img_2[:, m, :]), dim=-1)
                edge_features.append(edge_feat)

    return edge_features

def construct_siamese_dataset(input, graph_in_features, thres=0.01, mode='train', device=None):
    '''
    Initial Dataset construction for lookup at later epochs
    '''

    data_dict = {}

    false_positives = 0  # Predicted box does not match with any GT bbox based on threshold IoU
    num_combined_frames = len(graph_in_features) - 1
    edge_features = []
    targets = []
    vis_idxs = []
    unique_dets = []
    img_ids = [None] * len(graph_in_features)

    # Only for triplet loss
    anchors = []
    positive_samples = []
    negative_samples = []
    num_pos_miss = 0
    num_neg_miss = 0


    for t in range(num_combined_frames):  # always check two neighboring frames

        obj_ids = [] # per frame object ids

        gt_bbox_1 = input[t]['gt_3Dbbox']  # num inst x 8 pts x xyz
        gt_bbox_2 = input[t + 1]['gt_3Dbbox']

        gt_id_1 = input[t]['gt_object_id']  # num inst
        gt_id_2 = input[t + 1]['gt_object_id']

        img_1 = graph_in_features[t]
        img_2 = graph_in_features[t + 1]  # 1 x num instances x 16

        if img_1 is None:
            continue

        pred_bbox_1 = input[t]['pred_3Dbbox']  # num inst x 8pts x xyz
        pred_cls_1 = input[t]['classes']

        # Location for matching
        pred_loc_1 = input[t]['translations']

        # CHECK BOUNDING BOX OVERLAP GT AND PREDICTED IF IOU > THRES ASSIGN GT OBJ ID TO OBJECT
        for n in range(img_1.shape[1]):  # n = num instances in img 1
            assert gt_id_1.shape[0] == gt_bbox_1.shape[0]

            # Only usage for calculating triplet loss
            anchor = None
            positive_sample = None
            negative_sample = None

            try:
                obj_id_1 = check_pair(pred_bbox_1[n, :, :], gt_bbox_1, gt_id_1,
                                      thres=thres)  # ASSERT SAME SORT AS GT BOX AND ID
            except:
                obj_id_1 = None
                traceback.print_exc()
                print('Issue with convex hull', ', Bad scene:', input[0]['scene'])

            obj_ids.append(obj_id_1) # n obj ids with Nones

            if obj_id_1 is None:
                false_positives += 1  # No overlapping GT bounding box found
                continue  # SKIP THIS INSTANCE FOR LOSS COMPUTATION

            if img_2 is None:
                # For MOTA evaluation add, also for visualisations
                unique_dets.append(
                    {'image': t, 'obj_1': n, 'obj_2': None, 'obj_id_1': int(obj_id_1), 'obj_id_2': None,
                     'loc_id_1': pred_loc_1[n], 'loc_id_2': None, 'cls_id_1': pred_cls_1[n], 'cls_id_2': None })
                continue

            pred_bbox_2 = input[t + 1]['pred_3Dbbox']
            pred_loc_2 = input[t + 1]['translations']
            pred_cls_2 = input[t+1]['classes']

            for m in range(img_2.shape[1]):  # m = num instances in img 2
                assert gt_id_2.shape[0] == gt_bbox_2.shape[0]

                try:
                    obj_id_2 = check_pair(pred_bbox_2[m, :, :], gt_bbox_2, gt_id_2, thres=thres)
                except:
                    obj_id_2 = None
                    print('Issue with convex hull', ', Bad scene:', input[0]['scene'])

                # ONLY FOR LAST FRAME WHICH ISNT COVERED IN OUTER LOOP ADD FP
                if t == len(graph_in_features) - 2 and n == img_1.shape[1] - 1:
                    if obj_id_2 is None:
                        false_positives += 1
                        continue

                # GT targets: active (1) and non-active (0) connections
                if obj_id_1 == obj_id_2 and obj_id_1 is not None and obj_id_2 is not None:  # both objects exist and same id
                    target = 1
                    anchor = img_1[:, n, :]
                    positive_sample = img_2[:, m, :]
                elif obj_id_1 != obj_id_2 and obj_id_1 is not None and obj_id_2 is not None:
                    negative_sample = img_2[:, m, :]
                    target = 0
                elif obj_id_2 is None:  # false prediction for any object in consecutive frame -> exclude
                    continue

                vis_idxs.append(
                    {'image': t, 'obj_1': n, 'obj_2': m, 'obj_id_1': int(obj_id_1), 'obj_id_2': int(obj_id_2),
                     'loc_id_1': pred_loc_1[n], 'loc_id_2': pred_loc_2[m],
                     'cls_id_1': pred_cls_1[n], 'cls_id_2': pred_cls_2[m]})

                targets.append(target)  # obj1 img1 with all obj img2, obj2 img1 with all obj img2 ... per sequence

                # Concat object embeddings for edge features
                edge_feat = torch.cat((img_1[:, n, :], img_2[:, m, :]), dim=-1)  # 1 x 2*16
                edge_features.append(edge_feat)

            if positive_sample is None:
                num_pos_miss += 1
            if negative_sample is None:
                num_neg_miss += 1  # issue scenes with 1 gt object

            # Per object in img 1:
            if positive_sample is not None and negative_sample is not None:  # exist a positive and negative sample in consecutive frame
                anchors.append(anchor)
                positive_samples.append(positive_sample)
                negative_samples.append(negative_sample)

        img_ids[t] = obj_ids

    # Get last frame obj ids
    last_obj_ids = []
    gt_bbox_last = input[-1]['gt_3Dbbox']  # num inst x 8 pts x xyz
    gt_id_last = input[-1]['gt_object_id']  # num inst
    img_last = graph_in_features[-1]

    if img_last is not None:
        pred_bbox_last = input[-1]['pred_3Dbbox']

        for j in range(img_last.shape[1]):
            try:
                obj_id_last = check_pair(pred_bbox_last[j, :, :], gt_bbox_last, gt_id_last, thres=thres)
            except:
                obj_id_last = None
                print('Issue with convex hull', ', Bad scene:', input[0]['scene'])

            last_obj_ids.append(obj_id_last)

        img_ids[-1] = last_obj_ids

    # Assignment dict
    data_dict['edge_features'] = edge_features
    data_dict['targets'] = torch.tensor(targets, dtype=torch.float32, device=device)
    data_dict['false_positives'] = false_positives
    data_dict['obj_ids'] = img_ids
    data_dict['vis_idxs'] = vis_idxs
    data_dict['unique_dets'] = unique_dets
    '''
    if mode == 'train':
        data_dict['vis_idxs'] = None#vis_idxs
        data_dict['unique_dets'] = None#unique_dets
    else:
        data_dict['vis_idxs'] = vis_idxs
        data_dict['unique_dets'] = unique_dets
    #data_dict['anchors'] = anchors
    #data_dict['positive_samples'] = positive_samples
    #data_dict['negative_samples'] = negative_samples
    '''

    return data_dict


def construct_siamese_dataset_vis(input, graph_in_features, thres=0.01, device=None):
    '''
    Dataset construction for visualisation purposes
    '''

    data_dict = {}

    false_positives = 0  # Predicted box does not match with any GT bbox based on threshold IoU
    num_combined_frames = len(graph_in_features) - 1
    edge_features = []
    targets = []
    vis_idxs = []
    unique_dets = []
    img_ids = [None] * len(graph_in_features)

    # Only for triplet loss
    anchors = []
    positive_samples = []
    negative_samples = []
    num_pos_miss = 0
    num_neg_miss = 0


    for t in range(num_combined_frames):  # always check two neighboring frames

        obj_ids = [] # per frame object ids

        gt_bbox_1 = input[t]['gt_3Dbbox']  # num inst x 8 pts x xyz
        gt_bbox_2 = input[t + 1]['gt_3Dbbox']

        gt_id_1 = input[t]['gt_object_id']  # num inst
        gt_id_2 = input[t + 1]['gt_object_id']

        img_1 = graph_in_features[t]
        img_2 = graph_in_features[t + 1]  # 1 x num instances x 16

        if img_1 is None:
            continue

        # pose for vis
        pred_bbox_1 = input[t]['pred_3Dbbox']  # num inst x 8pts x xyz
        pred_loc_1 = input[t]['translations']
        pred_rot_1 = input[t]['rotations']
        pred_scales_1 = input[t]['scales']

        # Voxel for vis
        pred_vox_1 = input[t]['voxels']

        # CHECK BOUNDING BOX OVERLAP GT AND PREDICTED IF IOU > THRES ASSIGN GT OBJ ID TO OBJECT
        for n in range(img_1.shape[1]):  # n = num instances in img 1
            assert gt_id_1.shape[0] == gt_bbox_1.shape[0]

            # Cad2world mat
            cad2world_1 = cad2world_mat(pred_rot_1[n], pred_loc_1[n], pred_scales_1[n])

            # Only usage for calculating triplet loss
            anchor = None
            positive_sample = None
            negative_sample = None

            try:
                obj_id_1 = check_pair(pred_bbox_1[n, :, :], gt_bbox_1, gt_id_1,
                                      thres=thres)  # ASSERT SAME SORT AS GT BOX AND ID
            except:
                obj_id_1 = None
                traceback.print_exc()
                print('Issue with convex hull', ', Bad scene:', input[0]['scene'])

            obj_ids.append(obj_id_1) # n obj ids with Nones

            if obj_id_1 is None:
                false_positives += 1  # No overlapping GT bounding box found
                continue  # SKIP THIS INSTANCE FOR LOSS COMPUTATION

            if img_2 is None:
                # For Mota and Visualisation also append here otherwise Detections are missed
                unique_dets.append(
                    {'image': t, 'obj_1': n, 'obj_2': None, 'obj_id_1': int(obj_id_1), 'obj_id_2': None,
                     'cad2world_1': cad2world_1, 'cad2world_2': None, 'vox_1': pred_vox_1[n],
                     'vox_2': None, 'box_1': box2minmax(pred_bbox_1[n]), 'box_2': None})
                continue

            pred_bbox_2 = input[t + 1]['pred_3Dbbox']
            pred_loc_2 = input[t + 1]['translations']
            pred_rot_2 = input[t + 1]['rotations']
            pred_scales_2 = input[t + 1]['scales']
            pred_vox_2 = input[t + 1]['voxels']

            for m in range(img_2.shape[1]):  # m = num instances in img 2
                assert gt_id_2.shape[0] == gt_bbox_2.shape[0]

                try:
                    obj_id_2 = check_pair(pred_bbox_2[m, :, :], gt_bbox_2, gt_id_2, thres=thres)
                except:
                    obj_id_2 = None
                    print('Issue with convex hull', ', Bad scene:', input[0]['scene'])

                # ONLY FOR LAST FRAME WHICH ISNT COVERED IN OUTER LOOP ADD FP
                if t == len(graph_in_features) - 2 and n == img_1.shape[1] - 1:
                    if obj_id_2 is None:
                        false_positives += 1
                        continue

                # GT targets: active (1) and non-active (0) connections
                if obj_id_1 == obj_id_2 and obj_id_1 is not None and obj_id_2 is not None:  # both objects exist and same id
                    target = 1
                    anchor = img_1[:, n, :]
                    positive_sample = img_2[:, m, :]
                elif obj_id_1 != obj_id_2 and obj_id_1 is not None and obj_id_2 is not None:
                    negative_sample = img_2[:, m, :]
                    target = 0
                elif obj_id_2 is None:  # false prediction for any object in consecutive frame -> exclude
                    continue

                # Get Cad2world mat
                cad2world_2 = cad2world_mat(pred_rot_2[m], pred_loc_2[m], pred_scales_2[m])
                vis_idxs.append(
                    {'image': t, 'obj_1': n, 'obj_2': m, 'obj_id_1': int(obj_id_1), 'obj_id_2': int(obj_id_2),
                     'cad2world_1': cad2world_1, 'cad2world_2': cad2world_2, 'vox_1': pred_vox_1[n], 'vox_2': pred_vox_2[m],
                     'box_1': box2minmax(pred_bbox_1[n]), 'box_2': box2minmax(pred_bbox_2[m])})

                targets.append(target)  # obj1 img1 with all obj img2, obj2 img1 with all obj img2 ... per sequence

                # Concat object embeddings for edge features
                edge_feat = torch.cat((img_1[:, n, :], img_2[:, m, :]), dim=-1)  # 1 x 2*16
                edge_features.append(edge_feat)

            if positive_sample is None:
                num_pos_miss += 1
            if negative_sample is None:
                num_neg_miss += 1  # issue scenes with 1 gt object

            # Per object in img 1:
            if positive_sample is not None and negative_sample is not None:  # exist a positive and negative sample in consecutive frame
                anchors.append(anchor)
                positive_samples.append(positive_sample)
                negative_samples.append(negative_sample)

        img_ids[t] = obj_ids

    # Get last frame obj ids
    last_obj_ids = []
    gt_bbox_last = input[-1]['gt_3Dbbox']  # num inst x 8 pts x xyz
    gt_id_last = input[-1]['gt_object_id']  # num inst
    img_last = graph_in_features[-1]

    if img_last is not None:
        pred_bbox_last = input[-1]['pred_3Dbbox']
        for j in range(img_last.shape[1]):
            try:
                obj_id_last = check_pair(pred_bbox_last[j, :, :], gt_bbox_last, gt_id_last, thres=thres)
            except:
                obj_id_last = None
                print('Issue with convex hull', ', Bad scene:', input[0]['scene'])

            last_obj_ids.append(obj_id_last)

        img_ids[-1] = last_obj_ids

    # Assignment dict

    data_dict['edge_features'] = edge_features
    data_dict['targets'] = torch.tensor(targets, dtype=torch.float32, device=device)
    data_dict['false_positives'] = false_positives
    data_dict['obj_ids'] = img_ids
    data_dict['vis_idxs'] = vis_idxs
    data_dict['unique_dets'] = unique_dets
    data_dict['anchors'] = anchors
    data_dict['positive_samples'] = positive_samples
    data_dict['negative_samples'] = negative_samples

    return data_dict

def construct_siamese_dataset_office(input, graph_in_features):
    '''
    Initial Dataset construction for lookup at later epochs
    '''

    data_dict = {}

    num_combined_frames = len(graph_in_features) - 1
    edge_features = []
    vis_idxs = []

    for t in range(num_combined_frames):  # always check two neighboring frames

        pred_bbox_1 = input[t]['pred_3Dbbox']  # num inst x 8pts x xyz
        pred_bbox_2 = input[t + 1]['pred_3Dbbox']

        img_1 = graph_in_features[t]
        img_2 = graph_in_features[t + 1]  # 1 x num instances x 16

        # pose for vis
        pred_loc_1 = input[t]['translations']
        pred_loc_2 = input[t + 1]['translations']

        pred_rot_1 = input[t]['rotations']
        pred_rot_2 = input[t + 1]['rotations']

        pred_scales_1 = input[t]['scales']
        pred_scales_2 = input[t + 1]['scales']

        # Voxel for vis
        pred_vox_1 = input[t]['voxels']
        pred_vox_2 = input[t + 1]['voxels']

        for n in range(img_1.shape[1]):  # n = num instances in img 1
            for m in range(img_2.shape[1]):  # m = num instances in img 2

                # Get Cad2world mats
                cad2world_1 = cad2world_mat(pred_rot_1[n], pred_loc_1[n], pred_scales_1[n], with_scale=True)
                cad2world_2 = cad2world_mat(pred_rot_2[m], pred_loc_2[m], pred_scales_2[m], with_scale=True)
                vis_idxs.append(
                    {'image': t, 'obj_1': n, 'obj_2': m, 'obj_id_1': None, 'obj_id_2': None,
                     'cad2world_1': cad2world_1, 'cad2world_2': cad2world_2,
                     'vox_1': pred_vox_1[n], 'vox_2': pred_vox_2[m],
                     'box_1': box2minmax_axaligned(pred_bbox_1[n]), 'box_2': box2minmax_axaligned(pred_bbox_2[m])})

                # Concat object embeddings for edge features
                edge_feat = torch.cat((img_1[:, n, :], img_2[:, m, :]), dim=-1)  # 1 x 2*16
                edge_features.append(edge_feat)

    # Assignment dict
    data_dict['edge_features'] = edge_features
    data_dict['vis_idxs'] = vis_idxs

    return data_dict