import sys
import torch
import numpy as np
import open3d as o3d
import mathutils

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

from Tracking.utils.train_utils import convert_voxel_to_pc


def get_precision(predictions, targets):

    # Binarize predictions
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0

    precision = precision_score(targets, predictions, zero_division=0)
    return precision

def get_recall(predictions, targets):

    # Binarize predictions
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0

    recall = recall_score(targets, predictions, zero_division=0)
    return recall

def get_f1(predictions, targets):

    # Binarize predictions
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0

    f1 = f1_score(targets, predictions, zero_division='warn') # warn only once
    return f1

def get_MOTA(predictions, targets, gt_objects, misses, fps):
    '''
    Full val/test set MOTA calculations
    MOTA score: 1 - num_misses + false positives + id_switches / total num_objects in all frames
    false_positives: Predicted 3D BBOX does not overlap with any GT 3D BBOX more than a threshold e.g. 0.2 IoU
    num_misses: For a GT 3D BBOX there exist no predicted 3D BBOX which overlaps more than min threshold, or less pred than gt objects
    id_switches: GT trajectory and predicted trajectory have do not match in object identities, predicted active/nonactive edge incorrect
    '''

    # Binarize predictions
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0

    id_switches = np.count_nonzero(targets - predictions)
    MOTA = 1.0 - (float(misses + fps + id_switches) / float(gt_objects))

    return MOTA, id_switches

def get_mota_df(num_gt_objs, num_misses, num_fps, num_switches):
    '''
    Calculates a mota score over all frames seen
    '''
    mota = 1.0 - (float(num_misses + num_fps + num_switches) / float(num_gt_objs))
    return mota


