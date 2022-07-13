import os, sys
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2 import model_zoo

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

# Initialize training config
def init_cfg(num_classes, combined=False, run_test=False, office=False, office_train=False):
    '''
    Set parameters:
    run_test: for final test run
    eval_period: num iterations between each evaluation run
    ims_per_batch: batch size
    checkpoint period: save model after n iterations
    base_lr & weight_decay: training setup
    '''

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) # Loads only backbone weights

    # MOTFront Dataset
    if not office:
        cfg.DATASETS.TRAIN = ("front_train",)
        if not run_test:
            cfg.DATASETS.TEST = ("front_val",)
        else:
            cfg.DATASETS.TEST = ("front_test",)

    # Office Dataset
    else:
        cfg.DATASETS.TRAIN = ("office_train",)
        if office_train:
            cfg.DATASETS.TEST = ("office_train",)
        else:
            cfg.DATASETS.TEST = ("office_inference",)

    cfg.TEST.EVAL_PERIOD = 1000
    cfg.TEST.IMG_SAVE_FREQ = 4 # Every 4th evaluation run save pred images to tensorboard
    cfg.TEST.START_EVAL = 1  # Start evaluation after n iterations
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = False

    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = 0

    # Input
    cfg.INPUT.MIN_SIZE_TRAIN = (240,)
    # Sample size of smallest side by choice or random selection from range give by
    # INPUT.MIN_SIZE_TRAIN
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    # Maximum size of the side of the image during training
    cfg.INPUT.MAX_SIZE_TRAIN = 320
    # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
    cfg.INPUT.MIN_SIZE_TEST = 240
    # Maximum size of the side of the image during testing
    cfg.INPUT.MAX_SIZE_TEST = 320
    cfg.INPUT.RANDOM_FLIP = 'none'
    cfg.INPUT.FORMAT = "BGR" # Image input format -> will be transformed to rgb in mapper heads

    # ROI HEADS
    cfg.MODEL.ROI_HEADS.NAME = "VoxelNocsHeads"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.75]
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.20
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1

    if not office:
        cfg.MODEL.PIXEL_MEAN = [59.64, 61.96, 64.02] # MOTFront
    else:
        cfg.MODEL.PIXEL_MEAN = [92.0080866, 98.01352945, 121.7431208] # office

    cfg.MODEL.PIXEL_STD = [1, 1, 1]
    cfg.MODEL.MASK_ON = True

    # Voxel Head
    cfg.MODEL.VOXEL_ON = True
    cfg.MODEL.ROI_VOXEL_HEAD = CN()
    cfg.MODEL.ROI_VOXEL_HEAD.LOSS_WEIGHT = 0.75
    if office_train:
        cfg.MODEL.VOXEL_ON = True
        cfg.MODEL.ROI_VOXEL_HEAD.LOSS_WEIGHT = 0.015

    cfg.MODEL.ROI_VOXEL_HEAD.NAME = 'Pix2VoxDecoder'
    cfg.MODEL.ROI_VOXEL_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_VOXEL_HEAD.POOLER_TYPE = "ROIAlign"
    cfg.MODEL.ROI_VOXEL_HEAD.POOLER_SAMPLING_RATIO = 0

    # Nocs Head
    cfg.MODEL.NOCS_ON = True
    if office_train:
        cfg.MODEL.NOCS_ON = False
    cfg.MODEL.ROI_NOCS_HEAD = CN()
    cfg.MODEL.ROI_NOCS_HEAD.USE_BIN_LOSS = False # True for classification loss, False for smooth l1 loss
    cfg.MODEL.ROI_NOCS_HEAD.NUM_BINS = 32
    if cfg.MODEL.ROI_NOCS_HEAD.USE_BIN_LOSS:
        cfg.MODEL.ROI_NOCS_HEAD.LOSS_WEIGHT = 0.2
    else:
        cfg.MODEL.ROI_NOCS_HEAD.LOSS_WEIGHT = 3
    cfg.MODEL.ROI_NOCS_HEAD.IOU_THRES = 0.5
    cfg.MODEL.ROI_NOCS_HEAD.NAME = 'NocsDecoder'
    cfg.MODEL.ROI_NOCS_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_NOCS_HEAD.POOLER_TYPE = "ROIAlign"
    cfg.MODEL.ROI_NOCS_HEAD.POOLER_SAMPLING_RATIO = 0

    # Solver Options
    cfg.SOLVER.CHECKPOINT_PERIOD = 3000 #save model each n iterations
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.SOLVER.STEPS = []  # decay learning rate
    cfg.SOLVER.WARMUP_FACTOR = 1
    cfg.SOLVER.WARMUP_ITERS = 0
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.GAMMA = 1
    cfg.SOLVER.WEIGHT_DECAY = 0.0005 # L2-Regularization
    cfg.SOLVER.IMS_PER_BATCH = 2 # Batch size
    cfg.SOLVER.BASE_LR = 0.0008
    cfg.SOLVER.MAX_ITER = 240000

    # Combined settings
    if combined:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4  # Overlap threshold used for non-maximum suppression (suppress boxes with IoU >= this threshold)
    elif office:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2 # higher more suppression
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2  # lower more suppression, overlap threshold used for non-maximum suppression (suppress boxes with IoU >= this threshold)


    cfg.OUTPUT_DIR = CONF.PATH.DETECTOUTPUT

    return cfg


def inference_cfg(num_classes):

    train_cfg = init_cfg(num_classes)
    train_cfg.MODEL.WEIGHTS = os.path.join(CONF.PATH.DETECTMODEL, "best_model.pth")  # path to the model we just trained
    train_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    train_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4 # Overlap threshold used for non-maximum suppression (suppress boxes with IoU >= this threshold)

    return train_cfg
