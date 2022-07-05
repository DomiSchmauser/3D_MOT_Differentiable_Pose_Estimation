import logging
import os, sys, shutil, re, traceback, time, json, random
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from detectron2.data.samplers import TrainingSampler
import roi_heads #Required for call register()

import warnings
warnings.filterwarnings('ignore')

from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer

from detectron2.engine import default_argument_parser, default_writers, launch
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.modeling import build_model


from cfg_setup import init_cfg
from tracker.postprocess import postprocess_dets_office
from dvis import dvis

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

# Load tracking setup
from Tracking.options import Options
from Tracking.tracker.tracking_front import Tracker
from Tracking.visualise.visualise import visualise_pred_sequence_office, visualise_gt_sequence_office
from Tracking.utils.vis_utils import fuse_pose

options = Options()
opts = options.parse()
if opts.use_graph:
    from Tracking.mpn_trainer import Trainer
else:
    from Tracking.trainer import Trainer

from Detection.data.office_dataset import Office_dataset

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate


logger = logging.getLogger("front_logger")

class Office_Trainer(DefaultTrainer):
    '''
    Main Trainer class for End-to-End Detection, Pose Estimation and Tracking
    '''

    def __init__(self):
        self.dataset = Office_dataset
        DATA_DIR = CONF.PATH.OFFICEDATA

        infer_dataset = self.dataset(
            base_dir=DATA_DIR,
            split='infer')

        self.infer_loader = DataLoader(
            infer_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False)

    def fuse_pose(self, trajectories, seq_len=None, constraint=False):
        '''
        Pose fusion via slurp and spline interpolation
        '''

        def get_scale(m):
            if type(m) == torch.Tensor:
                return m.norm(dim=0)
            return np.linalg.norm(m, axis=0)

        def fill_last(fill_list, exp_dim=False):
            # fill with last value
            for t_idx, tt in enumerate(fill_list):
                if tt is None:
                    for i in range(t_idx - 1, -1, -1):
                        if fill_list[i] is not None:
                            if exp_dim:
                                fill_list[t_idx] = np.expand_dims(fill_list[i], axis=0)
                            else:
                                fill_list[t_idx] = fill_list[i]
                            break
            return fill_list

        def fill_last_t(fill_list, exp_dim=True):
            # fill with last value
            for t_idx, tt in enumerate(fill_list):
                if tt.sum() == 0:
                    for i in range(t_idx - 1, -1, -1):
                        if fill_list[i].sum() != 0:
                            if exp_dim:
                                fill_list[t_idx] = np.expand_dims(np.squeeze(fill_list[i]), axis=0)
                            else:
                                fill_list[t_idx] = fill_list[i]
                            break
                else:
                    fill_list[t_idx] = np.expand_dims(fill_list[t_idx], axis=0)

            for t_idx, tt in enumerate(fill_list):
                if len(tt.shape) == 1:
                    fill_list[t_idx] = np.expand_dims(fill_list[t_idx], axis=0)

            return fill_list

        def unscale_mat(cad2world):

            c2w_cpy = torch.clone(cad2world)
            rot = cad2world[:3, :3]
            scale = get_scale(rot)
            unscaled_rot = rot / scale
            c2w_cpy[:3, :3] = unscaled_rot
            return c2w_cpy

        new_trajectories = []

        times = np.arange(seq_len)
        for traj in trajectories:
            key_times = []
            key_trans = []
            key_rots = []
            t_trans = [np.zeros(3) for i in range(seq_len)]
            t_vox = [None for i in range(seq_len)]
            t_box = [None for i in range(seq_len)]
            t_id = [None for i in range(seq_len)]
            t_scale = [None for i in range(seq_len)]
            for pred in traj:
                key_rots.append(torch.unsqueeze(unscale_mat(pred['obj']['cad2world'][:3, :3]), dim=0))
                key_trans.append(torch.unsqueeze(pred['obj']['cad2world'][:3, 3], dim=0))
                key_times.append(pred['scan_idx'])
                t_trans[pred['scan_idx']] = pred['obj']['cad2world'][:3, 3].numpy()
                t_vox[pred['scan_idx']] = pred['obj']['voxel']
                t_id[pred['scan_idx']] = pred['obj']['obj_idx']
                t_box[pred['scan_idx']] = pred['obj']['compl_box']
                t_scale[pred['scan_idx']] = get_scale(pred['obj']['cad2world'][:3, :3])

            times = np.linspace(key_times[0], key_times[-1], num=key_times[-1] - key_times[0] + 1).astype(np.int)
            traj_rots = torch.cat(key_rots, dim=0).numpy()
            key_trans = torch.cat(key_trans, dim=0).numpy()

            t_trans = np.concatenate(fill_last_t(t_trans, exp_dim=True), axis=0)
            t_trans[:, 0] = gaussian_filter1d(t_trans[:, 0], 3)
            t_trans[:, 1] = gaussian_filter1d(t_trans[:, 1], 3)
            t_trans[:, 2] = gaussian_filter1d(t_trans[:, 2], 3)
            if constraint:
                t_trans[:, 1] = 0
            t_vox = fill_last(t_vox)
            t_id = fill_last(t_id)
            t_box = fill_last(t_box)
            t_scale = fill_last(t_scale)

            r = R.from_matrix(traj_rots)
            slerp = Slerp(key_times, r)
            interp_rots = slerp(times)
            interp_rotmat = interp_rots.as_matrix()
            euler_rots = interp_rots.as_euler('xyz')
            euler_rots[:, -1] = gaussian_filter1d(euler_rots[:, -1], 3)  # 3 = sigma = standard deviation
            euler_rots[:, -1] = np.clip(euler_rots[:, -1], euler_rots[0, -1] - (0.2 * euler_rots[0, -1]),
                                        euler_rots[0, -1] + (0.2 * euler_rots[0, -1]))
            euler_rots[:, 0] = gaussian_filter1d(euler_rots[:, 0], 3)  # 3 = sigma = standard deviation
            euler_rots[:, 0] = np.clip(euler_rots[:, 0], euler_rots[0, 0] - (0.2 * euler_rots[0, 0]),
                                        euler_rots[0, 0] + (0.2 * euler_rots[0, 0]))
            euler_rots[:, 1] = gaussian_filter1d(euler_rots[:, 1], 3)  # 3 = sigma = standard deviation
            euler_rots[:, 1] = np.clip(euler_rots[:, 1], euler_rots[0, 1] - (0.2 * euler_rots[0, 1]),
                                        euler_rots[0, 1] + (0.2 * euler_rots[0, 1]))
            if constraint:
                euler_rots[:, 0] = 0
                euler_rots[:, -1] = 0

            r_e = R.from_euler('xyz', euler_rots, degrees=False)
            interp_rotmat = r_e.as_matrix()

            # test = np.diag(t_scale[0]) @ interp_rotmat[0,:,:]
            constraint_flip = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            new_traj = []
            for t in times:
                t_dict = dict()
                t_dict['scan_idx'] = t
                t_dict['obj'] = dict()
                t_dict['obj']['cad2world'] = np.identity(4)
                t_dict['obj']['cad2world'][:3, :3] = (np.diag(t_scale[t]) @ interp_rotmat[t - key_times[0], :, :])
                t_dict['obj']['cad2world'][:3, 3] = t_trans[t]
                t_dict['obj']['voxel'] = t_vox[t]
                t_dict['obj']['obj_idx'] = t_id[t]
                t_dict['obj']['compl_box'] = t_box[t]
                new_traj.append(t_dict)

            new_trajectories.append(new_traj)

        return new_trajectories

    def backproject_rgb(self, rgb, depth, intrinsics, flip=False):
        '''
        Backproject depth map to camera space, with additional rgb values
        Returns: Depth PC with according RGB values in camspace, used idxs in pixel space
        '''

        intrinsics_inv = np.linalg.inv(intrinsics)
        non_zero_mask = (depth > 0)

        idxs = np.where(non_zero_mask)
        grid = np.array([idxs[1], idxs[0]])

        length = grid.shape[1]
        ones = np.ones([1, length])
        uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]

        xyz = intrinsics_inv @ uv_grid  # [3, num_pixel]
        xyz = np.transpose(xyz)  # [num_pixel, 3]

        z = depth[idxs[0], idxs[1]]

        pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
        pts[:, 1] = -pts[:, 1]
        pts[:, 2] = -pts[:, 2]

        if flip:
            dvis_flip = np.diag([1, -1, -1])
            pts = pts@dvis_flip

        rgb_vals = rgb[idxs[0], idxs[1]]

        rgb_pts = np.concatenate((pts, rgb_vals), axis=-1)

        return rgb_pts

    def vis_inference(self, im, outputs):
        '''
        Visualize 2D object detections and segmentations
        '''

        im = im.numpy()
        v = Visualizer(im[:, :, ::-1],
                       metadata=None,
                       scale=2.5,
                       instance_mode=ColorMode.IMAGE_BW
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("office_inference", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)

    def inference_office(self, cfg, model, resume=False, max_window_len=30, vis_seg=False, vis_gt=False, start_idx=160):
        '''
        Main inference pipeline with object detection and tracklet association
        '''

        # Tracking classes
        trainer = Trainer(opts, combined=True)
        if resume:
            trainer.load_model()

        # Evaluation Tracking
        MOTracker = Tracker(seq_len=max_window_len)
        model.eval()
        trainer.set_eval()

        seq_inputs = []
        seq_outputs = []
        seq_scan_pcs = []
        seq_pattern = 'office_dataset' + "/(.*?)/"

        for idx, data in enumerate(self.infer_loader):

            if int(idx + 1) % 40 == 0:
                print('Scan {} processed ...'.format(int(idx+1)))

            if idx == 0:
                prev_seq_name = re.search(seq_pattern, data['seq_id'][0]).group(1)

            seq_name = re.search(seq_pattern, data['seq_id'][0]).group(1)

            with torch.no_grad():
                im = torch.squeeze(data['rgb']) #todo is fed as bgr image to the model

                # Scan pointcloud for visualisation
                if idx == 0:
                    pc = self.backproject_rgb(torch.squeeze(data['rgb_fs']).numpy()[:, :, ::-1], torch.squeeze(data['depth_fs']).numpy(),
                                              torch.squeeze(data['camera_intrinsics_fs']).numpy())
                    #dvis(pc, fmt='xyzrgb', vs=0.1, l=[0, 4], vis_conf={'opacity': 0.6},
                    #     name='background')  # set opacity to 0.5

                # For sequences not starting directly with object movement
                if idx < start_idx:
                    continue

                img_input = [{'image': im.permute(2, 0, 1).contiguous()}]
                outputs = model(img_input) #C x H x W as list(dict('image'))

                # Only for 2D Segmentation Visualization
                if vis_seg:
                    if int(idx + 1) % 20 == 0:
                        self.vis_inference(im, outputs[0])
                    continue

                # Set sampling rate
                sampling_rate = 10
                if vis_gt and (int(idx + 1) % sampling_rate == 0 or idx==0):
                    pc = self.backproject_rgb(torch.squeeze(data['rgb_fs']).numpy()[:, :, ::-1],
                                              torch.squeeze(data['depth_fs']).numpy(),
                                              torch.squeeze(data['camera_intrinsics_fs']).numpy())
                    seq_scan_pcs.append(pc)

                seq_inputs.append(data)
                seq_outputs.append(outputs)
                # Process sequences independently
                if seq_name == prev_seq_name and len(seq_inputs) < max_window_len and int(idx + 1) != len(self.infer_loader):
                    prev_seq_name = seq_name
                    continue

                prev_seq_name = seq_name
                seq_len = len(seq_inputs)

                # Postprocess and run Tracking network
                window_seq_data = postprocess_dets_office(seq_inputs, seq_outputs, obj_threshold=0.19, mode='val') #0.05 for office 1, 0.24 for office v3

                #if None in window_seq_data:
                #    print('Sequence {} contains image with no predicted objects skipping ...'.format(seq_name))
                #    continue

                window_seq_data = [window_seq_data]
                tracking_outputs = trainer.process_batch_office(window_seq_data, mode='val')
                if not tracking_outputs:
                    print('Empty predictions skipping ...')
                    continue

                # Visualisation with DVIS
                if vis_gt:
                    visualise_gt_sequence_office(seq_scan_pcs, seq_name=seq_name, seq_len=int(seq_len/sampling_rate)+1)
                else:
                    pred_trajectories = MOTracker.analyse_trajectories_office_new(tracking_outputs[0], seq_len=seq_len)
                    pred_trajectories = self.fuse_pose(pred_trajectories, seq_len=seq_len)
                    visualise_pred_sequence_office(pred_trajectories, pc, seq_name=seq_name, seq_len=seq_len)


                # Reset storage
                seq_inputs = []
                seq_outputs = []
                seq_scan_pcs = []

## ------------------------------ Class methods end ------------------
def setup():
    num_classes = 7
    cfg = init_cfg(num_classes, office=True)
    return cfg


def main(args, use_pretrained=True, use_finetuned=True):
    cfg = setup()

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    model_path = os.path.join(CONF.PATH.DETECTMODEL, 'best_model.pth')
    # model_path = cfg.MODEL.WEIGHTS

    # FIRST LOAD NOCS AND VOXEL WEIGHTS
    if use_pretrained:
        DetectionCheckpointer(model, save_dir='').resume_or_load(
            model_path, resume=True
        )

    '''
    # SECOND LOAD MASK AND BOX WEIGHTS
    backbone_path = os.path.join(CONF.PATH.DETECTMODEL, 'best_model_backbone.pth')
    if use_finetuned:
        DetectionCheckpointer(model, save_dir='').resume_or_load(
            backbone_path, resume=True
        )
    '''

    # Load model only Backbone
    backbone_path = os.path.join(CONF.PATH.DETECTMODEL, 'best_model_backbone.pth')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(backbone_path)
    keys_to_skip = ['nocs', 'noc']#, 'fpn', 'res5']
    pretrained_dict = {k: v for k, v in pretrained_dict['model'].items() if
                       k in model_dict and not any(ks in k for ks in keys_to_skip)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


    office_infer = Office_Trainer()
    office_infer.inference_office(cfg, model, resume=use_pretrained, max_window_len=220) # 210 - 540 = 330 frames

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
