import sys
import os
import cv2
import numpy as np
import json
import h5py
import torch
import open3d as o3d

from torchvision import transforms
from torch.utils.data import Dataset

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

class Front_dataset(Dataset):
    def __init__(self, base_dir, split='train', transform=None, with_scene_pc=False):
        self.transform = transform  # using transform in torch
        self.split = split
        self.scenes_skip = []
        self.data_dir = os.path.join(base_dir, self.split)
        self.hdf5_dir = os.path.join(CONF.PATH.DETECTDATA, self.split)
        self.scenes = [f for f in os.listdir(os.path.abspath(self.data_dir)) if f not in self.scenes_skip]
        self.json_dir = os.path.join(CONF.PATH.DETECTDATA, self.split)
        self.camera_intrinsics = np.array([[292.87803547399, 0, 0], [0, 292.87803547399, 0], [0, 0, 1]])
        self.with_scene_pc = with_scene_pc

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):

        scene = self.scenes[idx]

        data_path = os.path.join(self.data_dir, scene)
        #json_path = os.path.join(self.json_dir, scene)

        unsorted_imgs = [f for f in os.listdir(os.path.abspath(data_path)) if 'feat' not in f]
        img_ints = [int(img[:-3]) for img in unsorted_imgs]
        imgs = [im for _, im in sorted(zip(img_ints, unsorted_imgs))]

        output = []
        for idx_, img in enumerate(imgs):

            # Load scan pointcloud
            if self.with_scene_pc:
                hdf5_path = os.path.join(self.hdf5_dir, scene, str(idx_) + '.hdf5')
                rgb_path = os.path.join(self.hdf5_dir, scene, 'coco_data', 'rgb_' + str(idx_).zfill(4) + '.png')
                depth_map, campose, cx, cy = self.load_hdf5(hdf5_path)
                self.camera_intrinsics[0, 2] = cx
                self.camera_intrinsics[1, 2] = cy
                rgb_img = self.load_rgb(rgb_path)
                cam_rgb_pc = self.backproject_rgb(rgb_img, depth_map, self.camera_intrinsics)
                world_pc = self.cam2world(cam_rgb_pc, campose)


            img_path = os.path.join(data_path, img)
            hf = h5py.File(img_path, 'r')

            # Unpack GT data
            gt_object_id = np.array(hf.get("gt_objid"))
            gt_voxels = np.array(hf.get("gt_voxels"))
            gt_3Dbbox = np.array(hf.get("gt_3Dbbox"))
            gt_locations = np.array(hf.get("gt_locations"))
            gt_rotations = np.array(hf.get("gt_rotations"))
            gt_compl_box = np.array(hf.get("gt_compl_box"))
            gt_scales = np.array(hf.get("gt_scales"))
            gt_classes = np.array(hf.get("gt_cls")) - 1 # -1 because predicted starts at 0 and gt at 1

            # Unpack predicted data
            classes = np.array(hf.get("classes")) #from 0 to 6
            objectness_scores = np.array(hf.get("objectness_scores"))
            rotations = np.array(hf.get("rotations"))
            translations = np.array(hf.get("translations"))
            scales = np.array(hf.get("scales"))
            voxels = np.array(hf.get("voxels"))
            pred_3Dbbox = np.array(hf.get("pred_3Dbbox"))

            img_dict = {'classes': torch.tensor(classes, dtype=torch.int),
                'objectness_scores': objectness_scores,
                'rotations': torch.tensor(rotations),
                'translations': torch.tensor(translations),
                'scales': torch.tensor(scales),
                'voxels': torch.tensor(voxels),
                'pred_3Dbbox': torch.tensor(pred_3Dbbox),
                'gt_object_id': torch.tensor(gt_object_id),
                'gt_locations': torch.tensor(gt_locations),
                'gt_rotations': torch.tensor(gt_rotations),
                'gt_3Dbbox': torch.tensor(gt_3Dbbox),
                'gt_compl_box': torch.tensor(gt_compl_box),
                'gt_scales': torch.tensor(gt_scales),
                'gt_classes': torch.tensor(gt_classes),
                'gt_voxels': gt_voxels,
                'image': img,
                'scene': scene
            }
            if self.with_scene_pc:
                img_dict['world_pc'] = world_pc
            output.append(img_dict)

        return output # list of parameters of n images

    def load_hdf5(self, hdf5_path):
        '''
        Loads campose and depth map from an hdf5 file
        returns additional camera intrinsics cx, cy
        '''

        with h5py.File(hdf5_path, 'r') as data:
            for key in data.keys():
                if key == 'depth':
                    depth = np.array(data[key])
                elif key == 'campose':
                    campose = np.array(data[key])

        img_width = depth.shape[1]
        img_height = depth.shape[0]

        cx = (img_width / 2) - 0.5  # 0,0 is center top-left pixel -> -0,5
        cy = (img_height / 2) - 0.5  # 0,0 is center top-left pixel -> -0,5

        return depth, campose, cx, cy

    def load_rgb(self, rgb_path):
        '''
        Loads a rgb image from a png file
        '''
        bgr_img = cv2.imread(rgb_path)
        rgb_img = bgr_img[:, :, ::-1]
        rgb_img = np.array(rgb_img, dtype=np.float32)

        return rgb_img

    def backproject_rgb(self, rgb, depth, intrinsics, debug_mode=False):
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

        rgb_vals = rgb[idxs[0], idxs[1]]

        rgb_pts = np.concatenate((pts, rgb_vals), axis=-1)

        if debug_mode:
            depth_pc_obj = o3d.geometry.PointCloud()
            nocs_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
            depth_pc_obj.points = o3d.utility.Vector3dVector(pts)
            o3d.visualization.draw_geometries([depth_pc_obj, nocs_origin])

        return rgb_pts

    def cam2world(self, rgb_pts, campose):
        '''
        transform camera space pc to world space pc
        '''
        trans = campose[:3, 3:]
        rot = campose[:3, :3]

        cam_pts = rgb_pts[:, :3]
        world_pc = np.dot(rot, cam_pts.transpose()) + trans
        world_pc = world_pc.transpose()

        rgb_world = np.concatenate((world_pc, rgb_pts[:, 3:]), axis=-1)

        return rgb_world