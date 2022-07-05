import sys
import os
import cv2
import numpy as np

from torch.utils.data import Dataset

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

class Office_dataset(Dataset):
    def __init__(self, base_dir, split='infer'):
        self.split = split
        self.data_dir = base_dir
        self.scenes = [f for f in os.listdir(os.path.abspath(self.data_dir))]
        self.scenes.sort()
        self.imgs = []
        for scene in self.scenes:
            scene_path = os.path.join(self.data_dir, scene, 'rgb')
            scene_imgs = [os.path.join(scene_path, img) for img in os.listdir(scene_path)]
            scene_imgs.sort()
            self.imgs += scene_imgs
        self.mask_person = False
        self.resize_img = True

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        img_dict = dict()

        img_path = self.imgs[idx]
        seq_path = img_path[:img_path.find('rgb')]
        img_name = img_path[img_path.find('rgb')+4:]
        depth_path = os.path.join(seq_path, 'depth', img_name)
        densepose_path = os.path.join(seq_path, 'denseposes', img_name)

        # RGB
        rgb_img = self.load_rgb(img_path, fmt='bgr') #todo needs loading as bgr
        rgb_img_fs = rgb_img

        # Depth
        depth_img = self.load_depth(depth_path)
        depth_img_fs = depth_img

        # Densepose
        densepose_mask = self.load_depth(densepose_path)
        bin_mask = (densepose_mask == 0.0).astype(int) # BG 1, Person 0
        bin_mask = np.expand_dims(bin_mask, axis=-1).repeat(3, axis=-1)

        if self.mask_person:
            rgb_img *= bin_mask
            #rgb_img[rgb_img == 0] = 255

        if self.resize_img:
            rgb_img = cv2.resize(rgb_img, dsize=(320, 240), interpolation=cv2.INTER_LINEAR)
            depth_img = cv2.resize(depth_img, dsize=(320, 240), interpolation=cv2.INTER_LINEAR)

        # Camera calibration
        calibration = os.path.join(seq_path, 'calibration.txt')
        with open(calibration) as f:
            tmp = f.readlines()

        calibration_list = tmp[0].split()
        fx, fy = float(calibration_list[0]), float(calibration_list[1])
        cx, cy = float(calibration_list[2]), float(calibration_list[3])
        camera_intrinsics_fs = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]) #fullsized
        if self.resize_img:
            fx *= 0.5
            fy *= 0.5
            cx *= 0.5
            cy *= 0.5
        camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Output
        img_dict['seq_id'] = seq_path
        img_dict['img_id'] = img_name
        img_dict['rgb'] = rgb_img
        img_dict['rgb_fs'] = rgb_img_fs
        img_dict['depth'] = depth_img
        img_dict['depth_fs'] = depth_img_fs
        img_dict['densepose'] = densepose_mask
        img_dict['camera_intrinsics'] = camera_intrinsics
        img_dict['camera_intrinsics_fs'] = camera_intrinsics_fs

        return img_dict

    def load_rgb(self, rgb_path, fmt='bgr'):
        '''
        Loads a rgb image from a png file
        Detectron uses BGR!
        '''
        bgr_img = cv2.imread(rgb_path)
        if fmt == 'rgb':
            rgb_img = bgr_img[:, :, ::-1]
        elif fmt == 'bgr':
            rgb_img = bgr_img
        rgb_img = np.array(rgb_img, dtype=np.float32)

        return rgb_img

    def load_depth(self, depth_path):
        '''
        Loads a depth image or a densepose image from a png file
        '''
        depth_img = cv2.imread(depth_path)
        depth_img = np.array(depth_img[:,:,0], dtype=np.float32) #all channels equal only use first

        return depth_img