# import some common libraries
import torch
import numpy as np
import os, json, cv2, random, csv, pickle, sys
import h5py
from pycocotools.coco import COCO
import matplotlib.pyplot as plt


# import some common detectron2 utilities
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.structures import polygons_to_bitmask
from detectron2.utils.visualizer import GenericMask

from PIL import Image

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

from BlenderProc.utils import binvox_rw
from Detection.utils.train_utils import get_voxel


# Define directory to images
IMG_DIR = CONF.PATH.DETECTDATA

# custom dataset registration
class RegisterDataset:

    def __init__(self, mapping_list, name_list, img_dir=IMG_DIR):
        self.img_dir = img_dir
        self.mapping_list = list(mapping_list)
        self.name_list = list(name_list)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_front_dicts(self, img_path):

        mapping_file = os.path.join(self.img_dir, "3D_front_mapping.csv")
        _, csv_dict = self.read_csv_mapping(mapping_file)

        folders = os.listdir(img_path)

        dataset_dicts = []
        for folder in folders:

            json_file = os.path.join(img_path, folder, "coco_data/coco_annotations.json")

            with open(json_file) as f:
                imgs_anns = json.load(f)

            camposes = []
            all_objs = []
            for idx, v in enumerate(imgs_anns['images']):

                record = {}

                filename = os.path.join(img_path, folder, 'coco_data', v["file_name"])
                depth_name = os.path.join(img_path, folder, str(idx) + '.hdf5')

                record["file_name"] = filename
                record["image_id"] = str(v['id']) + '_' + folder[:8]
                record["height"] = v['height']
                record["width"] = v['width']
                # record["nocs_map"] = self.get_nocs(v["file_name"], img_path, folder)
                #record["depth_map"], record['campose'] = self.load_hdf5(depth_name)
                record["nocs_map"] = filename.replace('rgb', 'nocs')
                record["depth_map"] = depth_name
                record["campose"] = self.load_campose(depth_name)

                depth = []
                objs = []
                voxels = []
                boxes = []
                segmap_store = []
                category = []
                object_ids = []
                gt_rotations = []
                gt_locations = []
                gt_3dbox = []
                gt_scales = []

                for anno in imgs_anns['annotations']:
                    if anno['image_id'] == v['id']:
                        cat_id = anno['category_id']
                        object_id = anno['id']
                        jid = anno['jid']
                        scale = np.array(anno['3Dscale'])

                        #voxel = os.path.join(CONF.PATH.FUTURE3D, jid, 'model.binvox')
                        voxel = os.path.join(CONF.PATH.VOXELDATA, jid, 'model.binvox')
                        name = csv_dict[cat_id]

                        #nocs_obj = self.crop_segmask(record["nocs_map"], anno['bbox'], anno['segmentation'])
                        #depth_obj = self.crop_depth(record["depth_map"], anno['bbox'], anno['segmentation'])

                        if not name in self.name_list:
                            self.name_list.append(name)

                        if cat_id in self.mapping_list:
                            id = self.mapping_list.index(cat_id)
                        else:
                            self.mapping_list.append(cat_id)
                            id = self.mapping_list.index(cat_id)

                        obj = {
                            "bbox": anno['bbox'],
                            "bbox_mode": BoxMode.XYWH_ABS,
                            "segmentation": anno['segmentation'],
                            "category_id": id,
                            "voxel": voxel,
                            "scale": scale,
                            "jid": jid,
                            "id": object_id,
                        }
                        objs.append(obj)
                        segmap_store.append(anno['segmentation'])
                        voxels.append(voxel)
                        category.append(id)
                        boxes.append(anno['bbox'])
                        #depth.append(depth_obj)
                        object_ids.append(object_id)
                        gt_rotations.append(anno['3Drot'])
                        anno_3dloc = self.add_halfheight(anno['3Dloc'].copy(), anno['3Dbbox'])
                        gt_locations.append(anno_3dloc)
                        gt_3dbox.append(np.array(anno['3Dbbox']))
                        gt_scales.append(scale)

                record['cat_id'] = category # starts at 0
                record['vox'] = voxels
                record['segmap'] = segmap_store
                record['boxes'] = boxes
                record["annotations"] = objs
                record['object_id'] = object_ids
                record['rotations'] = gt_rotations
                record['locations'] = gt_locations
                record['3dboxes'] = gt_3dbox
                record['3dscales'] = gt_scales
                #all_objs.append(objs)
                #camposes.append(record['campose'])
                dataset_dicts.append(record)

        '''
        with open('optimization.pickle', 'wb') as handle:
            all_objs.append(camposes)
            pickle.dump(all_objs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        sys.exit()
        '''

        return dataset_dicts

    def get_eval_dicts(self, img_path):

        mapping_file = os.path.join(self.img_dir, "3D_front_mapping.csv")
        _, csv_dict = self.read_csv_mapping(mapping_file)

        folders = os.listdir(img_path)

        dataset_dicts = []
        for folder in folders:

            json_file = os.path.join(img_path, folder, "coco_data/coco_annotations.json")

            with open(json_file) as f:
                imgs_anns = json.load(f)

            for idx, v in enumerate(imgs_anns['images']):
                if idx == 0:
                    record = {}

                    filename = os.path.join(img_path, folder, 'coco_data', v["file_name"])

                    record["file_name"] = filename
                    record["image_id"] = str(v['id']) + '_' + folder[:8]
                    record["height"] = v['height']
                    record["width"] = v['width']
                    record["nocs_map"] = self.get_nocs(v["file_name"], img_path, folder)

                    objs = []
                    for anno in imgs_anns['annotations']:
                        if anno['image_id'] == v['id']:
                            jid = anno['jid']
                            voxel = get_voxel(os.path.join(CONF.PATH.VOXELDATA, jid, 'model.binvox'), np.array(anno['3Dscale']))
                            cat_id = anno['category_id']
                            name = csv_dict[cat_id]
                            nocs_obj = self.crop_segmask(record["nocs_map"], anno['bbox'], anno['segmentation'])
                            if not name in self.name_list:
                                self.name_list.append(name)

                            if cat_id in self.mapping_list:
                                id = self.mapping_list.index(cat_id)
                            else:
                                self.mapping_list.append(cat_id)
                                id = self.mapping_list.index(cat_id)

                            obj = {
                                "bbox": anno['bbox'],
                                "bbox_mode": BoxMode.XYWH_ABS,
                                "segmentation": anno['segmentation'],
                                "category_id": id,
                                "voxel": voxel,
                                "nocs": nocs_obj,
                            }
                            objs.append(obj)

                    record["annotations"] = objs
                    dataset_dicts.append(record)

        return dataset_dicts

    # register train and val dataset
    def reg_dset(self):
        for d in ["train", "val", "test"]:
            DatasetCatalog.register("front_" + d, lambda d=d: self.get_front_dicts(self.img_dir + d))
            MetadataCatalog.get("front_" + d).set(thing_classes=self.name_list)
        print("Registered Dataset")

    # data mean, std
    def calculate_mean_std(self):
        dataset_dicts = self.get_front_dicts(os.path.join(self.img_dir, 'train'))

        data_mean = np.zeros((1, 3))
        data_std = np.zeros((1, 3))
        data_len = len(dataset_dicts)

        for idx, d in enumerate(dataset_dicts):
            img = cv2.imread(d["file_name"])
            data_mean = data_mean + np.mean(img, axis=(0, 1)) / data_len
            data_std = data_std + np.std(img, axis=(0, 1)) / data_len
            print("data mean", data_mean)
        return data_mean, data_std

    # visualize annotations
    def vis_annotation(self, num_imgs=1):
        front_metadata = MetadataCatalog.get("front_train")
        dataset_dicts = self.get_front_dicts(os.path.join(self.img_dir, 'train'))

        for d in random.sample(dataset_dicts, num_imgs):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=front_metadata, scale=1)
            out = visualizer.draw_dataset_dict(d)
            cv2.imshow('image', out.get_image()[:, :, ::-1])
            cv2.waitKey(500)

    # evaluate annotations
    def eval_annotation(self):
        front_metadata = MetadataCatalog.get("front_train")
        dataset_dicts = self.get_eval_dicts(os.path.join(self.img_dir, 'train'))

        for idx, d in enumerate(dataset_dicts):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=front_metadata, scale=1)
            out = visualizer.draw_dataset_dict(d)
            cv2.imshow('image', out.get_image()[:, :, ::-1])
            print("image id: ", idx, " image name: ", d["file_name"])
            cv2.waitKey(0)

    @staticmethod
    def read_csv_mapping(path):
        """ Loads an idset mapping from a csv file, assuming the rows are sorted by their ids.
        :param path: Path to csv file
        """

        with open(path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            new_id_label_map = []
            new_label_id_map = {}

            for row in reader:
                new_id_label_map.append(row["name"])
                new_label_id_map[int(row["id"])] = row["name"]

            return new_id_label_map, new_label_id_map

    @staticmethod
    def write_pickle(img_dir, filename, pickle_data):

        filepath = os.path.join(img_dir, filename + ".pkl")
        print("PATH",filepath)
        if 'train' in img_dir:
            print('intrain')
            with open(filepath, 'wb') as f:
                pickle.dump(pickle_data, f)

    @staticmethod
    def load_pickle(self,img_dir, filename):

        if 'val' in img_dir:
            filepath = os.path.join(img_dir[:-4],'train', filename + ".pkl")
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            return data[0], data[1]
        else:
            return [],[]

    @staticmethod
    def get_nocs(filename, img_path, folder):
        nocs_name =  filename.replace('rgb', 'nocs')
        nocs_path = os.path.join(img_path, folder, 'coco_data', nocs_name)
        nocs = cv2.imread(nocs_path) #BGRA
        nocs = nocs[:,:,:3]
        nocs = nocs[:, :, ::-1] # RGB

        nocs = np.array(nocs, dtype=np.float32) / 255

        return nocs

    @staticmethod
    def crop_segmask(nocs_img, bbox, segmap):

        abs_bbox = torch.tensor(BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS), dtype=torch.float32)
        # width = torch.abs(abs_bbox[2] - abs_bbox[0])
        # height = torch.abs(abs_bbox[3] - abs_bbox[1])

        gm = GenericMask(segmap, 240, 320)
        bin_mask = gm.polygons_to_mask(segmap)
        binary_mask = bin_mask[:,:, None]
        crop_im = np.multiply(nocs_img,binary_mask)
        cropped_im = np.array(crop_im[int(abs_bbox[1]):int(abs_bbox[3]),int(abs_bbox[0]):int(abs_bbox[2]),:])
        # cropped_im = np.clip(cropped_im, 0, 1)

        cropped_im[cropped_im == 0] = 1

        return torch.from_numpy(cropped_im).to(torch.float32)

    @staticmethod
    def load_campose(path):

        with h5py.File(path, 'r') as data:
            for key in data.keys():
                if key == 'campose':
                    campose = np.array(data[key])

        return campose

    @staticmethod
    def load_hdf5(path):

        with h5py.File(path, 'r') as data:
            for key in data.keys():
                if key == 'depth':
                    depth = np.array(data[key])
                elif key == 'campose':
                    campose = np.array(data[key])

        return depth, campose

    @staticmethod
    def crop_depth(depth_img, bbox, segmap):

        abs_bbox = torch.tensor(BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS), dtype=torch.float32)

        gm = GenericMask(segmap, 240, 320)
        bin_mask = gm.polygons_to_mask(segmap)
        binary_mask = bin_mask[:, :]
        crop_im = np.multiply(depth_img, binary_mask)
        #crop_im[crop_im == 0] = 255
        cropped_im = np.array(crop_im[int(abs_bbox[1]):int(abs_bbox[3]),int(abs_bbox[0]):int(abs_bbox[2])])

        return torch.from_numpy(cropped_im).to(torch.float32)

    @staticmethod
    def add_halfheight(location, box):
        '''
        Object location z-center is at bottom, calculate half height of the object
        and add to shift z-center to correct location
        '''
        z_coords = []
        for pt in box:
            z = pt[-1]
            z_coords.append(z)
        z_coords = np.array(z_coords)
        half_height = np.abs(z_coords.max() - z_coords.min()) / 2
        location[-1] = half_height  # Center location is at bottom object

        return location