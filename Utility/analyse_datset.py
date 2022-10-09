import os, json, cv2, csv, sys
import shutil
sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

def get_dataset_info(img_path, combined=False):

    mapping_file = os.path.join(img_path[:-6], "3D_front_mapping.csv")
    _, csv_dict = read_csv_mapping(mapping_file)
    mapping_list, name_list = [], []

    folders = os.listdir(img_path)
    bad_folder = []
    img_count = 0

    for folder in folders:

        json_file = os.path.join(img_path, folder, "coco_data/coco_annotations.json")

        with open(json_file) as f:
            imgs_anns = json.load(f)

        for idx, v in enumerate(imgs_anns['images']):
            img_count += 1
            for anno in imgs_anns['annotations']:
                if anno['image_id'] == v['id']:
                    cat_id = anno['category_id']
                    try:
                        name = csv_dict[cat_id]
                    except:
                        bad_folder.append(folder)
                    if not name in name_list:
                        name_list.append(name)

                    if cat_id in mapping_list:
                        pass
                    else:
                        mapping_list.append(cat_id)
    for l in list(set(bad_folder)):
        print("remove folder", os.path.join(CONF.PATH.DETECTTRAIN, l))
        #shutil.rmtree(os.path.join(CONF.PATH.DETECTTRAIN, l), ignore_errors=True)

    if combined:
        return mapping_list, name_list, img_count
    else:
        return mapping_list, name_list

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
