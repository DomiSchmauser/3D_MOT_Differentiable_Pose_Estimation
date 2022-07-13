
import os, sys
import argparse

# the directory that options.py resides in
file_dir = os.path.dirname(__file__)

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Tracking options")

        # PATHS
        self.parser.add_argument("--base_dir",
                                 type=str,
                                 help="path to the training data",
                                 default=CONF.PATH.TRACKDATA)
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=CONF.PATH.TRACKOUTPUT)

        # Network
        self.parser.add_argument("--use_graph",
                                 type=bool,
                                 help="Use Graph Neural Network for edge classification",
                                 default=True)
        self.parser.add_argument("--seq_len",
                                 type=int,
                                 help="Length of the input sequence",
                                 default=25)
        self.parser.add_argument("--no_pose",
                                 type=bool,
                                 help="Exclude pose for edge classification",
                                 default=False)
        self.parser.add_argument("--no_geo",
                                 type=bool,
                                 help="Exclude pose for edge classification",
                                 default=False)
        self.parser.add_argument("--rel_app",
                                 type=bool,
                                 help="Use a relative appearance feature for graph edges",
                                 default=False)
        self.parser.add_argument("--as_quaternion",
                                 type=bool,
                                 help="Use quaternion angles for rotation",
                                 default=False)
        self.parser.add_argument("--precompute_feats",
                                 type=bool,
                                 help="Precompute Siamese features and store as hdf5",
                                 default=False)


        # Model Parameters
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-3)
        self.parser.add_argument("--weight_decay", # L2 Regularization
                                 type=float,
                                 help="weight decay",
                                 default=1e-4) # 1e-4
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=100)
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=2)
        self.parser.add_argument("--use_augmentation",
                                 type=bool,
                                 help="use data augmentation",
                                 default=False)
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=0)
        self.parser.add_argument("--use_triplet",
                                 type=bool,
                                 help="Use triplet loss for edge classification",
                                 default=False)
        self.parser.add_argument("--use_l1",
                                 type=bool,
                                 help="Use l1 loss for edge classification",
                                 default=False)

        # Logging
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=15)

        self.parser.add_argument("--start_saving",
                                 type=int,
                                 help="epoch start to save weights",
                                 default=15)

        self.parser.add_argument("--start_saving_optimizer",
                                 type=int,
                                 help="epoch start to save weights",
                                 default=14)

        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=10)

        self.parser.add_argument("--save_model",
                                 type=bool,
                                 help="save model",
                                 default=True)

        self.parser.add_argument("--resume",
                                 type=bool,
                                 help="resume training",
                                 default=False)

        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="folder of pretrain model",
                                 default=os.path.join(file_dir, "model/pretrained"))

        self.parser.add_argument("--models_to_load",
                                 type=list,
                                 help="pretrained model to load",
                                 default=['edge_classifier', 'edge_encoder', 'voxel_encoder', 'graph_net'])

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
