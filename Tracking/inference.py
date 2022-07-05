from __future__ import absolute_import, division, print_function

from options import Options
import os
import argparse


# the directory that options.py resides in
file_dir = os.path.dirname(__file__)

options = Options()
opts = options.parse()

if opts.use_graph:
    from mpn_trainer import Trainer
else:
    from trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.inference(vis_pose=False, classwise=True)
