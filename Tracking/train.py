from __future__ import absolute_import, division, print_function

from options import Options
import os, shutil,sys
import argparse

# the directory that options.py resides in
file_dir = os.path.dirname(__file__)

options = Options()
opts = options.parse()

if opts.use_graph:
    from mpn_trainer import Trainer
else:
    from trainer import Trainer

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF

if __name__ == "__main__":

    # Remove old files
    if os.path.exists(CONF.PATH.TRACKOUTPUT):
        print('Removing old outputs ...')
        shutil.rmtree(CONF.PATH.TRACKOUTPUT)
        os.mkdir(CONF.PATH.TRACKOUTPUT)

    trainer = Trainer(opts)
    if opts.precompute_feats:
        trainer.precompute()
    else:
        trainer.train()
