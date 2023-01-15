import os
from easydict import EasyDict

CONF = EasyDict()
CONF.PATH = EasyDict()

# Base Folder
CONF.PATH.BASE = os.path.abspath(os.path.dirname(__file__)) #Base Graph3DMOT path
CONF.PATH.BPROC = os.path.join(CONF.PATH.BASE, "BlenderProc")
CONF.PATH.DETECT = os.path.join(CONF.PATH.BASE, "Detection")
CONF.PATH.PROJ = os.path.join(CONF.PATH.BASE, "Detection/pose")
CONF.PATH.TRACK = os.path.join(CONF.PATH.BASE, 'Tracking')

# Front Data Generation
CONF.PATH.FRONTDATA = os.path.join(CONF.PATH.BPROC, "resources/front_3D")
CONF.PATH.FRONT3D = os.path.join(CONF.PATH.FRONTDATA, "3D-FRONT")
CONF.PATH.FUTURE3D = os.path.join(CONF.PATH.FRONTDATA, "3D-FUTURE-model")
CONF.PATH.FRONTTEXT = os.path.join(CONF.PATH.FRONTDATA, "3D-FRONT-texture")

# Detection

# MOTFront storage folder
#CONF.PATH.DETECTDATA = os.path.join('/home/dominik/Schreibtisch/Graph3DMOT/Detection', 'front_dataset/')
#CONF.PATH.DETECTDATA = os.path.join(CONF.PATH.DETECT, 'front_dataset/')
CONF.PATH.DETECTDATA = '/mnt/raid/schmauser/'
CONF.PATH.DETECTTRAIN = os.path.join(CONF.PATH.DETECTDATA, 'train')
CONF.PATH.DETECTVAL = os.path.join(CONF.PATH.DETECTDATA, 'val')
CONF.PATH.DETECTTEST = os.path.join(CONF.PATH.DETECTDATA, 'test')
CONF.PATH.DETECTVIS = os.path.join(CONF.PATH.DETECTDATA, 'vis')
CONF.PATH.VOXELDATA = os.path.join(os.path.join(CONF.PATH.DETECT, 'front_dataset/'), 'voxel')
#CONF.PATH.VOXELDATA = os.path.join(CONF.PATH.DETECTDATA, 'voxel') # storage for binvox model folder
# Pretrained Detection network folder
CONF.PATH.DETECTMODEL = os.path.join(CONF.PATH.DETECT, 'model/')

# Projection (for debugging)
CONF.PATH.PROJDATA = os.path.join(CONF.PATH.PROJ, 'data')

# Tracking (Data folder for seperate Tracking pipeline training)
CONF.PATH.TRACKDATA = os.path.join(CONF.PATH.DETECT, 'predicted_data')

# Outputs/ Logging
CONF.PATH.DETECTOUTPUT = os.path.join(CONF.PATH.DETECT, 'outputs')
CONF.PATH.BPROCOUTPUT = os.path.join(CONF.PATH.BPROC, 'output')
CONF.PATH.TRACKOUTPUT = os.path.join(CONF.PATH.TRACK, 'output')

