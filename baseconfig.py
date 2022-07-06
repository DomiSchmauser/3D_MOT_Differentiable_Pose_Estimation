import os
from easydict import EasyDict

CONF = EasyDict()
CONF.PATH = EasyDict()

# Base Folder
CONF.PATH.BASE = os.path.abspath(os.path.dirname(__file__)) #Base Graph3DMOT path
CONF.PATH.BPROC = os.path.join(CONF.PATH.BASE, "BlenderProc")
CONF.PATH.DETECT = os.path.join(CONF.PATH.BASE, "Detection")
CONF.PATH.PROJ = os.path.join(CONF.PATH.BASE, "PoseEst")
CONF.PATH.TRACK = os.path.join(CONF.PATH.BASE, 'Tracking')

# Front Data
CONF.PATH.FRONTDATA = os.path.join(CONF.PATH.BPROC, "resources/front_3D")
CONF.PATH.FRONT3D = os.path.join(CONF.PATH.FRONTDATA, "3D-FRONT")
CONF.PATH.FUTURE3D = os.path.join(CONF.PATH.FRONTDATA, "3D-FUTURE-model")
CONF.PATH.FRONTTEXT = os.path.join(CONF.PATH.FRONTDATA, "3D-FRONT-texture")

# Detection
CONF.PATH.DETECTDATA = os.path.join(CONF.PATH.DETECT, 'front_dataset/')
CONF.PATH.DETECTTRAIN = os.path.join(CONF.PATH.DETECTDATA, 'train')
CONF.PATH.DETECTVAL = os.path.join(CONF.PATH.DETECTDATA, 'val')
CONF.PATH.DETECTTEST = os.path.join(CONF.PATH.DETECTDATA, 'test')
CONF.PATH.DETECTVIS = os.path.join(CONF.PATH.DETECTDATA, 'vis')
CONF.PATH.DETECTMODEL = os.path.join(CONF.PATH.DETECT, 'model/')

# Projection
CONF.PATH.PROJDATA = os.path.join(CONF.PATH.PROJ, 'data')

# Tracking
CONF.PATH.TRACKDATA = os.path.join(CONF.PATH.DETECT, 'predicted_data')

# Outputs/ Logging
CONF.PATH.DETECTOUTPUT = os.path.join(CONF.PATH.DETECT, 'outputs')
CONF.PATH.BPROCOUTPUT = os.path.join(CONF.PATH.BPROC, 'output')
CONF.PATH.UPOUTPUT = os.path.join(CONF.PATH.UPLIFT, 'output')
CONF.PATH.TRACKOUTPUT = os.path.join(CONF.PATH.TRACK, 'output')

