import os, sys
sys.path.append(os.path.abspath('./networks'))
from networks.flownet3d import FlowNet3D
from networks.flowstep3d import FlowStep3D
from networks.recurrent_rigid_scene_flow import SequenceWeights
from networks.pointpwc import PointConvSceneFlowPWC8192selfglobalPointConv as PointPWC
def get_model(mode):
    if mode == 'FlowNet3D':
        return FlowNet3D
    elif mode == 'FlowStep3D':
        return FlowStep3D
    elif mode == 'SequenceWeights':
        return SequenceWeights
    elif mode == 'PointPWC':
        return PointPWC
    else:
        raise ValueError('Mode {} not found.'.format(mode))
