import sys, os
import os.path as osp
import numpy as np
import pptk
import open3d
import torch.utils.data as data
import MinkowskiEngine as ME

# root = '/data/1015323606/FlyingThing3D_subset/FlyingThings3D_subset_processed_35m'
# root = osp.join(root, 'train')
root = '/data/1015323606/Weekly_Supervised_Scene_Flow/lidar_kitti/'
# number_str = '000007'
# root = osp.join(root, number_str)
all_paths = sorted(os.listdir(root))

for i in range(len(all_paths)):
    path = osp.join(root, all_paths[i])

    data = np.load(path)
    positions1 = data['pc1']
    positions2 = data['pc2']
    pose = data['pose_s']

    near_mask_1 = positions1[:, 2] < 35.0
    near_mask_2 = positions2[:, 2] < 35.0

    positions1 = positions1[near_mask_1, :]
    positions2 = positions2[near_mask_2, :]

    #
    # is_not_ground_s = (positions1[:, 1] > -0.0)
    # is_not_ground_t = (positions2[:, 1] > -0.0)
    #
    # positions1 = positions1[is_not_ground_s,:]
    # positions2 = positions2[is_not_ground_t,:]

    # pose = np.load(osp.join(path, 'trans.npy')).astype(np.float32)
    # pose = np.reshape(pose, (4, 4))
    # rot = pose[:3, :3]
    # trans = np.expand_dims(pose[:3, 3], axis=-1)
    # aligned_pc1 = rot @ pc1.transpose([1,0]) + trans
    # aligned_pc1 = aligned_pc1.transpose([1,0])
    # ego_flow = aligned_pc1-pc1
    # flow = pc2-pc1

    sel1 = ME.utils.sparse_quantize(np.ascontiguousarray(positions1) / 0.1, return_index=True)
    sel2 = ME.utils.sparse_quantize(np.ascontiguousarray(positions2) / 0.1, return_index=True)

    print(sel1.shape)


