import sys, os
import os.path as osp
import numpy as np
import pptk
import open3d
import torch.utils.data as data
import MinkowskiEngine as ME

# root = '/data/1015323606/FlyingThing3D_subset/FlyingThings3D_subset_processed_35m'
# root = osp.join(root, 'train')
root = '/data/1015323606/FlyingThing3D_subset/FlyingThings3D_subset_processed_35m'
root = osp.join(root, 'train')
all_paths = sorted(os.listdir(root))

for i in range(len(all_paths)):
    path = osp.join(root, all_paths[i])

    pc1 = np.load(osp.join(path, 'pc1.npy')).astype(np.float32)
    pc2 = np.load(osp.join(path, 'pc2.npy')).astype(np.float32)
    pose = np.load(osp.join(path, 'trans.npy')).astype(np.float32)
    pose = np.reshape(pose, (4, 4))
    rot = pose[:3, :3]
    trans = np.expand_dims(pose[:3, 3], axis=-1)
    aligned_pc1 = rot @ pc1.transpose([1,0]) + trans
    aligned_pc1 = aligned_pc1.transpose([1,0])
    ego_flow = aligned_pc1-pc1
    flow = pc2-pc1

    sel1 = ME.utils.sparse_quantize(np.ascontiguousarray(pc1) / 0.1, return_index=True)
    sel2 = ME.utils.sparse_quantize(np.ascontiguousarray(pc2) / 0.1, return_index=True)

    print(sel1.shape)

########Visualization#######################
# visual_pos1 = open3d.geometry.PointCloud()
# visual_pos2 = open3d.geometry.PointCloud()
# visual_pos3 = open3d.geometry.PointCloud()
# aligned_pos1 = gt_rots @ pos1 + gt_trans
# visual_pos1.points = open3d.utility.Vector3dVector(pc1)
# visual_pos2.points = open3d.utility.Vector3dVector(pc2)
# visual_pos3.points = open3d.utility.Vector3dVector(aligned_pc1)
# visual_pos1.paint_uniform_color([0.0, 0.0, 1.0])
# visual_pos2.paint_uniform_color([1.0, 0.0, 0.0])
# visual_pos3.paint_uniform_color([0.0, 1.0, 0.0])
# open3d.visualization.draw_geometries([visual_pos1, visual_pos2, visual_pos3])
########Visualization#######################


# multiply -1 only for subset datasets
# pc1[..., -1] *= -1
# pc1[..., 0] *= -1
# pc2[..., 0] *= -1
# pc2[..., -1] *= -1
# aligned_pc1[..., -1] *= -1
# aligned_pc1[..., 0] *= -1

