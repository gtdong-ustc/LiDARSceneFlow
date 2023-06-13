import sys, os
import os.path as osp
import numpy as np
import pptk
import open3d

import torch.utils.data as data

__all__ = ['FlyingThings3DSubset']


class FlyingThings3DSubset(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        args:
    """
    def __init__(self,
                 train,
                 transform,
                 num_points,
                 data_root,
                 full = True):
        self.root = osp.join(data_root, 'FlyingThings3D_subset_processed_35m')
        self.train = train
        self.transform = transform
        self.num_points = num_points

        self.samples = self.make_dataset(full)

        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pc1_loaded, pc2_loaded, aligned_pc1_loaded, rot_loaded, trans_loaded = self.pc_loader(self.samples[index])
        #import ipdb; ipdb.set_trace()
        #fn = '/media/wenxuan/Large/Code2019_9/HPLFlowNet/data_preprocess/FlyingThings3D_subset_processed_35m/train/0004781'
        #pc1_loaded, pc2_loaded = self.pc_loader(fn)
        pc1_transformed, pc2_transformed, aligned_pc1_transformed, sf_transformed, rf_tranformed = self.transform([pc1_loaded, pc2_loaded, aligned_pc1_loaded])
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        #pc1_norm = pptk.estimate_normals(pc1_transformed, k = 16, r = np.inf, verbose = False)
        #pc2_norm = pptk.estimate_normals(pc2_transformed, k = 16, r = np.inf, verbose = False)
        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed
        aligned_pc1_norm = aligned_pc1_transformed
        return pc1_transformed, pc2_transformed, aligned_pc1_transformed, pc1_norm, pc2_norm, aligned_pc1_norm,\
               sf_transformed, rf_tranformed, rot_loaded, trans_loaded, self.samples[index] ##rf: residual scene flow

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def make_dataset(self, full):

        print('full {}'.format(full))

        root = osp.realpath(osp.expanduser(self.root))

        root = osp.join(root, 'train') if self.train else osp.join(root, 'val')

        print(root)

        all_paths = sorted(os.listdir(root))

        print('all_paths {}'.format(len(all_paths)))

        useful_paths = [osp.join(root, item) for item in all_paths]

        try:
            if self.train:
                assert (len(useful_paths) <= 19640)
            else:
                assert (len(useful_paths) <= 3824)
                # assert (len(useful_paths) == 19640)
        except AssertionError:
            print('len(useful_paths) assert error', len(useful_paths))
            sys.exit(1)

        if not full:
            res_paths = useful_paths[::4]
        else:
            res_paths = useful_paths

        if not self.train:
            res_paths = useful_paths[::10]
        else:
            res_paths = useful_paths

        print(len(res_paths))

        # print(res_paths[0])
        # print(res_paths[-1])

        return res_paths

    def pc_loader(self, path):
        """
        Args:
            path: path to a dir, e.g., home/xiuye/share/data/Driving_processed/35mm_focallength/scene_forwards/slow/0791
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        pc1 = np.load(osp.join(path, 'pc1.npy')).astype(np.float32)
        pc2 = np.load(osp.join(path, 'pc2.npy')).astype(np.float32)
        pose = np.load(osp.join(path, 'trans.npy')).astype(np.float32)
        pose = np.reshape(pose, (4, 4))

        rot = pose[:3, :3]
        trans = np.expand_dims(pose[:3, 3], axis=-1)
        aligned_pc1 = rot @ pc1.transpose([1,0]) + trans
        aligned_pc1 = aligned_pc1.transpose([1,0])

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
        pc1[..., -1] *= -1
        pc1[..., 0] *= -1
        pc2[..., 0] *= -1
        pc2[..., -1] *= -1
        aligned_pc1[..., -1] *= -1
        aligned_pc1[..., 0] *= -1

        return pc1, pc2, aligned_pc1, rot, trans
