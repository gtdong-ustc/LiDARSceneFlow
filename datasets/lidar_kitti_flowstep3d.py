import sys, os
import os.path as osp
import numpy as np

import torch.utils.data as data
import MinkowskiEngine as ME

__all__ = ['LidarKITTI']


class LidarKITTI(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        gen_func (callable):
        args:
    """

    def __init__(self,
                 train,
                 transform,
                 num_points,
                 data_root,
                 remove_ground = True,
                 full=True):
        # self.root = osp.join(data_root, 'lidar_kitti_with_ground')
        self.root = osp.join(data_root, 'lidar_kitti_Before')
        #assert train is False
        self.train = train
        self.transform = transform
        self.num_points = num_points
        self.remove_ground = remove_ground

        self.samples = self.make_dataset()
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pc1_loaded, pc2_loaded, aligned_pc1_loaded, sf_loaded, rot_loaded, trans_loaded, label1_loaded, label2_loaded = self.pc_loader(self.samples[index])

        pc1_transformed, \
        pc2_transformed, \
        aligned_pc1_transformed, \
        sf_transformed, \
        label1_transformed, \
        label2_transformed, \
        fg_labels1_transformed, \
        fg_labels2_transformed = self.transform(
            [pc1_loaded, pc2_loaded, aligned_pc1_loaded, sf_loaded, label1_loaded, label2_loaded])
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed
        aligned_pc1_norm = aligned_pc1_transformed
        return pc1_transformed, pc2_transformed, aligned_pc1_transformed, \
               pc1_norm, pc2_norm, aligned_pc1_norm, \
               sf_transformed, rot_loaded, trans_loaded, \
               label1_transformed, label2_transformed, \
               fg_labels1_transformed, fg_labels2_transformed, self.samples[index]  ##rf: residual scene flow

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is removing ground: {}\n'.format(self.remove_ground)
        fmt_str += '    is training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str

    def make_dataset(self):
        root = osp.realpath(osp.expanduser(self.root))
        all_paths = sorted(os.listdir(root))
        useful_paths = [os.path.join(root, item) for item in all_paths]

        return useful_paths

    def pc_loader(self, path):
        """
        Args:
            path:
        Returns:
            pc1: ndarray (N1, 3) np.float32
            pc2: ndarray (N2, 3) np.float32
            gt: ndarray (N1, 3) np.float32
        """
        data = np.load(path)

        pc1 = data['pc1'].astype(np.float32)
        pc2 = data['pc2'].astype(np.float32)
        flow = data['flow'].astype(np.float32)
        rel_trans = data['pose_s'].astype(np.float32)

        seg_label_1 = data['sem_label_s']
        seg_label_2 = data['sem_label_t']
        # mot_label_1 = np.zeros((seg_label_1.shape[0])).astype(np.float32)
        # mot_label_1[((seg_label_1 < 40) | (seg_label_1 > 99))] = 1.0
        # mot_label_1 = np.expand_dims(mot_label_1, axis=1)
        # mot_label_2 = np.ones(pc2.shape).astype(np.float32)

        mot_label_1 = np.expand_dims(seg_label_1, axis=1)
        mot_label_2 = np.expand_dims(seg_label_2, axis=1)

        rot = rel_trans[:3, :3]
        trans = np.expand_dims(rel_trans[:3, 3], axis=-1)

        aligned_pc1 = rot @ pc1.transpose([1, 0]) + trans
        aligned_pc1 = aligned_pc1.transpose([1, 0])

        return pc1, pc2, aligned_pc1, flow, rot, trans, mot_label_1, mot_label_2
