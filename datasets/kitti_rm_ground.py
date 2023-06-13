import sys, os
import os.path as osp
import numpy as np

import torch.utils.data as data

__all__ = ['KITTI']


class KITTI(data.Dataset):
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
        self.root = osp.join(data_root, 'kitti_rm_ground/train')
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
        pc1_loaded, pc2_loaded, sf_loaded = self.pc_loader(self.samples[index])
        pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded, sf_loaded])
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed
        return pc1_transformed, pc2_transformed, pc1_norm, pc2_norm, sf_transformed, self.samples[index]

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
        # do_mapping = True
        root = osp.realpath(osp.expanduser(self.root))
        # root = osp.join(root, 'train') if self.train else osp.join(root, 'val')

        all_paths = sorted(os.walk(root))
        # print(all_paths)
        useful_paths = [os.path.join(self.root, item) for item in all_paths[0][2]]
        # print(useful_paths)
        # try:
        #     assert (len(useful_paths) == 200)
        # except AssertionError:
        #     print('assert (len(useful_paths) == 200) failed!', len(useful_paths))
        #
        # if do_mapping:
        #     mapping_path = osp.join(osp.dirname(__file__), 'KITTI_mapping.txt')
        #     print('mapping_path', mapping_path)
        #
        #     with open(mapping_path) as fd:
        #         lines = fd.readlines()
        #         lines = [line.strip() for line in lines]
        #     useful_paths = [path for path in useful_paths if lines[int(osp.split(path)[-1])] != '']

        res_paths = useful_paths

        return res_paths

    def pc_loader(self, path):
        """
        Args:
            path:
        Returns:
            pc1: ndarray (N1, 3) np.float32
            pc2: ndarray (N2, 3) np.float32
            gt: ndarray (N1, 3) np.float32
        """
        # print(path)
        scan = np.load(path)
        gt = scan['gt'].astype(np.float32)
        pc1 = scan['pos1'].astype(np.float32)
        pc2 = scan['pos2'].astype(np.float32)

        return pc1, pc2, gt
